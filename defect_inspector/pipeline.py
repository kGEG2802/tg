"""Оркестрація інспекції: запуск детекторів і агрегація результату."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import InspectionConfig
from .detectors import (
    EdgeChipDetector,
    ReferenceDetector,
    ScratchDetector,
    SpotDetector,
)
from .detectors.base import Defect
from .image_io import compute_roi_mask, load_image, preprocess


@dataclass
class InspectionResult:
    """Підсумок інспекції однієї деталі."""

    source: str
    passed: bool
    defects: List[Defect]
    image_size: tuple
    elapsed_ms: float
    reasons: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def max_severity(self) -> float:
        return max((d.severity for d in self.defects), default=0.0)

    def counts_by_kind(self) -> dict:
        out: dict = {}
        for d in self.defects:
            out[d.kind] = out.get(d.kind, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "verdict": self.verdict,
            "passed": self.passed,
            "image_size": {"w": self.image_size[0], "h": self.image_size[1]},
            "defect_count": len(self.defects),
            "max_severity": round(self.max_severity, 4),
            "counts_by_kind": self.counts_by_kind(),
            "reasons": self.reasons,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "defects": [d.to_dict() for d in self.defects],
        }


class Inspector:
    """Високорівневий фасад над набором детекторів.

    Приклад:
        >>> inspector = Inspector()
        >>> result = inspector.inspect("part.jpg", reference="golden.jpg")
        >>> print(result.verdict, len(result.defects))
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        cfg = self.config
        self._detectors = []
        if cfg.scratch.enabled:
            self._detectors.append(ScratchDetector(cfg.scratch))
        if cfg.spot.enabled:
            self._detectors.append(SpotDetector(cfg.spot))
        if cfg.edge.enabled:
            self._detectors.append(EdgeChipDetector(cfg.edge))
        if cfg.reference.enabled:
            self._detectors.append(ReferenceDetector(cfg.reference))

    # --------------------------------------------------------------- public
    def inspect(
        self,
        image: str | Path | np.ndarray,
        reference: Optional[str | Path | np.ndarray] = None,
    ) -> InspectionResult:
        """Проінспектувати одну деталь.

        Args:
            image: Шлях до зображення або вже завантажений BGR-масив.
            reference: Необов'язковий еталон для режиму порівняння.
        """
        start = time.perf_counter()
        source = str(image) if not isinstance(image, np.ndarray) else "<array>"
        bgr = image if isinstance(image, np.ndarray) else load_image(image)
        gray, _ = preprocess(bgr, self.config.preprocess)

        ref_gray = None
        if reference is not None:
            ref_bgr = reference if isinstance(reference, np.ndarray) else load_image(reference)
            ref_gray, _ = preprocess(ref_bgr, self.config.preprocess)

        roi = compute_roi_mask(gray)
        raw: List[Defect] = []
        for detector in self._detectors:
            raw.extend(detector.detect(gray, ref_gray, roi))

        defects = self._postprocess(self._filter_roi(raw, roi))
        passed, reasons = self._decide(defects, gray.shape)
        elapsed = (time.perf_counter() - start) * 1000.0

        return InspectionResult(
            source=source,
            passed=passed,
            defects=defects,
            image_size=(gray.shape[1], gray.shape[0]),
            elapsed_ms=elapsed,
            reasons=reasons,
        )

    # -------------------------------------------------------------- internal
    @staticmethod
    def _filter_roi(defects: List[Defect], roi: np.ndarray) -> List[Defect]:
        """Лишити поверхневі дефекти лише в межах деталі.

        ``edge_chip`` пропускається завжди (він за природою на межі деталі).
        """
        h, w = roi.shape
        kept: List[Defect] = []
        for d in defects:
            if d.kind == "edge_chip":
                kept.append(d)
                continue
            cx, cy = d.centroid
            if 0 <= cy < h and 0 <= cx < w and roi[cy, cx] > 0:
                kept.append(d)
        return kept

    def _postprocess(self, defects: List[Defect]) -> List[Defect]:
        """Відсіяти незначні дефекти та прибрати дублікати (NMS по IoU)."""
        min_sev = self.config.verdict.min_severity
        kept = [d for d in defects if d.severity >= min_sev]
        kept.sort(key=lambda d: d.severity, reverse=True)

        result: List[Defect] = []
        for d in kept:
            if any(d.iou(k) > 0.5 for k in result):
                continue
            result.append(d)
        return result

    def _decide(self, defects: List[Defect], shape) -> tuple[bool, List[str]]:
        """Застосувати правила вердикту; повернути ``(passed, reasons)``."""
        v = self.config.verdict
        reasons: List[str] = []

        critical = [d for d in defects if d.severity >= v.fail_severity]
        if critical:
            reasons.append(
                f"{len(critical)} критичних дефект(ів) із тяжкістю ≥ {v.fail_severity}"
            )
        if len(defects) > v.max_defect_count:
            reasons.append(
                f"кількість дефектів {len(defects)} > дозволених {v.max_defect_count}"
            )
        if v.max_area_ratio > 0:
            part_area = float(shape[0] * shape[1])
            ratio = sum(d.area for d in defects) / part_area if part_area else 0.0
            if ratio > v.max_area_ratio:
                reasons.append(
                    f"площа дефектів {ratio:.3f} > дозволеної {v.max_area_ratio}"
                )

        return (len(reasons) == 0), reasons
