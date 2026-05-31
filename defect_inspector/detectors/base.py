"""Базові абстракції для детекторів дефектів."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Defect:
    """Один виявлений дефект.

    Attributes:
        kind: Тип дефекту ("scratch", "spot", "edge_chip", "anomaly", ...).
        bbox: Обмежувальний прямокутник (x, y, w, h) у пікселях.
        severity: Тяжкість у діапазоні 0..1 (вища — критичніша).
        area: Площа дефекту в пікселях.
        detector: Ім'я детектора, що знайшов дефект.
        meta: Додаткові числові характеристики для звіту/відлагодження.
    """

    kind: str
    bbox: Tuple[int, int, int, int]
    severity: float
    area: float
    detector: str
    meta: dict = field(default_factory=dict)

    @property
    def centroid(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    def iou(self, other: "Defect") -> float:
        """Перетин-над-об'єднанням обмежувальних прямокутників (0..1)."""
        ax, ay, aw, ah = self.bbox
        bx, by, bw, bh = other.bbox
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "kind": self.kind,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]},
            "severity": round(float(self.severity), 4),
            "area": round(float(self.area), 2),
            "detector": self.detector,
            "meta": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                     for k, v in self.meta.items()},
        }


class Detector(ABC):
    """Спільний інтерфейс усіх детекторів.

    Реалізації приймають підготовлене **сіре** зображення (uint8) і
    повертають список :class:`Defect`. Детектор, що порівнює з еталоном,
    додатково використовує ``reference``.
    """

    #: Коротка машинна назва детектора.
    name: str = "detector"

    @abstractmethod
    def detect(
        self,
        gray: np.ndarray,
        reference: Optional[np.ndarray] = None,
        roi: Optional[np.ndarray] = None,
    ) -> List[Defect]:
        """Знайти дефекти на зображенні ``gray``.

        Args:
            gray: Підготовлене сіре зображення (uint8).
            reference: Еталон (лише для :class:`ReferenceDetector`).
            roi: Необов'язкова маска корисної площі деталі (255 — поверхня).
        """
        raise NotImplementedError


def clamp01(value: float) -> float:
    """Обмежити значення діапазоном [0, 1]."""
    return float(max(0.0, min(1.0, value)))
