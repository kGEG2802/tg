"""Формування звітів: анотоване зображення + JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .image_io import load_image, preprocess
from .config import PreprocessConfig
from .pipeline import InspectionResult

#: Кольори рамок (BGR) за типом дефекту.
KIND_COLORS = {
    "scratch": (0, 165, 255),    # помаранчевий
    "spot": (0, 0, 255),         # червоний
    "edge_chip": (255, 0, 255),  # пурпуровий
    "anomaly": (0, 255, 255),    # жовтий
}
_DEFAULT_COLOR = (0, 255, 0)


def annotate(image_bgr: np.ndarray, result: InspectionResult,
             preprocess_cfg: PreprocessConfig) -> np.ndarray:
    """Намалювати дефекти на копії зображення (у масштабі інспекції)."""
    # Інспекція виконується на зменшеному зображенні; приводимо до того ж розміру.
    canvas = cv2.resize(image_bgr, result.image_size)
    for d in result.defects:
        x, y, w, h = d.bbox
        color = KIND_COLORS.get(d.kind, _DEFAULT_COLOR)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = f"{d.kind} {d.severity:.2f}"
        ytxt = max(0, y - 6)
        cv2.putText(canvas, label, (x, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    banner = f"{result.verdict}  defects: {len(result.defects)}  max_sev: {result.max_severity:.2f}"
    bcolor = (0, 180, 0) if result.passed else (0, 0, 220)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), bcolor, -1)
    cv2.putText(canvas, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_reports(image_path: str | Path, result: InspectionResult,
                  out_dir: str | Path, preprocess_cfg: PreprocessConfig) -> Tuple[Path, Path]:
    """Зберегти ``<stem>.json`` та ``<stem>_annotated.png`` у ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source).stem or "result"

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    annotated_path = out_dir / f"{stem}_annotated.png"
    image_bgr = load_image(image_path)
    canvas = annotate(image_bgr, result, preprocess_cfg)
    cv2.imwrite(str(annotated_path), canvas)
    return json_path, annotated_path
