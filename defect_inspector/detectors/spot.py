"""Детектор плям, забруднень, раковин і точкових дефектів."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..config import SpotConfig
from .base import Defect, Detector, clamp01


class SpotDetector(Detector):
    """Знаходить локальні відхилення яскравості від фону.

    Фон оцінюється великим медіанним вікном; усе, що сильно відхиляється
    від нього (темніше або світліше), вважається кандидатом. Кандидати
    фільтруються за площею та округлістю, щоб відрізнити компактні плями
    від видовжених подряпин (їх обробляє :class:`ScratchDetector`).
    """

    name = "spot"

    def __init__(self, config: SpotConfig):
        self.cfg = config

    def detect(self, gray: np.ndarray, reference: Optional[np.ndarray] = None,
               roi: Optional[np.ndarray] = None) -> List[Defect]:
        cfg = self.cfg
        bs = cfg.block_size if cfg.block_size % 2 == 1 else cfg.block_size + 1
        background = cv2.medianBlur(gray, bs)
        diff = cv2.absdiff(gray, background)
        # Поза деталлю (фон, межа) спрацювань не розглядаємо.
        if roi is not None:
            diff = cv2.bitwise_and(diff, diff, mask=roi)

        mask = (diff >= cfg.intensity_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects: List[Defect] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area or area > cfg.max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < cfg.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            comp_mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            contrast = float(diff[comp_mask > 0].mean()) / 255.0
            severity = clamp01(
                0.5 * min(1.0, area / 1500.0)
                + 0.5 * min(1.0, contrast / 0.4)
            )
            defects.append(
                Defect(
                    kind="spot",
                    bbox=(int(x), int(y), int(w), int(h)),
                    severity=severity,
                    area=float(area),
                    detector=self.name,
                    meta={"circularity": float(circularity), "contrast": contrast},
                )
            )
        return defects
