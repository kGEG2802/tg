"""Детектор подряпин і тріщин — тонких лінійних дефектів."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from skimage.filters import sato

from ..config import ScratchConfig
from .base import Defect, Detector, clamp01


class ScratchDetector(Detector):
    """Знаходить тонкі видовжені дефекти (подряпини, тріщини, волосовини).

    Використовує ridge-фільтр Sato, що підкреслює лінійні (тубулярні)
    структури незалежно від їх орієнтації, після чого відбирає компоненти
    за довжиною та видовженістю, відсікаючи плями й точки.
    """

    name = "scratch"

    def __init__(self, config: ScratchConfig):
        self.cfg = config

    def detect(self, gray: np.ndarray, reference: Optional[np.ndarray] = None,
               roi: Optional[np.ndarray] = None) -> List[Defect]:
        cfg = self.cfg
        # Sato підсилює як темні, так і світлі лінії; беремо обидві полярності.
        ridges = sato(
            gray.astype(np.float32) / 255.0,
            sigmas=cfg.sigmas,
            black_ridges=True,
            mode="reflect",
        )
        ridges_bright = sato(
            gray.astype(np.float32) / 255.0,
            sigmas=cfg.sigmas,
            black_ridges=False,
            mode="reflect",
        )
        response = np.maximum(ridges, ridges_bright)
        # Краї силуету деталі дають сильний відгук і "забивають" нормалізацію;
        # обмежуємось поверхнею деталі, щоб бачити саме поверхневі подряпини.
        if roi is not None:
            response[roi == 0] = 0.0
        if response.max() > 0:
            response = response / response.max()

        mask = (response >= cfg.response_threshold).astype(np.uint8) * 255
        # З'єднуємо розриви вздовж лінії.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects: List[Defect] = []
        for cnt in contours:
            # Елонгацію міряємо ПОВЕРНУТИМ прямокутником: інакше діагональна
            # подряпина має майже квадратний bbox і хибно відсіюється.
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            length = max(rw, rh)
            thickness = max(1.0, min(rw, rh))
            aspect = length / thickness
            if length < cfg.min_length or aspect < cfg.min_aspect_ratio:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            comp = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(comp, [cnt], -1, 255, -1)
            strength = float(response[comp > 0].mean())
            area = float(cv2.contourArea(cnt)) or float(length * thickness)
            # Тяжкість росте з довжиною, контрастом і видовженістю.
            severity = clamp01(
                0.45 * min(1.0, length / 200.0)
                + 0.35 * strength
                + 0.20 * min(1.0, aspect / 12.0)
            )
            defects.append(
                Defect(
                    kind="scratch",
                    bbox=(int(x), int(y), int(w), int(h)),
                    severity=severity,
                    area=area,
                    detector=self.name,
                    meta={"length": float(length), "aspect": float(aspect),
                          "response": strength},
                )
            )
        return defects
