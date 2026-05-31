"""Детектор сколів і вищербин по контуру деталі."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..config import EdgeConfig
from ..image_io import binarize_part
from .base import Defect, Detector, clamp01


class EdgeChipDetector(Detector):
    """Знаходить сколи/вищербини на краю деталі.

    Силует деталі відокремлюється від фону (Otsu), для нього будується
    опукла оболонка, а западини між контуром і оболонкою (convexity
    defects) глибші за поріг трактуються як відсутність матеріалу —
    сколи, тріщини з краю, вищерблення.
    """

    name = "edge"

    def __init__(self, config: EdgeConfig):
        self.cfg = config

    def detect(self, gray: np.ndarray, reference: Optional[np.ndarray] = None,
               roi: Optional[np.ndarray] = None) -> List[Defect]:
        # ``roi`` свідомо не використовується: цей детектор аналізує саме
        # межу деталі, тож працює з повним силуетом.
        cfg = self.cfg
        binary = binarize_part(gray)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        cnt = max(contours, key=cv2.contourArea)
        part_area = cv2.contourArea(cnt)
        if part_area < cfg.min_part_area:
            return []

        diag = float(np.hypot(*gray.shape))
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) <= 3:
            return []
        cv_defects = cv2.convexityDefects(cnt, hull_idx)
        if cv_defects is None:
            return []

        defects: List[Defect] = []
        for k in range(cv_defects.shape[0]):
            s, e, f, depth_fp = cv_defects[k, 0]
            depth = depth_fp / 256.0
            if depth < cfg.min_defect_depth:
                continue
            start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])
            xs = [start[0], end[0], far[0]]
            ys = [start[1], end[1], far[1]]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x + 1, max(ys) - y + 1
            severity = clamp01(min(1.0, depth / (0.10 * diag)))
            defects.append(
                Defect(
                    kind="edge_chip",
                    bbox=(int(x), int(y), int(w), int(h)),
                    severity=severity,
                    area=float(w * h),
                    detector=self.name,
                    meta={"depth": float(depth)},
                )
            )
        return defects
