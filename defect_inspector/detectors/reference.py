"""Детектор аномалій порівнянням з еталоном (golden sample)."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..config import ReferenceConfig
from .base import Defect, Detector, clamp01


class ReferenceDetector(Detector):
    """Порівнює деталь із завідомо годним зразком.

    Це найнадійніший режим для серійного контролю: знімок геометрично
    вирівнюється під еталон (ORB-фічі + гомографія), після чого будь-яке
    значиме відхилення яскравості трактується як дефект — незалежно від
    його типу (відсутній отвір, зайвий матеріал, пляма, деформація).
    """

    name = "reference"

    def __init__(self, config: ReferenceConfig):
        self.cfg = config

    # --------------------------------------------------------------- align
    def _align(self, gray: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Вирівняти ``gray`` під ``reference``; за невдачі повернути як є."""
        try:
            orb = cv2.ORB_create(2000)
            k1, d1 = orb.detectAndCompute(gray, None)
            k2, d2 = orb.detectAndCompute(reference, None)
            if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
                return gray
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
            matches = matches[: max(10, len(matches) // 2)]
            if len(matches) < 8:
                return gray
            src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            h_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if h_mat is None:
                return gray
            return cv2.warpPerspective(gray, h_mat, (reference.shape[1], reference.shape[0]))
        except cv2.error:
            return gray

    # -------------------------------------------------------------- detect
    def detect(self, gray: np.ndarray, reference: Optional[np.ndarray] = None,
               roi: Optional[np.ndarray] = None) -> List[Defect]:
        if reference is None:
            return []
        cfg = self.cfg
        if gray.shape != reference.shape:
            gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]))
        if cfg.align:
            gray = self._align(gray, reference)

        blur = cfg.blur if cfg.blur % 2 == 1 else cfg.blur + 1
        a = cv2.GaussianBlur(gray, (blur, blur), 0)
        b = cv2.GaussianBlur(reference, (blur, blur), 0)
        diff = cv2.absdiff(a, b)

        mask = (diff >= cfg.diff_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects: List[Defect] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            comp_mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            contrast = float(diff[comp_mask > 0].mean()) / 255.0
            severity = clamp01(
                0.5 * min(1.0, area / 2000.0) + 0.5 * min(1.0, contrast / 0.5)
            )
            defects.append(
                Defect(
                    kind="anomaly",
                    bbox=(int(x), int(y), int(w), int(h)),
                    severity=severity,
                    area=float(area),
                    detector=self.name,
                    meta={"contrast": contrast},
                )
            )
        return defects
