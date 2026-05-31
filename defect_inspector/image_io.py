"""Завантаження та попередня обробка зображень."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .config import PreprocessConfig

#: Підтримувані розширення вхідних зображень.
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Зчитати кольорове зображення (BGR). Кидає ``FileNotFoundError``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Зображення не знайдено: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не вдалося прочитати зображення: {path}")
    return image


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, float]:
    """Підготувати сіре зображення для детекторів.

    Повертає кортеж ``(gray, scale)``, де ``scale`` — коефіцієнт, на який
    помножено оригінал (потрібен, щоб перерахувати координати назад).
    """
    scale = 1.0
    if cfg.max_side and max(image.shape[:2]) > cfg.max_side:
        scale = cfg.max_side / float(max(image.shape[:2]))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cfg.denoise and cfg.denoise >= 3:
        k = cfg.denoise if cfg.denoise % 2 == 1 else cfg.denoise + 1
        gray = cv2.medianBlur(gray, k)
    if cfg.equalize:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray, scale


def binarize_part(gray: np.ndarray) -> np.ndarray:
    """Бінаризувати деталь так, що передній план (деталь) = 255.

    Полярність визначається за пікселями на межі кадру: фон зазвичай
    торкається країв зображення. Це надійніше за припущення про площу
    деталі (деталь може займати й більшу частину кадру).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    if (border == 255).mean() > 0.5:  # 255 переважає на межі → це фон
        binary = cv2.bitwise_not(binary)
    return binary


def compute_roi_mask(gray: np.ndarray, erode_frac: float = 0.04) -> np.ndarray:
    """Маска корисної площі деталі (uint8, 255 — поверхня деталі).

    Силует деталі відокремлюється від фону (Otsu), береться найбільша
    зв'язна компонента й трохи стискається, щоб межа деталі не сприймалась
    детекторами поверхні як дефект. Якщо фон не вдалося відокремити,
    повертається повністю біла маска (вся площа вважається корисною).
    """
    binary = binarize_part(gray)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, np.uint8)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * gray.size:
        return np.full(gray.shape, 255, np.uint8)
    # Заливаємо силует суцільно — темні дефекти всередині деталі не повинні
    # ставати "дірками" в масці, інакше їх потім помилково відсіє ROI-фільтр.
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    k = max(3, int(erode_frac * max(gray.shape)) | 1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask
