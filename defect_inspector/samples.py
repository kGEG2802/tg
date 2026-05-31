"""Генератор синтетичних зразків деталей для демонстрації та тестів.

Дозволяє запустити та перевірити інспектор без реальних фотографій:
створює "годну" деталь і її дефектну версію з контрольованими дефектами.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def make_good_part(size: int = 480, seed: int = 0) -> np.ndarray:
    """Зібрати зображення бездефектної деталі (металева пластина з отвором)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 60, np.uint8)  # темний фон

    # Корпус деталі — світло-сірий прямокутник із заокругленням.
    cv2.rectangle(img, (60, 60), (size - 60, size - 60), (160, 160, 160), -1)
    # Легка текстура поверхні.
    noise = rng.normal(0, 6, (size, size, 1)).astype(np.int16)
    body = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = body > 100
    img[mask] = np.clip(img[mask].astype(np.int16) + noise[mask], 0, 255).astype(np.uint8)
    return img


def make_defective_part(size: int = 480, seed: int = 0) -> np.ndarray:
    """Взяти годну деталь і додати подряпину, пляму та скол краю."""
    img = make_good_part(size, seed)
    # Подряпина — тонка темна лінія.
    cv2.line(img, (110, 130), (330, 250), (40, 40, 40), 2, cv2.LINE_AA)
    # Пляма/забруднення — темна заливка.
    cv2.circle(img, (340, 150), 16, (70, 70, 70), -1)
    # Скол краю — вирізаємо матеріал до кольору фону.
    cv2.fillPoly(
        img,
        [np.array([[size - 60, 300], [size - 60, 360], [size - 95, 330]], np.int32)],
        (60, 60, 60),
    )
    return img


def write_demo_pair(out_dir: str | Path) -> Tuple[Path, Path]:
    """Записати ``good.png`` і ``defective.png`` у каталог; повернути шляхи."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    good_path = out_dir / "good.png"
    bad_path = out_dir / "defective.png"
    cv2.imwrite(str(good_path), make_good_part())
    cv2.imwrite(str(bad_path), make_defective_part())
    return good_path, bad_path
