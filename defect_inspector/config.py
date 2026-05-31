"""Конфігурація інспекції.

Усі пороги зібрані в одному місці у вигляді dataclass-ів. Конфіг можна
завантажити з JSON-файлу (:meth:`InspectionConfig.from_file`) або задати
в коді. Значення за замовчуванням підібрані під загальну поверхневу
інспекцію та підходять для швидкого старту.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PreprocessConfig:
    """Попередня обробка зображення перед детекцією."""

    #: До цього розміру масштабується довша сторона (0 — без масштабування).
    max_side: int = 1280
    #: Сила медіанного фільтра для прибирання шуму (0 — вимкнено, непарне).
    denoise: int = 3
    #: Вирівнювати освітлення через CLAHE.
    equalize: bool = True


@dataclass
class ScratchConfig:
    """Детектор тонких лінійних дефектів: подряпин і тріщин."""

    enabled: bool = True
    #: Масштаби ridge-фільтра Sato (товщина ліній у пікселях).
    sigmas: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    #: Поріг на нормалізовану відповідь фільтра (0..1).
    response_threshold: float = 0.40
    #: Мінімальна довжина дефекту в пікселях.
    min_length: int = 22
    #: Мінімальне співвідношення сторін (елонгація) — відсікає плями.
    min_aspect_ratio: float = 3.5


@dataclass
class SpotConfig:
    """Детектор плям, забруднень, раковин і точкових дефектів."""

    enabled: bool = True
    #: Розмір вікна оцінки локального фону (непарне).
    block_size: int = 51
    #: Поріг відхилення яскравості від локального фону.
    intensity_threshold: float = 22.0
    #: Допустимий діапазон площі дефекту (пікселі).
    min_area: int = 14
    max_area: int = 6000
    #: Мінімальна "округлість" (4*pi*area/perimeter^2).
    min_circularity: float = 0.25


@dataclass
class EdgeConfig:
    """Детектор сколів і вищербин по контуру деталі."""

    enabled: bool = True
    #: Мінімальна глибина дефекту опуклості (пікселі).
    min_defect_depth: int = 7
    #: Мінімальна площа силуету деталі, щоб вважати контур валідним.
    min_part_area: int = 1500


@dataclass
class ReferenceConfig:
    """Порівняння з еталоном (golden sample)."""

    enabled: bool = True
    #: Поріг абсолютної різниці яскравості.
    diff_threshold: int = 32
    #: Мінімальна площа аномалії.
    min_area: int = 18
    #: Розмиття перед різницею (зменшує хибні спрацювання, непарне).
    blur: int = 5
    #: Геометрично вирівнювати знімок під еталон (ORB + гомографія).
    align: bool = True


@dataclass
class VerdictConfig:
    """Правила фінального вердикту PASS / FAIL."""

    #: Показувати/враховувати лише дефекти з тяжкістю >= цього значення.
    min_severity: float = 0.20
    #: FAIL, якщо тяжкість будь-якого дефекту >= цього значення.
    fail_severity: float = 0.55
    #: FAIL, якщо кількість значимих дефектів перевищує це число.
    max_defect_count: int = 0
    #: FAIL, якщо сумарна площа дефектів / площа деталі перевищує частку.
    max_area_ratio: float = 0.0


@dataclass
class InspectionConfig:
    """Кореневий конфіг, що агрегує налаштування всіх етапів."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    scratch: ScratchConfig = field(default_factory=ScratchConfig)
    spot: SpotConfig = field(default_factory=SpotConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(cls, path: str | Path) -> "InspectionConfig":
        """Завантажити конфіг із JSON-файлу (часткове перевизначення дозволене)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionConfig":
        """Зібрати конфіг зі словника, доповнюючи відсутнє значеннями за замовч."""
        sections = {
            "preprocess": PreprocessConfig,
            "scratch": ScratchConfig,
            "spot": SpotConfig,
            "edge": EdgeConfig,
            "reference": ReferenceConfig,
            "verdict": VerdictConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            kwargs[name] = klass(**(data.get(name) or {}))
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
