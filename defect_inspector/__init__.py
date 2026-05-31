"""defect_inspector — промислова інспекція деталей на дефекти.

Модульний застосунок комп'ютерного зору для виявлення поверхневих дефектів
деталей: подряпини/тріщини, плями/забруднення, сколи країв, а також
аномалії при порівнянні з еталоном (golden sample).

Працює офлайн, без датасету. Архітектура розширювана — кожен детектор
ізольований і реалізує спільний інтерфейс :class:`detectors.base.Detector`,
тож згодом можна додати нейромережевий бекенд, не змінюючи решту коду.
"""
from .config import InspectionConfig
from .detectors.base import Defect
from .pipeline import Inspector, InspectionResult

__all__ = [
    "InspectionConfig",
    "Inspector",
    "InspectionResult",
    "Defect",
]

__version__ = "1.0.0"
