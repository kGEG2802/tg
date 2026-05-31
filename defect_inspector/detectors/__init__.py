"""Колекція детекторів дефектів."""
from .base import Defect, Detector
from .edge import EdgeChipDetector
from .reference import ReferenceDetector
from .scratch import ScratchDetector
from .spot import SpotDetector

__all__ = [
    "Defect",
    "Detector",
    "ScratchDetector",
    "SpotDetector",
    "EdgeChipDetector",
    "ReferenceDetector",
]
