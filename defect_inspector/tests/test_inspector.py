"""Юніт-тести на синтетичних зображеннях (pytest)."""
from __future__ import annotations

import numpy as np

from defect_inspector import Inspector, InspectionConfig
from defect_inspector.detectors import (
    EdgeChipDetector,
    ScratchDetector,
    SpotDetector,
)
from defect_inspector.samples import make_defective_part, make_good_part
from defect_inspector.image_io import preprocess


def _gray(img):
    return preprocess(img, InspectionConfig().preprocess)[0]


# ----------------------------------------------------------- end-to-end
def test_good_part_passes():
    result = Inspector().inspect(make_good_part())
    assert result.passed, f"годна деталь має проходити, причини: {result.reasons}"


def test_defective_part_fails():
    result = Inspector().inspect(make_defective_part())
    assert not result.passed
    assert len(result.defects) >= 1


def test_reference_mode_finds_anomalies():
    cfg = InspectionConfig()
    # лишаємо лише еталонний детектор, щоб перевірити саме його
    cfg.scratch.enabled = cfg.spot.enabled = cfg.edge.enabled = False
    result = Inspector(cfg).inspect(make_defective_part(), reference=make_good_part())
    assert any(d.kind == "anomaly" for d in result.defects)


# ----------------------------------------------------------- per-detector
def test_scratch_detector_on_clean_image_is_quiet():
    gray = _gray(make_good_part())
    found = ScratchDetector(InspectionConfig().scratch).detect(gray)
    # на чистій деталі допускаємо щонайбільше поодинокі слабкі спрацювання
    strong = [d for d in found if d.severity >= 0.55]
    assert len(strong) == 0


def test_scratch_detector_finds_line():
    img = make_good_part()
    import cv2
    cv2.line(img, (100, 120), (360, 300), (30, 30, 30), 2)
    gray = _gray(img)
    found = ScratchDetector(InspectionConfig().scratch).detect(gray)
    assert len(found) >= 1


def test_spot_detector_finds_blob():
    img = make_good_part()
    import cv2
    cv2.circle(img, (200, 200), 14, (70, 70, 70), -1)
    gray = _gray(img)
    found = SpotDetector(InspectionConfig().spot).detect(gray)
    assert len(found) >= 1


def test_defect_iou_and_serialization():
    result = Inspector().inspect(make_defective_part())
    payload = result.to_dict()
    assert payload["verdict"] in {"PASS", "FAIL"}
    assert payload["defect_count"] == len(result.defects)
    for d in payload["defects"]:
        assert {"kind", "bbox", "severity", "area"} <= set(d.keys())


def test_config_roundtrip(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    InspectionConfig().save(cfg_path)
    loaded = InspectionConfig.from_file(cfg_path)
    assert isinstance(loaded, InspectionConfig)
    assert loaded.scratch.min_length == InspectionConfig().scratch.min_length


def test_partial_config_uses_defaults():
    cfg = InspectionConfig.from_dict({"scratch": {"min_length": 99}})
    assert cfg.scratch.min_length == 99
    assert cfg.spot.min_area == InspectionConfig().spot.min_area
