"""Tests for python_pipeline.ModelDetector using tools.model_detection conventions."""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from python_pipeline import ModelDetector  # noqa: E402


def test_detect_models_distinguishes_seed_prefixed_variants(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Eval workspace checkpoint naming convention.
    filenames = [
        "seed42_distill_l0p001.pth",
        "seed42_distill_l1.pth",
        "seed42_distill_lambda0p5.pth",
        "seed42_vanilla.pth",
        "seed43_distill_l0p001.pth",
        "seed43_distill_l1.pth",
        "seed43_distill_lambda0p5.pth",
        "seed43_vanilla.pth",
        "seed44_distill_l0p001.pth",
        "seed44_distill_l1.pth",
        "seed44_distill_lambda0p5.pth",
        "seed44_vanilla.pth",
    ]
    for name in filenames:
        (models_dir / name).write_text("")

    detector = ModelDetector(models_dir)
    detected = set(detector.detect_models())

    expected = {
        "distilled_l0p001_seed42",
        "distilled_l1_seed42",
        "distilled_l0p5_seed42",
        "vanilla_seed42",
        "distilled_l0p001_seed43",
        "distilled_l1_seed43",
        "distilled_l0p5_seed43",
        "vanilla_seed43",
        "distilled_l0p001_seed44",
        "distilled_l1_seed44",
        "distilled_l0p5_seed44",
        "vanilla_seed44",
    }

    assert detected == expected


def test_find_model_file_works_with_detected_names(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    (models_dir / "seed44_distill_l0p001.pth").write_text("")

    detector = ModelDetector(models_dir)
    model_path = detector.find_model_file("distilled_l0p001_seed44")

    assert model_path is not None
    assert model_path.name == "seed44_distill_l0p001.pth"
