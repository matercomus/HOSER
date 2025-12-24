from pathlib import Path

import pytest

from evaluation import evaluate_trajectories_programmatic


def test_evaluate_trajectories_programmatic_respects_data_dir(tmp_path: Path):
    # Provide an explicit dataset root that does not contain expected files.
    # The function should report missing files under tmp_path (not under repo_root/data/<dataset>).
    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate_trajectories_programmatic(
            generated_file=str(tmp_path / "dummy_generated.csv"),
            dataset="Beijing_abnormal_3",
            od_source="train",
            data_dir=tmp_path,
        )

    msg = str(excinfo.value)
    assert str(tmp_path) in msg
    assert "train.csv" in msg
