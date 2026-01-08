from __future__ import annotations

import csv
from pathlib import Path

from tools.plot_lmtad_results import _is_abnormal_label, _read_bool_labels_from_csv


def _write_labels_csv(path: Path, *, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["abnormality_info"])
        w.writeheader()
        for v in values:
            w.writerow({"abnormality_info": v})


def test_is_abnormal_label_null_like_and_normal():
    values = [
        "",
        "normal",
        "Normal",
        " NONE ",
        "null",
        "NaN",
        "{\"type\": \"detour\"}",
        "abnormal",
    ]
    got = [_is_abnormal_label(v) for v in values]
    assert got == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_read_bool_labels_from_csv_respects_normal_value(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    _write_labels_csv(csv_path, values=["ok", "normal", "{\"type\": \"route_switch\"}"])

    labels = _read_bool_labels_from_csv(
        csv_path,
        label_col="abnormality_info",
        normal_value="ok",
    )
    assert labels == [False, False, True]
