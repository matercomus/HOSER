import re
from typing import Dict, List


# Logic to be tested (will be moved to visualize_trajectories.py)
def extract_seed(model_name: str) -> str:
    # Matches "seed" followed by digits (e.g., "seed42" from "distilled_seed42")
    match = re.search(r"seed(\d+)", model_name)
    if match:
        return f"seed{match.group(1)}"
    return "no_seed"


def group_models_by_seed(models: List[str]) -> Dict[str, List[str]]:
    groups = {}
    for model_name in models:
        if model_name == "real":
            continue

        seed = extract_seed(model_name)
        if seed not in groups:
            groups[seed] = []
        groups[seed].append(model_name)

    return groups


def test_extract_seed():
    assert extract_seed("distilled_seed42") == "seed42"
    assert extract_seed("vanilla_seed100") == "seed100"
    assert extract_seed("distilled_seed42_epoch25") == "seed42"
    assert extract_seed("vanilla") == "no_seed"
    assert extract_seed("real") == "no_seed"
    assert extract_seed("distilled_phase2_seed5") == "seed5"


def test_group_models_by_seed():
    models = [
        "real",
        "distilled_seed42",
        "vanilla_seed42",
        "distilled_seed43",
        "vanilla_seed43",
        "baseline_no_seed",
    ]

    groups = group_models_by_seed(models)

    assert "seed42" in groups
    assert "seed43" in groups
    assert "no_seed" in groups

    assert set(groups["seed42"]) == {"distilled_seed42", "vanilla_seed42"}
    assert set(groups["seed43"]) == {"distilled_seed43", "vanilla_seed43"}
    assert set(groups["no_seed"]) == {"baseline_no_seed"}

    # Real is handled separately in the visualizer logic usually
    assert "real" not in groups.get("no_seed", [])
