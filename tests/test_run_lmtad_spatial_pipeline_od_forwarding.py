"""Test that run_lmtad_spatial_pipeline forwards the OD pairs file to evaluation."""

from unittest.mock import patch

import json


def test_pipeline_forwards_od_pairs_file(tmp_path):
    from tools.run_lmtad_spatial_pipeline import run_lmtad_spatial_pipeline

    eval_dir = tmp_path / "eval_dir"
    eval_dir.mkdir()
    # Create models dir (pipeline requires models/ existence)
    (eval_dir / "models").mkdir()

    dataset = "porto_hoser"

    # Create canonical OD pairs file that should be forwarded
    od_file = eval_dir / f"abnormal_od_pairs_lmtad_spatial_{dataset}.json"
    od_content = {
        "total_unique_od_pairs": 1,
        "od_pairs_by_type": {"route_switch": [[1, 2]]},
    }
    od_file.write_text(json.dumps(od_content))

    # Create generated trajectories dir and a dummy CSV so pipeline discovers a model
    gene_dir = eval_dir / "gene_abnormal_lmtad_spatial" / dataset / "seed42"
    gene_dir.mkdir(parents=True)
    model_csv = gene_dir / "vanilla_seed42_spatial_abnormal.csv"
    model_csv.write_text("# header\norigin,destination,gene_trace_road_id\n")

    # Mock LM-TAD source eval dir and checkpoint (existence is checked)
    lmtad_source = tmp_path / "lmtad_eval"
    lmtad_source.mkdir()
    checkpoint = tmp_path / "ckpt_best.pt"
    checkpoint.write_text("ckpt")

    # Capture the od_pairs_file argument passed to evaluator
    captured = {}

    def fake_evaluate(*args, **kwargs):
        # record the od_pairs_file kwarg value
        captured["od_pairs_file"] = kwargs.get("od_pairs_file")
        # return minimal result structure expected by pipeline
        return {"total_trajectories": 0, "trajectories": []}

    # Run pipeline with evaluation enabled but skip extraction/generation/agg/vis
    with patch(
        "tools.run_lmtad_spatial_pipeline.evaluate_spatial_abnormal_trajectories",
        side_effect=fake_evaluate,
    ):
        success = run_lmtad_spatial_pipeline(
            eval_dir=eval_dir,
            dataset=dataset,
            lmtad_source_eval_dir=lmtad_source,
            lmtad_checkpoint=checkpoint,
            skip_extraction=True,
            skip_generation=True,
            skip_aggregation=True,
            skip_visualization=True,
            force=False,
        )

    assert success is True
    # The pipeline should have forwarded the canonical od_file path (Path object)
    assert "od_pairs_file" in captured
    assert captured["od_pairs_file"] is not None
    # Normalize to str for comparison
    assert str(captured["od_pairs_file"]).endswith(str(od_file))
