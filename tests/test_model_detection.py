"""Tests for model detection utility."""

import pytest
from pathlib import Path
from tools.model_detection import (
    ModelFile,
    build_model_metadata,
    dataset_supports_phases,
    extract_model_name,
    format_phase_display,
    format_seed_label,
    get_display_name,
    get_model_color,
    get_model_line_style,
    parse_model_components,
)


class TestExtractModelName:
    """Tests for extract_model_name function."""

    def test_beijing_distilled_base(self):
        """Test detection of base distilled model."""
        assert extract_model_name("hoser_distilled_trainod.csv") == "distilled"
        assert extract_model_name("hoser_distilled_testod.csv") == "distilled"
        assert extract_model_name("distilled_results.csv") == "distilled"

    def test_beijing_distilled_with_seeds(self):
        """Test detection of distilled models with seed variants."""
        assert (
            extract_model_name("hoser_distilled_seed42_trainod.csv")
            == "distilled_seed42"
        )
        assert (
            extract_model_name("hoser_distilled_seed43_testod.csv")
            == "distilled_seed43"
        )
        assert (
            extract_model_name("hoser_distilled_seed44_trainod.csv")
            == "distilled_seed44"
        )

    def test_beijing_distilled_l1_checkpoints_detect(self):
        """Test that *_L1 distilled checkpoints map to the distilled_l1 family."""
        assert (
            extract_model_name("distilled_25epoch_seed42_L1.pth")
            == "distilled_l1_seed42"
        )
        assert (
            extract_model_name("distilled_25epoch_seed43_L1.pth")
            == "distilled_l1_seed43"
        )
        assert (
            extract_model_name("distilled_25epoch_seed44_L1.pth")
            == "distilled_l1_seed44"
        )
        assert (
            extract_model_name("hoser_distilled_25epoch_seed42_L1_trainod_gene.csv")
            == "distilled_l1_seed42"
        )

    def test_beijing_distilled_l0p5_checkpoints_detect(self):
        """Test that lambda=0.5 distilled checkpoints map to the distilled_l0p5 family."""

        assert (
            extract_model_name("distilled_25epoch_seed42_lambda0p5.pth")
            == "distilled_l0p5_seed42"
        )
        assert (
            extract_model_name("distilled_25epoch_seed42_L0p5.pth")
            == "distilled_l0p5_seed42"
        )
        assert (
            extract_model_name("hoser_distilled_lambda0p5_seed42_trainod_gene.csv")
            == "distilled_l0p5_seed42"
        )

    def test_abnormal_model_outputs(self):
        """Abnormal eval outputs should map to dedicated *_abnormal families."""

        assert extract_model_name("vanilla_abnormal_od.csv") == "vanilla_abnormal"
        assert (
            extract_model_name("vanilla_seed43_abnormal_od.csv")
            == "vanilla_abnormal_seed43"
        )

        assert extract_model_name("distilled_abnormal_od.csv") == "distilled_abnormal"
        assert (
            extract_model_name("distilled_seed44_abnormal_od.csv")
            == "distilled_abnormal_seed44"
        )

        assert (
            extract_model_name("distilled_seed44_L1_abnormal_od.csv")
            == "distilled_l1_abnormal_seed44"
        )

    def test_porto_phase1_base(self):
        """Test detection of base distill_phase1 model."""
        assert (
            extract_model_name("hoser_distill_phase1_trainod.csv") == "distill_phase1"
        )
        assert extract_model_name("distill_phase1_results.csv") == "distill_phase1"

    def test_porto_phase1_with_seeds(self):
        """Test detection of distill_phase1 models with seed variants."""
        assert (
            extract_model_name("hoser_distill_phase1_seed42_trainod.csv")
            == "distill_phase1_seed42"
        )
        assert (
            extract_model_name("hoser_distill_phase1_seed43_testod.csv")
            == "distill_phase1_seed43"
        )
        assert (
            extract_model_name("hoser_distill_phase1_seed44_trainod.csv")
            == "distill_phase1_seed44"
        )

    def test_porto_phase2_base(self):
        """Test detection of base distill_phase2 model."""
        assert (
            extract_model_name("hoser_distill_phase2_trainod.csv") == "distill_phase2"
        )
        assert extract_model_name("distill_phase2_results.csv") == "distill_phase2"

    def test_porto_phase2_with_seeds(self):
        """Test detection of distill_phase2 models with seed variants."""
        assert (
            extract_model_name("hoser_distill_phase2_seed42_trainod.csv")
            == "distill_phase2_seed42"
        )
        assert (
            extract_model_name("hoser_distill_phase2_seed43_testod.csv")
            == "distill_phase2_seed43"
        )
        assert (
            extract_model_name("hoser_distill_phase2_seed44_trainod.csv")
            == "distill_phase2_seed44"
        )

    def test_vanilla_base(self):
        """Test detection of base vanilla model."""
        assert extract_model_name("hoser_vanilla_trainod.csv") == "vanilla"
        assert extract_model_name("vanilla_results.csv") == "vanilla"

    def test_vanilla_with_seeds(self):
        """Test detection of vanilla models with seed variants."""
        assert (
            extract_model_name("hoser_vanilla_seed42_trainod.csv") == "vanilla_seed42"
        )
        assert extract_model_name("hoser_vanilla_seed43_testod.csv") == "vanilla_seed43"
        assert (
            extract_model_name("hoser_vanilla_seed44_trainod.csv") == "vanilla_seed44"
        )

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert (
            extract_model_name("HOSER_DISTILLED_SEED44_TRAINOD.CSV")
            == "distilled_seed44"
        )
        assert extract_model_name("Hoser_Vanilla_TestOD.csv") == "vanilla"
        assert (
            extract_model_name("DISTILL_PHASE2_SEED43.csv") == "distill_phase2_seed43"
        )

    def test_specificity_order(self):
        """Test that most specific patterns are matched first."""
        # Should match distilled_seed44, not just distilled
        assert (
            extract_model_name("hoser_distilled_seed44_trainod.csv")
            == "distilled_seed44"
        )

        # Should match distill_phase2_seed43, not just distill_phase2
        assert (
            extract_model_name("hoser_distill_phase2_seed43_testod.csv")
            == "distill_phase2_seed43"
        )

    def test_unknown_pattern(self):
        """Test that unknown patterns return 'unknown'."""
        assert extract_model_name("some_random_file.csv") == "unknown"
        assert extract_model_name("test_data.csv") == "unknown"
        assert extract_model_name("unknown_model.csv") == "unknown"

    def test_with_full_path(self):
        """Test that detection works with full paths."""
        path = "/home/user/eval/gene/porto/seed42/hoser_distilled_seed44_trainod.csv"
        assert extract_model_name(path) == "distilled_seed44"

    def test_new_phase_models(self):
        """Test automatic detection of new phase models not in predefined list."""
        # Phase 3 models (not explicitly in KNOWN_MODEL_PATTERNS)
        assert (
            extract_model_name("hoser_distill_phase3_trainod.csv") == "distill_phase3"
        )
        assert (
            extract_model_name("hoser_distill_phase3_seed42_testod.csv")
            == "distill_phase3_seed42"
        )
        assert (
            extract_model_name("hoser_distill_phase3_seed45_trainod.csv")
            == "distill_phase3_seed45"
        )

        # Phase 4 and beyond
        assert (
            extract_model_name("hoser_distill_phase4_seed99_trainod.csv")
            == "distill_phase4_seed99"
        )
        assert extract_model_name("distill_phase5_results.csv") == "distill_phase5"

    def test_new_seed_variants(self):
        """Test automatic detection of new seed numbers."""
        # Seed 45, 50, etc (not explicitly in KNOWN_MODEL_PATTERNS)
        assert (
            extract_model_name("hoser_distilled_seed45_trainod.csv")
            == "distilled_seed45"
        )
        assert (
            extract_model_name("hoser_distilled_seed50_testod.csv")
            == "distilled_seed50"
        )
        assert (
            extract_model_name("hoser_vanilla_seed99_trainod.csv") == "vanilla_seed99"
        )
        assert (
            extract_model_name("hoser_distill_phase2_seed100_testod.csv")
            == "distill_phase2_seed100"
        )

    def test_eval_seed_prefix_checkpoint_names(self):
        """Support eval-workspace checkpoint naming like seed42_distill_l1.pth."""

        assert (
            extract_model_name("seed42_distill_l0p001.pth") == "distilled_l0p001_seed42"
        )
        assert extract_model_name("seed43_distill_l1.pth") == "distilled_l1_seed43"
        assert (
            extract_model_name("seed44_distill_lambda0p5.pth")
            == "distilled_l0p5_seed44"
        )
        assert extract_model_name("seed42_vanilla.pth") == "vanilla_seed42"

    def test_checkpoint_default_seed42_when_missing(self):
        """Checkpoint filenames without an explicit seed default to seed42."""

        # Default-seed behavior should apply to checkpoints (.pth/.pt) only.
        assert extract_model_name("distilled.pth") == "distilled_seed42"
        assert extract_model_name("vanilla.pth") == "vanilla_seed42"
        assert extract_model_name("distilled_l1.pth") == "distilled_l1_seed42"
        assert extract_model_name("distill_phase1.pth") == "distill_phase1_seed42"

        # CSVs retain their unseeded meaning.
        assert extract_model_name("hoser_distilled_trainod.csv") == "distilled"


class TestGetDisplayName:
    """Tests for get_display_name function."""

    def test_beijing_models(self):
        """Test display names for Beijing models."""
        assert get_display_name("distilled") == "Distilled"
        assert get_display_name("distilled_seed42") == "Distilled (seed 42)"
        assert get_display_name("distilled_seed43") == "Distilled (seed 43)"
        assert get_display_name("distilled_seed44") == "Distilled (seed 44)"
        assert get_display_name("distilled_l1") == "Distilled L1"
        assert get_display_name("distilled_l1_seed42") == "Distilled L1 (seed 42)"
        assert get_display_name("distilled_l0p5") == "Distilled L0.5"
        assert get_display_name("distilled_l0p5_seed42") == "Distilled L0.5 (seed 42)"
        assert get_display_name("distilled_abnormal") == "Distilled Abnormal"
        assert (
            get_display_name("distilled_abnormal_seed42")
            == "Distilled Abnormal (seed 42)"
        )
        assert get_display_name("vanilla_abnormal") == "Vanilla Abnormal"

    def test_porto_phase1_models(self):
        """Test display names for Porto phase 1 models."""
        assert get_display_name("distill_phase1") == "Distill Phase 1"
        assert get_display_name("distill_phase1_seed42") == "Distill Phase 1 (seed 42)"
        assert get_display_name("distill_phase1_seed43") == "Distill Phase 1 (seed 43)"
        assert get_display_name("distill_phase1_seed44") == "Distill Phase 1 (seed 44)"

    def test_porto_phase2_models(self):
        """Test display names for Porto phase 2 models."""
        assert get_display_name("distill_phase2") == "Distill Phase 2"
        assert get_display_name("distill_phase2_seed42") == "Distill Phase 2 (seed 42)"
        assert get_display_name("distill_phase2_seed43") == "Distill Phase 2 (seed 43)"
        assert get_display_name("distill_phase2_seed44") == "Distill Phase 2 (seed 44)"

    def test_vanilla_models(self):
        """Test display names for vanilla models."""
        assert get_display_name("vanilla") == "Vanilla"
        assert get_display_name("vanilla_seed42") == "Vanilla (seed 42)"
        assert get_display_name("vanilla_seed43") == "Vanilla (seed 43)"
        assert get_display_name("vanilla_seed44") == "Vanilla (seed 44)"

    def test_special_cases(self):
        """Test display names for special cases."""
        assert get_display_name("real") == "Real"
        assert get_display_name("unknown") == "Unknown"

    def test_new_model_display_names(self):
        """Test automatic display name generation for new models."""
        # New phase models
        assert get_display_name("distill_phase3") == "Distill Phase 3"
        assert get_display_name("distill_phase3_seed45") == "Distill Phase 3 (seed 45)"
        assert get_display_name("distill_phase4_seed99") == "Distill Phase 4 (seed 99)"

        # New seed variants
        assert get_display_name("distilled_seed45") == "Distilled (seed 45)"
        assert get_display_name("vanilla_seed100") == "Vanilla (seed 100)"

        # Lambda=0.001 distilled family
        assert get_display_name("distilled_l0p001") == "Distilled L0.001"
        assert (
            get_display_name("distilled_l0p001_seed42") == "Distilled L0.001 (seed 42)"
        )
        # Legacy suffix form
        assert (
            get_display_name("distilled_seed42_L0p001") == "Distilled L0.001 (seed 42)"
        )

        # Lambda=0.5 distilled family
        assert get_display_name("distilled_l0p5") == "Distilled L0.5"
        assert get_display_name("distilled_l0p5_seed42") == "Distilled L0.5 (seed 42)"
        # Legacy suffix forms
        assert get_display_name("distilled_seed42_L0p5") == "Distilled L0.5 (seed 42)"
        assert (
            get_display_name("distilled_seed42_lambda0.5") == "Distilled L0.5 (seed 42)"
        )
        assert (
            get_display_name("distilled_seed42_lambda0p5") == "Distilled L0.5 (seed 42)"
        )


class TestGetModelColor:
    """Tests for get_model_color function."""

    def test_beijing_models_colors(self):
        """Test colors for Beijing models are deterministic and vary by seed."""

        colors = {
            "base": get_model_color("distilled"),
            "seed42": get_model_color("distilled_seed42"),
            "seed43": get_model_color("distilled_seed43"),
            "seed44": get_model_color("distilled_seed44"),
        }
        assert len(set(colors.values())) == len(colors)
        assert colors["seed42"] == get_model_color("distilled_seed42")

    def test_porto_phase1_colors(self):
        """Test colors for Porto phase 1 models are deterministic and vary by seed."""

        colors = {
            "base": get_model_color("distill_phase1"),
            "seed42": get_model_color("distill_phase1_seed42"),
            "seed43": get_model_color("distill_phase1_seed43"),
            "seed44": get_model_color("distill_phase1_seed44"),
        }
        assert len(set(colors.values())) == len(colors)
        assert colors["seed42"] == get_model_color("distill_phase1_seed42")

    def test_porto_phase2_colors(self):
        """Test colors for Porto phase 2 models are deterministic and vary by seed."""

        colors = {
            "base": get_model_color("distill_phase2"),
            "seed42": get_model_color("distill_phase2_seed42"),
            "seed43": get_model_color("distill_phase2_seed43"),
            "seed44": get_model_color("distill_phase2_seed44"),
        }
        assert len(set(colors.values())) == len(colors)
        assert colors["seed42"] == get_model_color("distill_phase2_seed42")

    def test_vanilla_colors(self):
        """Test colors for vanilla models are deterministic and vary by seed."""

        colors = {
            "base": get_model_color("vanilla"),
            "seed42": get_model_color("vanilla_seed42"),
            "seed43": get_model_color("vanilla_seed43"),
            "seed44": get_model_color("vanilla_seed44"),
        }
        assert len(set(colors.values())) == len(colors)
        assert colors["seed42"] == get_model_color("vanilla_seed42")

    def test_special_colors(self):
        """Test colors for special cases are deterministic."""

        real_color = get_model_color("real")
        unknown_color = get_model_color("unknown")
        assert real_color == get_model_color("real")
        assert unknown_color == get_model_color("unknown")
        assert real_color != unknown_color

    def test_color_format(self):
        """Test that all colors are valid hex codes."""
        from tools.model_detection import MODEL_COLORS

        for color in MODEL_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7
            # Check that it's a valid hex string
            int(color[1:], 16)

    def test_new_model_colors(self):
        """Test automatic color assignment for new models."""
        color_phase3 = get_model_color("distill_phase3")
        assert color_phase3.startswith("#")
        assert len(color_phase3) == 7
        assert color_phase3 == get_model_color("distill_phase3")

        color_distilled45 = get_model_color("distilled_seed45")
        assert color_distilled45.startswith("#")
        assert len(color_distilled45) == 7
        assert color_distilled45 == get_model_color("distilled_seed45")

        color_vanilla99 = get_model_color("vanilla_seed99")
        assert color_vanilla99.startswith("#")
        assert len(color_vanilla99) == 7
        assert color_vanilla99 == get_model_color("vanilla_seed99")


class TestGetModelLineStyle:
    """Tests for get_model_line_style function."""

    def test_known_models(self):
        """Test that known models return solid line."""
        assert get_model_line_style("distilled_seed44") == "-"
        assert get_model_line_style("distill_phase2_seed43") == "-"
        assert get_model_line_style("vanilla") == "-"
        assert get_model_line_style("real") == "-"

    def test_unknown_model(self):
        """Test that unknown models return dashed line."""
        assert get_model_line_style("unknown") == "--"

    def test_new_models(self):
        """Test that new models following conventions get solid lines."""
        assert get_model_line_style("distill_phase3") == "-"
        assert get_model_line_style("distill_phase3_seed45") == "-"
        assert get_model_line_style("distilled_seed99") == "-"
        assert get_model_line_style("vanilla_seed100") == "-"


class TestParseModelComponents:
    """Tests for parse_model_components function."""

    def test_models_with_seeds(self):
        """Test parsing models with seed variants."""
        result = parse_model_components("distilled_seed44")
        assert result["base_model"] == "distilled"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_seed44_L1")
        assert result["base_model"] == "distilled_l1"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_seed44_L0p5")
        assert result["base_model"] == "distilled_l0p5"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_seed44_lambda0.5")
        assert result["base_model"] == "distilled_l0p5"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_seed44_lambda0p5")
        assert result["base_model"] == "distilled_l0p5"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_l1_seed44")
        assert result["base_model"] == "distilled_l1"
        assert result["seed"] == "seed44"

        result = parse_model_components("distilled_l1_abnormal_seed44")
        assert result["base_model"] == "distilled_l1_abnormal"
        assert result["seed"] == "seed44"

        result = parse_model_components("vanilla_abnormal_seed43")
        assert result["base_model"] == "vanilla_abnormal"
        assert result["seed"] == "seed43"

        result = parse_model_components("distill_phase2_seed43")
        assert result["base_model"] == "distill_phase2"
        assert result["seed"] == "seed43"

        result = parse_model_components("vanilla_seed42")
        assert result["base_model"] == "vanilla"
        assert result["seed"] == "seed42"

    def test_models_without_seeds(self):
        """Test parsing models without seed variants."""
        result = parse_model_components("distilled")
        assert result["base_model"] == "distilled"
        assert result["seed"] is None

        result = parse_model_components("vanilla")
        assert result["base_model"] == "vanilla"
        assert result["seed"] is None

        result = parse_model_components("distill_phase1")
        assert result["base_model"] == "distill_phase1"
        assert result["seed"] is None

    def test_new_seed_parsing(self):
        """Test parsing new seed numbers automatically."""
        result = parse_model_components("distilled_seed45")
        assert result["base_model"] == "distilled"
        assert result["seed"] == "seed45"

        result = parse_model_components("distill_phase3_seed99")
        assert result["base_model"] == "distill_phase3"
        assert result["seed"] == "seed99"

        result = parse_model_components("vanilla_seed100")
        assert result["base_model"] == "vanilla"
        assert result["seed"] == "seed100"


class TestModelFile:
    """Tests for ModelFile dataclass."""

    def test_basic_creation(self):
        """Test basic ModelFile creation."""
        mf = ModelFile(
            path=Path("test.csv"),
            model_name="distilled_seed44",
            seed="seed44",
            base_model="distilled",
            filename="test.csv",
        )
        assert mf.path == Path("test.csv")
        assert mf.model_name == "distilled_seed44"
        assert mf.seed == "seed44"
        assert mf.base_model == "distilled"
        assert mf.filename == "test.csv"

    def test_auto_filename(self):
        """Test that filename is auto-filled from path."""
        mf = ModelFile(
            path=Path("/home/user/test.csv"),
            model_name="distilled_seed44",
            seed="seed44",
            base_model="distilled",
        )
        assert mf.filename == "test.csv"


class TestMetadataHelpers:
    """Tests for metadata helper utilities."""

    def test_build_model_metadata_porto(self):
        """Verify metadata extraction for Porto distillation models."""

        metadata = build_model_metadata("distill_phase2_seed43")

        assert metadata.base_model == "distill_phase2"
        assert metadata.normalized_base == "distilled"
        assert metadata.seed_label == "seed43"
        assert metadata.seed_number == 43
        assert metadata.phase_label == "phase2"

    def test_format_seed_label_default_seed(self):
        """Default seed token should map to Seed 42 label."""

        assert format_seed_label("default") == "Seed 42"

    def test_format_phase_display_spacing(self):
        """Phase labels should insert space before digits."""

        assert format_phase_display("phase2") == "Phase 2"

    def test_dataset_supports_phases(self):
        """Only Porto-family datasets report phase support."""

        assert dataset_supports_phases("porto_hoser") is True
        assert dataset_supports_phases("Beijing") is False


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete workflow from filename to display."""
        filename = "hoser_distilled_seed44_trainod.csv"

        # Extract model name
        model = extract_model_name(filename)
        assert model == "distilled_seed44"

        # Get display name
        display = get_display_name(model)
        assert display == "Distilled (seed 44)"

        # Get color (should be a hex string and deterministic)
        color = get_model_color(model)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7
        assert color == get_model_color(model)

        # Parse components
        components = parse_model_components(model)
        assert components["base_model"] == "distilled"
        assert components["seed"] == "seed44"

    def test_multiple_models(self):
        """Test detection and display for multiple models."""
        filenames = [
            "hoser_distilled_seed44_trainod.csv",
            "hoser_distill_phase2_seed43_testod.csv",
            "hoser_vanilla_trainod.csv",
        ]

        expected = [
            ("distilled_seed44", "Distilled (seed 44)"),
            ("distill_phase2_seed43", "Distill Phase 2 (seed 43)"),
            ("vanilla", "Vanilla"),
        ]

        for filename, (exp_model, exp_display) in zip(filenames, expected):
            model = extract_model_name(filename)
            assert model == exp_model
            assert get_display_name(model) == exp_display
            color = get_model_color(model)
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
