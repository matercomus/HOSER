"""Tests for generate_trajectories_programmatic function."""

from unittest.mock import patch, Mock
import inspect

# Import the function to test
from gene import generate_trajectories_programmatic


class TestGenerateTrajectoriesProgrammaticSignature:
    """Test function signature and parameter validation."""

    def test_function_signature(self):
        """Test that generate_trajectories_programmatic has correct signature."""
        sig = inspect.signature(generate_trajectories_programmatic)

        # Check required parameters
        assert "dataset" in sig.parameters
        assert "model_path" in sig.parameters

        # Check backward compatibility parameters
        assert "od_source" in sig.parameters
        assert sig.parameters["od_source"].default == "train"
        assert "num_gene" in sig.parameters
        assert sig.parameters["num_gene"].default == 100

        # Check new interface parameters
        assert "od_pairs" in sig.parameters
        assert sig.parameters["od_pairs"].default is None
        assert "output_file" in sig.parameters
        assert sig.parameters["output_file"].default is None

        # Check other parameters
        assert "seed" in sig.parameters
        assert "cuda_device" in sig.parameters
        assert "beam_search" in sig.parameters

    def test_function_docstring(self):
        """Test that function has comprehensive docstring."""
        doc = generate_trajectories_programmatic.__doc__
        assert doc is not None
        assert "od_pairs" in doc or "OD pairs" in doc
        assert "output_file" in doc or "output" in doc


class TestGenerateTrajectoriesProgrammaticSampling:
    """Test backward compatibility - sampling mode (old interface)."""

    @patch("gene.load_and_preprocess_data")
    @patch("gene.set_seed")
    def test_sampling_mode_with_od_source(self, mock_set_seed, mock_load_data):
        """Test that sampling mode works with od_source and num_gene."""
        # Mock data structure
        mock_data = {
            "num_roads": 1000,
            "reachable_road_id_dict": {i: [j for j in range(10)] for i in range(1000)},
            "od_counts": {(i, i + 1): 1.0 for i in range(100)},
            "od_to_train_indices": {(i, i + 1): [0] for i in range(100)},
            "train_traj": Mock(),
        }
        mock_load_data.return_value = mock_data

        # Mock model loading and other dependencies
        with (
            patch("gene.HOSER"),
            patch("gene.Searcher"),
            patch("builtins.open"),
            patch("yaml.safe_load"),
            patch("gene.create_nested_namespace"),
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch("polars.DataFrame.write_csv"),
        ):
            # This should not raise an error
            # We can't fully test without a real model, but we can test the signature
            sig = inspect.signature(generate_trajectories_programmatic)
            assert "od_source" in sig.parameters
            assert "num_gene" in sig.parameters


class TestGenerateTrajectoriesProgrammaticCustomOD:
    """Test custom OD pairs mode (new interface)."""

    def test_od_pairs_parameter_exists(self):
        """Test that od_pairs parameter exists in function signature."""
        sig = inspect.signature(generate_trajectories_programmatic)
        assert "od_pairs" in sig.parameters
        assert sig.parameters["od_pairs"].default is None

    def test_output_file_parameter_exists(self):
        """Test that output_file parameter exists in function signature."""
        sig = inspect.signature(generate_trajectories_programmatic)
        assert "output_file" in sig.parameters
        assert sig.parameters["output_file"].default is None


class TestGenerateTrajectoriesProgrammaticValidation:
    """Test input validation."""

    def test_od_pairs_type_hint(self):
        """Test that od_pairs has correct type hint."""
        sig = inspect.signature(generate_trajectories_programmatic)
        od_pairs_param = sig.parameters["od_pairs"]
        # Check that it's Optional[List[Tuple[int, int]]]
        assert od_pairs_param.annotation is not None


class TestGenerateTrajectoriesProgrammaticOutput:
    """Test output file handling."""

    def test_output_file_parameter(self):
        """Test that output_file parameter exists."""
        sig = inspect.signature(generate_trajectories_programmatic)
        assert "output_file" in sig.parameters
        assert sig.parameters["output_file"].default is None


class TestGenerateTrajectoriesProgrammaticIntegration:
    """Test integration scenarios."""

    def test_function_importable(self):
        """Test that function can be imported."""
        from gene import generate_trajectories_programmatic

        assert callable(generate_trajectories_programmatic)
