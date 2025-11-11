#!/usr/bin/env python3
"""
Test suite for translate_od_pairs module.

This test suite follows TDD principles - tests are written before implementation.
Tests cover all functions in the translate_od_pairs module with comprehensive
coverage including edge cases and error conditions.
"""

import json
import tempfile
from pathlib import Path
import pytest

# Import functions to be tested (will be implemented)
from tools.translate_od_pairs import (
    load_road_mapping,
    translate_od_pairs,
    filter_od_pairs_by_quality,
)


class TestLoadRoadMappingFormat:
    """Test suite for load_road_mapping function with the new format"""

    def test_load_road_mapping_with_distances(self):
        """Test loading mapping file with distance information (new format)"""
        mapping_data = {
            "1": {"target_road_id": 100, "distance_m": 15.5},
            "2": {"target_road_id": 200, "distance_m": 8.2},
            "3": {"target_road_id": 300, "distance_m": 22.1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mapping_data, f)
            temp_file = Path(f.name)

        try:
            result = load_road_mapping(temp_file)

            # Verify result structure
            assert isinstance(result, dict)
            assert len(result) == 3
            assert 1 in result and 2 in result and 3 in result

            # Verify each entry has required fields
            for road_id, mapping in result.items():
                assert isinstance(road_id, int)
                assert "target_road_id" in mapping
                assert "distance_m" in mapping
                assert isinstance(mapping["target_road_id"], int)
                assert isinstance(mapping["distance_m"], (int, float))

        finally:
            temp_file.unlink()

    def test_load_road_mapping_legacy_format(self):
        """Test loading mapping file without distance information (legacy format)"""
        mapping_data = {"1": 100, "2": 200, "3": 300}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mapping_data, f)
            temp_file = Path(f.name)

        try:
            result = load_road_mapping(temp_file)

            # Verify result structure
            assert isinstance(result, dict)
            assert len(result) == 3
            assert 1 in result and 2 in result and 3 in result

            # Verify each entry has target_road_id and distance_m (defaults to 0.0)
            for road_id, mapping in result.items():
                assert isinstance(road_id, int)
                assert "target_road_id" in mapping
                assert "distance_m" in mapping
                assert mapping["target_road_id"] in [100, 200, 300]
                assert (
                    mapping["distance_m"] == 0.0
                )  # Legacy format has no distance info

        finally:
            temp_file.unlink()

    def test_load_road_mapping_file_not_found(self):
        """Test that function fails fast when mapping file doesn't exist"""
        non_existent_file = Path("/non/existent/mapping.json")

        with pytest.raises(AssertionError) as exc_info:
            load_road_mapping(non_existent_file)

        assert "not found" in str(exc_info.value).lower()

    def test_load_road_mapping_empty_file(self):
        """Test loading empty mapping file"""
        mapping_data = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mapping_data, f)
            temp_file = Path(f.name)

        try:
            result = load_road_mapping(temp_file)
            assert result == {}
        finally:
            temp_file.unlink()

    def test_load_road_mapping_invalid_json(self):
        """Test handling of invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                load_road_mapping(temp_file)
        finally:
            temp_file.unlink()

    def test_load_road_mapping_invalid_structure(self):
        """Test handling of invalid mapping structure"""
        invalid_data = {
            "1": {"target_road_id": 100},  # Missing distance_m
            "2": {"distance_m": 8.2},  # Missing target_road_id
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = Path(f.name)

        try:
            # This should fail with clear error message for missing fields
            with pytest.raises(AssertionError) as exc_info:
                load_road_mapping(temp_file)

            # Check that the error message mentions the missing field
            error_msg = str(exc_info.value)
            assert "missing" in error_msg.lower() and "distance_m" in error_msg.lower()
        finally:
            temp_file.unlink()


class TestTranslateOdPairs:
    """Test suite for translate_od_pairs function"""

    def test_translate_od_pairs_basic(self):
        """Test basic translation of OD pairs"""
        od_pairs = [(1, 2), (3, 4), (5, 6)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
            3: {"target_road_id": 300, "distance_m": 5.0},
            4: {"target_road_id": 400, "distance_m": 20.0},
            5: {"target_road_id": 500, "distance_m": 18.0},
            6: {"target_road_id": 600, "distance_m": 12.0},
        }

        result = translate_od_pairs(od_pairs, mapping)

        expected = [(100, 200), (300, 400), (500, 600)]
        assert result == expected

    def test_translate_od_pairs_empty(self):
        """Test translation with empty OD pairs list"""
        od_pairs = []
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }

        result = translate_od_pairs(od_pairs, mapping)

        assert result == []

    def test_translate_od_pairs_unmapped_origin(self):
        """Test fail-fast when origin road not in mapping"""
        od_pairs = [(999, 2)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }

        with pytest.raises(AssertionError) as exc_info:
            translate_od_pairs(od_pairs, mapping)

        assert "not found in mapping" in str(exc_info.value).lower()

    def test_translate_od_pairs_unmapped_destination(self):
        """Test fail-fast when destination road not in mapping"""
        od_pairs = [(1, 999)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }

        with pytest.raises(AssertionError) as exc_info:
            translate_od_pairs(od_pairs, mapping)

        assert "not found in mapping" in str(exc_info.value).lower()

    def test_translate_od_pairs_single_pair(self):
        """Test translation with single OD pair"""
        od_pairs = [(1, 2)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }

        result = translate_od_pairs(od_pairs, mapping)

        assert result == [(100, 200)]

    def test_translate_od_pairs_large_input(self):
        """Test translation with larger input size"""
        # Create 1000 OD pairs
        od_pairs = [(i, i + 1000) for i in range(1, 1001)]
        mapping = {
            i: {"target_road_id": i + 10000, "distance_m": 10.0} for i in range(1, 2001)
        }

        result = translate_od_pairs(od_pairs, mapping)

        expected = [(i + 10000, i + 11000) for i in range(1, 1001)]
        assert result == expected
        assert len(result) == 1000

    def test_translate_od_pairs_duplicate_origins(self):
        """Test translation with duplicate origin roads"""
        od_pairs = [(1, 2), (1, 3), (1, 4)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
            3: {"target_road_id": 300, "distance_m": 5.0},
            4: {"target_road_id": 400, "distance_m": 20.0},
        }

        result = translate_od_pairs(od_pairs, mapping)

        expected = [(100, 200), (100, 300), (100, 400)]
        assert result == expected

    def test_translate_od_pairs_same_origin_destination(self):
        """Test translation where origin equals destination"""
        od_pairs = [(1, 1), (2, 2)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }

        result = translate_od_pairs(od_pairs, mapping)

        expected = [(100, 100), (200, 200)]
        assert result == expected


class TestFilterOdPairsByQuality:
    """Test suite for filter_od_pairs_by_quality function"""

    def test_filter_od_pairs_all_pass(self):
        """Test filtering when all OD pairs pass quality threshold"""
        od_pairs = [(1, 2), (3, 4), (5, 6)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
            3: {"target_road_id": 300, "distance_m": 5.0},
            4: {"target_road_id": 400, "distance_m": 20.0},
            5: {"target_road_id": 500, "distance_m": 18.0},
            6: {"target_road_id": 600, "distance_m": 12.0},
        }
        max_distance = 25.0

        result_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        # All pairs should pass
        assert result_pairs == od_pairs
        assert stats["total_pairs_before"] == 3
        assert stats["total_pairs_after"] == 3
        assert stats["filtered_pairs"] == 0
        assert stats["filter_rate_pct"] == 0.0

    def test_filter_od_pairs_all_filtered(self):
        """Test fail-fast when all OD pairs are filtered out"""
        od_pairs = [(1, 2), (3, 4)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 50.0},  # Exceeds 25m threshold
            2: {"target_road_id": 200, "distance_m": 40.0},  # Exceeds 25m threshold
            3: {"target_road_id": 300, "distance_m": 30.0},  # Exceeds 25m threshold
            4: {"target_road_id": 400, "distance_m": 60.0},  # Exceeds 25m threshold
        }
        max_distance = 25.0

        with pytest.raises(AssertionError) as exc_info:
            filter_od_pairs_by_quality(od_pairs, mapping, max_distance)

        assert "all od pairs filtered out" in str(exc_info.value).lower()

    def test_filter_od_pairs_partial_filter(self):
        """Test filtering when some pairs are filtered out"""
        od_pairs = [(1, 2), (3, 4), (5, 6)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},  # Good
            2: {"target_road_id": 200, "distance_m": 15.0},  # Good
            3: {"target_road_id": 300, "distance_m": 30.0},  # Bad (origin)
            4: {"target_road_id": 400, "distance_m": 5.0},  # Good
            5: {"target_road_id": 500, "distance_m": 8.0},  # Good
            6: {"target_road_id": 600, "distance_m": 50.0},  # Bad (destination)
        }
        max_distance = 25.0

        result_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        # Only first pair should pass (both origin and destination distances <= 25m)
        assert result_pairs == [(1, 2)]
        assert stats["total_pairs_before"] == 3
        assert stats["total_pairs_after"] == 1
        assert stats["filtered_pairs"] == 2
        assert stats["filter_rate_pct"] == pytest.approx(66.67, rel=0.01)

    def test_filter_od_pairs_boundary_distances(self):
        """Test filtering with distances exactly at threshold"""
        od_pairs = [(1, 2), (3, 4)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 25.0},  # Exactly at threshold
            2: {"target_road_id": 200, "distance_m": 25.0},  # Exactly at threshold
            3: {"target_road_id": 300, "distance_m": 25.1},  # Just over threshold
            4: {"target_road_id": 400, "distance_m": 24.9},  # Just under threshold
        }
        max_distance = 25.0

        result_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        # First pair should pass (both exactly at threshold), second should fail (origin over)
        assert result_pairs == [(1, 2)]
        assert stats["total_pairs_before"] == 2
        assert stats["total_pairs_after"] == 1
        assert stats["filtered_pairs"] == 1
        assert stats["filter_rate_pct"] == 50.0

    def test_filter_od_pairs_missing_road_in_mapping(self):
        """Test handling when road exists in mapping but has missing distance"""
        od_pairs = [(1, 2)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0}
            # Road 2 missing from mapping
        }
        max_distance = 25.0

        with pytest.raises(KeyError):
            filter_od_pairs_by_quality(od_pairs, mapping, max_distance)

    def test_filter_od_pairs_empty_input(self):
        """Test filtering with empty OD pairs list"""
        od_pairs = []
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
        }
        max_distance = 25.0

        result_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        assert result_pairs == []
        assert stats["total_pairs_before"] == 0
        assert stats["total_pairs_after"] == 0
        assert stats["filtered_pairs"] == 0
        assert stats["filter_rate_pct"] == 0.0

    def test_filter_od_pairs_zero_threshold(self):
        """Test filtering with zero distance threshold (only perfect matches)"""
        od_pairs = [(1, 2), (3, 4)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 0.0},  # Perfect match
            2: {
                "target_road_id": 200,
                "distance_m": 0.5,
            },  # Not perfect - should fail pair 1
            3: {"target_road_id": 300, "distance_m": 0.0},  # Perfect match
            4: {"target_road_id": 400, "distance_m": 0.0},  # Perfect match
        }
        max_distance = 0.0

        result_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        # Only (3,4) should pass (both distances = 0.0)
        # (1,2) should fail because road 2 has distance 0.5 > 0.0
        assert result_pairs == [(3, 4)]
        assert stats["total_pairs_before"] == 2
        assert stats["total_pairs_after"] == 1
        assert stats["filtered_pairs"] == 1
        assert stats["filter_rate_pct"] == 50.0


class TestIntegration:
    """Integration tests for the complete translation and filtering workflow"""

    def test_complete_translation_workflow(self):
        """Test complete workflow: filtering + translation (correct order)"""
        # Setup test data
        od_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 10.0},
            2: {"target_road_id": 200, "distance_m": 15.0},
            3: {"target_road_id": 300, "distance_m": 50.0},  # Will be filtered
            4: {"target_road_id": 400, "distance_m": 8.0},
            5: {"target_road_id": 500, "distance_m": 5.0},
            6: {"target_road_id": 600, "distance_m": 30.0},  # Will be filtered
            7: {"target_road_id": 700, "distance_m": 12.0},
            8: {"target_road_id": 800, "distance_m": 18.0},
        }
        max_distance = 25.0

        # Filter first (on source road IDs)
        filtered_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        # Should keep pairs that don't involve roads 3 or 6
        expected_filtered = [(1, 2), (7, 8)]
        assert filtered_pairs == expected_filtered
        assert stats["total_pairs_before"] == 4
        assert stats["total_pairs_after"] == 2
        assert stats["filtered_pairs"] == 2
        assert stats["filter_rate_pct"] == 50.0

        # Then translate the filtered pairs
        translated_pairs = translate_od_pairs(filtered_pairs, mapping)
        expected_translated = [(100, 200), (700, 800)]
        assert translated_pairs == expected_translated

    def test_workflow_with_mixed_quality(self):
        """Test workflow with mixed quality mappings"""
        od_pairs = [(1, 2), (2, 3), (3, 1)]  # Circular pattern
        mapping = {
            1: {"target_road_id": 101, "distance_m": 5.0},
            2: {"target_road_id": 202, "distance_m": 20.0},
            3: {"target_road_id": 303, "distance_m": 15.0},
        }
        max_distance = 20.0

        # Filter - all should pass (all distances <= 20.0)
        filtered_pairs, stats = filter_od_pairs_by_quality(
            od_pairs, mapping, max_distance
        )

        assert filtered_pairs == od_pairs
        assert stats["filter_rate_pct"] == 0.0

        # Then translate
        translated_pairs = translate_od_pairs(filtered_pairs, mapping)
        expected_translated = [(101, 202), (202, 303), (303, 101)]
        assert translated_pairs == expected_translated

    def test_workflow_with_varying_thresholds(self):
        """Test workflow with different distance thresholds"""
        od_pairs = [(1, 2), (3, 4)]
        mapping = {
            1: {"target_road_id": 100, "distance_m": 15.0},
            2: {"target_road_id": 200, "distance_m": 25.0},
            3: {"target_road_id": 300, "distance_m": 18.0},
            4: {"target_road_id": 400, "distance_m": 22.0},
        }

        # Test with threshold 20.0 (both pairs fail: road 2=25>20, road 4=22>20)
        with pytest.raises(AssertionError):
            filter_od_pairs_by_quality(od_pairs, mapping, 20.0)

        # Test with threshold 30.0 (both pass)
        filtered_pairs, _ = filter_od_pairs_by_quality(od_pairs, mapping, 30.0)
        assert len(filtered_pairs) == 2

        # Test with threshold 10.0 (both fail)
        with pytest.raises(AssertionError):
            filter_od_pairs_by_quality(od_pairs, mapping, 10.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
