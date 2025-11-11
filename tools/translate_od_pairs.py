#!/usr/bin/env python3
"""
OD Pair Translation and Quality Filtering Module

This module provides functions to translate OD pairs from source dataset road IDs
to target dataset road IDs using road network mapping, with quality filtering
based on mapping distance thresholds.

All functions use fail-fast assertions to ensure required files exist and
validation checks pass, with no fallbacks or graceful degradation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


def load_road_mapping(mapping_file: Path) -> Dict[int, Dict[str, float]]:
    """Load road network mapping from JSON file with fail-fast assertions

    Args:
        mapping_file: Path to mapping JSON file

    Returns:
        Dictionary mapping source road IDs to {target_road_id, distance_m} dicts

    Raises:
        AssertionError: If mapping file doesn't exist
    """
    # Fail fast: assert file exists
    assert mapping_file.exists(), (
        f"Mapping file not found: {mapping_file}. Translation is required."
    )

    logger.info(f"📂 Loading road mapping from {mapping_file}")

    # Load and parse JSON
    with open(mapping_file, "r") as f:
        raw_mapping = json.load(f)

    # Convert string keys to integers and ensure proper structure
    mapping = {}
    for source_id_str, mapping_info in raw_mapping.items():
        source_id = int(source_id_str)
        if isinstance(mapping_info, dict):
            # New format: {"target_road_id": target_id, "distance_m": distance}
            # Validate required fields
            assert "target_road_id" in mapping_info, (
                f"Missing 'target_road_id' for road {source_id} in mapping file"
            )
            assert "distance_m" in mapping_info, (
                f"Missing 'distance_m' for road {source_id} in mapping file"
            )

            mapping[source_id] = {
                "target_road_id": int(mapping_info["target_road_id"]),
                "distance_m": float(mapping_info["distance_m"]),
            }
        else:
            # Legacy format: just target_id - add default distance
            mapping[source_id] = {
                "target_road_id": int(mapping_info),
                "distance_m": 0.0,  # No distance info in legacy format
            }

    logger.info(f"  ✅ Loaded {len(mapping):,} road mappings")

    return mapping


def extract_target_ids_from_mapping(
    mapping: Dict[int, Dict[str, float]],
) -> Dict[int, int]:
    """Extract target road IDs from mapping for convenience

    Args:
        mapping: Full mapping with target IDs and distances

    Returns:
        Dictionary mapping source road IDs to target road IDs only
    """
    return {source_id: info["target_road_id"] for source_id, info in mapping.items()}


def translate_od_pairs(
    od_pairs: List[Tuple[int, int]], mapping: Dict[int, Dict[str, float]]
) -> List[Tuple[int, int]]:
    """Translate OD pairs from source road IDs to target road IDs

    Args:
        od_pairs: List of (origin, destination) tuples using source road IDs
        mapping: Dictionary mapping source road IDs to {target_road_id, distance_m}

    Returns:
        List of (origin, destination) tuples using target road IDs

    Raises:
        AssertionError: If any origin or destination road not found in mapping
    """
    if not od_pairs:
        logger.info("📂 No OD pairs to translate")
        return []

    logger.info(
        f"🗺️  Translating {len(od_pairs)} OD pairs from source to target road IDs"
    )

    # Extract target ID mapping for efficiency
    target_id_mapping = extract_target_ids_from_mapping(mapping)

    translated_pairs = []

    for origin, destination in od_pairs:
        # Fail fast: assert origin exists in mapping
        assert origin in mapping, (
            f"Origin road {origin} not found in mapping. All roads must be mapped."
        )

        # Fail fast: assert destination exists in mapping
        assert destination in mapping, (
            f"Destination road {destination} not found in mapping. All roads must be mapped."
        )

        # Translate both origin and destination
        translated_origin = target_id_mapping[origin]
        translated_destination = target_id_mapping[destination]

        translated_pairs.append((translated_origin, translated_destination))

    logger.info(f"  ✅ Successfully translated {len(translated_pairs)} OD pairs")

    return translated_pairs


def filter_od_pairs_by_quality(
    od_pairs: List[Tuple[int, int]],
    mapping: Dict[int, Dict[str, float]],
    max_distance_threshold: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Filter OD pairs based on quality thresholds using distance mapping

    Args:
        od_pairs: List of (origin, destination) tuples using source road IDs
        mapping: Dictionary mapping source road IDs to {target_road_id, distance_m}
        max_distance_threshold: Maximum acceptable mapping distance in meters

    Returns:
        Tuple of (filtered_pairs, statistics_dict)

    Raises:
        AssertionError: If all OD pairs are filtered out
    """
    if not od_pairs:
        logger.info("🔍 No OD pairs to filter")
        return [], {
            "total_pairs_before": 0,
            "total_pairs_after": 0,
            "filtered_pairs": 0,
            "filter_rate_pct": 0.0,
            "max_distance_threshold_m": max_distance_threshold,
        }

    logger.info(
        f"🔍 Filtering {len(od_pairs)} OD pairs with max distance threshold: {max_distance_threshold}m"
    )

    filtered_pairs = []
    filtered_origins = 0
    filtered_destinations = 0
    filtered_both = 0

    for origin, destination in od_pairs:
        # Get distances from mapping
        origin_distance = mapping[origin]["distance_m"]
        destination_distance = mapping[destination]["distance_m"]

        # Include pair only if BOTH distances are within threshold
        if (
            origin_distance <= max_distance_threshold
            and destination_distance <= max_distance_threshold
        ):
            filtered_pairs.append((origin, destination))
        else:
            # Track why pairs were filtered
            if (
                origin_distance > max_distance_threshold
                and destination_distance > max_distance_threshold
            ):
                filtered_both += 1
            elif origin_distance > max_distance_threshold:
                filtered_origins += 1
            else:
                filtered_destinations += 1

    total_pairs_before = len(od_pairs)
    total_pairs_after = len(filtered_pairs)
    filtered_pairs_count = total_pairs_before - total_pairs_after
    filter_rate_pct = (
        (filtered_pairs_count / total_pairs_before * 100)
        if total_pairs_before > 0
        else 0.0
    )

    # Fail fast: assert at least one pair passes quality filter
    assert total_pairs_after > 0, (
        f"All OD pairs filtered out. No pairs passed quality threshold {max_distance_threshold}m. "
        f"Consider increasing the threshold or checking mapping quality."
    )

    logger.info("  ✅ Quality filtering complete:")
    logger.info(f"    - Pairs before: {total_pairs_before}")
    logger.info(f"    - Pairs after: {total_pairs_after}")
    logger.info(f"    - Filtered out: {filtered_pairs_count} ({filter_rate_pct:.1f}%)")
    logger.info(f"    - Filtered (origin > threshold): {filtered_origins}")
    logger.info(f"    - Filtered (destination > threshold): {filtered_destinations}")
    logger.info(f"    - Filtered (both > threshold): {filtered_both}")

    stats = {
        "total_pairs_before": total_pairs_before,
        "total_pairs_after": total_pairs_after,
        "filtered_pairs": filtered_pairs_count,
        "filter_rate_pct": filter_rate_pct,
        "max_distance_threshold_m": max_distance_threshold,
    }

    return filtered_pairs, stats


def save_translated_od_pairs(
    original_data: Dict[str, Any],
    translated_pairs_by_category: Dict[str, List[Tuple[int, int]]],
    translation_stats: Dict[str, Any],
    source_dataset: str,
    target_dataset: str,
    output_file: Path,
) -> None:
    """Save translated and filtered OD pairs with metadata

    Args:
        original_data: Original OD pairs data (to preserve structure)
        translated_pairs_by_category: Translated OD pairs organized by category
        translation_stats: Statistics from translation/filtering process
        source_dataset: Source dataset name
        target_dataset: Target dataset name
        output_file: Path to save translated OD pairs JSON file
    """
    logger.info(f"💾 Saving translated OD pairs to {output_file}")

    # Create extended format with translation metadata
    translated_data = {
        "dataset": source_dataset,
        "translated_dataset": target_dataset,
        "translation_applied": True,
        "translation_stats": translation_stats,
        "od_pairs_by_category": {},
    }

    # Convert tuples back to lists for JSON serialization
    for category, pairs in translated_pairs_by_category.items():
        translated_data["od_pairs_by_category"][category] = [
            list(pair) for pair in pairs
        ]

    # Preserve other fields from original data
    for key, value in original_data.items():
        if key not in [
            "dataset",
            "od_pairs_by_category",
            "translation_applied",
            "translation_stats",
            "translated_dataset",
        ]:
            translated_data[key] = value

    # Save to file
    with open(output_file, "w") as f:
        json.dump(translated_data, f, indent=2)

    logger.info("  ✅ Saved translated OD pairs with metadata")
    logger.info(f"    - Source dataset: {source_dataset}")
    logger.info(f"    - Target dataset: {target_dataset}")
    logger.info(f"    - Categories: {list(translated_pairs_by_category.keys())}")
    logger.info(
        f"    - Total pairs after filtering: {translation_stats['total_pairs_after']}"
    )
