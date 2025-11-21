#!/usr/bin/env python3
"""
Test script to verify trajectory validation fixes for LM-TAD evaluation.

This script tests the trajectory validation logic to ensure it properly filters
out invalid trajectories before they reach LM-TAD evaluation.

Usage:
    uv run python test_trajectory_validation.py \\
        --trajectory-file gene_abnormal_lmtad_spatial/porto_hoser/seed42/vanilla_seed44_spatial_abnormal.csv \\
        --vocab-size 6167
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools.evaluate_lmtad_spatial_abnormal import (
    validate_trajectory_for_lmtad,
    filter_valid_trajectories,
)
from simple_evaluate_with_lmtad import load_hoser_trajectories

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_trajectory_validation(trajectory_file: Path, vocab_size: int = 6167):
    """Test trajectory validation on real trajectory data.

    Args:
        trajectory_file: Path to trajectory CSV file
        vocab_size: LM-TAD vocabulary size
    """
    logger.info(f"🧪 Testing trajectory validation on: {trajectory_file}")
    logger.info(f"   Vocabulary size: {vocab_size}")

    # Load trajectories
    trajectories = load_hoser_trajectories(trajectory_file)
    logger.info(f"📊 Loaded {len(trajectories)} trajectories")

    # Test individual trajectory validation
    logger.info("🔍 Testing individual trajectory validation...")
    valid_count = 0
    invalid_count = 0
    invalid_reasons = []

    for i, trajectory in enumerate(trajectories):
        is_valid, reason = validate_trajectory_for_lmtad(trajectory, vocab_size)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_reasons.append((i, reason))

            if invalid_count <= 3:  # Show first few failures
                logger.warning(f"   Trajectory {i} failed: {reason}")
                logger.warning(
                    f"   Trajectory data: {trajectory[:10]}{'...' if len(trajectory) > 10 else ''}"
                )

    logger.info("📈 Individual validation results:")
    logger.info(
        f"   Valid: {valid_count} ({valid_count / len(trajectories) * 100:.1f}%)"
    )
    logger.info(
        f"   Invalid: {invalid_count} ({invalid_count / len(trajectories) * 100:.1f}%)"
    )

    # Test batch filtering
    logger.info("🔄 Testing batch trajectory filtering...")
    od_pair_labels = {}  # Empty for this test
    valid_trajectories, validation_failures, filtered_labels = (
        filter_valid_trajectories(trajectories, od_pair_labels, vocab_size)
    )

    logger.info("📊 Batch filtering results:")
    logger.info(f"   Input: {len(trajectories)} trajectories")
    logger.info(f"   Output: {len(valid_trajectories)} valid trajectories")
    logger.info(
        f"   Success rate: {len(valid_trajectories) / len(trajectories) * 100:.1f}%"
    )

    # Analyze failure patterns
    if validation_failures:
        logger.info("🔍 Analyzing failure patterns...")

        # Count failure types
        failure_types = {}
        for failure in validation_failures:
            # Extract failure type from reason
            if "Invalid road IDs" in failure:
                failure_types["Invalid road IDs"] = (
                    failure_types.get("Invalid road IDs", 0) + 1
                )
            elif "too short" in failure:
                failure_types["Too short"] = failure_types.get("Too short", 0) + 1
            elif "duplicates" in failure:
                failure_types["Duplicate roads"] = (
                    failure_types.get("Duplicate roads", 0) + 1
                )
            else:
                failure_types["Other"] = failure_types.get("Other", 0) + 1

        logger.info("   Failure breakdown:")
        for failure_type, count in sorted(
            failure_types.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = count / len(validation_failures) * 100
            logger.info(f"     {failure_type}: {count} ({percentage:.1f}%)")

    # Test edge cases
    logger.info("🧪 Testing edge cases...")
    test_cases = [
        # Valid case
        ([1, 2, 3, 4], True, "Valid short trajectory"),
        # Invalid road ID
        ([1, 2, vocab_size + 100, 4], False, "Road ID exceeds vocab_size"),
        # Empty trajectory
        ([], False, "Empty trajectory"),
        # Too short
        ([1], False, "Too short"),
        # Consecutive duplicates
        ([1, 1, 2, 3], False, "Consecutive duplicates"),
        # Excessive duplicates
        ([1, 2, 1, 2, 1, 2], False, "Excessive duplicates"),
    ]

    for trajectory, expected_valid, description in test_cases:
        is_valid, reason = validate_trajectory_for_lmtad(trajectory, vocab_size)
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
        logger.info(f"   {status} {description}: {trajectory} -> {is_valid}")

        if is_valid != expected_valid:
            logger.error(
                f"      Expected: {expected_valid}, Got: {is_valid}, Reason: {reason}"
            )

    logger.info("🎉 Trajectory validation testing completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Test trajectory validation for LM-TAD evaluation"
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        required=True,
        help="Path to trajectory CSV file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=6167,
        help="LM-TAD vocabulary size (default: 6167 for Porto)",
    )

    args = parser.parse_args()

    if not args.trajectory_file.exists():
        logger.error(f"Trajectory file not found: {args.trajectory_file}")
        return 1

    test_trajectory_validation(args.trajectory_file, args.vocab_size)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
