"""Tests for prepare_training.py negative sample sampling logic."""

import tempfile
from pathlib import Path

import pytest

# Import will be added after refactoring
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestClassifySamples:
    """Test classify_samples() function."""

    def test_classify_samples_basic(self, tmp_path):
        """Correctly separates positive and negative samples."""
        from prepare_training import classify_samples

        # Images and labels in the same directory
        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create 3 positive samples (non-empty labels)
        for i in range(3):
            (data_dir / f"pos_{i}.jpg").touch()
            (data_dir / f"pos_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Create 5 negative samples (empty labels)
        for i in range(5):
            (data_dir / f"neg_{i}.jpg").touch()
            (data_dir / f"neg_{i}.txt").write_text("")

        positives, negatives = classify_samples(data_dir, data_dir)

        assert len(positives) == 3
        assert len(negatives) == 5
        assert all("pos_" in p.stem for p in positives)
        assert all("neg_" in n.stem for n in negatives)

    def test_classify_samples_whitespace_only_is_negative(self, tmp_path):
        """Labels with only whitespace are classified as negative."""
        from prepare_training import classify_samples

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        (data_dir / "whitespace.jpg").touch()
        (data_dir / "whitespace.txt").write_text("   \n\n  \n")

        positives, negatives = classify_samples(data_dir, data_dir)

        assert len(positives) == 0
        assert len(negatives) == 1

    def test_classify_samples_no_label_file_excluded(self, tmp_path):
        """Images without label files are excluded."""
        from prepare_training import classify_samples

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        (data_dir / "no_label.jpg").touch()
        # No corresponding label file

        positives, negatives = classify_samples(data_dir, data_dir)

        assert len(positives) == 0
        assert len(negatives) == 0


class TestSampleNegatives:
    """Test sample_negatives() function."""

    def test_sample_negatives_basic(self):
        """Sample correct number of negatives based on ratio."""
        from prepare_training import sample_negatives

        positives = [Path(f"pos_{i}.jpg") for i in range(100)]
        negatives = [Path(f"neg_{i}.jpg") for i in range(900)]

        # 10% negative ratio means: neg / (pos + neg) = 0.1
        # So neg = pos * ratio / (1 - ratio) = 100 * 0.1 / 0.9 ≈ 11
        sampled = sample_negatives(negatives, positives, ratio=0.1, seed=42)

        expected_count = int(len(positives) * 0.1 / 0.9)
        assert len(sampled) == expected_count

    def test_sample_negatives_ratio_zero(self):
        """Ratio 0 returns no negatives."""
        from prepare_training import sample_negatives

        positives = [Path(f"pos_{i}.jpg") for i in range(10)]
        negatives = [Path(f"neg_{i}.jpg") for i in range(90)]

        sampled = sample_negatives(negatives, positives, ratio=0.0, seed=42)

        assert len(sampled) == 0

    def test_sample_negatives_ratio_one(self):
        """Ratio 1 returns all negatives (edge case, all negative)."""
        from prepare_training import sample_negatives

        positives = [Path(f"pos_{i}.jpg") for i in range(10)]
        negatives = [Path(f"neg_{i}.jpg") for i in range(90)]

        # ratio=1.0 means 100% negatives, which is impossible with positives
        # The function should return all available negatives
        sampled = sample_negatives(negatives, positives, ratio=1.0, seed=42)

        assert len(sampled) == len(negatives)

    def test_sample_negatives_not_enough_negatives(self):
        """When not enough negatives available, return all."""
        from prepare_training import sample_negatives

        positives = [Path(f"pos_{i}.jpg") for i in range(100)]
        negatives = [Path(f"neg_{i}.jpg") for i in range(5)]

        # Want 11 negatives but only 5 available
        sampled = sample_negatives(negatives, positives, ratio=0.1, seed=42)

        assert len(sampled) == 5  # All available

    def test_sample_negatives_reproducible(self):
        """Same seed produces same results."""
        from prepare_training import sample_negatives

        positives = [Path(f"pos_{i}.jpg") for i in range(10)]
        negatives = [Path(f"neg_{i}.jpg") for i in range(100)]

        sampled1 = sample_negatives(negatives, positives, ratio=0.5, seed=123)
        sampled2 = sample_negatives(negatives, positives, ratio=0.5, seed=123)

        assert sampled1 == sampled2


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end(self, tmp_path):
        """Full workflow produces correct train/valid splits."""
        from prepare_training import classify_samples, sample_negatives

        # Images and labels in the same directory
        data_dir = tmp_path / "sliced"
        output_dir = tmp_path / "output"
        data_dir.mkdir(parents=True)

        # Create 10 positive samples
        for i in range(10):
            (data_dir / f"pos_{i}.jpg").touch()
            (data_dir / f"pos_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Create 90 negative samples
        for i in range(90):
            (data_dir / f"neg_{i}.jpg").touch()
            (data_dir / f"neg_{i}.txt").write_text("")

        # Classify
        positives, negatives = classify_samples(data_dir, data_dir)
        assert len(positives) == 10
        assert len(negatives) == 90

        # Sample negatives at 10% ratio
        sampled_negatives = sample_negatives(negatives, positives, ratio=0.1, seed=42)

        # Expected: 10 * 0.1 / 0.9 ≈ 1
        expected_neg_count = int(10 * 0.1 / 0.9)
        assert len(sampled_negatives) == expected_neg_count

        # Total samples
        total = positives + sampled_negatives
        actual_ratio = len(sampled_negatives) / len(total)
        assert actual_ratio <= 0.15  # Allow some tolerance

    def test_negative_ratio_preserves_all_positives(self, tmp_path):
        """All positive samples are preserved regardless of ratio."""
        from prepare_training import classify_samples, sample_negatives

        # Images and labels in the same directory
        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create positives
        for i in range(50):
            (data_dir / f"pos_{i}.jpg").touch()
            (data_dir / f"pos_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Create negatives
        for i in range(500):
            (data_dir / f"neg_{i}.jpg").touch()
            (data_dir / f"neg_{i}.txt").write_text("")

        positives, negatives = classify_samples(data_dir, data_dir)
        sampled_negatives = sample_negatives(negatives, positives, ratio=0.1, seed=42)

        # All 50 positives must be preserved
        assert len(positives) == 50
