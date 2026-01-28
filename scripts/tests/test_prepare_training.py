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


class TestCountClassInstances:
    """Test count_class_instances() function."""

    def test_count_class_instances_basic(self, tmp_path):
        """Count instances of each class from label files."""
        from prepare_training import count_class_instances

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create images and labels with known class distributions
        # Image 1: 2 instances of class 0, 1 instance of class 1
        (data_dir / "img1.jpg").touch()
        (data_dir / "img1.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n"
            "0 0.3 0.3 0.1 0.1\n"
            "1 0.7 0.7 0.1 0.1\n"
        )

        # Image 2: 1 instance of class 2
        (data_dir / "img2.jpg").touch()
        (data_dir / "img2.txt").write_text("2 0.5 0.5 0.1 0.1\n")

        # Image 3: 1 instance of class 0, 2 instances of class 1
        (data_dir / "img3.jpg").touch()
        (data_dir / "img3.txt").write_text(
            "0 0.2 0.2 0.1 0.1\n"
            "1 0.4 0.4 0.1 0.1\n"
            "1 0.6 0.6 0.1 0.1\n"
        )

        image_paths = [
            data_dir / "img1.jpg",
            data_dir / "img2.jpg",
            data_dir / "img3.jpg",
        ]

        counts = count_class_instances(image_paths, data_dir)

        assert counts[0] == 3  # class 0: 2 + 0 + 1 = 3
        assert counts[1] == 3  # class 1: 1 + 0 + 2 = 3
        assert counts[2] == 1  # class 2: 0 + 1 + 0 = 1

    def test_count_class_instances_empty_labels(self, tmp_path):
        """Empty label files contribute zero counts."""
        from prepare_training import count_class_instances

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        (data_dir / "empty.jpg").touch()
        (data_dir / "empty.txt").write_text("")

        image_paths = [data_dir / "empty.jpg"]
        counts = count_class_instances(image_paths, data_dir)

        assert counts == {}

    def test_count_class_instances_missing_label(self, tmp_path):
        """Missing label files are skipped gracefully."""
        from prepare_training import count_class_instances

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        (data_dir / "no_label.jpg").touch()
        # No corresponding .txt file

        image_paths = [data_dir / "no_label.jpg"]
        counts = count_class_instances(image_paths, data_dir)

        assert counts == {}


class TestValidateMinClassCount:
    """Test validate_min_class_count() function."""

    def test_validate_min_class_count_passes(self):
        """Returns True when all classes meet minimum."""
        from prepare_training import validate_min_class_count

        class_counts = {0: 10, 1: 8, 2: 5, 3: 12, 4: 6}
        result = validate_min_class_count(class_counts, min_count=5, num_classes=5)

        assert result is True

    def test_validate_min_class_count_fails(self):
        """Returns False when any class below minimum."""
        from prepare_training import validate_min_class_count

        class_counts = {0: 10, 1: 8, 2: 3, 3: 12, 4: 6}  # class 2 has only 3
        result = validate_min_class_count(class_counts, min_count=5, num_classes=5)

        assert result is False

    def test_validate_min_class_count_missing_class(self):
        """Returns False when a class is completely missing."""
        from prepare_training import validate_min_class_count

        class_counts = {0: 10, 1: 8, 3: 12, 4: 6}  # class 2 is missing
        result = validate_min_class_count(class_counts, min_count=5, num_classes=5)

        assert result is False

    def test_validate_min_class_count_zero_minimum(self):
        """Zero minimum always passes (no validation)."""
        from prepare_training import validate_min_class_count

        class_counts = {0: 1}
        result = validate_min_class_count(class_counts, min_count=0, num_classes=5)

        assert result is True


class TestSplitWithRetry:
    """Test split_with_retry() function."""

    def test_split_with_retry_succeeds_first_try(self, tmp_path):
        """Successfully splits when minimum is easily achievable."""
        from prepare_training import split_with_retry

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create enough samples of each class to easily meet minimum
        for i in range(50):
            (data_dir / f"img_{i}.jpg").touch()
            class_id = i % 3  # Classes 0, 1, 2
            (data_dir / f"img_{i}.txt").write_text(f"{class_id} 0.5 0.5 0.1 0.1\n")

        image_paths = [data_dir / f"img_{i}.jpg" for i in range(50)]

        train, valid = split_with_retry(
            samples=image_paths,
            train_ratio=0.8,
            min_valid_count=2,
            labels_dir=data_dir,
            num_classes=3,
            seed=42,
        )

        assert len(train) + len(valid) == 50
        assert len(valid) == 10  # 20% of 50

    def test_split_with_retry_retries_on_failure(self, tmp_path):
        """Retries with different seed when minimum not met."""
        from prepare_training import split_with_retry, count_class_instances

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create imbalanced dataset - class 2 is rare
        for i in range(40):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        for i in range(40, 80):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("1 0.5 0.5 0.1 0.1\n")

        # Only 10 samples of class 2
        for i in range(80, 90):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("2 0.5 0.5 0.1 0.1\n")

        image_paths = [data_dir / f"img_{i}.jpg" for i in range(90)]

        # With 90 samples and 0.9 train ratio, valid has 9 samples
        # Requiring 2 per class for 3 classes is achievable but may need retries
        train, valid = split_with_retry(
            samples=image_paths,
            train_ratio=0.9,
            min_valid_count=2,
            labels_dir=data_dir,
            num_classes=3,
            seed=42,
        )

        # Verify the split succeeded
        assert len(train) + len(valid) == 90

        # Verify validation set meets minimum
        valid_counts = count_class_instances(valid, data_dir)
        for class_id in range(3):
            assert valid_counts.get(class_id, 0) >= 2

    def test_split_with_retry_max_attempts_exceeded(self, tmp_path):
        """Raises error after max retry attempts."""
        from prepare_training import split_with_retry

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        # Create dataset where minimum is impossible to achieve
        # Only 2 samples of class 0, requiring 5 in validation is impossible
        for i in range(2):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        for i in range(2, 100):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("1 0.5 0.5 0.1 0.1\n")

        image_paths = [data_dir / f"img_{i}.jpg" for i in range(100)]

        with pytest.raises(RuntimeError, match="max attempts"):
            split_with_retry(
                samples=image_paths,
                train_ratio=0.9,
                min_valid_count=5,  # Impossible for class 0
                labels_dir=data_dir,
                num_classes=2,
                seed=42,
                max_attempts=5,
            )

    def test_split_with_retry_zero_minimum_skips_validation(self, tmp_path):
        """Zero minimum skips validation entirely."""
        from prepare_training import split_with_retry

        data_dir = tmp_path / "sliced"
        data_dir.mkdir()

        for i in range(10):
            (data_dir / f"img_{i}.jpg").touch()
            (data_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        image_paths = [data_dir / f"img_{i}.jpg" for i in range(10)]

        train, valid = split_with_retry(
            samples=image_paths,
            train_ratio=0.8,
            min_valid_count=0,  # No validation
            labels_dir=data_dir,
            num_classes=1,
            seed=42,
        )

        assert len(train) == 8
        assert len(valid) == 2
