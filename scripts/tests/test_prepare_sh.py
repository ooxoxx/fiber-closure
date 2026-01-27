"""Tests for prepare.sh data preparation script."""

import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parent.parent.parent / "prepare.sh"


class TestPrepareScript:
    """Tests for the prepare.sh shell script."""

    def test_script_exists_and_executable(self):
        """prepare.sh exists and is executable."""
        assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"
        assert SCRIPT_PATH.stat().st_mode & 0o111, "Script is not executable"

    def test_help_option(self):
        """--help displays usage information."""
        result = subprocess.run(
            [str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Usage" in result.stdout or "usage" in result.stdout

    def test_validates_missing_raw_images(self, tmp_path):
        """Exits with error when raw images directory missing."""
        result = subprocess.run(
            [str(SCRIPT_PATH), "--raw-images", str(tmp_path / "nonexistent")],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "Error" in output or "error" in output

    def test_validates_missing_raw_labels(self, tmp_path):
        """Exits with error when raw labels directory missing."""
        # Create images dir but not labels
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "test.jpg").touch()

        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--raw-images", str(images_dir),
                "--raw-labels", str(tmp_path / "nonexistent")
            ],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "Error" in output or "error" in output

    def test_validates_empty_images_directory(self, tmp_path):
        """Exits with error when images directory is empty."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--raw-images", str(images_dir),
                "--raw-labels", str(labels_dir)
            ],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "Error" in output or "error" in output or "No images" in output


class TestPrepareScriptEndToEnd:
    """End-to-end integration tests for prepare.sh."""

    @pytest.fixture
    def setup_raw_data(self, tmp_path):
        """Create minimal raw data structure for testing."""
        raw_images = tmp_path / "raw" / "images"
        raw_labels = tmp_path / "raw" / "labels"
        raw_images.mkdir(parents=True)
        raw_labels.mkdir(parents=True)

        # Create a small test image (10x10 black image)
        import numpy as np
        try:
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(raw_images / "test_001.jpg"), img)
        except ImportError:
            pytest.skip("cv2 not available for creating test images")

        # Create corresponding label
        (raw_labels / "test_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        return tmp_path

    def test_end_to_end_creates_outputs(self, setup_raw_data):
        """Full pipeline creates sliced images and train/valid files."""
        tmp_path = setup_raw_data
        sliced_dir = tmp_path / "sliced"
        output_dir = tmp_path

        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--raw-images", str(tmp_path / "raw" / "images"),
                "--raw-labels", str(tmp_path / "raw" / "labels"),
                "--sliced-dir", str(sliced_dir),
                "--output-dir", str(output_dir),
                "--tile-size", "64",  # Small tiles for test
                "--overlap", "0.2"
            ],
            capture_output=True,
            text=True
        )

        # Print output for debugging if failed
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check outputs exist
        assert sliced_dir.exists(), "Sliced directory not created"
        assert (output_dir / "train.txt").exists(), "train.txt not created"
        assert (output_dir / "valid.txt").exists(), "valid.txt not created"

        # Check sliced images were created
        sliced_images = list(sliced_dir.glob("*.jpg"))
        assert len(sliced_images) > 0, "No sliced images created"

        # Check train.txt has content
        train_content = (output_dir / "train.txt").read_text()
        valid_content = (output_dir / "valid.txt").read_text()
        total_lines = len(train_content.strip().split('\n')) + len(valid_content.strip().split('\n'))
        assert total_lines > 0, "train.txt and valid.txt are empty"
