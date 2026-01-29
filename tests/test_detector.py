"""Tests for the DarknetDetector module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from inference.config import Settings


class TestDarknetDetectorSingleton:
    """Test cases for DarknetDetector singleton pattern."""

    def test_get_instance_returns_same_object(self):
        """Test that get_instance returns the same instance."""
        from inference.detector import DarknetDetector

        # Reset singleton for test isolation
        DarknetDetector._instance = None

        with patch.object(DarknetDetector, '__init__', return_value=None):
            instance1 = DarknetDetector.get_instance()
            instance2 = DarknetDetector.get_instance()
            assert instance1 is instance2

    def test_is_loaded_false_initially(self):
        """Test is_loaded returns False when no instance exists."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None
        assert DarknetDetector.is_loaded() is False

    def test_is_loaded_true_after_init(self):
        """Test is_loaded returns True after instance created."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = MagicMock()
        assert DarknetDetector.is_loaded() is True
        DarknetDetector._instance = None  # cleanup


class TestDarknetDetectorValidation:
    """Test cases for model file validation."""

    def test_missing_cfg_file_raises_error(self, tmp_path):
        """Test that missing cfg file raises FileNotFoundError."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "nonexistent.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"

        # Create weights and data files
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        (tmp_path / "classes.names").write_text("class1\n")

        with pytest.raises(FileNotFoundError, match="cfg"):
            DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )

    def test_missing_weights_file_raises_error(self, tmp_path):
        """Test that missing weights file raises FileNotFoundError."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "nonexistent.weights"
        data_path = tmp_path / "model.data"

        # Create cfg and data files
        cfg_path.touch()
        data_path.write_text("names = classes.names\n")
        (tmp_path / "classes.names").write_text("class1\n")

        with pytest.raises(FileNotFoundError, match="weights"):
            DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )

    def test_missing_data_file_raises_error(self, tmp_path):
        """Test that missing data file raises FileNotFoundError."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "nonexistent.data"

        # Create cfg and weights files
        cfg_path.touch()
        weights_path.touch()

        with pytest.raises(FileNotFoundError, match="data"):
            DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )


class TestDarknetLibLoading:
    """Test cases for darknet library loading."""

    def test_darknet_import_error_gives_clear_message(self, tmp_path):
        """Test that darknet import failure gives clear error message."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        (tmp_path / "classes.names").write_text("class1\n")

        with patch.dict('sys.modules', {'darknet': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'darknet'")):
                with pytest.raises(RuntimeError, match="darknet"):
                    DarknetDetector(
                        cfg_path=cfg_path,
                        weights_path=weights_path,
                        data_path=data_path,
                    )

    def test_darknet_lib_not_found_gives_clear_message(self, tmp_path):
        """Test that missing .so file gives clear error message."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        (tmp_path / "classes.names").write_text("class1\n")

        with patch.dict('sys.modules', {'darknet': None}):
            with patch('builtins.__import__', side_effect=OSError("libdarknet.so: cannot open")):
                with pytest.raises(RuntimeError, match="libdarknet"):
                    DarknetDetector(
                        cfg_path=cfg_path,
                        weights_path=weights_path,
                        data_path=data_path,
                    )


class TestClassNamesLoading:
    """Test cases for class names loading from .data file."""

    def test_load_class_names_success(self, tmp_path):
        """Test successful loading of class names."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        # Create test files
        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"
        names_path = tmp_path / "classes.names"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        names_path.write_text("class1\nclass2\nclass3\n")

        # Mock darknet module
        mock_dn = MagicMock()
        mock_dn.load_network.return_value = MagicMock()
        mock_dn.network_width.return_value = 416
        mock_dn.network_height.return_value = 416

        with patch.dict('sys.modules', {'darknet': mock_dn}):
            detector = DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )
            assert detector.class_names == ["class1", "class2", "class3"]

    def test_load_class_names_missing_names_entry(self, tmp_path):
        """Test error when .data file has no names entry."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("classes = 5\n")  # No names entry

        mock_dn = MagicMock()
        with patch.dict('sys.modules', {'darknet': mock_dn}):
            with pytest.raises(ValueError, match="No 'names' entry"):
                DarknetDetector(
                    cfg_path=cfg_path,
                    weights_path=weights_path,
                    data_path=data_path,
                )

    def test_load_class_names_absolute_path(self, tmp_path):
        """Test loading class names with absolute path in .data file."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"
        names_path = tmp_path / "absolute_classes.names"

        cfg_path.touch()
        weights_path.touch()
        names_path.write_text("abs_class1\nabs_class2\n")
        data_path.write_text(f"names = {names_path}\n")

        mock_dn = MagicMock()
        mock_dn.load_network.return_value = MagicMock()
        mock_dn.network_width.return_value = 416
        mock_dn.network_height.return_value = 416

        with patch.dict('sys.modules', {'darknet': mock_dn}):
            detector = DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )
            assert detector.class_names == ["abs_class1", "abs_class2"]


class TestDetectorDetectMethod:
    """Test cases for DarknetDetector.detect method."""

    def test_detect_returns_scaled_coordinates(self, tmp_path):
        """Test that detect scales coordinates correctly."""
        from inference.detector import DarknetDetector
        import numpy as np

        DarknetDetector._instance = None

        # Setup files
        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"
        names_path = tmp_path / "classes.names"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        names_path.write_text("class1\n")

        # Mock darknet
        mock_dn = MagicMock()
        mock_dn.load_network.return_value = MagicMock()
        mock_dn.network_width.return_value = 416
        mock_dn.network_height.return_value = 416
        mock_dn.detect_image.return_value = [
            ("class1", "95.5", (208, 208, 100, 100))
        ]

        with patch.dict('sys.modules', {'darknet': mock_dn}):
            detector = DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )

            # Test with 832x832 image (2x network size)
            image = np.zeros((832, 832, 3), dtype=np.uint8)
            results = detector.detect(image)

            assert len(results) == 1
            class_name, conf, bbox = results[0]
            assert class_name == "class1"
            # Coordinates should be scaled by 2
            assert bbox[0] == 416  # 208 * 2
            assert bbox[1] == 416  # 208 * 2

    def test_get_network_size(self, tmp_path):
        """Test get_network_size returns correct dimensions."""
        from inference.detector import DarknetDetector

        DarknetDetector._instance = None

        cfg_path = tmp_path / "model.cfg"
        weights_path = tmp_path / "model.weights"
        data_path = tmp_path / "model.data"
        names_path = tmp_path / "classes.names"

        cfg_path.touch()
        weights_path.touch()
        data_path.write_text("names = classes.names\n")
        names_path.write_text("class1\n")

        mock_dn = MagicMock()
        mock_dn.load_network.return_value = MagicMock()
        mock_dn.network_width.return_value = 608
        mock_dn.network_height.return_value = 608

        with patch.dict('sys.modules', {'darknet': mock_dn}):
            detector = DarknetDetector(
                cfg_path=cfg_path,
                weights_path=weights_path,
                data_path=data_path,
            )
            w, h = detector.get_network_size()
            assert w == 608
            assert h == 608
