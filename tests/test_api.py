"""Tests for the FastAPI endpoints."""

import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from inference.main import app
from inference import __version__


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response has correct format."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == __version__
