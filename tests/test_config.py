"""Tests for configuration management."""


class TestSettingsFromEnvFile:
    """Test loading settings from .env file."""

    def test_loads_from_env_file(self, tmp_path, monkeypatch):
        """Settings loads values from .env file."""
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("FIBER_PORT=9999\nFIBER_CONFIDENCE_THRESHOLD=0.8\n")

        # Change to tmp_path so .env is found
        monkeypatch.chdir(tmp_path)

        # Re-import to pick up new .env
        from inference.config import Settings

        settings = Settings()

        assert settings.port == 9999
        assert settings.confidence_threshold == 0.8

    def test_env_var_overrides_env_file(self, tmp_path, monkeypatch):
        """Environment variables take precedence over .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("FIBER_PORT=9999\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FIBER_PORT", "7777")

        from inference.config import Settings

        settings = Settings()

        assert settings.port == 7777  # env var wins
