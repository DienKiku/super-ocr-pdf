"""
Configuration Manager for Super OCR & High-Res PDF Studio.
Persists application settings, API keys, and user preferences.
"""

import json
import os
from typing import Any, Optional


class ConfigManager:
    """Manages application settings and credentials stored in ~/.super_ocr_pdf/config.json."""

    _instance: Optional["ConfigManager"] = None

    def __init__(self):
        home_dir = os.path.expanduser("~")
        self.config_dir = os.path.join(home_dir, ".super_ocr_pdf")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.data = {
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
            "gemini_model": "gemini-3.6-flash",
            "preferred_engine": "offline",
        }
        self.load()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def load(self):
        """Load configuration from JSON file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        self.data.update(saved)
                # Auto-migrate deprecated models
                if self.data.get("gemini_model") in (
                    "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp"
                ):
                    self.data["gemini_model"] = "gemini-3.6-flash"
                    self.save()
                # Auto-migrate deprecated engine names
                if self.data.get("preferred_engine") in ("vietnamese", "rapid", "vietocr"):
                    self.data["preferred_engine"] = "offline"
                    self.save()
                elif self.data.get("preferred_engine") == "gemini":
                    self.data["preferred_engine"] = "online"
                    self.save()
            except Exception as e:
                print(f"Warning loading config: {e}")

    def save(self):
        """Save current configuration to JSON file."""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()

    def get_gemini_api_key(self) -> str:
        key = self.data.get("gemini_api_key", "").strip()
        if not key:
            key = os.environ.get("GEMINI_API_KEY", "").strip()
        return key

    def set_gemini_api_key(self, key: str):
        self.data["gemini_api_key"] = key.strip()
        self.save()

    def get_gemini_model(self) -> str:
        return self.data.get("gemini_model", "gemini-2.5-flash")

    def set_gemini_model(self, model: str):
        self.data["gemini_model"] = model.strip()
        self.save()
