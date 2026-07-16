"""Configuration and preferences management for AIDaS."""

import json
import os
from pathlib import Path
import tempfile


class Config:
    """Handle loading and saving user preferences."""
    
    CONFIG_DIR = Path.home() / ".aidas"
    CONFIG_FILE = CONFIG_DIR / "preferences.json"
    
    # Default preferences
    DEFAULTS = {
        "theme": "clam",
        "rscript_path": "",
        "check_for_updates": True,
        "last_successful_update_check": 0,
    }
    
    def __init__(self):
        self._ensure_config_dir()
        self.prefs = self._load_prefs()
    
    @classmethod
    def _ensure_config_dir(cls):
        """Create config directory if it doesn't exist."""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_prefs(self):
        """Load preferences, retaining defaults added by newer AIDaS versions."""
        prefs = self.DEFAULTS.copy()
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    prefs.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass
        return prefs
    
    def save(self):
        """Atomically save preferences so an interrupted update cannot corrupt them."""
        temp_path = None
        try:
            self._ensure_config_dir()
            descriptor, temp_name = tempfile.mkstemp(
                prefix="preferences.",
                suffix=".tmp",
                dir=self.CONFIG_DIR,
            )
            temp_path = Path(temp_name)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
                json.dump(self.prefs, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.CONFIG_FILE)
            temp_path = None
        except OSError as e:
            print(f"Warning: Could not save preferences: {e}")
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
    
    def get(self, key, default=None):
        """Get a preference value."""
        return self.prefs.get(key, default)
    
    def set(self, key, value):
        """Set a preference value."""
        self.prefs[key] = value
        self.save()
