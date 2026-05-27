"""Configuration and preferences management for AIDaS."""

import json
from pathlib import Path


class Config:
    """Handle loading and saving user preferences."""
    
    CONFIG_DIR = Path.home() / ".aidas"
    CONFIG_FILE = CONFIG_DIR / "preferences.json"
    
    # Default preferences
    DEFAULTS = {
        "theme": "clam",
        "rscript_path": "",
    }
    
    def __init__(self):
        self._ensure_config_dir()
        self.prefs = self._load_prefs()
    
    @classmethod
    def _ensure_config_dir(cls):
        """Create config directory if it doesn't exist."""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_prefs(self):
        """Load preferences from file or return defaults."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self.DEFAULTS.copy()
        return self.DEFAULTS.copy()
    
    def save(self):
        """Save current preferences to file."""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.prefs, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save preferences: {e}")
    
    def get(self, key, default=None):
        """Get a preference value."""
        return self.prefs.get(key, default)
    
    def set(self, key, value):
        """Set a preference value."""
        self.prefs[key] = value
        self.save()
