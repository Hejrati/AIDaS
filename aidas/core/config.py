"""Configuration and preferences management for AIDaS."""

import json
import math
import os
from pathlib import Path
import tempfile


STEP4_AUTO_DETECTION_DEFAULTS = {
    "step4_auto_start_min": 70,
    "step4_auto_start_max": 90,
    "step4_auto_end_min": 90,
    "step4_auto_end_max": 110,
    "step4_auto_savgol_window": 9,
    "step4_auto_confidence_percent": 66.0,
    "step4_auto_min_quadratic_r2": 0.55,
    "step4_auto_consistency_tolerance": 6.0,
}


def validate_step4_auto_detection_preferences(values):
    """Parse and validate the user-tunable Step 4 auto-detection settings."""

    try:
        parsed = {
            "step4_auto_start_min": int(values["step4_auto_start_min"]),
            "step4_auto_start_max": int(values["step4_auto_start_max"]),
            "step4_auto_end_min": int(values["step4_auto_end_min"]),
            "step4_auto_end_max": int(values["step4_auto_end_max"]),
            "step4_auto_savgol_window": int(values["step4_auto_savgol_window"]),
            "step4_auto_confidence_percent": float(
                values["step4_auto_confidence_percent"]
            ),
            "step4_auto_min_quadratic_r2": float(
                values["step4_auto_min_quadratic_r2"]
            ),
            "step4_auto_consistency_tolerance": float(
                values["step4_auto_consistency_tolerance"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every auto-detection parameter must be numeric.") from exc

    start_min = parsed["step4_auto_start_min"]
    start_max = parsed["step4_auto_start_max"]
    end_min = parsed["step4_auto_end_min"]
    end_max = parsed["step4_auto_end_max"]
    window = parsed["step4_auto_savgol_window"]
    confidence = parsed["step4_auto_confidence_percent"]
    minimum_r2 = parsed["step4_auto_min_quadratic_r2"]
    tolerance = parsed["step4_auto_consistency_tolerance"]

    if not all(math.isfinite(value) for value in (confidence, minimum_r2, tolerance)):
        raise ValueError("Confidence, R-squared, and tolerance values must be finite numbers.")

    if start_min < 1 or start_max < start_min:
        raise ValueError("The start search range must contain positive samples in ascending order.")
    if end_min < 1 or end_max < end_min:
        raise ValueError("The end search range must contain positive samples in ascending order.")
    if start_min >= end_max:
        raise ValueError("The start search range must begin before the end search range finishes.")
    if window < 3 or window % 2 == 0:
        raise ValueError("The Savitzky-Golay window length must be an odd integer of 3 or greater.")
    if not 0.0 <= confidence <= 100.0:
        raise ValueError("The acceptance confidence must be between 0% and 100%.")
    if not 0.0 <= minimum_r2 <= 1.0:
        raise ValueError("The minimum quadratic R-squared value must be between 0 and 1.")
    if tolerance < 0.0:
        raise ValueError("The consistency tolerance must be zero samples or greater.")
    return parsed


class Config:
    """Handle loading and saving user preferences."""
    
    CONFIG_DIR = Path.home() / ".aidas"
    CONFIG_FILE = CONFIG_DIR / "preferences.json"
    
    # Default preferences
    DEFAULTS = {
        "theme": "clam",
        # CustomTkinter appearance is intentionally separate from the legacy
        # ttk theme preference.  Keeping both keys lets older preference files
        # load unchanged while the UI moves to System/Light/Dark modes.
        "appearance_mode": "System",
        # Presentation is independent of color appearance. Classic restores
        # the pre-v3 native shell while every processing workflow stays on the
        # current implementation.
        "interface_mode": "Modern",
        "sdb_raw_width": 768,
        "sdb_raw_height": 1200,
        "sdb_raw_offset": 1050,
        "sdb_little_endian": True,
        "rscript_path": "",
        "r_main_script_path": "",
        "r_output_script_path": "",
        "check_for_updates": True,
        "last_successful_update_check": 0,
        **STEP4_AUTO_DETECTION_DEFAULTS,
    }
    
    def __init__(self):
        self._ensure_config_dir()
        self.prefs = self._load_prefs()

    @classmethod
    def peek(cls, key, default=None):
        """Read one saved value without creating directories or raising I/O errors.

        Startup uses this lightweight path to choose the splash presentation.
        Full preference initialization still happens under the visible splash.
        """

        fallback = cls.DEFAULTS.get(key, default)
        try:
            if not cls.CONFIG_FILE.is_file():
                return fallback
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                return loaded.get(key, fallback)
        except (json.JSONDecodeError, OSError):
            pass
        return fallback
    
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
                    if (
                        "step4_auto_confidence_percent" not in loaded
                        and "step4_auto_confidence" in loaded
                    ):
                        try:
                            legacy_confidence = float(
                                loaded["step4_auto_confidence"]
                            )
                        except (TypeError, ValueError):
                            pass
                        else:
                            prefs["step4_auto_confidence_percent"] = (
                                legacy_confidence * 100.0
                                if 0.0 <= legacy_confidence <= 1.0
                                else legacy_confidence
                            )
                    if (
                        "step4_auto_consistency_tolerance" not in loaded
                        and "step4_auto_consistency_tolerance_percent" in loaded
                    ):
                        prefs["step4_auto_consistency_tolerance"] = loaded[
                            "step4_auto_consistency_tolerance_percent"
                        ]
                    # Polynomial order is fixed at degree 2. Confidence is a
                    # percentage, while consistency is measured in samples.
                    for obsolete_key in (
                        "step4_auto_savgol_polyorder",
                        "step4_auto_confidence",
                        "step4_auto_consistency_tolerance_percent",
                    ):
                        prefs.pop(obsolete_key, None)
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
