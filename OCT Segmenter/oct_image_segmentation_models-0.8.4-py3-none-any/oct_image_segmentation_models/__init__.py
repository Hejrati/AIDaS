import importlib.metadata
import os

# Semantic Versioning 2.0.0: https://semver.org/
# 1. MAJOR version when you make incompatible API changes;
# 2. MINOR version when you add functionality in a backwards-compatible manner;
# 3. PATCH version when you make backwards-compatible bug fixes.
__version__ = "0.8.2"
# __version__ = importlib.metadata.version("oct_image_segmentation_models")

import matplotlib

# There is a memory buildup in matplotlib. For now current solution is to use "Agg"
# For reference: https://github.com/matplotlib/mplfinance/issues/386
matplotlib.use("Agg")

# Some systems have the TCL_LIBRARY environment set which causes version conflicts. Thus, we unset
# it during initialization
if os.environ.get("TCL_LIBRARY"):
    del os.environ["TCL_LIBRARY"]
