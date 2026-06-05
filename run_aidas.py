#!/usr/bin/env python3
"""Entry point for the AIDaS OCT Image Processing application.

Usage:
    python run_aidas.py
"""

import sys
import os

# Ensure the workspace root is on the path so `aidas` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aidas.app import AIDaSApp


def main() -> None:
    '''Launch the AIDaS application.'''
    app = AIDaSApp()
    app.mainloop()

if __name__ == "__main__":
    main()
