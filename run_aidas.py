#!/usr/bin/env python3
"""Entry point for the AIDaS OCT Image Processing application.

Usage:
    python run_aidas.py
"""

import sys
import os

# Ensure the workspace root is on the path so `aidas` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AI_WORKER_ARG = "--aidas-ai-worker"


def main() -> None:
    '''Launch the AIDaS application or a private AI worker process.'''
    if len(sys.argv) > 1 and sys.argv[1] == AI_WORKER_ARG:
        from aidas.ai_for_aidas_cli import main as ai_worker_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        ai_worker_main()
        return

    from aidas.app import AIDaSApp

    app = AIDaSApp()
    app.mainloop()

if __name__ == "__main__":
    main()
