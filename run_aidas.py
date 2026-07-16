#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Machine Vision and Pattern Recognition Lab, Wayne State University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Entry point for the AIDaS OCT Image Processing application.

Usage:
    python run_aidas.py
"""

import sys
import os
import traceback

# Ensure the workspace root is on the path so `aidas` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AI_WORKER_ARG = "--aidas-ai-worker"
AI_WORKER_ERROR_LOG_ENV = "AIDAS_AI_WORKER_ERROR_LOG"


def _record_worker_startup(message: str) -> None:
    """Write frozen-worker bootstrap diagnostics into its temporary directory."""
    log_path = os.environ.get(AI_WORKER_ERROR_LOG_ENV)
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(str(message).rstrip() + "\n")
    except OSError:
        pass


def main() -> int:
    '''Launch the AIDaS application or a private AI worker process.'''
    if len(sys.argv) > 1 and sys.argv[1] == AI_WORKER_ARG:
        _record_worker_startup("Worker entry point selected.")
        try:
            from aidas.ai.worker import main as ai_worker_main

            _record_worker_startup("Worker module imported.")
            sys.argv = [sys.argv[0], *sys.argv[2:]]
            return int(ai_worker_main() or 0)
        except BaseException:
            _record_worker_startup(traceback.format_exc())
            raise

    from aidas.app import main as app_main

    return app_main()

if __name__ == "__main__":
    raise SystemExit(main())
