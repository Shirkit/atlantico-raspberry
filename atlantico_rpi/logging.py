"""Centralized logging setup for Atlantico RPi.

This module exposes a single idempotent helper `setup_logging()` which ensures
there is a stdout StreamHandler and (optionally) a FileHandler writing to
`run/logs/device.log`. The function is safe to call multiple times.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

LOG_PATH = os.environ.get("ATLANTICO_DEVICE_LOG", os.path.join("run", "logs", "device.log"))

_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")


def setup_logging(force_file: bool = False) -> None:
    """Idempotently add a stdout StreamHandler and optional FileHandler.

    If force_file is True or the env var ATLANTICO_DEVICE_CREATE_FILE is set to
    '1', ensure a FileHandler is added writing to LOG_PATH.
    """

    if '--connect' in sys.argv or (__name__ == '__main__'):
        # Prefer creating the file handler for CLI runs
        os.environ.setdefault('ATLANTICO_DEVICE_CREATE_FILE', '1')

    rl = logging.getLogger()
    rl.setLevel(logging.INFO)

    # ensure stdout stream handler
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) is sys.stdout for h in rl.handlers):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(_fmt)
        sh.setLevel(logging.INFO)
        rl.addHandler(sh)

    want_file = force_file or os.environ.get('ATLANTICO_DEVICE_CREATE_FILE', '0') == '1'
    abs_log_path = os.path.abspath(LOG_PATH)
    if want_file:
        # create directory first
        try:
            os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
        except Exception:
            pass
        # add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == abs_log_path for h in rl.handlers):
            try:
                fh = logging.FileHandler(LOG_PATH)
                fh.setFormatter(_fmt)
                fh.setLevel(logging.INFO)
                rl.addHandler(fh)
            except Exception:
                # best-effort
                pass


__all__ = ['setup_logging', 'LOG_PATH']
