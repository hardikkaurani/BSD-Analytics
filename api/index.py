"""Vercel Serverless Function Entrypoint."""

import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import handler  # noqa: E402

__all__ = ["handler"]
