"""API gateway entrypoint for Vercel Python runtime."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, application  # noqa: E402

__all__ = ["app", "application"]
