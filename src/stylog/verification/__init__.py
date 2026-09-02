"""Stylog verification — deterministic pure-Python verifier fitting.

The fitting stack (``spec`` + ``fit``) imports nothing heavier than ``math``:
no NumPy, no BLAS, no sklearn. ``analysis/verify.py`` holds the scoring core.
"""

from __future__ import annotations
