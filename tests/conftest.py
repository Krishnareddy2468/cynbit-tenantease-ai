"""Shared pytest fixtures: generate the portfolio once per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_generator  # noqa: E402


@pytest.fixture(scope="session")
def portfolio():
    """A freshly generated synthetic portfolio (seeded, deterministic)."""
    return data_generator.generate()
