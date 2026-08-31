"""Shared test plumbing: the saved dataset rows, and the opt-in live tests.

Everything runs offline by default. `--live` additionally runs the tests that
hit external services — the HuggingFace datasets-server and the infini-gram
API — which is where an upstream schema change gets caught before it reaches
a sampling run.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS = Path(__file__).resolve().parent / "fixtures" / "rows"


def row_fixture(target: str, stage: str) -> dict:
    """The saved row for one registry stage, captured by
    scripts/capture_row_fixtures.py."""
    path = ROWS / f"{target}.{stage}.json"
    assert path.exists(), (
        f"no saved row for {target}/{stage} — run scripts/capture_row_fixtures.py"
    )
    return json.loads(path.read_text())


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="also run tests that fetch from the HuggingFace datasets-server",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: fetches from the HuggingFace datasets-server")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live (fetches from the datasets-server)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
