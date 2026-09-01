"""Run every JavaScript suite under tests/site/ inside the pytest run.

The page is one file with no build step, so the assertions about it are
JavaScript. This puts them in the same `pytest` invocation as everything else,
which is what makes one command — and one CI step — the whole check.

Discovered by glob rather than listed. There were four wrapper modules here,
byte-identical apart from the filename they pointed at, and a fifth suite
therefore needed somebody to remember to write a fifth wrapper. A suite nobody
remembered would sit in the tree looking like coverage and never run, which is
a worse failure than not having written it.
"""

from pathlib import Path

import pytest

import sitejs

SITE_DIR = Path(__file__).resolve().parent / "site"
SUITES = sorted(SITE_DIR.glob("*.test.mjs"))


def test_the_glob_still_finds_the_suites():
    """A rename that breaks the pattern would otherwise read as everything
    passing, because zero parametrized cases is a green run."""
    assert SUITES, f"no *.test.mjs under {SITE_DIR} — the suites moved or were renamed"


@sitejs.needs_node
@pytest.mark.parametrize("suite", SUITES, ids=lambda p: p.name[: -len(".test.mjs")])
def test_site_suite(suite):
    sitejs.run_suite(suite)
