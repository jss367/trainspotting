"""Run the site's declaration-collision check under pytest.

The check reads docs/index.html, so it is JavaScript; this wrapper puts it in
the same `pytest` run as everything else. Skipped where node is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent / "site" / "declarations.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_no_duplicate_top_level_declarations():
    proc = subprocess.run(
        [shutil.which("node"), str(SUITE)], capture_output=True, text=True, timeout=60
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAIL")]
    assert proc.returncode == 0, "\n".join(failed[:20] + [proc.stderr[-2000:]])
