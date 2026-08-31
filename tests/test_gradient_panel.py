"""Run the site's DPO gradient-panel assertions under pytest.

The panel lives in docs/index.html, so its tests are JavaScript; this wrapper
puts them in the same `pytest` run as everything else. Skipped where node is
not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parent / "site" / "gradient_panel.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_gradient_panel_claims():
    proc = subprocess.run(
        [shutil.which("node"), str(SUITE)], capture_output=True, text=True, timeout=300
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAIL")]
    assert proc.returncode == 0, "\n".join(failed[:20] + [proc.stdout[-2000:], proc.stderr[-2000:]])
