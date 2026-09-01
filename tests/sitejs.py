"""Call a function defined in docs/index.html from a Python test.

The page is a single file with no build step, so the only way to test the code
the browser runs is to evaluate the page and call into it. `tests/site/page.mjs`
knows how; this is the Python door to it, so a test here can hand records in as
JSON and get a result back without each one growing its own copy of the node
boilerplate.

Prefer this over pulling a function's source out of the file by name. That was
tried, and it broke the first time the function it lifted called a helper
defined elsewhere in the page — which is what happened when the DPO branch rule
became one shared `branchPoint` instead of a loop copied into every caller.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent / "docs" / "index.html"
HARNESS = Path(__file__).resolve().parent / "site" / "page.mjs"

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def call(body: str, payload=None, timeout: int = 300):
    """Run `body` with the page's functions bound as `page`, return its JSON.

    `body` is JavaScript. It gets `page` (every function the harness exports)
    and `input` (whatever `payload` was, parsed back from JSON), and whatever it
    assigns to `output` comes back as Python.
    """
    if not SITE.exists():
        pytest.skip("no docs/index.html in this checkout")
    script = f"""
import fs from "node:fs";
import {{ loadPage }} from {json.dumps(str(HARNESS))};
const page = loadPage();
const input = JSON.parse(fs.readFileSync(0, "utf8"));
let output;
{body}
console.log(JSON.stringify(output === undefined ? null : output));
"""
    proc = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout)
