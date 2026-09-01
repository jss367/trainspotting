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

# One string, because several things read it. It lands in pytest's skip summary
# and the CI workflow greps that summary for it, failing the job if a site suite
# skipped on a runner that has node. It was spelled out separately in each of
# the four wrapper files this module replaced; any one of them drifting would
# have made the guard quietly stop guarding for that suite.
NODE_MISSING = "node is not installed"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason=NODE_MISSING)


def _node_or_skip() -> str:
    """The node binary, or skip the test.

    Every path into node goes through here rather than through each caller's
    marker. A caller that forgets `needs_node` does not skip on a machine
    without node — it passes `None` to `subprocess.run` as the executable and
    dies with `TypeError: expected str, bytes or os.PathLike object, not
    NoneType`. That is what happened to `TestPageFieldsAgainstCommittedSamples`
    when its own node-checking fixture was replaced by this module: three tests
    that used to skip started erroring instead.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip(NODE_MISSING)
    return node


def run_suite(path, timeout: int = 300):
    """Run one `tests/site/*.test.mjs` file, failing with the lines it flagged.

    The suites print `ok`/`FAIL` per assertion and exit non-zero if any failed,
    so the useful part of a failure is the FAIL lines rather than the tail of a
    long transcript. Those come first, with the raw output after them.
    """
    node = _node_or_skip()
    proc = subprocess.run(
        [node, str(path)], capture_output=True, text=True, timeout=timeout
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAIL")]
    assert proc.returncode == 0, "\n".join(
        failed[:20] + [proc.stdout[-2000:], proc.stderr[-2000:]]
    )


def call(body: str, payload=None, timeout: int = 300):
    """Run `body` with the page's functions bound as `page`, return its JSON.

    `body` is JavaScript. It gets `page` (every function the harness exports)
    and `input` (whatever `payload` was, parsed back from JSON), and whatever it
    assigns to `output` comes back as Python.
    """
    node = _node_or_skip()
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
        [node, "--input-type=module", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout)
