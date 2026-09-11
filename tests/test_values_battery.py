"""The values battery selects only capabilities the registered target has."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "values_battery.sh"


@pytest.fixture
def battery(tmp_path):
    commands = tmp_path / "commands.jsonl"
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'trainspotting.cli']:\n"
        f"    with open({str(commands)!r}, 'a') as log:\n"
        "        log.write(json.dumps(sys.argv[3:]) + '\\n')\n"
        "else:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)

    def run(target, phase, slug="refusal"):
        result = subprocess.run(
            [shutil.which("bash"), str(SCRIPT), target, phase, slug],
            cwd=tmp_path, env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "PYTHONPATH": str(ROOT)},
            text=True, capture_output=True,
        )
        calls = [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("target,corpus,stance", [
    ("olmo-3-7b-think", True, True),
    ("wildchat-1m", False, False),
    ("pythia-12b-deduped", True, False),
])
@pytest.mark.parametrize("phase", ["all", "ask", "stance", "budget"])
def test_each_phase_uses_the_targets_supported_capabilities(battery, target, corpus, stance, phase):
    result, calls = battery(target, phase)
    assert result.returncode == 0, result.stderr
    expected = []
    if phase in ("all", "ask"):
        expected.append("ask")
    if phase in ("all", "stance") and stance:
        expected.append("stance")
    if phase in ("all", "budget"):
        expected.append("budget")
    assert [call[0] for call in calls] == expected
    for call in calls:
        assert call[1] == target
        assert ("--pretrain" in call) == (call[0] == "ask" and corpus)
        assert "refusal" in call
    if phase in ("all", "stance") and not stance:
        assert "skipping direction" in result.stderr


def test_invalid_target_fails_before_any_paid_command(battery):
    result, calls = battery("not-a-target", "ask")
    assert result.returncode != 0
    assert calls == []


@pytest.mark.parametrize("phase", ["all", "ask", "stance", "budget"])
def test_unknown_question_slug_fails_with_available_choices_before_commands(battery, phase):
    result, calls = battery("olmo-3-7b-think", phase, "refusla")
    assert result.returncode == 2
    assert calls == []
    assert "unknown question slug: refusla" in result.stderr
    for known in ["self-identity", "knowledge-cutoff", "refusal", "sycophancy", "admitting-uncertainty"]:
        assert known in result.stderr
    assert "done" not in result.stderr


def test_omitting_the_question_slug_still_runs_the_whole_battery(battery):
    result, calls = battery("olmo-3-7b-think", "ask", "")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 5
    assert {c[c.index("--slug") + 1] for c in calls} == {
        "self-identity", "knowledge-cutoff", "refusal", "sycophancy", "admitting-uncertainty",
    }
