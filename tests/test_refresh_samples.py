"""Refresh existing questions offline, regardless of which artifact copy exists."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refresh_samples.sh"


@pytest.fixture
def refresh(tmp_path):
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

    def write(directory, stage, slug, question, kind="ask"):
        path = tmp_path / directory / f"target.{stage}.{kind}-{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"question": question}))

    def run():
        result = subprocess.run(
            [shutil.which("bash"), str(SCRIPT), "target", "asks"],
            cwd=tmp_path, env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
            text=True, capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        assert "done" in result.stderr
        return [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []

    return write, run


@pytest.mark.parametrize("directory", ["results", "docs/data"])
def test_question_with_only_one_copy_is_rerun(refresh, directory):
    write, run = refresh
    question = 'Does it say "hello"?\nOr use $(a shell command)?'
    write(directory, "sft", "greeting", question)
    assert run() == [["ask", "target", question, "--slug", "greeting"]]


def test_no_questions_is_a_successful_noop(refresh):
    _, run = refresh
    assert run() == []


def test_each_slug_runs_once_and_results_wins_across_stages_and_copies(refresh):
    write, run = refresh
    write("docs/data", "dpo", "topic", "exported wording")
    write("docs/data", "sft", "topic", "exported wording")
    write("results", "sft", "topic", "fresh wording")
    write("results", "rlvr", "topic", "fresh wording")
    write("docs/data", "sft", "export-only", "only in export")
    write("results", "sft", "new", "only in results")
    # Kind is part of the identity, even when ask and stance reuse a slug.
    write("results", "sft", "topic", "which direction?", kind="stance")
    write("docs/data", "sft", "topic", "old direction?", kind="stance")
    assert sorted(run()) == sorted([
        ["ask", "target", "fresh wording", "--slug", "topic"],
        ["ask", "target", "only in export", "--slug", "export-only"],
        ["ask", "target", "only in results", "--slug", "new"],
        ["stance", "target", "which direction?", "--slug", "topic"],
    ])
