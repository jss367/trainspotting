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
        "elif sys.argv[1:] == ['scripts/export_site_data.py']:\n"
        f"    with open({str(commands)!r}, 'a') as log:\n"
        "        log.write(json.dumps(['export']) + '\\n')\n"
        "else:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)

    def write(directory, stage, slug=None, question=None, kind="ask", target="target", **metadata):
        suffix = kind if kind in ("labels", "labels-replicate") else f"{kind}-{slug}"
        path = tmp_path / directory / f"{target}.{stage}.{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"question": question, **metadata}))

    def run(target="target", phase="asks", expected_status=0):
        result = subprocess.run(
            [shutil.which("bash"), str(SCRIPT), target, phase],
            cwd=tmp_path, env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "PYTHONPATH": str(SCRIPT.parent.parent)},
            text=True, capture_output=True,
        )
        assert result.returncode == expected_status, result.stderr
        if expected_status == 0:
            assert "done" in result.stderr
        return [json.loads(line) for line in commands.read_text().splitlines()] if commands.exists() else []

    return write, run


@pytest.mark.parametrize("directory", ["results", "docs/data"])
def test_question_with_only_one_copy_is_rerun(refresh, directory):
    write, run = refresh
    question = 'Does it say "hello"?\nOr use $(a shell command)?'
    write(directory, "sft", "greeting", question)
    assert run() == [["ask", "target", question, "--slug", "greeting", "--stage", "sft"]]


def test_no_questions_is_a_successful_noop(refresh):
    _, run = refresh
    assert run() == []


def test_each_stage_slug_runs_once_and_results_wins_over_its_exported_copy(refresh):
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
        ["ask", "target", "fresh wording", "--slug", "topic", "--stage", "sft"],
        ["ask", "target", "fresh wording", "--slug", "topic", "--stage", "rlvr"],
        ["ask", "target", "exported wording", "--slug", "topic", "--stage", "dpo"],
        ["ask", "target", "only in export", "--slug", "export-only", "--stage", "sft"],
        ["ask", "target", "only in results", "--slug", "new", "--stage", "sft"],
        ["stance", "target", "which direction?", "--slug", "topic", "--stage", "sft"],
    ])


@pytest.mark.parametrize("stage", ["pretrain", "midtrain", "long-context"])
def test_corpus_only_question_refreshes_only_its_saved_stage(refresh, stage):
    write, run = refresh
    write("docs/data", stage, "topic", "old corpus wording")
    write("results", stage, "topic", "fresh corpus wording")
    assert run() == [[
        "ask", "target", "fresh corpus wording", "--slug", "topic",
        "--stage", stage, "--pretrain-only",
    ]]


def test_mixed_question_keeps_each_stage_and_its_own_wording(refresh):
    write, run = refresh
    write("docs/data", "pretrain", "topic", "old corpus wording")
    write("results", "pretrain", "topic", "fresh corpus wording")
    write("docs/data", "long-context", "topic", "long document wording")
    write("results", "sft", "topic", "prompt wording")
    write("docs/data", "sft", "topic", "old prompt wording")
    assert sorted(run()) == sorted([
        ["ask", "target", "fresh corpus wording", "--slug", "topic", "--stage", "pretrain", "--pretrain-only"],
        ["ask", "target", "long document wording", "--slug", "topic", "--stage", "long-context", "--pretrain-only"],
        ["ask", "target", "prompt wording", "--slug", "topic", "--stage", "sft"],
    ])


def test_dataset_stage_does_not_receive_a_corpus_flag(refresh):
    write, run = refresh
    write("results", "chat", "topic", "chat wording")
    assert run() == [["ask", "target", "chat wording", "--slug", "topic", "--stage", "chat"]]


@pytest.mark.parametrize("kind", ["ask", "stance"])
def test_refresh_preserves_the_saved_classifier_from_the_preferred_copy(refresh, kind):
    write, run = refresh
    write("docs/data", "sft", "topic", "old wording", kind=kind, classifier="old-judge")
    write("results", "sft", "topic", "saved wording", kind=kind, classifier="alternate-judge")
    assert run() == [[
        kind, "target", "saved wording", "--slug", "topic", "--stage", "sft",
        "--classifier", "alternate-judge",
    ]]


@pytest.mark.parametrize("kind", ["ask", "stance"])
@pytest.mark.parametrize("metadata", [{}, {"classifier": None}])
def test_legacy_questions_without_a_classifier_keep_the_cli_default(refresh, kind, metadata):
    write, run = refresh
    write("results", "sft", "topic", "legacy wording", kind=kind, **metadata)
    assert run() == [[kind, "target", "legacy wording", "--slug", "topic", "--stage", "sft"]]


@pytest.mark.parametrize("phase", ["labels", "all"])
def test_labels_keep_each_stages_classifier_and_still_run_new_stages(refresh, phase):
    write, run = refresh
    target = "olmo-3.1-32b-instruct"
    write("results", "sft", kind="labels", target=target, classifier="fresh-judge")
    write("docs/data", "sft", kind="labels", target=target, classifier="old-judge")
    write("docs/data", "dpo", kind="labels", target=target, classifier="exported-judge")
    write("results", "rlvr", kind="labels-replicate", target=target, classifier="replicate-judge")
    calls = run(target=target, phase=phase)
    assert [c for c in calls if c[0] == "classify"] == [
        ["classify", target, "--stage", "sft", "--classifier", "fresh-judge"],
        ["classify", target, "--stage", "dpo", "--classifier", "exported-judge"],
        ["classify", target, "--stage", "rlvr"],
    ]
    if phase == "all":
        assert [c[0] for c in calls] == ["context", "languages", "classify", "classify", "classify", "export"]


@pytest.mark.parametrize("metadata", [{}, {"classifier": None}])
def test_legacy_main_labels_keep_defaults_instead_of_using_an_older_copy(refresh, metadata):
    write, run = refresh
    write("results", "chat", kind="labels", target="wildchat-1m", **metadata)
    write("docs/data", "chat", kind="labels", target="wildchat-1m", classifier="old-judge")
    assert run(target="wildchat-1m", phase="labels") == [["classify", "wildchat-1m", "--stage", "chat"]]


def test_label_classifier_lookup_uses_the_canonical_target_key(refresh):
    write, run = refresh
    write("results", "chat", kind="labels", target="wildchat-1m", classifier="saved-judge")
    assert run(target="WILDCHAT-1M", phase="labels") == [
        ["classify", "wildchat-1m", "--stage", "chat", "--classifier", "saved-judge"],
    ]


def test_labels_for_a_base_only_target_still_fail_before_any_classification(refresh):
    _, run = refresh
    assert run(target="pythia-12b-deduped", phase="labels", expected_status=1) == []
