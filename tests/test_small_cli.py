"""Formatting and argument checks too small for a module of their own."""

import json
import sys
from types import SimpleNamespace

import pytest

from trainspotting import cli
from trainspotting.commands import languages as cmd_languages
from trainspotting.commands.common import _fmt_est


@pytest.mark.parametrize(("n", "want"), [
    (999_700, "1M"), (999_400, "999K"), (1_234_567, "1.2M"), (99_960, "100K"),
    (950, "950"), (1e12, "1T"), (None, "—"),
])
def test_an_estimate_rounds_before_it_picks_a_unit(n, want):
    assert _fmt_est(n) == want


def test_languages_sample_must_be_positive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trainspotting", "languages", "olmo-3-7b-think", "--sample", "0"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2  # argparse's usage error, before any handler runs


def test_from_labels_refuses_a_run_drawn_from_another_dataset(tmp_path, monkeypatch):
    stage = {"stage": "sft", "hf_dataset": "allenai/new"}
    (tmp_path / "olmo-3-7b-think.sft.labels.json").write_text(json.dumps(
        {"dataset": "allenai/old", "sample": 3, "seed": 0, "records": [{"prompt": "hi"}]}))
    monkeypatch.setattr(cmd_languages, "RESULTS", tmp_path)
    monkeypatch.setattr(cmd_languages, "_select_stages", lambda *a: [stage])
    args = SimpleNamespace(target="olmo-3-7b-think", sample=3, seed=0, from_labels=True)
    with pytest.raises(SystemExit, match="allenai/old"):
        cmd_languages.cmd_languages(args)
