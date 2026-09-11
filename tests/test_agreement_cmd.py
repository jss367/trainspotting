"""`agreement` compares a replicate run with the main one, or says there is none.

It reads committed labels runs and writes under results/, so the tests point it
at a temporary directory and build the two runs there.
"""

import json

import pytest

from trainspotting import paths
from trainspotting.commands import agreement as cmd


def _args(target="olmo-3-7b-think", stage="sft"):
    return type("A", (), {"target": target, "stage": stage})()


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(cmd, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SITE_DATA", tmp_path / "nowhere")
    labels = ["honesty", "capability", "capability", "helpfulness", "helpfulness", "tool_use"]
    (tmp_path / "olmo-3-7b-think.sft.labels.json").write_text(json.dumps({
        "dataset": "allenai/Dolci-Think-SFT-7B", "sample": 6, "seed": 0, "classifier": "c",
        "system_sha": "abc", "revision": "rev1", "records": [
            {"row": i, "prompt": f"p{i}", "label": lab} for i, lab in enumerate(labels)
        ],
    }))
    return tmp_path


def test_nothing_to_check_exits_nonzero_and_says_what_to_run(results, capsys):
    with pytest.raises(SystemExit):
        cmd.cmd_agreement(_args())
    assert "--replicate" in capsys.readouterr().err


def test_compares_the_replicate_and_records_the_draw(results, capsys):
    rep = json.loads((results / "olmo-3-7b-think.sft.labels.json").read_text())
    rep["records"][5]["label"] = "other"
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(rep))

    cmd.cmd_agreement(_args())

    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["labels_run"]["classifier"] == "c"
    r = out["replicate"]
    assert (r["n"], r["agree"], r["classifier"], r["same_draw"]) == (6, 5, "c", True)
    assert r["shares"]["tool_use"] == {"first": 1 / 6, "second": 0}
    assert r["comparison_issues"] == []
    assert r["revision"] == out["labels_run"]["revision"] == "rev1"
    assert "second run of c" in capsys.readouterr().err


def test_a_replicate_from_another_draw_is_flagged(results, capsys):
    rep = json.loads((results / "olmo-3-7b-think.sft.labels.json").read_text())
    rep["seed"] = 1
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(rep))
    cmd.cmd_agreement(_args())
    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["replicate"]["same_draw"] is False
    assert "different seed" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("dataset", "another/dataset"), ("revision", "rev2"),
    ("system_sha", "changed-rubric"), ("classifier", "c2"), ("sample", 7),
])
def test_changed_provenance_cannot_claim_repeatability(results, capsys, field, value):
    rep = json.loads((results / "olmo-3-7b-think.sft.labels.json").read_text())
    rep[field] = value
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(rep))
    cmd.cmd_agreement(_args())
    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["replicate"]["same_draw"] is False
    assert out["replicate"]["comparison_issues"] == [f"different {field}"]
    assert "not a repeatability check" in capsys.readouterr().err


@pytest.mark.parametrize("field", cmd.PROVENANCE)
@pytest.mark.parametrize("missing", [None, "", "absent"])
def test_equal_but_unknown_provenance_does_not_pass(results, field, missing):
    labels_path = results / "olmo-3-7b-think.sft.labels.json"
    run = json.loads(labels_path.read_text())
    if missing == "absent":
        run.pop(field)
    else:
        run[field] = missing
    labels_path.write_text(json.dumps(run))
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(run))
    cmd.cmd_agreement(_args())
    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["replicate"]["same_draw"] is False
    assert out["replicate"]["comparison_issues"] == [f"unknown {field}"]


@pytest.mark.parametrize("suffix", ["labels", "labels-replicate"])
def test_mixed_revision_run_does_not_pass(results, suffix):
    run = json.loads((results / "olmo-3-7b-think.sft.labels.json").read_text())
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(run))
    run["revision_moved_to"] = "rev2"
    (results / f"olmo-3-7b-think.sft.{suffix}.json").write_text(json.dumps(run))
    cmd.cmd_agreement(_args())
    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["replicate"]["same_draw"] is False
    assert out["replicate"]["comparison_issues"] == ["dataset revision moved during a run"]
