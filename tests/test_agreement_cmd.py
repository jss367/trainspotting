"""`gold` draws, refuses to overwrite hand labels, and `agreement` scores.

Both commands read committed labels runs and write under results/, so the
tests point them at a temporary directory and build a labels file there.
"""

import json

import pytest

from trainspotting import paths
from trainspotting.commands import agreement as cmd


def _args(target="olmo-3-7b-think", stage="sft", **kw):
    return type("A", (), {"target": target, "stage": stage, "per_label": 2, "seed": 0, **kw})()


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(cmd, "RESULTS", tmp_path)
    monkeypatch.setattr(cmd, "GOLD", tmp_path / "gold")
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SITE_DATA", tmp_path / "nowhere")
    labels = ["honesty", "capability", "capability", "helpfulness", "helpfulness", "tool_use"]
    (tmp_path / "olmo-3-7b-think.sft.labels.json").write_text(json.dumps({
        "dataset": "allenai/Dolci-Think-SFT-7B", "sample": 6, "seed": 0, "classifier": "c",
        "system_sha": "abc", "records": [
            {"row": i, "prompt": f"p{i}", "label": lab} for i, lab in enumerate(labels)
        ],
    }))
    return tmp_path


def test_gold_draws_blind_and_stratified(results, capsys):
    cmd.cmd_gold(_args())
    gold = json.loads((results / "gold" / "olmo-3-7b-think.sft.gold.json").read_text())
    assert gold["dataset"] == "allenai/Dolci-Think-SFT-7B"
    assert gold["labels_run"]["classifier"] == "c"
    assert "label" not in gold["items"][0] and gold["items"][0]["human_label"] is None
    # 2 per label where there are 2, 1 where there is 1: 1+2+2+1
    assert len(gold["items"]) == 6
    assert "harmlessness" in gold["rubric"]
    assert "fill in `human_label`" in capsys.readouterr().err


def test_gold_refuses_to_overwrite_hand_labels(results):
    cmd.cmd_gold(_args())
    path = results / "gold" / "olmo-3-7b-think.sft.gold.json"
    gold = json.loads(path.read_text())
    gold["items"][0]["human_label"] = "honesty"
    path.write_text(json.dumps(gold))
    with pytest.raises(SystemExit, match="already holds hand labels"):
        cmd.cmd_gold(_args())
    # An unlabeled gold file is fair game.
    gold["items"][0]["human_label"] = None
    path.write_text(json.dumps(gold))
    cmd.cmd_gold(_args())


def test_agreement_with_nothing_to_check_exits_nonzero(results, capsys):
    with pytest.raises(SystemExit):
        cmd.cmd_agreement(_args())
    assert "nothing to check against" in capsys.readouterr().err


def test_agreement_scores_gold_and_replicate(results, capsys):
    cmd.cmd_gold(_args())
    path = results / "gold" / "olmo-3-7b-think.sft.gold.json"
    gold = json.loads(path.read_text())
    by_row = {0: "honesty", 1: "capability", 2: "helpfulness", 3: "helpfulness", 4: "helpfulness", 5: "tool_use"}
    for item in gold["items"][:5]:
        item["human_label"] = by_row[item["row"]]
    path.write_text(json.dumps(gold))
    rep = json.loads((results / "olmo-3-7b-think.sft.labels.json").read_text())
    rep["classifier"] = "c2"
    rep["records"][5]["label"] = "other"
    (results / "olmo-3-7b-think.sft.labels-replicate.json").write_text(json.dumps(rep))

    cmd.cmd_agreement(_args())

    out = json.loads((results / "olmo-3-7b-think.sft.agreement.json").read_text())
    assert out["gold"]["n"] == 5 and out["gold"]["unlabeled_gold"] == 1
    assert out["gold"]["stratified"] is True
    assert out["replicate"]["n"] == 6 and out["replicate"]["agree"] == 5
    assert out["replicate"]["classifier"] == "c2" and out["replicate"]["same_draw"] is True
    err = capsys.readouterr().err
    assert "vs 5 hand labels" in err and "second run of c2" in err and "1 gold items still unlabeled" in err
