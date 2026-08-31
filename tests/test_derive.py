"""The numbers the size cards draw, and the two joins they depend on.

Three things can break silently here and none of them raises:

  * `example_chars` mis-attributing text — counting a DPO pair's shared history
    as something the model was fit to would inflate every target number
  * the prompt-key hash drifting between derive.py and the copy in
    docs/index.html, which empties the crosstab grid rather than erroring
  * a committed profile going stale against the labels file it is crossed with,
    which shows up as prompts silently missing from the grid

So the joins are tested against the committed samples themselves, not fixtures.
"""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from trainspotting import derive

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
SITE = Path(__file__).resolve().parent.parent / "docs" / "index.html"

PROFILES = sorted(DATA.glob("*.profile.json"))
PROFILE_IDS = [p.name.replace(".profile.json", "") for p in PROFILES]


def turn(role, chars, text="x", reasoning=None):
    t = {"role": role, "text": text, "chars": chars}
    if reasoning:
        t["reasoning"] = {"text": "r", "chars": reasoning}
    return t


def test_sft_target_is_the_assistant_turns_including_reasoning():
    rec = {
        "kind": "sft",
        "turns": [turn("system", 10), turn("user", 100), turn("assistant", 200, reasoning=50),
                  turn("environment", 30)],
    }
    total, target = derive.example_chars(rec)
    assert total == 10 + 100 + 200 + 50 + 30
    # Tool output is text the model reads and is not scored on, same as the prompt.
    assert target == 250


def test_dpo_counts_the_shared_history_once_and_only_the_branch_as_target():
    shared = [turn("user", 100, "ask"), turn("assistant", 60, "first reply"), turn("user", 40, "again")]
    rec = {
        "kind": "dpo",
        "chosen": {"turns": shared + [turn("assistant", 300, "good")]},
        "rejected": {"turns": shared + [turn("assistant", 90, "bad")]},
    }
    total, target = derive.example_chars(rec)
    assert target == 390                     # both completions, neither shared turn
    assert total == 200 + 390                # the shared conversation, counted once


def test_dpo_with_identical_sides_still_has_a_target():
    same = [turn("user", 10, "q"), turn("assistant", 20, "a")]
    rec = {"kind": "dpo", "chosen": {"turns": list(same)}, "rejected": {"turns": list(same)}}
    total, target = derive.example_chars(rec)
    assert target == 40 and total == 50


def test_rl_stores_no_target_and_chat_is_not_a_training_example():
    rl = {"kind": "rlvr", "prompt_full": {"chars": 500}, "reward": {}}
    assert derive.example_chars(rl) == (500, 0)
    chat = {"kind": "chat", "turns": [turn("user", 10), turn("assistant", 90)]}
    assert derive.example_chars(chat) == (100, 0)


def test_histogram_bins_are_shared_and_the_tail_is_kept_not_dropped():
    edges = derive.hist_edges()
    assert len(edges) == derive.HIST_BINS + 1
    assert edges[0] == pytest.approx(10)
    bins = derive.histogram([1, 10, 5000, 10 ** 9])
    assert sum(bins) == 4                 # nothing dropped, including the under- and overflow
    assert bins[-1] == 1                  # a document past the top edge lands in the last bin
    assert bins[0] == 2                   # 1 and 10 both clamp into the first


def test_estimate_scales_the_sampled_mean_and_carries_its_interval():
    stats = derive.summarize([400] * 50 + [800] * 50)
    est = derive._estimate(stats, 1_000_000)
    assert est["per_example"] == pytest.approx(600 / derive.CHARS_PER_TOKEN)
    assert est["tokens"] == pytest.approx(150_000_000)
    assert est["lo"] < est["tokens"] < est["hi"]
    # The interval is the sample's own standard error, not a fixed fraction.
    assert est["hi"] - est["tokens"] == pytest.approx(1.96 * stats["se"] / 4 * 1_000_000)


def test_a_stage_with_no_row_count_measures_shape_but_claims_no_budget():
    ctx = {"stage": "sft", "records": [{"kind": "sft", "key": "a", "turns": [turn("assistant", 40)], "meta": {}}]}
    p = derive.stage_profile(ctx, None)
    assert p["chars"]["n"] == 1
    assert p["tokens"] is None and p["target_tokens"] is None


def test_stage_profile_reports_ambiguous_keys_rather_than_hiding_them():
    rec = lambda key: {"kind": "sft", "key": key, "turns": [turn("assistant", 10)], "meta": {"source": "s"}}
    p = derive.stage_profile({"stage": "sft", "records": [rec("same"), rec("same"), rec("other")]}, 10)
    assert p["ambiguous_keys"] == 1
    assert p["columns"] == {"source": 1}


# --------------------------------------------------------------- the joins ---

@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_profile_matches_the_context_sample_it_was_derived_from(profile):
    """The profile is a summary of a bulk file the site also ships. If the two
    fall out of step the page draws one sample's lengths over another's."""
    d = json.loads(profile.read_text())
    ctx = json.loads(profile.with_name(profile.name.replace(".profile", ".context")).read_text())
    assert d["dataset"] == ctx["dataset"] and d["sample"] == ctx["sample"]
    assert len(d["records"]) == len(ctx["records"]) == d["chars"]["n"]
    assert [r["k"] for r in d["records"]] == [derive.prompt_key(c["key"]) for c in ctx["records"]]


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_labeled_prompts_reach_their_sampled_row(profile):
    """The crosstab grid crosses a labels file against a profile through this
    hash. A drift makes the grid empty, not wrong-looking, so hold it to the
    committed data: sampling is deterministic, so the overlap should be total
    but for prompts sharing a 400-character opening."""
    labels = profile.with_name(profile.name.replace(".profile", ".labels"))
    if not labels.exists():
        pytest.skip("no committed classify run for this stage")
    d = json.loads(profile.read_text())
    keys = {r["k"] for r in d["records"]}
    records = json.loads(labels.read_text())["records"]
    joined = sum(derive.prompt_key(r["prompt"]) in keys for r in records)
    assert joined >= len(records) - d["ambiguous_keys"]


def test_the_page_computes_the_same_prompt_key_as_python():
    """Two implementations of one hash, in two languages, in two files. The one
    in docs/index.html is what actually draws the grid."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    src = SITE.read_text()
    fn = re.search(r"function promptKey\(prompt\)\{.*?\n\}", src, re.S)
    assert fn, "docs/index.html no longer defines promptKey — the crosstab join is gone"
    samples = ["", "hello", "a" * 1000, "日本語のプロンプト", "emoji 🙂 and \\ quotes \" '",
               "line\nbreak\ttab", "Ω" * 401]
    script = (
        f"const KEY_CHARS = {derive.KEY_CHARS};\n{fn.group(0)}\n"
        f"console.log(JSON.parse(process.argv[1]).map(promptKey).join(','))"
    )
    out = subprocess.run([node, "-e", script, json.dumps(samples)],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out.split(",") == [derive.prompt_key(s) for s in samples]


def test_the_page_bins_lengths_the_same_way_python_does():
    src = SITE.read_text()
    m = re.search(r"const HIST_EDGES = Array\.from\(\{length: (\d+)\}, \(_, i\) => 10 \*\* \(([\d.]+) \+ i \* ([\d.]+)\)\)", src)
    assert m, "docs/index.html no longer derives its histogram edges the same way"
    count, start, step = int(m.group(1)), float(m.group(2)), float(m.group(3))
    assert (count, start, step) == (derive.HIST_BINS + 1, derive.HIST_MIN_LOG, derive.HIST_STEP)


@pytest.mark.parametrize("profile", PROFILES, ids=PROFILE_IDS)
def test_committed_profiles_are_internally_consistent(profile):
    d = json.loads(profile.read_text())
    assert d["target_chars"]["mean"] <= d["chars"]["mean"] + 1e-9
    if d["tokens"]:
        assert d["tokens"]["lo"] <= d["tokens"]["tokens"] <= d["tokens"]["hi"]
        assert d["tokens"]["tokens"] == pytest.approx(
            d["chars"]["mean"] / derive.CHARS_PER_TOKEN * d["examples"])
    assert sum(d["chars"]["hist"]) <= d["chars"]["n"]
    assert not math.isnan(d["chars"]["se"])
