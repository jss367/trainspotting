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


def test_dpo_with_identical_sides_has_no_target_because_it_has_no_signal():
    """DPO reads the difference between the two sequences' log probabilities, so
    a pair whose sides are identical cancels exactly and carries no gradient.
    Manufacturing a final-turn branch would report two copies of one answer as
    gradient-bearing. Four sampled Instruct DPO pairs are identical in full."""
    same = [turn("user", 10, "q"), turn("assistant", 20, "a")]
    rec = {"kind": "dpo", "chosen": {"turns": list(same)}, "rejected": {"turns": list(same)}}
    total, target = derive.example_chars(rec)
    assert target == 0
    assert total == 30      # the conversation is there; none of it is a target


def test_dpo_with_one_side_a_prefix_of_the_other_scores_only_the_extra_turns():
    """Everything up to where the shorter side ends is conditioned identically
    on both sequences and cancels, so the difference between them is exactly
    what only the longer side has."""
    shared = [turn("user", 10, "q"), turn("assistant", 20, "a")]
    rec = {
        "kind": "dpo",
        "chosen": {"turns": shared + [turn("assistant", 30, "more")]},
        "rejected": {"turns": list(shared)},
    }
    total, target = derive.example_chars(rec)
    assert target == 30
    assert total == 30 + 30


def test_dpo_branches_where_the_reasoning_diverges_even_if_the_answer_matches():
    """A turn is its answer and the reasoning span stored beside it. Two turns
    that reach the same answer at the same length by different reasoning are a
    branch, not shared history — counting them as shared would keep one copy of
    a turn that is really two and score neither reasoning span."""
    opening = turn("user", 50, "ask")
    rec = {
        "kind": "dpo",
        "chosen": {"turns": [opening, turn("assistant", 30, "same answer", reasoning=70),
                             turn("assistant", 10, "tail")]},
        "rejected": {"turns": [opening, turn("assistant", 30, "same answer", reasoning=70),
                               turn("assistant", 10, "tail")]},
    }
    # Identical on both sides: shared all the way down, so nothing is a target.
    assert derive._shared_turns(rec["chosen"]["turns"], rec["rejected"]["turns"]) == 3

    rec["rejected"]["turns"][1] = turn("assistant", 30, "same answer", reasoning=70)
    rec["rejected"]["turns"][1]["reasoning"]["text"] = "a different route to it"
    total, target = derive.example_chars(rec)
    assert derive._shared_turns(rec["chosen"]["turns"], rec["rejected"]["turns"]) == 1
    # Both divergent turns and both reasoning spans are targets now.
    assert target == (30 + 70) * 2 + 10 * 2
    assert total == 50 + target


def test_dpo_branches_on_a_digest_where_the_stored_text_was_cut():
    """`context._text` keeps the first 4,000 characters and the true length, so
    two long turns can be indistinguishable in the record and different in the
    dataset. The digest of the whole field settles it; without one, a pair of
    4,001-character responses differing in their last character scans as fully
    shared and comes back with no target."""
    opening = turn("user", 20, "ask")
    long_a = {"role": "assistant", "text": "x" * 4000, "chars": 4001, "sha": "aaaa1111"}
    long_b = {"role": "assistant", "text": "x" * 4000, "chars": 4001, "sha": "bbbb2222"}

    rec = {"kind": "dpo", "chosen": {"turns": [opening, long_a]},
           "rejected": {"turns": [opening, long_b]}}
    total, target = derive.example_chars(rec)
    assert derive._shared_turns(rec["chosen"]["turns"], rec["rejected"]["turns"]) == 1
    assert target == 4001 * 2      # two responses, not one shared one

    # Same digest is the same text however long it is.
    rec["rejected"]["turns"][1] = dict(long_a)
    assert derive._shared_turns(rec["chosen"]["turns"], rec["rejected"]["turns"]) == 2

    # A record written before digests existed compares on what it has.
    no_sha = [{k: v for k, v in t.items() if k != "sha"} for t in (long_a, long_b)]
    assert derive._shared_turns([opening, no_sha[0]], [opening, no_sha[1]]) == 2


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


def test_clusters_are_the_samplers_draws_not_its_rows():
    """`hf.sample_rows_with_truncation` takes ten consecutive rows per random
    offset, so maximal runs of consecutive indices are the draws."""
    assert derive.clusters_of([10, 11, 12, 40, 41]) == [[0, 1, 2], [3, 4]]
    # Order of arrival doesn't matter; the row index does.
    assert derive.clusters_of([41, 11, 40, 10, 12]) == [[3, 1, 4], [2, 0]]
    # Two offsets landing next to each other merge, which claims nothing false.
    assert derive.clusters_of([1, 2, 3]) == [[0, 1, 2]]
    # Records committed before the sampler recorded a row index.
    assert derive.clusters_of([1, None, 3]) is None


def test_the_interval_widens_when_the_draws_are_correlated():
    """Neighbouring rows share a source dataset and a length profile. Scoring
    300 correlated rows as 300 independent ones makes every whisker on the page
    too narrow — about 2x on the committed samples."""
    # Ten draws of ten, alike inside a draw and far apart between them.
    values, rows = [], []
    for draw in range(10):
        for i in range(10):
            values.append(100 if draw % 2 else 900)
            rows.append(draw * 1000 + i)

    clustered = derive.summarize(values, rows)
    independent = derive.summarize(values)

    assert clustered["clusters"] == 10
    assert clustered["deff"] > 5
    assert clustered["se"] > independent["se"] * 2
    # The point estimate is untouched; only the claim about its precision moves.
    assert clustered["mean"] == independent["mean"]


def test_an_uncorrelated_sample_is_not_widened_for_nothing():
    """The design effect has a floor of 1: clustering can only cost precision,
    and a draw whose neighbours are unalike has not cost any."""
    values = [(i * 37) % 100 for i in range(100)]
    rows = list(range(100))

    stats = derive.summarize(values, rows)

    assert stats["deff"] == pytest.approx(1.0, abs=0.5)


def test_clustering_never_narrows_the_interval():
    """The floor has to apply to the error, not only to the design effect
    printed beside it. A sample whose draws happen to look unalike has not
    bought precision, and a page that says "widened for clustering" must not be
    reporting an interval that clustering narrowed."""
    # Twenty separate draws of ten, deliberately anti-correlated inside each.
    values, rows = [], []
    for draw in range(20):
        for i in range(10):
            values.append(0 if i % 2 else 1000)
            rows.append(draw * 1000 + i)

    clustered = derive.summarize(values, rows)
    independent = derive.summarize(values)

    assert clustered["deff"] == 1.0
    assert clustered["se"] >= independent["se"]
    assert clustered["se"] == pytest.approx(independent["se"])


def test_a_sample_with_no_row_indices_falls_back_to_the_independent_error():
    stats = derive.summarize([1, 2, 3, 4], [None, None, None, None])
    assert stats["clusters"] is None and stats["deff"] == 1.0


def test_one_cluster_reports_the_independent_error_rather_than_none():
    """A sample that arrived as a single run leaves the design effect
    unestimable, not 1. Taking the cluster variance there would print a
    zero-width interval and call it certainty."""
    values = [10, 20, 30, 40, 50, 60]

    stats = derive.summarize(values, list(range(6)))

    assert stats["clusters"] == 1
    assert stats["se"] == pytest.approx(derive.summarize(values)["se"])
    assert stats["se"] > 0


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


def site_function(name: str) -> str:
    """One top-level function lifted out of docs/index.html by name.

    The site is a single file with no build step and no module boundary, so the
    only way to test its logic is to read it back out. A top-level declaration
    ends at the first line that is exactly `}`, which is what the file's own
    formatting guarantees.
    """
    src = SITE.read_text().splitlines()
    start = next(
        (i for i, line in enumerate(src) if line.startswith(f"function {name}(")), None
    )
    assert start is not None, f"docs/index.html no longer defines {name}()"
    end = next(i for i in range(start, len(src)) if src[i] == "}")
    return "\n".join(src[start : end + 1])


def site_const(name: str) -> str:
    """A one-line `const name = ...` lifted out of docs/index.html, so a test
    exercising a function that closes over it runs the file's own definition."""
    src = SITE.read_text().splitlines()
    line = next((l for l in src if l.startswith(f"const {name} = ")), None)
    assert line is not None, f"docs/index.html no longer defines {name}"
    return line


def run_node(script: str, *args: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    return subprocess.run(
        [node, "-e", script, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_the_page_computes_the_same_prompt_key_as_python():
    """Two implementations of one hash, in two languages, in two files. The one
    in docs/index.html is what actually draws the grid."""
    fn = site_function("promptKey")
    samples = ["", "hello", "a" * 1000, "日本語のプロンプト", "emoji 🙂 and \\ quotes \" '",
               "line\nbreak\ttab", "Ω" * 401,
               # Non-BMP characters before the cutoff, in a prompt long enough to
               # be cut. This is the case that shipped broken: JavaScript's
               # String.slice counts UTF-16 units, so one emoji moves the cut a
               # character earlier than Python's code-point slice and the two
               # sides hash different text. Three committed prompts hit it.
               "🙂" + "a" * 500,
               "🇰🇬 " + "Манас эпосу " * 60,
               "a" * 399 + "🙂" + "b" * 100,
               "𝗖𝗵𝗮𝘁𝗚𝗣𝗧 " * 80]
    script = (
        f"const KEY_CHARS = {derive.KEY_CHARS};\n{site_const('keyPrefix')}\n{fn}\n"
        f"console.log(JSON.parse(process.argv[1]).map(promptKey).join(','))"
    )
    out = run_node(script, json.dumps(samples))
    assert out.split(",") == [derive.prompt_key(s) for s in samples]


def test_the_page_refuses_to_guess_when_two_sampled_rows_share_a_key():
    """A prompt key is a prompt's opening, so two sampled rows can collide on
    one. Where they carry different metadata, no join can say which row a
    labeled prompt came from — filing it under whichever row was sampled last
    is a wrong answer dressed as a right one. Those keys have to drop out."""
    script = (
        f"const KEY_CHARS = {derive.KEY_CHARS};\n{site_const('keyPrefix')}\n"
        f"{site_function('promptKey')}\n{site_function('valueByKey')}\n{site_function('crossRows')}\n"
        """
        const records = [
          {k: "aaa", m: {src: "one"}},            // collides, disagrees
          {k: "aaa", m: {src: "two"}},
          {k: "bbb", m: {src: "three"}},          // collides, agrees
          {k: "bbb", m: {src: "three"}},
          {k: "ccc", m: {src: "four"}},           // clean
          {k: "ddd", m: {}},                      // no value for this column
        ];
        const {map, dropped} = valueByKey(records, "src");
        // Three prompts: one on a clean key, one on the key thrown out as
        // undecidable, one on a key the sample never had.
        const labeled = [
          {prompt: "clean prompt", label: "helpfulness"},
          {prompt: "conflicted prompt", label: "honesty"},
          {prompt: "unknown prompt", label: "capability"},
        ];
        const kv = new Map([[promptKey("clean prompt"), "four"]]);
        const conflictedKey = promptKey("conflicted prompt");
        const cross = crossRows(labeled, kv, r => r.label, 12, new Set([conflictedKey]));
        console.log(JSON.stringify({
          kept: [...map.entries()].sort(),
          dropped: [...dropped],
          rows: cross.rows.map(r => [r.name, r.total]),
          undecidable: cross.undecidable,
          unmatched: cross.unmatched,
        }));
        """
    )
    out = json.loads(run_node(script))
    # The disagreeing key is gone entirely — the row that claimed it first is
    # no more decidable than the one that collided with it.
    assert out["kept"] == [["bbb", "three"], ["ccc", "four"]]
    assert out["dropped"] == ["aaa"]
    # A prompt whose key was dropped is missing from the grid, never misfiled,
    # and it is counted apart from one the sample simply never had.
    assert out["rows"] == [["four", 1]]
    assert (out["undecidable"], out["unmatched"]) == (1, 1)


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
