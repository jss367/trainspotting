"""The training-order layer, offline.

The ways this breaks are all silent: a step address off by a shard boundary
decodes into real text from the wrong step, a sample drawn uniformly can miss
half the run, a rate summed over steps without the clustering claims a
precision the draw does not have. Everything here pins the arithmetic against
the published layout; the one `--live` test pins the layout against the hub.
"""

import array
import re
from types import SimpleNamespace

import pytest

from trainspotting import cli, registry, steps

ORDER = registry.PYTHIA_TRAINING_ORDER


def test_the_published_layout_closes():
    """143,000 steps of 1,024 × 2,049 uint16 has to be exactly the bytes the
    shards hold, or every offset is an address into the wrong text."""
    steps.check_layout(ORDER)
    assert steps.step_bytes(ORDER) == 4_196_352
    assert steps.total_bytes(ORDER) == 20 * 30_000_000_000 + 78_336_000


def test_a_layout_that_does_not_close_is_refused():
    with pytest.raises(ValueError):
        steps.check_layout({**ORDER, "steps": ORDER["steps"] - 1})


def test_step_zero_is_the_front_of_shard_zero():
    assert steps.segments(ORDER, 0) == [(0, 0, 4_196_351)]


def test_a_step_across_a_shard_boundary_is_two_contiguous_ranges():
    step = 30_000_000_000 // steps.step_bytes(ORDER)  # the first step past the seam
    segs = steps.segments(ORDER, step)
    assert len(segs) == 2
    (s0, a0, b0), (s1, a1, b1) = segs
    assert (s0, s1) == (0, 1)
    assert b0 == 30_000_000_000 - 1
    assert a1 == 0
    assert (b0 - a0 + 1) + (b1 - a1 + 1) == steps.step_bytes(ORDER)


def test_the_last_step_ends_on_the_last_byte_of_the_last_shard():
    (shard, _, last), = steps.segments(ORDER, ORDER["steps"] - 1)
    assert shard == ORDER["shards"] - 1
    assert last == ORDER["last_shard_bytes"] - 1


def test_a_step_outside_the_run_is_an_error_not_an_address():
    with pytest.raises(ValueError):
        steps.segments(ORDER, ORDER["steps"])
    with pytest.raises(ValueError):
        steps.draw_steps(ORDER["steps"], 4, 0, at=[ORDER["steps"]])


def test_fetch_assembles_the_ranges_in_order_and_checks_their_length():
    calls = []

    class R:
        status_code = 206

        def __init__(self, content):
            self.content = content

    def get(url, headers):
        calls.append((url, headers["Range"]))
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        n = last - first + 1
        return R((bytes(range(256)) * (n // 256 + 1))[:n])

    step = 30_000_000_000 // steps.step_bytes(ORDER)
    tokens = steps.fetch_step(ORDER, step, "abc123", get=get)
    assert len(tokens) == ORDER["sequences_per_step"] * ORDER["sequence_tokens"]
    assert [u.rsplit("/", 1)[1] for u, _ in calls] == ["document-00000-of-00020.bin", "document-00001-of-00020.bin"]
    assert all("/resolve/abc123/" in u for u, _ in calls)

    def short(url, headers):
        return R(b"\x00\x01")

    with pytest.raises(RuntimeError):
        steps.fetch_step(ORDER, 0, "abc123", get=short)


def test_sequences_cut_a_step_back_into_its_rows():
    order = {**ORDER, "sequence_tokens": 3}
    tokens = array.array("H", [1, 2, 3, 4, 5, 6])
    assert steps.sequences(tokens, order) == [[1, 2, 3], [4, 5, 6]]


def test_draw_is_one_step_per_slice_of_the_run_and_deterministic():
    total, n = ORDER["steps"], 16
    picks = steps.draw_steps(total, n, seed=0)
    assert picks == sorted(picks) and len(picks) == n
    for i, step in enumerate(picks):
        assert i * total // n <= step < (i + 1) * total // n
    assert picks == steps.draw_steps(total, n, seed=0)
    assert picks != steps.draw_steps(total, n, seed=1)


def test_named_steps_are_merged_and_not_doubled():
    picks = steps.draw_steps(ORDER["steps"], 4, seed=0, at=[1000, 1000, 143_000 - 1])
    assert picks.count(1000) == 1 and picks[-1] == 142_999
    assert len(picks) == len(set(picks))


def test_a_literal_is_a_literal_and_case_folds_by_default():
    assert steps.compile_pattern("a.b").search("A.B")
    assert not steps.compile_pattern("a.b").search("axb")
    assert steps.compile_pattern("a.b", regex=True).search("axb")
    assert not steps.compile_pattern("OpenAI", case_sensitive=True).search("openai")


def test_a_step_is_counted_by_sequence_and_by_occurrence():
    rx = re.compile("cat")
    examples = []
    texts = ["a cat", "no", "cat cat cat", "dog"]
    c = steps.count_step(7, texts, rx, examples, limit=1)
    assert c == {"step": 7, "sequences": 4, "matched": 2, "occurrences": 4}
    # The cap holds, and the one kept says where it came from.
    assert len(examples) == 1
    assert examples[0]["step"] == 7 and examples[0]["sequence"] == 0 and "cat" in examples[0]["snippet"]


def _per_step(counts, sequences=1024):
    return [
        {"step": s, "sequences": sequences, "matched": k, "occurrences": k}
        for s, k in counts
    ]


def test_the_interval_is_clustered_by_step_not_over_sequences():
    """Sixteen steps agreeing on 1% get about the binomial interval; the same
    total concentrated in two steps has a design effect that widens it."""
    even = steps.summarize(_per_step([(s * 1000, 10) for s in range(16)]))
    lumpy = steps.summarize(_per_step([(s * 1000, 80 if s < 2 else 0) for s in range(16)]))
    assert even["rate"] == lumpy["rate"] == pytest.approx(160 / (16 * 1024))
    assert even["n_effective"] == pytest.approx(16 * 1024, rel=0.02)
    assert lumpy["n_effective"] < even["n_effective"] / 4
    assert lumpy["hi"] - lumpy["lo"] > 2 * (even["hi"] - even["lo"])


def test_a_unanimous_zero_counts_every_sequence_not_every_step():
    """The shard sampler's fallback — a unanimous cluster is one observation —
    would put the upper bound on a never-seen string at a third of the corpus.
    The stream is shuffled, so here a step is 1,024 draws."""
    zeros = steps.summarize(_per_step([(s * 1000, 0) for s in range(8)]))
    assert zeros["design_effect"] == 1.0 and zeros["n_effective"] == 8 * 1024
    assert zeros["hi"] < 0.001
    assert steps.design_effect(_per_step([(s * 1000, 0) for s in range(8)])) is None
    assert steps.design_effect(_per_step([(0, 3)])) is None


def test_slices_borrow_the_whole_runs_design_effect():
    per = _per_step([(s * 1000, 80 if s < 2 else 0) for s in range(16)])
    whole = steps.summarize(per)
    assert whole["design_effect"] > 4
    for sl in steps.by_slice(per, ORDER["steps"], 4, deff=whole["design_effect"]):
        if sl["sequences"]:
            assert sl["design_effect"] == whole["design_effect"]


def test_no_steps_summarize_to_nothing_rather_than_a_zero_rate():
    empty = steps.summarize([])
    assert empty["rate"] is None and empty["lo"] is None and empty["sequences"] == 0


def test_slices_cover_the_run_and_an_unsampled_slice_says_so():
    per = _per_step([(500, 1), (100_000, 3)])
    slices = steps.by_slice(per, ORDER["steps"], 4)
    assert slices[0]["from_step"] == 0 and slices[-1]["to_step"] == ORDER["steps"] - 1
    for a, b in zip(slices, slices[1:]):
        assert b["from_step"] == a["to_step"] + 1
    assert [sl["steps"] for sl in slices] == [1, 0, 1, 0]
    assert slices[1]["rate"] is None
    assert slices[2]["matched"] == 3


def test_exposure_is_the_rate_carried_along_the_run_with_its_interval():
    summary = steps.summarize(_per_step([(s * 1000, 10) for s in range(16)]))
    exp = steps.exposure(summary, [0, 1000, 143_000], 1024)
    assert exp[0] == {"step": 0, "sequences_seen": 0, "expected": 0.0, "lo": 0.0, "hi": 0.0}
    assert exp[1]["expected"] == pytest.approx(summary["rate"] * 1000 * 1024)
    assert exp[1]["lo"] < exp[1]["expected"] < exp[1]["hi"]
    assert exp[2]["expected"] == pytest.approx(exp[1]["expected"] * 143)


def test_a_string_never_seen_still_has_an_upper_bound_at_every_checkpoint():
    summary = steps.summarize(_per_step([(s * 1000, 0) for s in range(16)]))
    exp = steps.exposure(summary, [143_000], 1024)
    assert exp[0]["expected"] == 0 and exp[0]["hi"] > 0


def test_the_checkpoint_schedule_is_pythias():
    cps = ORDER["checkpoints"]
    assert len(cps) == 154
    assert cps[:12] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    assert cps[-1] == 143_000 and cps == sorted(set(cps))


def test_the_second_pass_starts_where_the_corpus_runs_out():
    step = steps.second_pass_step(ORDER)
    assert 98_000 < step < 100_000
    assert steps.second_pass_step({**ORDER, "corpus_tokens": None}) is None
    assert steps.second_pass_step({**ORDER, "corpus_tokens": 10**15}) is None


def test_only_pythia_has_an_order_and_the_command_says_so(monkeypatch):
    assert registry.training_order(registry.resolve("pythia-12b-deduped"))[1] is ORDER
    for name in registry.targets():
        if name != "pythia-12b-deduped":
            assert registry.training_order(registry.resolve(name)) is None

    class Args:
        target, pattern = "olmo-3-7b-instruct", "x"

    with pytest.raises(SystemExit, match="no published training order"):
        cli.cmd_steps(Args())


def test_the_scan_walks_steps_in_order_with_a_fake_fetch(monkeypatch):
    order = {**ORDER, "sequence_tokens": 2, "sequences_per_step": 2}
    monkeypatch.setattr(
        steps, "fetch_step", lambda o, step, rev: array.array("H", [step, 0, step, 1])
    )
    decode = lambda seqs: [f"step{a}-{b}" for a, b in seqs]  # noqa: E731
    per, ex = steps.scan(order, [5, 900], "rev", re.compile("step900"), decode, workers=2)
    assert [c["step"] for c in per] == [5, 900]
    assert [c["matched"] for c in per] == [0, 2]
    assert ex[0]["step"] == 900


def test_an_explicit_step_gets_the_bounded_example_slot_first(monkeypatch):
    order = {**ORDER, "sequence_tokens": 2, "sequences_per_step": 1}
    monkeypatch.setattr(
        steps,
        "fetch_step",
        lambda o, step, rev: array.array("H", [step, 0]),
    )
    decode = lambda seqs: [f"needle at step {seqs[0][0]}"]  # noqa: E731

    _, examples = steps.scan(
        order,
        [1, 10],
        "rev",
        re.compile("needle"),
        decode,
        examples_limit=1,
        workers=1,
        priority_steps=[10],
    )

    assert len(examples) == 1
    assert examples[0]["step"] == 10


def test_the_bounded_budget_keeps_one_example_from_each_explicit_step(monkeypatch):
    order = {**ORDER, "sequence_tokens": 2, "sequences_per_step": 2}
    monkeypatch.setattr(
        steps,
        "fetch_step",
        lambda o, step, rev: array.array("H", [step, 0, step, 1]),
    )
    decode = lambda seqs: [f"needle at step {seq[0]}" for seq in seqs]  # noqa: E731

    _, examples = steps.scan(
        order,
        [1, 10, 20],
        "rev",
        re.compile("needle"),
        decode,
        examples_limit=2,
        workers=1,
        priority_steps=[10, 20],
    )

    assert len(examples) == 2
    assert [example["step"] for example in examples] == [10, 20]


def test_explicit_steps_are_read_but_do_not_enter_population_estimates(
    monkeypatch, tmp_path
):
    """A checkpoint named with --at has a different inclusion probability from
    the stratified draw. It is evidence for inspection, never another sample."""
    captured = {}
    args = SimpleNamespace(
        target="pythia-12b-deduped",
        pattern="needle",
        sample=2,
        seed=0,
        at=[10],
        regex=False,
        case_sensitive=False,
        examples=0,
        slices=2,
        slug=None,
    )
    monkeypatch.setattr(steps, "draw_steps", lambda total, n, seed, at=(): [1, 2, *at])
    monkeypatch.setattr(steps, "decoder", lambda order, revision: lambda rows: rows)
    monkeypatch.setattr(steps, "resolve_tokenizer_revision", lambda repo: "token-rev")
    monkeypatch.setattr(cli.pretrain, "resolve_revision", lambda dataset: "data-rev")
    monkeypatch.setattr(
        steps,
        "scan",
        lambda *a, **k: (
            _per_step([(1, 0), (2, 0), (10, 1024)]),
            [],
        ),
    )

    def write(path, payload):
        captured.update(payload)
        return tmp_path / path.name

    monkeypatch.setattr(cli, "_write_json", write)
    cli.cmd_steps(args)

    assert captured["sample"] == 2
    assert captured["at"] == [10]
    assert captured["steps"] == 2
    assert captured["matched"] == 0
    assert len(captured["per_step"]) == 3
    assert captured["tokenizer_revision"] == "token-rev"


def test_tokenizer_is_fetched_and_cached_by_immutable_revision(monkeypatch, tmp_path):
    calls = []

    class Response:
        content = b'{"version":"1.0"}'

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            calls.append(path)
            return cls()

        def decode_batch(self, seqs, skip_special_tokens):
            return ["decoded"]

    monkeypatch.setattr(steps.pretrain, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(steps.pretrain, "_get", get)
    monkeypatch.setitem(__import__("sys").modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))

    decode = steps.decoder(ORDER, "abc123")
    assert decode([[1]]) == ["decoded"]
    assert "/resolve/abc123/tokenizer.json" in calls[0]
    assert "@abc123.json" in calls[1]

    steps.decoder(ORDER, "abc123")
    assert sum("/resolve/abc123/tokenizer.json" in call for call in calls) == 1


def test_tokenizer_revision_resolves_from_the_model_repository(monkeypatch):
    seen = {}

    class Response:
        def json(self):
            return {"sha": "abc123"}

    def get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        return Response()

    monkeypatch.setattr(steps.pretrain, "_get", get)
    assert steps.resolve_tokenizer_revision("org/model") == "abc123"
    assert seen["url"] == "https://huggingface.co/api/models/org/model/revision/main"
    assert seen["headers"] is steps.hf.HEADERS


@pytest.mark.live
def test_step_zero_still_decodes_to_the_same_text():
    """The layout, pinned against the hub: if the repo were republished with a
    different cut, this is the sentence that would change."""
    from trainspotting import pretrain

    revision = pretrain.resolve_revision(ORDER["dataset"])
    tokens = steps.fetch_step(ORDER, 0, revision)
    tokenizer_revision = steps.resolve_tokenizer_revision(ORDER["tokenizer"])
    first = steps.decoder(ORDER, tokenizer_revision)(steps.sequences(tokens, ORDER)[:1])[0]
    assert "Belle whispered" in first
