"""The second route into a pretraining corpus: paging the datasets-server.

Dolma 3 has to be read shard by shard because the viewer indexes only its head,
and that route buys its coverage with a position bias every result file has to
disclose. A corpus the viewer serves whole needs none of that, and these tests
are mostly about the two routes staying distinguishable — a rows sample that
inherits the shard sampler's caveat, or a shard stage that silently defaults
into the rows route, would each be a result file making a claim about how it
was drawn that is not true of it.
"""

import pytest

from trainspotting import cli, hf, pretrain, registry


def test_a_stage_declares_its_route_and_shards_is_the_default():
    """Defaulting the other way would let a stage that forgot to declare one
    sample the first 5 GB of a 450 GB repo and call it uniform."""
    pile = registry.pretrain_stages(registry.resolve("pythia-12b-deduped"))[0]
    assert registry.sample_route(pile) == "rows"
    for name in ("olmo-3-7b-think", "olmo-3-32b-think"):
        for stage in registry.pretrain_stages(registry.resolve(name)):
            assert registry.sample_route(stage) == "shards"
    assert registry.sample_route({}) == "shards"


def test_only_the_shard_caveat_discloses_a_position_bias():
    """The shard route always samples a shard's head and has to say so; the rows
    route reaches any offset and has to not say so. Letting one caveat stand in
    for the other would either invent a bias or hide one."""
    shard, rows = pretrain.sampling_caveat(), pretrain.rows_sampling_caveat()
    assert "never the tail" in shard
    assert "no position bias to correct for" in rows
    assert "never the tail" not in rows
    # The rows caveat still owes the reader the one thing that *is* clustered
    # about it: the draw is by page, and it is only safe because the corpus was
    # shuffled before release.
    assert "shuffled before release" in rows


def test_a_rows_document_carries_a_row_index_and_no_invented_provenance(monkeypatch):
    """The deduplicated Pile is one `text` column: no source, no topic, no
    shard, no filter metadata. Filling those with plausible-looking values is
    the failure mode — the site would render provenance the corpus never had."""
    monkeypatch.setattr(hf, "num_rows", lambda *a, **k: 134_318_121)
    monkeypatch.setattr(
        hf,
        "sample_rows_with_truncation",
        lambda *a, **k: [
            (5, {"text": "a document"}, []),
            (9, {"text": "clipped"}, ["text"]),
            (11, {"text": "   "}, []),      # whitespace only
            (12, {"other": "no text"}, []),
        ],
    )

    docs, total = pretrain.sample_rows_documents("EleutherAI/the_pile_deduplicated", 4)

    assert total == 134_318_121
    # The empty and column-less rows drop rather than becoming empty documents.
    assert [d["row"] for d in docs] == [5, 9]
    assert [d["id"] for d in docs] == ["row-5", "row-9"]
    assert all(d["source"] == "" and d["topic"] == "" and d["shard"] == "" for d in docs)
    assert all(d["metadata"] == {} for d in docs)
    # A cell the server shortened travels flagged: `chars` is then the length of
    # what arrived, not of the document.
    assert [d["truncated"] for d in docs] == [False, True]


def test_a_rows_run_writes_the_corpus_size_and_not_shard_facts(tmp_path, monkeypatch):
    """`shards: 0`, `bytes: 0` or `groups: {}` would read on the site as a
    measured result rather than as a breakdown this route cannot produce."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(hf, "num_rows", lambda *a, **k: 100)
    monkeypatch.setattr(hf, "dataset_revision", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(
        hf, "sample_rows_with_truncation", lambda *a, **k: [(3, {"text": "doc"}, [])]
    )
    args = type("A", (), {"target": "pythia-12b-deduped", "stage": None, "sample": 1, "seed": 0})()

    cli.cmd_pretrain(args)

    import json

    written = json.loads((tmp_path / "pythia-12b-deduped.pretrain.docs.json").read_text())
    assert written["route"] == "rows"
    assert written["rows_total"] == 100
    assert written["revision"] == "deadbeef"
    assert not {"shards", "bytes", "groups", "short_draws", "docs_per_shard"} & set(written)
    assert written["records"][0]["row"] == 3
    assert written["caveat"] == pretrain.rows_sampling_caveat()


def test_a_base_only_model_has_no_post_training_stages():
    """Pythia's defining shape here. Every layer that reads prompts keys on this
    being empty, so a stage sneaking in would give them rows to sample from a
    pipeline that does not exist."""
    pythia = registry.resolve("pythia-12b-deduped")
    assert pythia["is_model"]
    assert registry.post_training_stages(pythia) == []
    assert [s["stage"] for s in pythia["stages"]] == ["pretrain"]


def test_ask_without_pretrain_on_a_base_model_says_what_to_pass():
    """It would otherwise exit on "has no post-training stages", which is true
    and useless: the corpus it can score is the whole reason to ask."""
    args = type("A", (), {"target": "pythia-12b-deduped", "question": "q", "slug": None, "pretrain": False})()
    with pytest.raises(SystemExit, match="--pretrain"):
        cli.cmd_ask(args)


def test_the_pile_composition_is_in_bytes_and_says_what_it_describes():
    """It is EleutherAI's table for the corpus before deduplication, while the
    documents come from after it. Rendering those shares unqualified next to
    that sample would claim a measurement of the sampled corpus."""
    stage = registry.pretrain_stages(registry.resolve("pythia-12b-deduped"))[0]
    assert stage["composition_unit"] == "bytes"
    assert all("bytes" in c and "tokens" not in c for c in stage["composition"])
    assert len(stage["composition"]) == 22
    # 825.18 GiB, the total EleutherAI reports.
    total = sum(c["bytes"] for c in stage["composition"]) / 1024**3
    assert 825.0 < total < 825.3
    assert "before deduplication" in stage["composition_scope"]


def test_the_pile_index_is_not_given_the_no_dolma3_caveat():
    """It is the corpus Pythia actually trained on, so telling a reader the
    count is "over a different corpus than the one this tool samples" would be
    a stronger warning than the truth."""
    from trainspotting import infinigram

    assert "v4_piletrain_llama" in infinigram.INDEXES
    caveat = infinigram.caveat_for("v4_piletrain_llama")
    assert "deduplicat" in caveat
    assert caveat != infinigram.NO_OLMO3_CAVEAT
    assert infinigram.caveat_for("v4_dolma-v1_7_llama") == infinigram.NO_OLMO3_CAVEAT
    # An index this module does not know still must not be characterized.
    assert infinigram.caveat_for("v4_some_future_dolma3_index") is None


@pytest.mark.live
def test_live_the_pile_is_still_served_whole():
    """The one upstream fact the rows route rests on. If HuggingFace ever
    re-indexes this corpus partially, the sample silently narrows to whatever
    prefix it kept and the caveat this route prints becomes false."""
    import requests

    r = requests.get(
        f"{hf.BASE}/rows",
        params={
            "dataset": "EleutherAI/the_pile_deduplicated",
            "config": "default",
            "split": "train",
            "offset": 130_000_000,
            "length": 1,
        },
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    assert body["partial"] is False
    assert body["num_rows_total"] > 134_000_000
    assert body["rows"][0]["row"]["text"]


def test_find_runs_without_a_target(monkeypatch):
    """`find` searches an index, not a registered model, so it is the one
    subcommand with no target to canonicalize. Canonicalizing unconditionally
    crashed it with an AttributeError before it ran at all — which is how the
    Pile index went unexercised from the CLI."""
    import sys

    seen = {}
    monkeypatch.setattr(cli, "cmd_find", lambda args: seen.setdefault("index", args.index))
    monkeypatch.setattr(
        sys, "argv", ["trainspotting", "find", "a phrase", "--index", "v4_piletrain_llama"]
    )
    cli.main()
    assert seen["index"] == "v4_piletrain_llama"
