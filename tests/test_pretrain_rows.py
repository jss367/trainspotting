"""The second route into a pretraining corpus: paging the datasets-server.

Dolma 3 has to be read shard by shard because the viewer indexes only its head,
and that route buys its coverage with a position bias every result file has to
disclose. A corpus the viewer serves whole needs none of that, and these tests
are mostly about the two routes staying distinguishable — a rows sample that
inherits the shard sampler's caveat, or a shard stage that silently defaults
into the rows route, would each be a result file making a claim about how it
was drawn that is not true of it.
"""

import json

import pytest

from trainspotting import classify, cli, hf, paths, pretrain, registry


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
        "sample_rows_with_pages",
        lambda *a, **k: [
            (5, {"text": "a document"}, [], 5),
            (9, {"text": "clipped"}, ["text"], 5),
            (11, {"text": "   "}, [], 5),      # whitespace only
            (12, {"other": "no text"}, [], 5),
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
    # `cluster` is the exception to "leave it empty": these two rows arrived in
    # one page, and that is the unit every interval over them has to widen for.
    assert all(d["cluster"] == "page-5" for d in docs)


def test_the_page_is_the_rows_route_cluster_and_survives_to_the_written_record(
    tmp_path, monkeypatch
):
    """The empty `shard` a rows document carries is honest — there is no file to
    name — but it is not a cluster identity, and `_cluster_wilson` groups on one.
    Writing the sample without the page collapsed all 300 Pile documents into a
    single cluster and reported n_effective = 1 at any sample size."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(hf, "num_rows", lambda *a, **k: 1_000)
    monkeypatch.setattr(hf, "dataset_revision", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(
        hf,
        "sample_rows_with_pages",
        # Two pages of two, interleaved: a grid over the row index would not
        # recover this grouping, which is why the offset travels with the row.
        lambda *a, **k: [
            (48, {"text": "a"}, [], 47),
            (114, {"text": "b"}, [], 113),
            (47, {"text": "c"}, [], 47),
            (113, {"text": "d"}, [], 113),
        ],
    )
    args = type("A", (), {"target": "pythia-12b-deduped", "stage": None, "sample": 4, "seed": 0})()

    cli.cmd_pretrain(args)

    written = json.loads((tmp_path / "pythia-12b-deduped.pretrain.docs.json").read_text())
    assert [r["cluster"] for r in written["records"]] == [
        "page-47",
        "page-113",
        "page-47",
        "page-113",
    ]


def test_a_shard_document_keeps_the_shard_as_its_only_cluster_identity(tmp_path, monkeypatch):
    """A shard sample must not grow a second field naming the same thing. The
    committed Olmo samples have no `cluster`, and the interval reads `shard`
    for them exactly as it did before this field existed."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(pretrain, "list_shards", lambda *a, **k: ([{"size": 1}], "cafe123"))
    monkeypatch.setattr(pretrain, "group_sizes", lambda *a, **k: {"common_crawl/art": 1})
    monkeypatch.setattr(
        pretrain,
        "sample_documents",
        lambda *a, **k: (
            [
                {
                    "id": "d0",
                    "text": "a document",
                    "source": "common_crawl",
                    "topic": "art",
                    "shard": "data/common_crawl-art-0001/shard_00000000.jsonl.zst",
                    "metadata": {},
                }
            ],
            0,
        ),
    )
    args = type(
        "A",
        (),
        {
            "target": "olmo-3-7b-think",
            "stage": "pretrain",
            "sample": 1,
            "seed": 0,
            "docs_per_shard": 1,
        },
    )()

    cli.cmd_pretrain(args)

    written = json.loads((tmp_path / "olmo-3-7b-think.pretrain.docs.json").read_text())
    assert "cluster" not in written["records"][0]
    assert written["records"][0]["shard"].endswith(".jsonl.zst")


def _write_docs(path, records, **facts):
    path.write_text(
        json.dumps(
            {
                "dataset": "EleutherAI/the_pile_deduplicated",
                "stage": "pretrain",
                "name": "The Pile (deduplicated)",
                "sample": len(records),
                "seed": 0,
                "revision": "deadbeef",
                **facts,
                "records": records,
            }
        )
    )


def test_a_rows_sample_does_not_collapse_to_one_effective_observation(tmp_path, monkeypatch):
    """The regression. `ask --pretrain` on a rows sample used to report
    n_effective = 1 and a 5–95% interval over 300 documents, because every one of
    them carried the same empty `shard`. Thirty pages that mostly agree are worth
    far more than one observation, and the file has to say so."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SITE_DATA", tmp_path)
    # Thirty pages of ten, the shape a 300-document draw actually has. Matches
    # are spread across pages rather than aligned with them, which is what a
    # shuffled corpus looks like and where the design effect should land near 1.
    records = [
        {
            "id": f"row-{page * 1_000 + i}",
            "row": page * 1_000 + i,
            "text": f"document {page}-{i}",
            "chars": 20,
            "source": "",
            "topic": "",
            "shard": "",
            "cluster": f"page-{page * 1_000}",
            "metadata": {},
        }
        for page in range(30)
        for i in range(10)
    ]
    _write_docs(
        tmp_path / "pythia-12b-deduped.pretrain.docs.json",
        records,
        route="rows",
        rows_total=134_318_121,
        caveat=pretrain.rows_sampling_caveat(),
    )
    monkeypatch.setattr(
        classify,
        "classify_prompts",
        lambda prompts, **k: (["yes" if i % 7 == 0 else "no" for i in range(len(prompts))], {}),
    )
    args = type("A", (), {"target": "pythia-12b-deduped", "classifier": "test-model"})()

    cli._label_pretrain_docs(args, "does this mention anything?", "slug")

    scored = json.loads((tmp_path / "pythia-12b-deduped.pretrain.ask-slug.json").read_text())
    lo, hi = scored["ci"]
    # 30 clusters, not 1, and not 300 either: the correction is applied, it just
    # has almost nothing to correct.
    assert scored["n_effective"] > 100
    assert hi - lo < 0.15
    # And the cluster identity is what did it — the same records clustered by
    # their (empty) shard are the bug this test exists for.
    assert cli._cluster_wilson(scored["records"], key="shard")[2] == pytest.approx(1.0)


def test_a_shard_sample_still_clusters_by_shard_without_a_cluster_field(tmp_path, monkeypatch):
    """The committed Olmo samples predate `cluster`, and the interval they show
    on the site must not move. `shard` is the fallback, so it does not."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SITE_DATA", tmp_path)
    records = [
        {
            "id": f"d{s}-{i}",
            "text": f"document {s}-{i}",
            "chars": 20,
            "source": "common_crawl",
            "topic": "art",
            "shard": f"data/common_crawl-art-000{s}/shard_0000000{s}.jsonl.zst",
            "metadata": {},
        }
        for s in range(4)
        for i in range(5)
    ]
    _write_docs(
        tmp_path / "olmo-3-7b-think.pretrain.docs.json", records, route="shards", short_draws=0
    )
    monkeypatch.setattr(
        classify,
        "classify_prompts",
        # Matches aligned with shard boundaries: the whole reason clustering
        # exists. Twenty documents that disagree along shard lines are not twenty
        # observations.
        lambda prompts, **k: (["yes" if "document 0" in p or "document 1" in p else "no" for p in prompts], {}),
    )
    args = type("A", (), {"target": "olmo-3-7b-think", "classifier": "test-model"})()

    cli._label_pretrain_docs(args, "q", "slug")

    scored = json.loads((tmp_path / "olmo-3-7b-think.pretrain.ask-slug.json").read_text())
    # Four shards of five, two unanimously matching: deff = 20/3, so twenty
    # documents carry the information of three. What matters here is that the
    # number is the shard one — clustering on the absent `cluster` field would
    # have made it 1.
    assert scored["n_effective"] == pytest.approx(3.0)
    assert all(r["cluster"] == r["shard"] for r in scored["records"])


def test_a_rows_run_writes_the_corpus_size_and_not_shard_facts(tmp_path, monkeypatch):
    """`shards: 0`, `bytes: 0` or `groups: {}` would read on the site as a
    measured result rather than as a breakdown this route cannot produce."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(hf, "num_rows", lambda *a, **k: 100)
    monkeypatch.setattr(hf, "dataset_revision", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(
        hf, "sample_rows_with_pages", lambda *a, **k: [(3, {"text": "doc"}, [], 0)]
    )
    args = type("A", (), {"target": "pythia-12b-deduped", "stage": None, "sample": 1, "seed": 0})()

    cli.cmd_pretrain(args)

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
    args = type("A", (), {"target": "pythia-12b-deduped", "question": "q", "slug": None,
                          "pretrain": False, "pretrain_only": False, "stage": None})()
    with pytest.raises(SystemExit, match="--pretrain"):
        cli.cmd_ask(args)


@pytest.mark.parametrize("as_json", [False, True])
def test_sources_on_a_base_only_model_fails_instead_of_writing_an_empty_audit(
    tmp_path, monkeypatch, as_json
):
    """It used to iterate an empty stage list, exit 0 having printed nothing, and
    with --json write `{}` — an audit file the site would serve as a measured
    empty breakdown. Every other prompt-reading command fails through
    `_select_stages`; this one now does too."""
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    args = type("A", (), {"target": "pythia-12b-deduped", "json": as_json})()

    with pytest.raises(SystemExit, match="no post-training stages"):
        cli.cmd_sources(args)

    assert not list(tmp_path.glob("*.sources.json"))


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
