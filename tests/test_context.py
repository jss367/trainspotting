"""What a stored training example keeps about where it came from.

A context record carries a small metadata dict, and the site's provenance grid
is built entirely out of it. The per-kind lists in `context.build` are the
columns these mixes happened to use when they were written, so a mix that names
its provenance something else drops out of the record silently: Dolci Think 32B
calls it `source` where the 7B mixes call it `source_dataset`, and the result
was a stage whose grid the site suppressed as "records nothing about where its
examples came from" while `sources.json` listed fourteen source groups for it.

The registry already declares each stage's `source_columns` — the sources layer
counts them — so this holds `build` to them against the saved rows.
"""

import pytest
from conftest import row_fixture

from trainspotting import context, registry

STAGES = [
    (name, stage)
    for name in registry.targets()
    for stage in registry.post_training_stages(registry.resolve(name))
]
STAGE_IDS = [f"{t}.{s['stage']}" for t, s in STAGES]


@pytest.mark.parametrize(("target_name", "stage"), STAGES, ids=STAGE_IDS)
def test_a_record_keeps_the_source_columns_the_registry_declares(target_name, stage):
    saved = row_fixture(target_name, stage["stage"])
    columns = stage.get("source_columns") or ()

    rec = context.build(saved["row"], registry.stage_kind(stage), "prompt", 0, columns)

    # Only the columns this row actually carries: the registry names the columns
    # the stage is counted by, and a row is allowed to leave one null.
    expected = {c for c in columns if saved["row"].get(c) not in (None, "", [], {})}
    assert expected <= set(rec["meta"]), (
        f"{stage['hf_dataset']}: {sorted(expected - set(rec['meta']))} declared in the registry "
        "but dropped from the stored example, so the site cannot group by it"
    )


@pytest.mark.parametrize(("target_name", "stage"), STAGES, ids=STAGE_IDS)
def test_every_stage_can_be_grouped_by_something(target_name, stage):
    """The end the grid actually needs: some column survives into the record.
    A stage with an empty metadata dict has no provenance view at all."""
    saved = row_fixture(target_name, stage["stage"])

    rec = context.build(
        saved["row"], registry.stage_kind(stage), "prompt", 0, stage.get("source_columns") or ()
    )

    assert rec["meta"], f"{stage['hf_dataset']}: nothing recorded about where this example came from"


def test_source_columns_do_not_displace_the_extras_a_kind_already_kept():
    """The registry's columns are added to the per-kind list, not swapped for
    it. Dolci Instruct SFT is grouped by `source_dataset` and also carries a
    `domain` bucket the language card reads."""
    row = {"messages": [{"role": "user", "content": "hi"}], "source_dataset": "Wildchat", "domain": "Chat"}

    rec = context.build(row, "sft", "hi", 0, ["source_dataset"])

    assert rec["meta"] == {"source_dataset": "Wildchat", "domain": "Chat"}


def test_a_stage_with_no_declared_columns_still_records_what_it_can():
    row = {"messages": [{"role": "user", "content": "hi"}], "source_dataset": "Wildchat"}

    assert context.build(row, "sft", "hi", 0)["meta"] == {"source_dataset": "Wildchat"}


class TestTheAnswerKey:
    """What a record keeps of what the verifier scores against.

    An RL row's `ground_truth` is a list wherever the mix accepts more than one
    form of the answer. The record kept the first element, so the alternatives
    were invisible to the site's search while `search.fields` and whole-mix
    `grep` both matched them — the count could prove a string was in the answer
    key and the drill-down said it was not.
    """

    def build(self, row):
        return context.build(row, "rlvr", "prompt", 0)["reward"]["ground_truth"]

    def test_every_alternative_is_kept(self):
        got = self.build({"ground_truth": ["7", "seven", "VII"]})
        assert got["text"] == "7\nseven\nVII"

    def test_it_renders_a_list_the_way_a_search_would(self):
        """The same function `search` and `grep` use, so the record holds the
        text a search would have matched against rather than a second opinion
        about what a cell says."""
        from trainspotting import search

        row = {"ground_truth": ["alpha", "beta"]}
        assert self.build(row)["text"] == search.flatten(row["ground_truth"])

    def test_a_plain_string_is_unchanged(self):
        assert self.build({"ground_truth": "42"})["text"] == "42"

    def test_it_falls_back_to_the_reward_model(self):
        assert self.build({"reward_model": {"ground_truth": "x"}})["text"] == "x"

    def test_an_empty_list_falls_back_rather_than_storing_nothing(self):
        row = {"ground_truth": [], "reward_model": {"ground_truth": "x"}}
        assert self.build(row)["text"] == "x"

    def test_no_answer_key_at_all_is_none(self):
        assert self.build({}) is None
