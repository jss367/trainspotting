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
