"""Grouping committed `grep` runs into the searches they actually ran.

The report reads result files off disk and hands each group to the influence
layer. The group key is the interesting part: a slug is a filename, so two runs
sharing one can be two different searches, and ranking those against each other
under whichever pattern sorted first is a wrong answer with no visible symptom.
"""

import json

from trainspotting import cli


def write(tmp_path, model, stage, slug, **over):
    payload = {
        "dataset": f"allenai/{stage}", "stage": stage, "slug": slug,
        "pattern": "ChatGPT", "regex": False, "case_sensitive": False,
        "fields": ["prompt"], "available_fields": ["prompt"],
        "matched": 5, "total_rows": 1_000, "by_group": {"prompt": 5},
        "by_source": {}, "rows_by_source": {}, "by_source_group": {},
        "revision": "abc123def456", "partial": False, "unsearched_columns": [],
    }
    payload.update(over)
    (tmp_path / f"{model}.{stage}.grep-{slug}.json").write_text(json.dumps(payload))


def traces(tmp_path, monkeypatch, model="olmo-3-7b-think"):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    return cli._grep_traces(model, cli.registry.get_model(model))


def test_stages_of_one_search_are_grouped_together(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "chatgpt")
    out = traces(tmp_path, monkeypatch)
    assert len(out) == 1
    slug, split, trace = out[0]
    assert (slug, split) == ("chatgpt", False)
    assert sorted(r["stage"] for r in trace["ranked"]) == ["dpo", "rlvr"]


def test_one_slug_over_two_patterns_is_split_and_flagged(tmp_path, monkeypatch):
    # Refining a regex without changing --slug leaves the old stage's file in
    # place under the same name.
    write(tmp_path, "olmo-3-7b-think", "dpo", "identity", pattern="I am ChatGPT")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "identity", pattern="I am (ChatGPT|GPT-4)")
    out = traces(tmp_path, monkeypatch)
    assert len(out) == 2
    assert all(split for _, split, _ in out)
    assert {t["pattern"] for _, _, t in out} == {"I am ChatGPT", "I am (ChatGPT|GPT-4)"}
    # And neither group ranks the other search's stage.
    assert all(len(t["ranked"]) == 1 for _, _, t in out)


def test_a_matching_flag_alone_splits_the_group(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "chatgpt", case_sensitive=True)
    out = traces(tmp_path, monkeypatch)
    assert len(out) == 2 and all(split for _, split, _ in out)


def test_distinct_slugs_are_not_reported_as_a_collision(tmp_path, monkeypatch):
    # The regression this guards: comparing a set of slugs against a set of
    # group keys is never equal, which printed the collision note on every run.
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "openai", pattern="OpenAI")
    out = traces(tmp_path, monkeypatch)
    assert len(out) == 2
    assert not any(split for _, split, _ in out)


def test_a_run_written_before_the_slug_was_recorded_takes_it_from_its_name(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    path = tmp_path / "olmo-3-7b-think.dpo.grep-chatgpt.json"
    payload = json.loads(path.read_text())
    del payload["slug"]
    path.write_text(json.dumps(payload))
    _, _, trace = traces(tmp_path, monkeypatch)[0]
    assert trace["slug"] == "chatgpt"


def test_another_model_s_runs_are_not_folded_in(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-instruct", "dpo", "chatgpt")
    assert len(traces(tmp_path, monkeypatch)) == 1


def test_the_preamble_does_not_claim_full_coverage_for_a_partial_conversion(tmp_path, monkeypatch, capsys):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt", partial=True)
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.cmd_report(type("A", (), {"model": "olmo-3-7b-think"})())
    out = capsys.readouterr().out
    assert "converted only part of dpo ('ChatGPT')" in out
    assert "cover the converted subset alone" in out


def test_the_qualifier_is_absent_when_every_conversion_is_complete(tmp_path, monkeypatch, capsys):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.cmd_report(type("A", (), {"model": "olmo-3-7b-think"})())
    assert "converted subset alone" not in capsys.readouterr().out


def test_split_traces_get_free_slugs_rather_than_the_colliding_one(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "identity", pattern="I am ChatGPT")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "identity", pattern="I am GPT-4")
    out = traces(tmp_path, monkeypatch)
    assert all(t["slug_collides"] for _, _, t in out)
    assert {t["slug_suggest"] for _, _, t in out} == {"identity-1", "identity-2"}
    from trainspotting import influence
    for _, _, t in out:
        text = " ".join(influence.render(t, "olmo-3-7b-think"))
        assert f"--slug {t['slug_suggest']}" in text


def test_a_flag_only_collision_still_gets_distinct_slugs(tmp_path, monkeypatch):
    # Same pattern, so dropping `--slug` would derive the same filename twice.
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "chatgpt", case_sensitive=True)
    out = traces(tmp_path, monkeypatch)
    suggested = {t["slug_suggest"] for _, _, t in out}
    assert len(suggested) == 2 and "chatgpt" not in suggested


def test_a_suggested_slug_skips_one_already_on_disk(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "identity", pattern="a")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "identity", pattern="b")
    write(tmp_path, "olmo-3-7b-think", "sft", "identity-1", pattern="c")
    out = traces(tmp_path, monkeypatch)
    suggested = {t["slug_suggest"] for _, _, t in out if t["slug_collides"]}
    assert suggested == {"identity-2", "identity-3"}


def test_an_unrelated_slug_keeps_its_own(tmp_path, monkeypatch):
    # One contested slug must not strip `--slug` from every other trace.
    write(tmp_path, "olmo-3-7b-think", "dpo", "identity", pattern="a")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "identity", pattern="b")
    write(tmp_path, "olmo-3-7b-think", "sft", "chatgpt", pattern="ChatGPT")
    from trainspotting import influence
    for slug, _, t in traces(tmp_path, monkeypatch):
        if slug != "chatgpt":
            continue
        assert not t["slug_collides"]
        assert "--slug chatgpt" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_an_uncontested_slug_is_not_marked(tmp_path, monkeypatch):
    write(tmp_path, "olmo-3-7b-think", "dpo", "chatgpt")
    write(tmp_path, "olmo-3-7b-think", "rlvr", "chatgpt")
    _, _, t = traces(tmp_path, monkeypatch)[0]
    assert not t["slug_collides"]
