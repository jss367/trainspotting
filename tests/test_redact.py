import base64
import json

from trainspotting.commands.common import _write_json
from trainspotting.redact import MARKER, redact_credentials


def token(account=b"123456789012345678"):
    return base64.urlsafe_b64encode(account).decode().rstrip("=") + ".abcDEF." + "x" * 27


def test_masks_embedded_credential_and_preserves_surrounding_text():
    value = token()
    assert redact_credentials(f'client.run("{value}")') == f'client.run("{MARKER}")'
    assert redact_credentials(value + " " + value) == MARKER + " " + MARKER
    assert redact_credentials(MARKER) == MARKER


def test_dotted_text_without_a_numeric_account_id_is_unchanged():
    value = token(b"not-a-discord-user!")
    assert redact_credentials(value) == value
    assert redact_credentials("ordinary.training.example") == "ordinary.training.example"


def test_result_writer_redacts_nested_text_without_changing_metadata(tmp_path):
    path = tmp_path / "sample.json"
    _write_json(path, {"records": [{"row": 3, "prompt": token(), "label": "other"}], "sample": 1000})
    assert json.loads(path.read_text()) == {
        "records": [{"row": 3, "prompt": MARKER, "label": "other"}], "sample": 1000,
    }


def test_case_study_redacts_sampled_document_fields_before_storage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from trainspotting.commands import case_study

    fake = token()
    result = {
        "probe": {"query": "safe query", "occurrences": 1, "exhaustive": True,
                  "documents": [{"excerpt": f"before {fake} after", "title": fake,
                                 "url": f"https://example.com/{fake}"}]},
        "spread": {"query": "safe spread", "occurrences": 1, "drawn": 1,
                   "domains": [{"domain": "example.com", "share": 1}]},
    }
    monkeypatch.setattr(case_study, "RESULTS", tmp_path / "nested" / "results")
    monkeypatch.setattr(case_study.casestudy, "run", lambda *a, **k: result)
    slug = next(iter(case_study.casestudy.CASE_STUDIES))
    case_study.cmd_case_study(SimpleNamespace(slug=slug))
    stored = (case_study.RESULTS / f"case-study.{slug}.json").read_text()
    assert fake not in stored
    assert json.loads(stored) == json.loads(redact_credentials(json.dumps(result)))


def test_backfill_redacts_freshly_fetched_prompts(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    from trainspotting import hf

    script = Path(__file__).resolve().parents[1] / "scripts" / "backfill_prompts.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    path = results / "olmo-3-7b-think.sft.labels.json"
    prefix = "safe opening " * 20
    path.write_text(json.dumps({"sample": 1, "seed": 0, "records": [{"prompt": prefix}]}))
    fake = token()
    monkeypatch.setattr(hf, "sample_rows", lambda *a, **k: [{"messages": [{"role": "user", "content": prefix + fake}]}])
    main()
    saved = json.loads(path.read_text())
    assert saved["records"][0]["prompt"] == prefix + MARKER
    assert saved["sample"] == 1


def test_fixture_capture_redacts_before_computing_prompt_goldens(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    from trainspotting import hf, registry

    script = Path(__file__).resolve().parents[1] / "scripts" / "capture_row_fixtures.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "FIXTURES", tmp_path)
    monkeypatch.setattr(registry, "targets", lambda: ["olmo-3-7b-think"])
    stage = registry.post_training_stages(registry.resolve("olmo-3-7b-think"))[0]
    monkeypatch.setattr(registry, "post_training_stages", lambda target: [stage])
    fake = token()
    row = {"messages": [{"role": "user", "content": "before " + fake + " after"}]}
    monkeypatch.setattr(hf, "_get", lambda *a, **k: {"rows": [{"row": row}]})
    main()
    stored = (tmp_path / f"olmo-3-7b-think.{stage['stage']}.json").read_text()
    assert fake not in stored
    saved = json.loads(stored)
    expected = "before " + MARKER + " after"
    assert saved["row"]["messages"][0]["content"] == expected
    assert saved["prompt_chars"] == len(expected)
    assert saved["prompt_head"] == expected[:120]
    assert saved["prompt_tail"] == expected[-60:]


def test_backfill_skips_summaries_and_nonprompt_records_before_fetching(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path

    from trainspotting import hf

    script = Path(__file__).resolve().parents[1] / "scripts" / "backfill_prompts.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    files = {
        "agreement": {"replicate": {"same_draw": True}},
        "context": {"sample": 1000, "seed": 0, "records": [{"row": 1, "key": "opening"}]},
        "profile": {"sample": 1000, "seed": 0, "records": [{"row": 1, "k": "hash", "m": {}}]},
    }
    for suffix, data in files.items():
        (results / f"olmo-3-7b-think.sft.{suffix}.json").write_text(json.dumps(data))

    def no_fetch(*args, **kwargs):
        raise AssertionError("nonprompt artifacts must not trigger upstream fetches")

    monkeypatch.setattr(hf, "sample_rows", no_fetch)
    main()
    for suffix, expected in files.items():
        assert json.loads((results / f"olmo-3-7b-think.sft.{suffix}.json").read_text()) == expected
