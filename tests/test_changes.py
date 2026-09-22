"""Scientific invariants: reversals, normalization, probability scale, and coverage."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainspotting import changes
from trainspotting.commands.changes import cmd_changes
from sitejs import run_suite


def test_path_can_move_and_return_to_its_start():
    torch = pytest.importorskip("torch")
    start = {"model.layers.0.weight": torch.tensor([0., 0.])}
    middle = {"model.layers.0.weight": torch.tensor([3., 4.])}
    outward = changes.parameter_distance(start, middle)
    back = changes.parameter_distance(middle, start)
    net = changes.parameter_distance(start, start)
    assert outward["rms"] == pytest.approx(5 / 2**0.5)
    assert outward["relative_l2"] is None
    assert outward["layers"][0]["name"] == "Transformer block 1"
    assert outward["rms"] + back["rms"] > net["rms"] == 0
    assert back["relative_l2"] == 1


def test_distance_weights_parameters_not_layers_equally():
    torch = pytest.importorskip("torch")
    a = {"layers.0.weight": torch.ones(1), "layers.1.weight": torch.ones(3)}
    b = {"layers.0.weight": torch.tensor([3.]), "layers.1.weight": torch.ones(3)}
    d = changes.parameter_distance(a, b)
    assert d["parameters"] == 4
    assert d["rms"] == 1
    assert d["relative_l2"] == 1


@pytest.mark.parametrize("other,match", [({"other": [1.]}, "names differ"),
                                       ({"weight": [1., 2.]}, "shape differs"),
                                       ({"weight": [float('nan')]}, "Non-finite")])
def test_incompatible_or_corrupt_weights_are_not_silently_scored(other, match):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match=match):
        changes.parameter_distance({"weight": torch.tensor([1.])},
                                   {k: torch.tensor(v) for k, v in other.items()})


def test_probe_divergence_has_known_endpoints_and_is_symmetric():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1., 0.], [0.5, 0.5]])
    b = torch.tensor([[0., 1.], [0.5, 0.5]])
    probes = [{"topic": "Changes"}, {"topic": "Same"}]
    result = changes.probability_change(a, b, probes)
    assert result["js_bits"] == 0.5
    assert result["topics"][0]["js_bits"] == 1
    assert result["topics"][1]["js_bits"] == 0
    assert result == changes.probability_change(b, a, probes)
    assert changes.probability_change(a, a, probes)["js_bits"] == 0


@pytest.mark.parametrize("edit,match", [
    (lambda p: p['stages'].append(copy.deepcopy(p['stages'][0])), "duplicate"),
    (lambda p: p['stages'][0].update(coverage='unknown'), "coverage"),
    (lambda p: p['stages'][0].update(lineage_source=''), "lineage_source"),
    (lambda p: p['stages'][0]['checkpoints'].reverse(), "increasing"),
    (lambda p: p['stages'][0]['checkpoints'][0].update(step=None), "integers"),
    (lambda p: p['stages'][0]['checkpoints'][0].update(revision=''), "revision"),
])
def test_invalid_plans_fail_before_loading(edit, match):
    plan = changes.default_plan('pythia-70m-deduped')
    edit(plan)
    with pytest.raises(ValueError, match=match):
        changes.validate_plan(plan, {'pretrain'})


def test_default_plan_covers_initialization_and_final_checkpoint():
    plan = changes.default_plan('pythia-70m-deduped')
    changes.validate_plan(plan, {'pretrain'})
    assert plan['stages'][0]['checkpoints'][0]['step'] == 0
    assert plan['stages'][0]['checkpoints'][-1]['step'] == 143000
    with pytest.raises(ValueError, match='verified'):
        changes.default_plan('olmo-3-7b-think')


def test_standalone_dataset_is_rejected_before_inference():
    with pytest.raises(SystemExit, match='standalone dataset'):
        cmd_changes(SimpleNamespace(target='wildchat-1m'))


def test_failed_measurement_preserves_previous_artifact(tmp_path, monkeypatch):
    from trainspotting import paths
    output = tmp_path / 'pythia-70m-deduped.changes.json'
    output.write_text('{"previous":true}')
    monkeypatch.setattr(paths, 'RESULTS', tmp_path)
    def fail(*a):
        raise ValueError('Tokenizer changed')
    monkeypatch.setattr(changes, 'measure', fail)
    with pytest.raises(SystemExit, match='Tokenizer changed'):
        cmd_changes(SimpleNamespace(target='pythia-70m-deduped', plan=None, probes=None, device='cpu'))
    assert json.loads(output.read_text()) == {'previous': True}


def test_probe_file_and_fingerprint_inputs_are_nonempty():
    probes = changes.read_probes(changes.PROBES)
    assert len(probes) == 18
    assert len({p['topic'] for p in probes}) == 6


def test_training_change_site():
    run_suite(Path(__file__).parent / 'site' / 'training_change.test.mjs')


def test_published_evaluation_parser_keeps_missing_baseline_and_categories():
    from scripts.capture_training_evaluations import parse_table
    text = '''| Benchmark | A | B | C |
|---|---|---|---|
| **Math** | | | |
| Test | 0 | 20 | 10 |
| **Safety** | 40 | 50 | 45 |
'''
    rows = parse_table(text, 1, ['A', 'B', 'C'])
    assert rows[0] == {'name': 'Test', 'category': 'Mathematics',
                       'scores': {'sft': 0, 'dpo': 20, 'rlvr': 10}}
    assert rows[1]['category'] == 'Safety'
    assert 'pretrain' not in rows[0]['scores']
    with pytest.raises(ValueError, match='columns changed'):
        parse_table(text, 1, ['Wrong', 'B', 'C'])


def test_published_blank_benchmark_name_uses_source_category():
    from scripts.capture_training_evaluations import parse_table
    text = '''| Skill | Benchmark | A | B | C |
|---|---|---|---|---|
| **Safety** | | 40 | 50 | 45 |
'''
    assert parse_table(text, 2, ['A', 'B', 'C'])[0]['name'] == 'Safety'


def test_supplied_olmo_plan_has_all_six_phases():
    from trainspotting import registry
    plan = json.loads((Path(__file__).parents[1] / 'measurement-plans' /
                       'olmo-7b-instruct-training-change.json').read_text())
    stages = {s['stage'] for s in registry.resolve('olmo-3-7b-instruct')['stages']}
    changes.validate_plan(plan, stages)
    assert {s['stage'] for s in plan['stages']} == stages
    for before, after in zip(plan['stages'], plan['stages'][1:]):
        end, start = before['checkpoints'][-1], after['checkpoints'][0]
        assert (end['repo'], end['revision']) == (start['repo'], start['revision'])


def test_pythia_output_head_is_not_an_embedding():
    """GPT-NeoX calls its unembedding `embed_out`; matching on "embed" merged it
    into the input embeddings and the output group vanished."""
    assert changes.layer_name("gpt_neox.embed_in.weight") == "Embeddings"
    assert changes.layer_name("embed_out.weight") == "Output and other parameters"
    assert changes.layer_name("model.embed_tokens.weight") == "Embeddings"
    assert changes.layer_name("lm_head.weight") == "Output and other parameters"
