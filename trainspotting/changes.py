"""Checkpoint measurements, independent of training objective.

Distances describe aligned parameters within one model lineage, not knowledge
or causal attribution. Probe divergence describes one next-token distribution
per fixed, untemplated input, not whether responses are better or safer.
Heavy dependencies are imported only when running a measurement.
"""

import gc
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PROBES = Path(__file__).parent / "probes" / "training-change.json"


def default_plan(target):
    if not target.startswith("pythia-"):
        raise ValueError("This model needs --plan with verified phase-boundary checkpoints; see docs/training-change.md")
    return {"stages": [{"stage": "pretrain", "coverage": "full_stage",
                        "lineage_source": "https://github.com/EleutherAI/pythia",
                        "checkpoints": [{"repo": f"EleutherAI/{target}", "revision": f"step{s}",
                                         "step": s, "label": f"Step {s:,}"}
                                        for s in (0, 1000, 10000, 143000)]}]}


def validate_plan(plan, stages):
    seen = set()
    if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list) or not plan["stages"]:
        raise ValueError("Plan needs a nonempty stages list")
    for stage in plan["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("Each stage must be an object")
        name = stage.get("stage")
        if name not in stages or name in seen:
            raise ValueError(f"Unknown or duplicate stage: {name}")
        seen.add(name)
        if stage.get("coverage") not in ("full_stage", "partial"):
            raise ValueError("Each stage needs coverage: full_stage or partial")
        if not stage.get("lineage_source"):
            raise ValueError("Each stage needs a lineage_source documenting checkpoint ancestry")
        points = stage.get("checkpoints", [])
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("Each stage needs at least two checkpoints")
        for point in points:
            if not isinstance(point, dict) or not all(isinstance(point.get(k), str) and point[k] for k in ("repo", "revision")):
                raise ValueError("Each checkpoint needs repo and revision")
        steps = [p.get("step") for p in points]
        if any(s is not None for s in steps):
            if any(type(s) is not int or s < 0 for s in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
                raise ValueError("Checkpoint steps must all be nonnegative integers in increasing order")


def layer_name(name):
    match = re.search(r"(?:layers|h)\.(\d+)\.", name)
    if match:
        return f"Transformer block {int(match[1]) + 1}"
    # GPT-NeoX names its unembedding `embed_out`, which is the output head.
    if "embed" in name and "embed_out" not in name:
        return "Embeddings"
    if "norm" in name:
        return "Final normalization"
    return "Output and other parameters"


def parameter_distance(before, after):
    """RMS and relative L2 change; tied parameters count once in named_parameters.

    Float64 reductions in bounded chunks avoid both cancellation and a full
    double-precision copy of the model. Refuse partial-key comparisons.
    """
    import torch

    if before.keys() != after.keys():
        raise ValueError("Checkpoint parameter names differ; cannot compare aligned weights")
    groups = {}
    for name, a in before.items():
        b = after[name]
        if a.shape != b.shape:
            raise ValueError(f"Checkpoint parameter shape differs: {name}")
        sums = groups.setdefault(layer_name(name), [0, 0.0, 0.0])
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        for start in range(0, a.numel(), 1_000_000):
            x = flat_a[start:start + 1_000_000].to(dtype=torch.float64)
            y = flat_b[start:start + 1_000_000].to(dtype=torch.float64)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Non-finite weights: {name}")
            sums[0] += x.numel()
            sums[1] += (y - x).square().sum().item()
            sums[2] += x.square().sum().item()

    def summarize(sums):
        n, delta, base = sums
        if not n:
            raise ValueError("No parameters to compare")
        return {"parameters": n, "rms": math.sqrt(delta / n),
                "relative_l2": math.sqrt(delta / base) if base else None}

    total = summarize([sum(v[i] for v in groups.values()) for i in range(3)])
    total["layers"] = [{"name": k, **summarize(v)} for k, v in groups.items()]
    return total


def probability_change(before, after, probes):
    """Mean Jensen–Shannon divergence in bits, equally weighted by prompt."""
    import torch

    if before.shape != after.shape or before.shape[0] != len(probes):
        raise ValueError("Probe distributions do not align")
    a, b = before.double(), after.double()
    middle = (a + b) / 2
    tiny = torch.finfo(torch.float64).tiny
    js = ((a * (a.clamp_min(tiny).log2() - middle.clamp_min(tiny).log2())).sum(-1)
          + (b * (b.clamp_min(tiny).log2() - middle.clamp_min(tiny).log2())).sum(-1)) / 2
    values = js.clamp(0, 1).tolist()
    topics = {}
    for probe, value in zip(probes, values):
        topics.setdefault(probe["topic"], []).append(value)
    return {"js_bits": sum(values) / len(values), "prompts": len(values),
            "topics": [{"name": k, "js_bits": sum(v) / len(v), "prompts": len(v)} for k, v in topics.items()]}


def read_probes(path):
    probes = json.loads(Path(path).read_text())
    if not isinstance(probes, list) or not probes or any(
        not isinstance(p, dict) or not isinstance(p.get("prompt"), str) or not p["prompt"].strip()
        or not isinstance(p.get("topic"), str) or not p["topic"].strip() for p in probes
    ):
        raise ValueError("Probes must be a nonempty list of {topic, prompt} objects")
    return probes


def measure(target, plan, probes, device="cpu", progress=print):
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi()
    # A mutable branch used as both one phase's end and the next phase's start
    # must resolve to the same checkpoint throughout this run.
    revisions = {}
    for stage in plan["stages"]:
        for point in stage["checkpoints"]:
            key = (point["repo"], point["revision"])
            if key not in revisions:
                revisions[key] = api.model_info(key[0], revision=key[1]).sha
    result = {"schema_version": 1, "target": target,
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "method": "aligned-parameter-rms-and-next-token-js-v1",
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": device, "dtype": "float32", "prompt_format": "raw text; no chat template",
              "probe_sha256": hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest(),
              "probes": probes, "stages": []}
    for stage in plan["stages"]:
        points = []
        first_weights = previous_weights = first_probs = None
        vocabulary = None
        total_path = 0.0
        for index, point in enumerate(stage["checkpoints"]):
            revision = revisions[(point["repo"], point["revision"])]
            progress(f"{stage['stage']}: {point['repo']} @ {point['revision']} ({revision[:8]})", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(point["repo"], revision=revision, trust_remote_code=False)
            encoded = [tokenizer(p["prompt"], return_tensors="pt", add_special_tokens=False) for p in probes]
            signature = (tokenizer.get_vocab(), tokenizer.all_special_ids,
                         [e["input_ids"].tolist() for e in encoded])
            if vocabulary is not None and signature != vocabulary:
                raise ValueError("Tokenizer changed across checkpoints; probe distributions are not comparable")
            vocabulary = signature
            model = AutoModelForCausalLM.from_pretrained(point["repo"], revision=revision,
                                                       torch_dtype=torch.float32, trust_remote_code=False)
            model.eval().to(device)
            distributions, examples = [], []
            with torch.inference_mode():
                for probe, tokens in zip(probes, encoded):
                    inputs = {k: v.to(device) for k, v in tokens.items()}
                    logits = model(**inputs).logits[0, -1].float()
                    distributions.append(logits.softmax(-1).cpu())
                    # Greedy examples illustrate outputs; they are not scored as accuracy.
                    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                    continuation = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    examples.append({"topic": probe["topic"], "prompt": probe["prompt"], "continuation": continuation})
            probs = torch.stack(distributions)
            # Parameters are never modified after inference; on CPU a detached
            # view keeps their storage alive without another full-model copy.
            weights = {n: p.detach().cpu() for n, p in model.named_parameters()}
            del model
            gc.collect()
            if index == 0:
                first_weights, first_probs = weights, probs
            net = parameter_distance(first_weights, weights)
            interval = parameter_distance(previous_weights, weights)["rms"] if previous_weights is not None else 0.0
            total_path += interval
            points.append({**point, "resolved_revision": revision, "net_rms": net["rms"],
                           "relative_l2": net["relative_l2"], "interval_rms": interval,
                           "observed_path_rms": total_path,
                           "behavior": probability_change(first_probs, probs, probes), "examples": examples})
            previous_weights = weights
        result["stages"].append({"stage": stage["stage"], "coverage": stage["coverage"],
                                 "lineage_source": stage["lineage_source"], "checkpoints": points,
                                 "net": net, "observed_path_rms": total_path,
                                 "behavior": points[-1]["behavior"],
                                 "path_coverage": "checkpoint_lower_bound"})
        del first_weights, previous_weights, weights
        gc.collect()
    return result
