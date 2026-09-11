"""`facts`: the registry's stage table for a target."""

from .. import hf, registry
from .common import _fmt_tokens


def cmd_facts(args):
    target = registry.resolve(args.target)
    print(f"# {args.target} ({target['hf_model'] or target['name']})\n")
    for s in target["stages"]:
        line = f"- {s['stage']:12s} {s['name']}"
        if s.get("tokens"):
            line += f" — {_fmt_tokens(s['tokens'])} tokens"
        if s.get("hf_dataset"):
            n = hf.num_rows(s["hf_dataset"])
            line += f" — {n:,} examples ({s['hf_dataset']})"
        elif s.get("sample_dataset"):
            route = registry.sample_route(s)
            how = "by shard" if route == "shards" else "in full, uniformly"
            line += f" — samplable {how} ({s['sample_dataset']})"
        print(line)
        if s.get("note"):
            print(f"    {s['note']}")
