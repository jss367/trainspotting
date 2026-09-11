"""`languages`: detect the natural language of sampled prompts, locally."""

import json
import sys

from .. import extract, hf, languages, registry
from ..paths import RESULTS
from .common import _counts, _sample_prompts, _select_stages, _stamp, _write_json


def cmd_languages(args):
    """Detect which natural language each sampled prompt is written in.

    Sampling is deterministic in (sample, seed), so the defaults pull exactly
    the rows a classify run labeled and the drill-down lines up. Detection is
    local — no API key, no cost — because the Dolci datasets carry no language
    column of their own and the only label that comes close is Instruct-SFT's
    single `Multilingual` domain bucket.
    """
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        sample, seed = args.sample, args.seed
        if args.from_labels:
            # The same (sample, seed) draws the same rows, so a committed classify
            # run already holds the prompts verbatim — no reason to re-fetch them.
            labels_path = RESULTS / f"{args.target}.{s['stage']}.labels.json"
            if not labels_path.exists():
                sys.exit(f"{labels_path} not found — drop --from-labels to sample from HuggingFace")
            prior = json.loads(labels_path.read_text())
            # A classify run written before result records carried their row
            # index has no row to reuse; those records keep joining to their
            # context by prompt text, as they did before.
            pairs = [(r.get("row"), r["prompt"]) for r in prior["records"]]
            sample, seed = prior["sample"], prior["seed"]
            # Reusing a run's prompts means reusing its revision: those rows came
            # from that tree. A file written before the field existed has none,
            # and null is the honest answer — not today's `main`, which is a
            # different tree from the one nobody recorded.
            revision = prior.get("revision")
            # And its ambiguity, if it had any. These are that run's rows, so a
            # language file that named only the first tree would state as settled
            # what the classification run recorded as unresolved.
            moved = prior.get("revision_moved_to")
            print(f"reusing {len(pairs)} prompts from {labels_path.name}", file=sys.stderr)
        else:
            revision = hf.dataset_revision(s["hf_dataset"])
            # Clip before detecting, not after. A classify run stores the clipped
            # prompt, so detecting the full text here would make --from-labels
            # disagree with this path on the handful of prompts past the cutoff.
            pairs = [
                (i, extract.clip(p))
                for i, p in _sample_prompts(s, args.sample, args.seed)
            ]
            # Detection is local, but the draw feeding it is thirty paged
            # requests, so this path has the same republish window as `context`.
            moved = hf.dataset_revision(s["hf_dataset"])
        records = []
        for i, p in pairs:
            code, conf = languages.detect(p)
            rec = {"prompt": p, "label": code, "confidence": round(conf, 3)}
            if i is not None:
                rec = {"row": i, **rec}
            records.append(rec)
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.languages.json",
            {
                "dataset": s["hf_dataset"],
                **_stamp(s["hf_dataset"], revision=revision),
                # From a reused run, this is the ambiguity it recorded; from a
                # fresh draw, movement observed across that draw. Either way it
                # takes two known and different readings: a first lookup that
                # failed leaves the tree unknown, and a SHA read afterwards is
                # the first answer we got, not evidence of a republish.
                **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
                "sample": sample,
                "seed": seed,
                "detector": "py3langid",
                "records": records,
            },
        )
        counts = _counts(records)
        n = len(records)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        summary = ", ".join(f"{languages.name(c)} {k / n * 100:.1f}%" for c, k in top)
        print(f"{s['stage']}: n={n} — {summary} -> {path}", file=sys.stderr)
