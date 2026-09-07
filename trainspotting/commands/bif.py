"""`bif`: experimental standalone-text loss sensitivity on Pythia-70m."""

import re
import sys

from .. import bif, registry
from ..paths import RESULTS
from .common import _filename_part, _slug, _stamp, _write_json


def cmd_bif(args):
    """Experimental standalone-text sensitivity on the small supported checkpoint."""
    target = registry.resolve(args.target)
    model_id = args.model or target["hf_model"]
    if model_id != bif.SUPPORTED_MODEL:
        sys.exit(f"experimental bif supports only {bif.SUPPORTED_MODEL}; pass --model "
                 f"{bif.SUPPORTED_MODEL}. Other checkpoints and post-training objectives are deferred.")
    query = sys.stdin.read() if args.text == "-" else args.text
    if not query.strip():
        sys.exit("the query is empty")
    if args.match:
        try:
            re.compile(args.match)
        except re.error as e:
            sys.exit(f"--match {args.match!r} is not a valid regex: {e}")
    stages = [args.stage] if args.stage else None
    incomplete: dict[str, int] = {}
    cands, skipped = bif.candidates(
        args.target, target, stages=stages, match=args.match, limit=args.limit, seed=args.seed,
        incomplete=incomplete,
    )
    for stage, why in skipped.items():
        print(f"{stage}: not weighed — {why}", file=sys.stderr)
    for stage, n in incomplete.items():
        print(f"{stage}: {n} records skipped — the stored document is incomplete or excerpted", file=sys.stderr)
    if not cands:
        sys.exit("no candidate examples: nothing committed for this target that this layer can weigh")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("this command needs torch and transformers: pip install -e '.[bif]'")
    tokenizer = bif.load_tokenizer(model_id)
    if getattr(tokenizer, "chat_template", None):
        sys.exit("chat templates and post-training objectives are deferred in experimental bif")
    device = bif.pick_device(args.device)
    print(f"loading {model_id} on {device} ({args.dtype})", file=sys.stderr)
    model, tokenizer, revision = bif.load(model_id, device, args.dtype, tokenizer=tokenizer)
    if model_id != target["hf_model"]:
        print(
            f"note: weighing against {model_id}, not {args.target}'s own checkpoint"
            f" {target['hf_model']}; the result file records which",
            file=sys.stderr,
        )
    encoded, kept, dropped, unrenderable = [], [], 0, 0
    for c in cands:
        try:
            e = bif.encode(tokenizer, c["turns"], args.max_tokens)
        except bif.SlowTokenizer as e:
            sys.exit(str(e))
        except bif.Unrenderable:
            unrenderable += 1
            continue
        if e["fit_tokens"] == 0:
            dropped += 1
            continue
        kept.append(c)
        encoded.append(e)
    if dropped:
        print(f"{dropped} candidates have no fit tokens after encoding and were dropped", file=sys.stderr)
    if unrenderable:
        print(f"{unrenderable} candidates were dropped because {model_id}'s chat template cannot render "
              f"them turn by turn", file=sys.stderr)
    if not encoded:
        sys.exit("no candidate has any text the model was fit to")
    chat = bool(getattr(tokenizer, "chat_template", None))
    try:
        q = bif.encode(tokenizer, bif.query_candidate(query, args.prompt, chat=chat)["turns"], args.max_tokens)
    except bif.Unrenderable as e:
        sys.exit(f"{e}: the query cannot be rendered through {model_id}'s template")
    if q["fit_tokens"] == 0:
        sys.exit("the query has no tokens to score")
    localize = list(range(len(kept)))
    nbeta = args.nbeta if args.nbeta is not None else bif.default_nbeta(len(localize))
    settings = {
        "chains": args.chains,
        "draws": args.draws,
        "burn_in": args.burn_in,
        "every": args.every,
        "lr": args.lr,
        "nbeta": nbeta,
        "gamma": args.gamma,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": device,
        "seed": args.seed,
        "match": args.match,
        "limit": args.limit,
        "stage": args.stage,
    }
    print(
        f"{len(encoded)} candidates ({len(localize)} localized on), query {q['fit_tokens']} fit tokens; "
        f"{args.chains} chains × {args.burn_in + (args.draws - 1) * args.every + 1} total steps; "
        "experimental settings, convergence not established",
        file=sys.stderr,
    )
    run = bif.sample(
        model, encoded, q, device=device, chains=args.chains, draws=args.draws,
        burn_in=args.burn_in, every=args.every, lr=args.lr, nbeta=nbeta, gamma=args.gamma,
        batch=args.batch, eval_batch=args.eval_batch, seed=args.seed, localize=localize,
        log=lambda m: print(m, file=sys.stderr),
    )
    # Through `_filename_part` like every other explicit slug: a slash in it
    # would write a file `bif.committed` never globs, and the run would vanish
    # from the report.
    slug = _filename_part(args.slug) if args.slug else _slug(query)
    res = bif.result(args.target, model_id, revision, query, args.prompt, kept, encoded, run, skipped, settings)
    res = {"slug": slug, **_stamp(), "dropped": dropped, "unrenderable": unrenderable,
           "incomplete": incomplete, **res}
    path = _write_json(RESULTS / f"{args.target}.bif-{slug}.json", res)
    for line in bif.render(res):
        print(line)
    print(f"-> {path}", file=sys.stderr)
