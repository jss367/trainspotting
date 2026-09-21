"""Parse the command line and hand off. Every command body lives in `commands/`.

This module is argparse and nothing else: the type validators the parsers use,
the shared `target` help string, and one `main` that builds the subparsers and
calls the `cmd_*` function each one names. Tests that swap a command out
(`monkeypatch.setattr(cli, "cmd_facts", ...)`) rely on the names below being
module globals that `main` reads at call time, which is why they are imported by
name rather than reached through the package.
"""

import argparse
import math
import sys
from pathlib import Path

from . import benchmarks, casestudy, grep, infinigram, lookup, registry
from .commands.agreement import cmd_agreement
from .commands.bif import cmd_bif
from .commands.budget import cmd_budget
from .commands.changes import cmd_changes
from .commands.case_study import cmd_case_study
from .commands.contaminate import CONTAM_DEFAULTS, cmd_contaminate
from .commands.context import cmd_context
from .commands.facts import cmd_facts
from .commands.find import cmd_find
from .commands.grep import cmd_grep
from .commands.labeling import cmd_ask, cmd_classify
from .commands.languages import cmd_languages
from .commands.lookup import cmd_lookup
from .commands.pretrain import cmd_pretrain
from .commands.report import cmd_report
from .commands.search import cmd_search
from .commands.sources import cmd_sources
from .commands.stance import cmd_stance
from .commands.steps import cmd_steps
from .commands.trace import cmd_trace


# Every command takes one of these. A model walks its whole pipeline; a dataset
# is a single samplable dataset with no pipeline around it, and the layers that
# read rows cannot tell the difference (see registry.resolve).
TARGET_HELP = "model or dataset: " + ", ".join(registry.targets())

# Rows drawn per post-training stage by every layer that samples prompts. 300
# gave roughly ±5% on the common labels and swamped the rare ones — `honesty`
# and `tool_use` are a few percent of a stage, and at n=300 their intervals were
# wider than the point estimate. 1,000 prompts at 1,500 characters is under a
# million input tokens per stage, so cost was never the constraint. `pretrain`
# keeps its own default: a corpus document is up to 200k characters and the
# committed sample is what the site ships.
SAMPLE = 1000


def _count_int(value: str) -> int:
    """argparse type for a count where zero is a real choice — `lookup --docs 0`
    asks for counts without documents. A negative one is not: it reaches the
    sampler as a negative budget, skips retrieval, and reports "0 documents from
    0 draws" for a phrase with thousands of occurrences."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {n}")
    return n


def _docs_arg(value: str) -> int | str:
    """argparse type for `lookup --docs`: a count, or `all` to walk every
    occurrence by rank instead of sampling ten at a time. The word rather than
    a separate flag, because "how many documents" is the one question and
    "all of them" is one of its answers."""
    if value.strip().lower() == "all":
        return "all"
    return _count_int(value)


def _positive_int(value: str) -> int:
    """argparse type for counts. Zero divides by zero deep inside the sampler and
    a negative one silently returns nothing; both should be a usage error."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _probe_words(value: str) -> int:
    """argparse type for `contaminate --words`. A probe shorter than
    `benchmarks.MIN_WORDS` matches by chance — a one-word window is a common
    word, and a chance match reads as contamination — so the floor `probe()`
    applies to items applies to the window too."""
    n = _positive_int(value)
    if n < benchmarks.MIN_WORDS:
        raise argparse.ArgumentTypeError(
            f"a probe needs at least {benchmarks.MIN_WORDS} words, got {n}: a shorter "
            "window matches by chance, and a chance match reads as contamination"
        )
    return n


def _draws(value: str) -> int:
    """argparse type for `bif --draws`. One retained draw is a series of one
    observation: centering it gives every covariance, correlation and partial
    exactly zero, and the file written would rank the candidates in an arbitrary
    tie and look like a result."""
    n = _positive_int(value)
    if n < 2:
        raise argparse.ArgumentTypeError(
            f"a covariance needs at least 2 retained draws, got {n}"
        )
    return n


def _positive_float(value: str) -> float:
    """argparse type for the sampler's real-valued settings. A negative or
    non-finite step size reaches `math.sqrt` inside the sampler after the
    checkpoint has been loaded; a negative γ makes the prior repulsive and a
    negative nβ drives the chain up the loss, and either writes a result for a
    distribution that is not the documented posterior."""
    x = float(value)
    if not (x > 0 and math.isfinite(x)):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value}")
    return x


def _nonnegative_int(value: str) -> int:
    """argparse type for `--examples`. Zero is meaningful — count without keeping
    any evidence — but a negative limit reaches the example heap with nothing in
    it and raises IndexError, after the multi-gigabyte scan has already run."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {n}")
    return n


def main():
    ap = argparse.ArgumentParser(prog="trainspotting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("changes", help="measure weight and behavior changes between training checkpoints")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--plan", type=Path, help="JSON phase/checkpoint plan (Pythia has a default)")
    p.add_argument("--probes", type=Path, help="JSON list of fixed {topic, prompt} probes")
    p.add_argument("--device", default="cpu", help="inference device: cpu, mps, or cuda")
    p.set_defaults(fn=cmd_changes)

    for name, fn in [("facts", cmd_facts), ("sources", cmd_sources), ("report", cmd_report)]:
        p = sub.add_parser(name)
        p.add_argument("target", help=TARGET_HELP)
        p.set_defaults(fn=fn)
        if name == "sources":
            p.add_argument("--json", action="store_true", help="also write results/<target>.sources.json")

    p = sub.add_parser("ask", help="score sampled prompts against a free-form yes/no question")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("question")
    p.add_argument(
        "--stage",
        help="only this stage — a post-training one (sft/dpo/rlvr), or with --pretrain "
        "a corpus one (pretrain/midtrain/long-context)",
    )
    p.add_argument("--sample", type=_positive_int, default=SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.add_argument(
        "--pretrain",
        action="store_true",
        help="also score pretraining documents sampled by `trainspotting pretrain`",
    )
    p.add_argument(
        "--pretrain-only",
        action="store_true",
        help="score only the pretraining documents — for extending a question already "
        "answered over post-training without paying for those stages again",
    )
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser(
        "stance",
        help="judge which way each stored training example pushes on a question "
        "(toward / away / neither) — reads whole examples, not prompts",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("question")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr)")
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.set_defaults(fn=cmd_stance)

    p = sub.add_parser(
        "budget",
        help="roll an ask question up into a share of the training budget, in tokens",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("slug", help="the --slug of the ask runs to roll up")
    p.add_argument("--json", action="store_true", help="also write results/<target>.budget-<slug>.json")
    p.set_defaults(fn=cmd_budget)

    p = sub.add_parser(
        "find",
        help="exact occurrence count + example documents for a phrase, via infini-gram",
    )
    p.add_argument("phrase", help="the exact string to look up (matched on token boundaries)")
    p.add_argument(
        "--index",
        default=infinigram.DEFAULT_INDEX,
        help="infini-gram index to search; known: "
        + ", ".join(infinigram.INDEXES)
        + " (no Dolma 3 / OLMo 3 index exists publicly yet; other names are passed through)",
    )
    p.add_argument("--docs", type=_positive_int, default=5, help="example documents to retrieve")
    p.add_argument(
        "--maxlen",
        type=_positive_int,
        default=200,
        help="tokens of each document to display around the match",
    )
    p.add_argument("--json", action="store_true", help="also write results/find.<index>.<slug>.json")
    p.add_argument("--slug", help="short name for the result file (default: derived from the phrase)")
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser(
        "trace",
        help="find which stages a pasted behavior's distinctive phrases occur in",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("text", help="transcript or description to trace ('-' reads stdin)")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument(
        "--max-queries",
        type=_positive_int,
        default=6,
        help="most distinctive phrases to extract and search for",
    )
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser(
        "bif",
        help="experimental standalone-text sensitivity on Pythia-70m (needs the bif extra)",
        description="Experimental standalone-text loss sensitivity on Pythia-70m-deduped only. "
        "Other checkpoints and post-training objectives are deferred. Sampling diagnostics "
        "can withhold rankings; passing them does not validate historical training attribution.",
    )
    p.add_argument("target", help="registered target whose committed corpus documents define the sample")
    p.add_argument("text", help="query continuation whose loss is measured ('-' reads stdin)")
    p.add_argument("--prompt", help="what it was replying to, scored as context rather than as target")
    p.add_argument("--stage", help="only this stage's committed sample")
    p.add_argument("--match", help="keep only candidates whose text holds this regex (case-insensitive)")
    p.add_argument("--limit", type=_positive_int, help="at most this many candidates per stage, drawn at random")
    p.add_argument("--model", help="use EleutherAI/pythia-70m-deduped; other checkpoints are deferred")
    p.add_argument("--chains", type=_positive_int, default=4)
    p.add_argument("--draws", type=_draws, default=100, help="retained draws per chain (at least 2)")
    p.add_argument("--burn-in", type=_nonnegative_int, default=50, help="SGLD steps discarded per chain")
    p.add_argument("--every", type=_positive_int, default=1, help="SGLD steps between retained draws")
    p.add_argument("--lr", type=_positive_float, default=5e-8,
                   help="SGLD step size ε; experimental, requires sampling diagnostics and stability checks")
    p.add_argument("--nbeta", type=_positive_float,
                   help="inverse temperature nβ (default: n / ln n over the localized candidates)")
    p.add_argument("--gamma", type=_positive_float, default=100.0, help="localization strength γ")
    p.add_argument("--batch", type=_positive_int, default=8,
                   help="candidates per SGLD minibatch (a memory setting; it does not change the posterior)")
    p.add_argument("--eval-batch", type=_positive_int, default=16, help="examples per forward pass when recording losses")
    p.add_argument("--max-tokens", type=_positive_int, default=512, help="tokens kept per example (the front is dropped)")
    p.add_argument("--dtype", default="float32", choices=["float32"],
                   help="the small-model experiment uses float32")
    p.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slug", help="short name for the result file (default: derived from the text)")
    p.set_defaults(fn=cmd_bif)

    p = sub.add_parser("pretrain", help="sample documents from a model's pretraining corpora")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (pretrain/midtrain/long-context)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--docs-per-shard",
        type=_positive_int,
        default=1,
        help="documents kept per sampled shard; >1 is faster but the documents "
        "are correlated, which widens any interval computed over them",
    )
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser(
        "search",
        help="find a regex anywhere in the sampled examples — prompt, response, "
        "and for DPO which side of the pair",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="Python regex, case-insensitive unless --case-sensitive")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--slug", help="short name for the result files (default: derived from the pattern)")
    p.add_argument("--show", type=int, default=3, help="matching rows to print (0 for none)")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("context", help="store the full training example behind each sampled prompt")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("languages", help="detect the natural language of sampled prompts (local, no API key)")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-labels", action="store_true",
                   help="read prompts from the committed classify run instead of re-sampling HuggingFace")
    p.set_defaults(fn=cmd_languages)

    p = sub.add_parser("lookup", help="count an exact string in the public corpora that have an index")
    p.add_argument("query", help="the exact string to look for")
    p.add_argument(
        "--index",
        action="append",
        help=f"restrict to one index (repeatable); default all of: {', '.join(i['id'] for i in lookup.INDEXES)}",
    )
    p.add_argument(
        "--docs",
        type=_docs_arg,
        default=0,
        metavar="N|all",
        help="also pull up to N documents behind each count (the index caps a single call at 10), "
        "or `all` to fetch every occurrence one request at a time",
    )
    p.set_defaults(fn=cmd_lookup)

    p = sub.add_parser("case-study", help="run a committed lookup study and write its result file")
    p.add_argument("slug", nargs="?", default="marginal-revolution",
                   help=f"one of: {', '.join(casestudy.CASE_STUDIES)}")
    p.set_defaults(fn=cmd_case_study)

    p = sub.add_parser("grep", help="exact string search over every row of each post-training mix")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="literal substring, or a regex with --regex")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument(
        "--field",
        action="append",
        choices=list(grep.GROUPS),
        help="which part of the example to search; repeatable, default all three",
    )
    p.add_argument(
        "--by",
        help="column to break the counts down by; default is the registry's "
        "source_columns, so the breakdown matches `sources`",
    )
    p.add_argument("--regex", action="store_true", help="treat the pattern as an RE2 regex")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep in the result file; 0 counts without keeping any")
    p.add_argument(
        "--max-gb",
        type=float,
        default=5.0,
        help="refuse to read more than this over the network; the plan is printed either way",
    )
    p.add_argument("--yes", action="store_true", help="scan whatever it costs")
    p.add_argument("--slug", help="short name for the result files (default: derived from the pattern)")
    p.set_defaults(fn=cmd_grep)

    p = sub.add_parser(
        "contaminate",
        help="is a benchmark's test set in the training data — which stage, which side",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("benchmark", help="one of: " + ", ".join(sorted(benchmarks.BENCHMARKS)))
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--items", type=_positive_int, default=CONTAM_DEFAULTS["items"],
                   help="test items to probe; a seeded draw when the set is larger")
    p.add_argument("--seed", type=int, default=CONTAM_DEFAULTS["seed"])
    p.add_argument("--words", type=_probe_words, default=CONTAM_DEFAULTS["words"],
                   help="consecutive words per probe, cut from the middle of the item; "
                        f"at least {benchmarks.MIN_WORDS}")
    p.add_argument(
        "--field",
        action="append",
        choices=list(grep.GROUPS),
        help="which part of the example to search; repeatable, default every side",
    )
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep per stage; 0 counts without keeping any")
    p.add_argument("--max-gb", type=float, default=5.0,
                   help="refuse to read more than this over the network; the plan is printed either way")
    p.add_argument("--yes", action="store_true", help="scan whatever it costs")
    p.add_argument("--index", help="infini-gram index for the corpus side (default: the "
                   "registry's closest index for the model; a dataset has no corpus side "
                   "and takes none)")
    # Together these would fetch the benchmark, read nothing, and exit 0 with a
    # summary of stages not scanned — a run that asked no question.
    side = p.add_mutually_exclusive_group()
    side.add_argument("--no-corpus", action="store_true", help="skip the corpus side")
    side.add_argument("--corpus-only", action="store_true", help="skip the post-training scans")
    p.add_argument("--slug", help="short name for the result files (default: the benchmark, "
                   "with a hash of --items/--seed/--words/--field/--case-sensitive when any "
                   "is not at its default)")
    p.set_defaults(fn=cmd_contaminate)

    p = sub.add_parser(
        "steps",
        help="count a string in sampled training batches, in the order the model saw them (Pythia)",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="literal substring, or a regex with --regex")
    p.add_argument(
        "--sample",
        type=_positive_int,
        default=64,
        help="training steps to read, one from each equal slice of the run; 4.2 MB each",
    )
    p.add_argument("--at", type=_nonnegative_int, action="append", help="also read this exact step; repeatable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slices", type=_positive_int, default=8, help="stretches of the run to report the rate over")
    p.add_argument("--regex", action="store_true", help="treat the pattern as a Python regex")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep in the result file; 0 counts without keeping any")
    p.add_argument("--slug", help="short name for the result file (default: derived from the pattern)")
    p.set_defaults(fn=cmd_steps)

    p = sub.add_parser("classify")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=SAMPLE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument(
        "--replicate",
        action="store_true",
        help="label the same draw again and write <target>.<stage>.labels-replicate.json, "
        "for `agreement` to compare against the main run",
    )
    p.set_defaults(fn=cmd_classify)

    p = sub.add_parser(
        "agreement",
        help="compare a `classify --replicate` run with the main labels run: agreement, kappa, share drift",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.set_defaults(fn=cmd_agreement)

    args = ap.parse_args()
    # Canonicalize once, here, so every result path and every lookup downstream
    # agrees. `resolve` accepts case variants; writing the raw argument into the
    # filename meant `classify WildChat-1M` produced a file the site — which
    # indexes the registry key — never asks for, and the run silently didn't
    # exist.
    # Only the commands that take one: `find`, `lookup` and `case-study` are
    # about corpora rather than a registered target, and canonicalizing an
    # argument they never parsed raised AttributeError before their handler ever
    # ran. `find` has been unusable on main since this canonicalization landed.
    if getattr(args, "target", None) is not None:
        try:
            args.target = registry.resolve(args.target)["target"]
        except KeyError as e:
            sys.exit(e.args[0])
    args.fn(args)


if __name__ == "__main__":
    main()
