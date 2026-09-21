"""`report`: everything committed about a target, on one page of text."""

import json

from .. import bif, budget, classify, influence, languages, registry, paths
from ..paths import RESULTS
from ..stats import wilson as _wilson
from .budget import BUDGET_COLS, _budget_row, _share_phrase, _warn_mixed_questions
from .common import _counts, _fmt_est, _fmt_tokens


def _grep_traces(model_name, model):
    """Committed `grep` runs for one model, grouped by the search they ran.

    The slug is where the stages of one sweep line up, and it has to be, because
    a pattern too long to name itself is stored under a `--slug`. But a slug is a
    filename rather than a promise: rerun one stage with a refined regex under
    the same slug and the directory holds two different searches with one name.
    So the group key is the slug *and* the search definition, and a slug that
    turns out to name more than one gets each rendered separately rather than
    ranked against each other under whichever pattern sorted first.
    """
    groups = {}
    for path in sorted(RESULTS.glob(f"{model_name}.*.grep-*.json")):
        slug = path.name.split(".grep-", 1)[1][: -len(".json")]
        run = json.loads(path.read_text())
        # The filename wins over the recorded slug, which is only a note of what
        # was passed. Grouping keys on the filename and `--slug` is what decides
        # the filename, so a rerun needs the name to land back in this group —
        # and after a collision rename the payload still carries the contested
        # slug it was moved away from.
        run["slug"] = slug
        key = (slug, run.get("pattern"), bool(run.get("regex")), bool(run.get("case_sensitive")))
        groups.setdefault(key, []).append(run)
    # Collision is a property of one slug, not of the directory: marking every
    # trace because some other slug is contested would strip a valid `--slug`
    # from commands that need it to land in their own group.
    per_slug = {}
    for slug, *_ in groups:
        per_slug[slug] = per_slug.get(slug, 0) + 1
    taken = {slug for slug, *_ in groups}

    out = []
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        slug = key[0]
        trace = influence.compare(runs, model["stages"])
        trace["slug_collides"] = per_slug[slug] > 1
        if trace["slug_collides"]:
            # Dropping `--slug` is not enough: two searches differing only in
            # `--regex` or `--case-sensitive` share a pattern, so `grep` would
            # derive the same filename for both. Hand out a free one instead,
            # skipping any slug already on disk.
            n = 1
            while f"{slug}-{n}" in taken:
                n += 1
            trace["slug_suggest"] = f"{slug}-{n}"
            taken.add(trace["slug_suggest"])
        out.append((slug, trace["slug_collides"], trace))
    return out


def cmd_report(args):
    target = registry.resolve(args.target)
    kind = "Training-data audit" if target["is_model"] else "Dataset audit"
    print(f"# {kind}: {args.target}\n")
    if target["is_model"]:
        print("## Stage sizes\n")
        for s in target["stages"]:
            if s.get("tokens"):
                print(f"- {s['stage']}: {s['name']}, {_fmt_tokens(s['tokens'])} tokens")
            else:
                print(f"- {s['stage']}: {s['name']} ({s['hf_dataset']})")
    elif target.get("note"):
        print(target["note"])
    if not registry.post_training_stages(target):
        # Pythia is the case: a base model with no post-training at all. Both
        # prompt sections below would print an empty heading each, which says
        # "we have not run this yet" — a very different claim from "this model
        # has no such stage to run it on". Say the latter once and skip them.
        print(
            f"\n{target['hf_model'] or target['name']} has no post-training"
            " stages — it was released as a base model, so there are no prompts"
            " to classify, no responses to classify them against, and no"
            " language to detect. `trainspotting pretrain` and `ask --pretrain`"
            " are what apply here."
        )
        # Not a stop, though: the corpus layers are what apply, and they are all
        # downstream of here. `ask --pretrain` on this target produces a rate per
        # corpus stage and a training-budget rollup over them, so returning at
        # this point would drop the one audit layer a base model *can* support
        # from the very report that just recommended running it.
        _report_influence(args.target)
        _report_questions(args.target, target)
        return
    # The same seven labels mean different things by kind, and the heading is
    # the only place the report says which.
    print(
        "\n## HHH classification (sampled)\n"
        if target["is_model"]
        else "\n## What the prompts ask for (sampled)\n"
    )
    for s in registry.post_training_stages(target):
        path = RESULTS / f"{args.target}.{s['stage']}.labels.json"
        if not path.exists():
            print(f"- {s['stage']}: no classification run yet (`trainspotting classify {args.target} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = [r for r in data["records"] if r["label"]]
        n = len(records)
        # Verifier-labeled rows never reached the classifier, so name both
        # counts rather than crediting the model for all of them.
        v = sum(1 for r in records if r.get("by") == "verifier")
        by = f", {v} by their verifier" if v else ""
        print(f"### {s['stage']} — {data['dataset']} (n={n} labeled{by})\n")
        # Every share below is over the labeled prompts. Refusals fall on
        # jailbreak-style content, so an unreported gap reads as a smaller
        # harmlessness share rather than as missing data.
        unlabeled = data.get("unlabeled", len(data["records"]) - n)
        if unlabeled:
            reasons = data.get("unlabeled_reasons") or {}
            detail = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
            print(
                f"- {unlabeled} of {n + unlabeled} sampled prompts went unlabeled"
                + (f" ({detail})" if detail else "")
                + " — excluded from every share below\n"
            )
        counts = _counts(records)
        for label in classify.LABELS:
            k = counts.get(label, 0)
            lo, hi = _wilson(k, n)
            print(f"- {label:22s} {k / n * 100 if n else 0:5.1f}%  (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
        print()

    print("\n## Language (sampled, detected locally)\n")
    for s in registry.post_training_stages(target):
        path = RESULTS / f"{args.target}.{s['stage']}.languages.json"
        if not path.exists():
            print(f"- {s['stage']}: no language run yet (`trainspotting languages {args.target} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = data["records"]
        n = len(records)
        counts = _counts(records)
        non_en = n - counts.get("en", 0) - counts.get(languages.UNDETERMINED, 0)
        lo, hi = _wilson(non_en, n)
        print(f"### {s['stage']} — {data['dataset']} (n={n} detected)\n")
        print(f"- not English: {non_en / n * 100 if n else 0:.1f}%  (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
        for code, k in sorted(counts.items(), key=lambda kv: -kv[1]):
            if code in ("en", languages.UNDETERMINED):
                continue
            print(f"  - {languages.name(code):20s} {k / n * 100:5.1f}%  ({k}/{n})")
        und = counts.get(languages.UNDETERMINED, 0)
        if und:
            print(f"- undetermined: {und / n * 100:.1f}%  ({und}/{n}) — too short, too much code, or too evenly mixed to call")
        print()

    _report_pairs(args.target, target)

    # String traces before the budget: main puts the budget last on purpose,
    # because the rates above it are per stage and not comparable to each other.
    if target["is_model"]:
        traces = _grep_traces(args.target, target)
        print("\n## String traces\n")
        if not traces:
            print(f"- no `grep` run yet (`trainspotting grep {args.target} \"some string\"`)")
        else:
            # Only true of stages the server converted in full, so it is said of
            # those rather than of the section: a partial conversion's
            # `total_rows` is the converted subset, and claiming otherwise here
            # would contradict the lower-bound warning printed under the stage.
            partial = sorted({f"{r['stage']} ({t['pattern']!r})"
                              for _, _, t in traces for r in t["stages"] if r["partial"]})
            print("Every count below is over all rows of the stage named, not a sample, so a "
                  "zero is the string being absent rather than merely unlikely — and a stage "
                  "listed as unsearched or inconclusive is neither.\n")
            if partial:
                print("Except where noted per stage: the datasets-server converted only part "
                      "of " + ", ".join(partial) + ", so those counts and denominators cover "
                      "the converted subset alone.\n")
            print(influence.BASIS_NOTE + "\n")
            if any(split for _, split, _ in traces):
                print("One slug below names more than one search — a pattern or a matching "
                      "flag was changed without changing the slug. Each is rendered on its "
                      "own; the stages under one heading are the stages that ran that exact "
                      "search.\n")
            for _, _, trace in traces:
                for line in influence.render(trace, args.target):
                    print(line)

    _report_influence(args.target)
    _report_questions(args.target, target)


def _report_influence(target_name: str) -> None:
    """Render experimental runs with diagnostics recomputed from saved draws."""
    runs = bif.committed(target_name)
    if not runs:
        return
    print("\n## Experimental loss sensitivity\n")
    print("Standalone-text sensitivity under a local sampling objective. "
          "This does not identify the historical cause of a model behavior.\n")
    for res in runs:
        for line in bif.render(res):
            print(line)
        print()


def _report_pairs(target_name: str, target: dict) -> None:
    """What separates the two sides of each preference pair, besides the answer.

    Printed under the sampled layers rather than beside the budget, because it
    is a property of the mix and not a share of training: it says what a policy
    could fit without reading a response, not what any policy did fit.
    """
    stages = [s for s in registry.post_training_stages(target) if registry.stage_kind(s) == "dpo"]
    if not stages:
        return
    print("\n## Preference pairs (sampled)\n")
    for s in stages:
        path = paths.find(f"{target_name}.{s['stage']}.pairs.json")
        if not path:
            print(f"- {s['stage']}: no pairs run yet (`trainspotting pairs {target_name} --stage {s['stage']}`)")
            continue
        d = json.loads(path.read_text())
        rule, delta = d["length_rule"], d["delta"]
        print(f"### {s['stage']} — {d['dataset']} (n={d['n']} pairs)\n")
        # A stage where every sampled pair ties has no pair the length rule can
        # answer, so `pairs._rate([])` carries back a denominator of zero and no
        # rate at all — reading `rate` here raised `KeyError` and took the whole
        # report down with it. The rule is undefined there rather than 0%: 0%
        # would say length never picks the chosen side, when what is true is that
        # length picks neither. `pairs._fmt_rate` says the same thing on the
        # command's own output.
        if not rule["n"]:
            print(
                f"- longer side chosen: undefined — all {d['n']} sampled pairs tie on length,"
                " so the rule has no pair to be right or wrong about"
            )
        else:
            print(
                f"- longer side chosen: {rule['rate'] * 100:.1f}%  ({rule['k']}/{rule['n']},"
                f" 95% CI {rule['lo'] * 100:.1f}–{rule['hi'] * 100:.1f}%)"
                + (f" — {d['ties']} pairs tie on length and are not counted" if d["ties"] else "")
            )
        print(f"- chosen − rejected: mean {delta['mean']:+,.0f} characters, median {delta['median']:+,.0f}")
        if d.get("split"):
            print(
                f"  - of which reasoning {d['split']['reasoning']['mean']:+,.0f}"
                f" and answer {d['split']['answer']['mean']:+,.0f}"
            )
        rules = d["models"]["rule"]
        if rules:
            # "Fits", not "predicts": the rule is read off the same rows it is
            # scored on, so it is a ceiling for this sample and not an estimate
            # for any other.
            print(
                f"- generator names alone fit the choice in {rules['rate'] * 100:.1f}%"
                f"  ({rules['k']}/{rules['n']} over {rules['matchups']} matchup(s), fitted in-sample)"
            )
            for m in d["models"]["matchups"][:3]:
                print(f"  - {m['chosen']} over {m['rejected']}: {m['n']} pairs")
        if d["degenerate"]:
            print(
                f"- {d['degenerate']} pairs have no gradient-bearing text on either side —"
                " the two completions are identical, so the DPO loss cancels"
            )
        if d["truncated_rows"]:
            print(
                f"- {d['truncated_rows']} rows arrived with a read column shortened upstream,"
                " so their lengths are lower bounds"
            )
        elif d["truncated_rows"] is None:
            print("- upstream truncation was not recorded for this sample, so the lengths are unchecked")
        print()


def _report_questions(target_name: str, target: dict) -> None:
    """The free-form layers: what was asked, which way it pushes, what it costs
    as a share of training.

    Ordered so the last thing a reader sees is the budget. The rates above it
    are per stage and not comparable to each other — that is the whole reason
    the budget table exists — so leading with them and stopping would leave the
    report saying "6% of DPO prompts" as if it answered how much training the
    model got.
    """
    asks = paths.runs(target_name, "ask")
    stances = paths.runs(target_name, "stance")
    corpus_names = {x["stage"] for x in registry.pretrain_stages(target)}
    if not asks and not stances:
        return
    order = [s["stage"] for s in target["stages"]]

    if asks:
        print("\n## Custom questions (sampled)\n")
        for slug, stages in asks.items():
            data = {st: budget.load(f"{target_name}.{st}.ask-{slug}.json") for st in stages}
            print(f"### {slug}\n")
            # Grouped by the wording each run actually stored, not by slug. A
            # slug is not a question — see `_warn_mixed_questions` — and
            # printing one question over every stage's rate attributes the
            # others' measurements to words they were never scored against.
            # Keyed on the classifier as well as the wording, as the site's ask
            # cards are: the same words put to two judges are two measurements,
            # and a block that lists both rates under one heading names neither.
            groups: dict[tuple, list[str]] = {}
            for st in sorted(stages, key=lambda x: order.index(x) if x in order else 99):
                if data[st]:
                    groups.setdefault((data[st]["question"], data[st].get("classifier")), []).append(st)
            for (question, classifier), group in groups.items():
                print(f"> {question}\n")
                if classifier:
                    print(f"judged by {classifier}\n")
                for st in group:
                    d = data[st]
                    records = d["records"]
                    k, n = sum(bool(r["match"]) for r in records), len(records)
                    # A corpus run stores its own cluster-corrected interval; the
                    # binomial one would be too narrow for documents drawn by shard.
                    lo, hi = d["ci"] if d.get("ci") else _wilson(k, n)
                    print(
                        f"- {st:14s} {k / n * 100 if n else 0:5.1f}%  ({k}/{n},"
                        f" 95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"
                    )
                print()
            if len(groups) > 1:
                differ = "wording" if len({q for q, _ in groups}) > 1 else "classifier"
                if len({q for q, _ in groups}) > 1 and len({c for _, c in groups}) > 1:
                    differ = "wording and classifier"
                print(
                    f"({len(groups)} instruments share the slug {slug!r}, differing by"
                    f" {differ}; the rates above are grouped under the one each stage was"
                    " actually scored by.)\n"
                )
            # The rubric is named rather than used as a grouping key, because a
            # corpus stage and a post-training stage are scored under different
            # rubrics on every `--pretrain` run by design — keying on it would
            # split every such block in two. What is worth saying is a rubric
            # that moved between stages judged the same way, which is the same
            # rule `budget.mixing` applies.
            for question, group in groups.items():
                by_family: dict[str, dict[str, list[str]]] = {}
                for st in group:
                    sha = data[st].get("system_sha")
                    if sha:
                        fam = "pretrain" if st in corpus_names else "post-training"
                        by_family.setdefault(fam, {}).setdefault(sha, []).append(st)
                for fam, shas in by_family.items():
                    if len(shas) > 1:
                        detail = "; ".join(
                            f"{', '.join(sts)} under {sha[:12]}" for sha, sts in shas.items()
                        )
                        print(
                            f"(the {fam} stages above were scored under"
                            f" {len(shas)} different rubrics — {detail} — so their rates"
                            " are not directly comparable.)\n"
                        )

    if stances:
        print("\n## Which way each example pushes (whole examples, sampled)\n")
        for slug, stages in stances.items():
            print(f"### {slug}\n")
            # Grouped by the instrument each run recorded, the same way the ask
            # section and the site's stance cards are. Printing every stage
            # under the slug alone lets two nets scored against different words,
            # or by different judges, read as one answer — and a net is signed,
            # so averaging incompatible ones by eye is worse than a rate.
            # Keyed on the rubric as well. `stance.SYSTEM` is the instrument
            # here in the way the question is — it is what tells the judge that
            # a DISPREFERRED completion means the model is trained *out* of that
            # text — so rewording it moves toward/away labels while the question
            # and the classifier stay identical. Unlike the budget's
            # family-scoped check, every stance run is one family, so the hash
            # can go straight into the key.
            groups: dict[tuple, list[tuple[str, dict]]] = {}
            for st in sorted(stages, key=lambda x: order.index(x) if x in order else 99):
                d = budget.load(f"{target_name}.{st}.stance-{slug}.json")
                if d:
                    key = (d["question"], d.get("classifier"), d.get("system_sha"))
                    groups.setdefault(key, []).append((st, d))
            shas = {k[2] for k in groups}
            for (question, classifier, sha), group in groups.items():
                print(f"> {question}\n")
                by = f"judged by {classifier}" if classifier else ""
                # Only worth printing when it is what separates two groups —
                # otherwise it is a hash on every report for no reason.
                if sha and len(shas) > 1:
                    by = (by + " " if by else "") + f"under rubric {sha[:12]}"
                if by:
                    print(f"{by}\n")
                for st, d in group:
                    c = d["counts"]
                    n = len(d["records"])
                    if not n:
                        print(f"- {st:14s} every sampled example went unjudged — no direction to show")
                        continue
                    lo, hi = _wilson(c["toward"], n)
                    print(
                        f"- {st:14s} toward {c['toward']}/{n} = {c['toward'] / n * 100:.1f}%"
                        f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%), away {c['away']},"
                        f" net {d['net']:+d}"
                    )
                print()
            if len(groups) > 1:
                why = "wording, classifier or rubric" if len(shas) > 1 else "wording or classifier"
                print(
                    f"({len(groups)} instruments share the slug {slug!r}, differing by"
                    f" {why}; the nets above are grouped under the one that produced them"
                    " and do not combine.)\n"
                )

    for slug in asks:
        est = budget.estimate(target_name, slug)
        if not any(s.get("measured") for s in est["stages"]):
            continue
        print(f"\n## Training budget — {slug}\n")
        mixed = _warn_mixed_questions(est)
        print(BUDGET_COLS)
        print("-" * len(BUDGET_COLS))
        for st in est["stages"]:
            print(_budget_row(st))
        t = est["totals"]["all"]
        # A measured stage that could not be sized is dropped from both the
        # denominator and the matching sum, so the share is as partial as an
        # unasked one — `unsized` has to count here too.
        print(
            f"\nwhole pipeline: {_fmt_est(t['size_tokens'])} fit tokens — no single"
            " total, see the warning above"
            if mixed
            else f"\nwhole pipeline: {_fmt_est(t['matching_tokens'])} of"
            f" {_fmt_est(t['size_tokens'])} fit tokens ({_share_phrase(t)})"
        )
        # "Never asked" and "asked, and nothing usable came back" need different
        # advice: `--pretrain-only` does not re-run a failed post-training stage,
        # and telling someone to ask a question that already ran and failed sends
        # them in a circle. The rows above already print each stage's reason.
        unasked = [x for x in est["stages"] if not x.get("measured") and not x.get("unusable")]
        unusable = [x for x in est["stages"] if x.get("unusable")]
        if unasked:
            corpora = [x["stage"] for x in unasked if x["family"] == "pretrain"]
            how = (
                " --pretrain-only` to close the gap" if corpora and len(corpora) == len(unasked)
                else "` for the stages below"
            )
            print(
                f"  {len(unasked)} of {t['stages']} stages were never asked this question"
                f" ({', '.join(x['stage'] for x in unasked)}) — run"
                f" `trainspotting ask {target_name} \"...\" --slug {slug}{how}"
            )
        if unusable:
            print(
                f"  {len(unusable)} stage(s) were asked and produced nothing usable"
                f" ({', '.join(x['stage'] for x in unusable)}) — see the reason on each"
                " row above; re-asking without fixing that will fail the same way"
            )
        if t["unsized"]:
            print(
                f"  {', '.join(t['unsized'])} could not be sized, so"
                " the share above is over the stages that could be"
            )
        print()
