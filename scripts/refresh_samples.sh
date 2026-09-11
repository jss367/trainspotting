#!/usr/bin/env bash
#
# Re-draw every sampled layer for a target at the current default sample size,
# in the order the joins require, then export the site.
#
# The layers join on row index within one draw: `context` stores the examples
# the site drills into, `classify` and `ask` and `languages` label rows drawn
# with the same --sample and --seed, and the site pairs a label with its stored
# example by that index. A labels file drawn at 1,000 rows next to a context
# file drawn at 300 pairs a third of its rows and the site says so on the card.
# So a change of sample size is a change to every layer of a stage at once,
# which is what this script is: one command for a consistent set.
#
# Usage:
#   scripts/refresh_samples.sh <target> [phase]
#
#   phase   all (default) | free | labels | asks | export
#           free    context + languages, no API key needed
#           labels  classify, plus existing replicate/agreement checks
#                   (needs ANTHROPIC_API_KEY; one extra judging run per replicate)
#           asks    re-run existing ask/stance questions in their saved stages
#                   (needs ANTHROPIC_API_KEY); corpus asks reuse stored documents
#           export  scripts/export_site_data.py
#
# Reruns overwrite the previous files. Commit the old ones first if you want the
# n=300 runs kept for comparison; `trainspotting agreement` can compare a
# --replicate run to the main one but does not read git history.

set -euo pipefail

TARGET="${1:?usage: refresh_samples.sh <target> [all|free|labels|asks|export]}"
PHASE="${2:-all}"
case "$PHASE" in
  all|free|labels|asks|export) ;;
  *) printf 'unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
# Resolve once so every phase discovers and writes the same canonical files.
selection=$(python3 - "$TARGET" <<'PYTHON'
import sys
from trainspotting import registry
target = registry.resolve(sys.argv[1])
print(target["target"], *(s["stage"] for s in registry.post_training_stages(target)))
PYTHON
)
read -r TARGET label_stages <<< "$selection"
TS=(python3 -m trainspotting.cli)
step() { printf '\n=== %s ===\n' "$1" >&2; }

# Match exact artifacts, preferring the working result to its exported copy.
saved_run() {
  local f
  for f in results/"$TARGET"."$1"."$2".json docs/data/"$TARGET"."$1"."$2".json; do
    if [[ -f "$f" ]]; then printf '%s\n' "$f"; return; fi
  done
}
classifier_in() {
  python3 - "$@" <<'PYTHON'
import json, sys
run = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    run = run.get(key) or {}
print(run.get("classifier") or "")
PYTHON
}

if [[ "$PHASE" == all || "$PHASE" == free ]]; then
  step "context $TARGET (the examples the site drills into)"
  "${TS[@]}" context "$TARGET"
  step "languages $TARGET"
  "${TS[@]}" languages "$TARGET"
fi

if [[ "$PHASE" == all || "$PHASE" == labels ]]; then
  # Visit every registered prompt stage, including stages without a prior run.
  if [[ -z "$label_stages" ]]; then
    printf '%s has no post-training stages\n' "$TARGET" >&2
    exit 1
  fi
  for stage in $label_stages; do
    # Capture both instruments before overwriting either run. Agreement alone
    # can retain them when the raw runs are missing from this checkout.
    labels_file=$(saved_run "$stage" labels)
    replicate_file=$(saved_run "$stage" labels-replicate)
    agreement_file=$(saved_run "$stage" agreement)
    classifier=""
    if [[ -n "$labels_file" ]]; then
      classifier=$(classifier_in "$labels_file")
    elif [[ -n "$agreement_file" ]]; then
      classifier=$(classifier_in "$agreement_file" labels_run)
    fi
    replicate_classifier=""
    if [[ -n "$replicate_file" ]]; then
      replicate_classifier=$(classifier_in "$replicate_file")
    elif [[ -n "$agreement_file" ]]; then
      replicate_classifier=$(classifier_in "$agreement_file" replicate)
    fi
    # With no recorded replicate instrument, repeat the main one. Preserve an
    # explicit different classifier; agreement flags it as a comparison instead
    # of claiming repeatability.
    replicate_classifier="${replicate_classifier:-$classifier}"
    args=(classify "$TARGET" --stage "$stage")
    # A new stage or a legacy file without a classifier uses the CLI default.
    if [[ -n "$classifier" ]]; then args+=(--classifier "$classifier"); fi
    step "classify $TARGET $stage (needs ANTHROPIC_API_KEY)"
    "${TS[@]}" "${args[@]}"
    if [[ -n "$replicate_file" || -n "$agreement_file" ]]; then
      args=(classify "$TARGET" --stage "$stage" --replicate)
      if [[ -n "$replicate_classifier" ]]; then args+=(--classifier "$replicate_classifier"); fi
      step "classify $TARGET $stage --replicate (additional judging run)"
      "${TS[@]}" "${args[@]}"
      step "agreement $TARGET $stage"
      "${TS[@]}" agreement "$TARGET" --stage "$stage"
    fi
  done
fi

if [[ "$PHASE" == all || "$PHASE" == asks ]]; then
  # Every question already asked of this target, by slug, with the wording the
  # committed file recorded. Rewording a question is a different measurement,
  # so the text and classifier are read back out of the result. Legacy files
  # without a classifier use the CLI default.
  # One run per stage and slug, however many copies (results/ and docs/data/)
  # name it. Keep each question in its existing stages: expanding a corpus-only
  # question to post-training would pay for an unrelated measurement.
  # Unmatched globs must disappear: either directory can be the sole copy,
  # and a target with no questions has nothing to rerun. Walk results first so
  # a freshly written question takes precedence over the exported copy.
  shopt -s nullglob
  for kind in ask stance; do
    seen_runs=("")
    for f in results/"$TARGET".*."$kind"-*.json docs/data/"$TARGET".*."$kind"-*.json; do
      slug="${f##*."$kind"-}"
      slug="${slug%.json}"
      name="${f##*/}"
      stage="${name#"$TARGET".}"
      stage="${stage%."$kind"-"$slug".json}"
      run="$stage.$slug"
      for seen_run in "${seen_runs[@]}"; do
        if [[ "$run" == "$seen_run" ]]; then continue 2; fi
      done
      seen_runs+=("$run")
      question=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["question"])' "$f")
      classifier=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("classifier") or "")' "$f")
      args=("$kind" "$TARGET" "$question" --slug "$slug" --stage "$stage")
      if [[ -n "$classifier" ]]; then args+=(--classifier "$classifier"); fi
      if [[ "$kind" == ask ]]; then
        case "$stage" in
          pretrain|midtrain|long-context) args+=(--pretrain-only) ;;
        esac
      fi
      step "$kind $TARGET $stage '$slug'"
      "${TS[@]}" "${args[@]}"
    done
  done
fi

if [[ "$PHASE" == all || "$PHASE" == export ]]; then
  step "export"
  python3 scripts/export_site_data.py
fi

step "done"
