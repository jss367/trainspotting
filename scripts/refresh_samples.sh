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
#           labels  classify (needs ANTHROPIC_API_KEY)
#           asks    re-run every ask question already committed for the target
#                   (needs ANTHROPIC_API_KEY); stance runs are re-run too
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
TS=(python3 -m trainspotting.cli)
step() { printf '\n=== %s ===\n' "$1" >&2; }

if [[ "$PHASE" == all || "$PHASE" == free ]]; then
  step "context $TARGET (the examples the site drills into)"
  "${TS[@]}" context "$TARGET"
  step "languages $TARGET"
  "${TS[@]}" languages "$TARGET"
fi

if [[ "$PHASE" == all || "$PHASE" == labels ]]; then
  step "classify $TARGET (needs ANTHROPIC_API_KEY)"
  "${TS[@]}" classify "$TARGET"
fi

if [[ "$PHASE" == all || "$PHASE" == asks ]]; then
  # Every question already asked of this target, by slug, with the wording the
  # committed file recorded. Rewording a question is a different measurement,
  # so the text is read back out of the result rather than typed here.
  # One run per slug, however many stages and copies (results/ and docs/data/)
  # name it: the same question asked twice is the same money spent twice.
  # Unmatched globs must disappear: either directory can be the sole copy,
  # and a target with no questions has nothing to rerun. Walk results first so
  # a freshly written question takes precedence over the exported copy.
  shopt -s nullglob
  for kind in ask stance; do
    seen_slugs=("")
    for f in results/"$TARGET".*."$kind"-*.json docs/data/"$TARGET".*."$kind"-*.json; do
      slug="${f##*."$kind"-}"
      slug="${slug%.json}"
      for seen_slug in "${seen_slugs[@]}"; do
        if [[ "$slug" == "$seen_slug" ]]; then continue 2; fi
      done
      seen_slugs+=("$slug")
      question=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["question"])' "$f")
      step "$kind $TARGET '$slug'"
      "${TS[@]}" "$kind" "$TARGET" "$question" --slug "$slug"
    done
  done
fi

if [[ "$PHASE" == all || "$PHASE" == export ]]; then
  step "export"
  python3 scripts/export_site_data.py
fi

step "done"
