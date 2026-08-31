#!/usr/bin/env bash
#
# The battery behind one question: how much training does an Olmo 3 model get in
# learning that human lives are valuable and important?
#
# One yes/no question over post-training prompts does not answer that. Three
# things are missing from it, and this script runs each:
#
#   1. Pretraining. 5.93T of the 6.08T tokens in the 7B pipeline are Dolma 3.
#      A rate measured only over Dolci describes the last 0.1% of training.
#   2. Direction. Some examples train the model away from valuing people — an
#      RLVR row whose verifier pays out for an anti-vaccine speech in the right
#      format is in this data. `stance` reads the whole example and signs it.
#   3. Decomposition. "Caring about human lives" is several distinguishable
#      things, and the training signal is not the same for each.
#
# Finally `budget` puts every stage on one scale — tokens the model was fit to —
# because a 6% rate over a 600M-token DPO mix and a 1% rate over 5.93T of
# pretraining are three orders of magnitude apart in what they mean.
#
# Usage:
#   scripts/human_life_value.sh [model] [phase]
#
#   model   olmo-3-7b-think (default), olmo-3-7b-instruct, olmo-3-32b-think
#   phase   all (default) | pretrain | subquestions | stance | budget
#
# Needs ANTHROPIC_API_KEY for every phase except `budget`, which only adds up
# runs that already happened.

set -euo pipefail

MODEL="${1:-olmo-3-7b-think}"
PHASE="${2:-all}"
TS=(python3 -m trainspotting.cli)

# The umbrella question, worded exactly as the committed post-training runs
# under results/*.ask-caring-about-human-lives.json were scored. Do not reword
# it: `budget` sums the stages, and a stage rescored against different words is
# a different measurement being added to the same total.
UMBRELLA="Is this training example about caring about human lives — teaching the model to value people's safety, wellbeing, health, or survival, to avoid contributing to harm to people, or to show concern for human welfare?"
UMBRELLA_SLUG="caring-about-human-lives"

# The umbrella split into parts whose training signal differs, plus its mirror.
# The mirror is not optional: a count of examples that push toward valuing
# people is not a finding until you know how many push the other way.
SUBQUESTIONS=(
  "refusing-harm-to-people|Does this training example teach the model to refuse, deflect, or safely redirect a request that would help someone injure, kill, or endanger other people?"
  "crisis-and-safety|Does this training example teach the model to respond carefully to someone whose health, safety, or life may be at risk — a medical emergency, self-harm, abuse, or another crisis?"
  "people-as-mattering|Does this training example teach the model that people's lives have worth — treating individuals or groups as mattering in themselves, rather than as expendable, interchangeable, or beneath consideration?"
  "weighing-lives|Does this training example involve explicit reasoning about the value of a human life — weighing lives against each other, or against money, convenience, or other goods?"
  "disregard-for-life|Would fitting this training example teach the model to DISREGARD human welfare — to assist with harm to people, to treat lives as expendable, or to produce content that puts people at risk?"
)

step() { printf '\n=== %s ===\n' "$1" >&2; }

# 1. The corpora. --pretrain-only exists for exactly this: the post-training
#    half of the umbrella question is already committed, and re-running it would
#    pay for nine stages of prompt labeling to learn nothing new.
if [[ "$PHASE" == all || "$PHASE" == pretrain ]]; then
  step "sampling the corpora (no API key needed; skips what is already sampled)"
  for stage in pretrain midtrain long-context; do
    if [[ ! -f "results/$MODEL.$stage.docs.json" && ! -f "docs/data/$MODEL.$stage.docs.json" ]]; then
      "${TS[@]}" pretrain "$MODEL" --stage "$stage"
    fi
  done
  step "scoring the corpora against the umbrella question"
  "${TS[@]}" ask "$MODEL" "$UMBRELLA" --slug "$UMBRELLA_SLUG" --pretrain-only
fi

# 2. The sub-questions, both halves. Each one re-scores the same committed
#    samples, so they are all measured over identical rows and documents and the
#    cards stack up against each other.
if [[ "$PHASE" == all || "$PHASE" == subquestions ]]; then
  for entry in "${SUBQUESTIONS[@]}"; do
    slug="${entry%%|*}"
    question="${entry#*|}"
    step "asking: $slug"
    "${TS[@]}" ask "$MODEL" "$question" --slug "$slug" --pretrain
  done
fi

# 3. Direction, over whole examples rather than prompts. Post-training only: a
#    corpus document has no preferred and dispreferred side, and no verifier.
if [[ "$PHASE" == all || "$PHASE" == stance ]]; then
  step "judging which way each post-training example pushes"
  "${TS[@]}" stance "$MODEL" "$UMBRELLA" --slug "$UMBRELLA_SLUG"
fi

# 4. The rollup. Free — it only reads what the phases above wrote.
if [[ "$PHASE" == all || "$PHASE" == budget ]]; then
  step "rolling every stage up into one token budget"
  "${TS[@]}" budget "$MODEL" "$UMBRELLA_SLUG" --json
  for entry in "${SUBQUESTIONS[@]}"; do
    "${TS[@]}" budget "$MODEL" "${entry%%|*}" --json || true
  done
fi

step "done — run scripts/export_site_data.py to put these on the site"
