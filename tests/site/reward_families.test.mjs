import assert from "node:assert/strict";
import { loadPage, read } from "./page.mjs";

const T = loadPage();
T.setRewards(read("reward-kinds.json"));

// An RL stage stays one stage; individual rewards carry the distinction.
assert.equal(T.stageLabel("rlvr"), "RL");
assert.equal(T.rewardFamily("LLM judge"), "rlaif");
assert.equal(T.rewardFamily("constraint checker"), "rlvr");
assert.equal(T.rewardFamily("future scorer"), "unknown");

for (const model of ["olmo-3-7b-instruct", "olmo-3-7b-think", "olmo-3-32b-think"]){
  const st = read(`${model}.sources.json`).rlvr;
  const groups = T.rewardComposition(st);
  assert.equal(groups.rlvr.count + groups.rlaif.count, st.total);
  assert.equal(Object.values(groups.rlvr.kinds).reduce((a, b) => a + b, 0), groups.rlvr.count);
  assert.equal(groups.rlaif.count, st.columns.dataset_source[
    model === "olmo-3-7b-think" ? "hamishivi/rlvr_general_mix"
      : "allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered"]);
  assert.equal(T.rewardComposition({...st, partial: true}), null);
}

const st = read("olmo-3-7b-instruct.sources.json").rlvr;
assert.equal(T.rewardComposition({...st, total: st.total + 1}), null);
assert.equal(T.rewardComposition({total: 1, columns: {source: {unidentified: 1}}}), null);
assert.equal(T.rewardComposition({total: 0, columns: {}}), null);
assert.equal(T.rewardComposition(null), null);
assert.match(T.renderRewardComposition(st), /not shares of training updates/);
assert.match(T.renderRewardComposition(st), /RLAIF · AI feedback/);
assert.match(T.renderRewardComposition({...st, partial: true}), /split is unavailable/);

// Exercise legacy saved records: no new family field is required to fix them.
const records = read("olmo-3-7b-instruct.rlvr.context.json").records;
const example = records.find(r => r.row === 127386);
assert.ok(example, "the user's linked example is in the committed sample");
const judged = T.renderRLVR(example, st.dataset);
assert.match(judged, /RLAIF · AI feedback/);
assert.match(judged, /Heart function classification/);
assert.match(judged, /3 · LLM judge/);
assert.doesNotMatch(judged, /No response is stored|No rule can grade/);
const checked = T.renderRLVR(records.find(r => r.reward.kind === "constraint checker"), st.dataset);
assert.match(checked, /RLVR · programmatic rewards/);
assert.match(checked, /3 · verifier/);
const unknown = T.renderRLVR({...example, reward: {kind: "unknown"}}, st.dataset);
assert.match(unknown, /reward type unknown/);
assert.doesNotMatch(unknown, /3 · verifier/);
console.log("reward family labels, counts, and legacy examples passed");
