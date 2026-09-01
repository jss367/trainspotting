// What "the whole pipeline as area" is allowed to leave out.
//
// The treemap's children come from a stage's published composition, and a
// composition states its per-source amounts in whichever unit its own release
// used: tokens for the Dolma 3 mixes, bytes for the Pile. `treemapLayout` drops
// any child without a positive value, so reading the wrong field does not
// produce a wrong box — it produces no boxes at all, and a card that silently
// omits the composition it exists to show. These tests hold the unit choice.
//
// Run via pytest (tests/test_pipeline_treemap.py) or directly: node <this file>

import { loadPage, read } from "./page.mjs";

const T = loadPage();

let failures = 0;
const ok = (cond, name) => {
  console.log((cond ? "ok   " : "FAIL ") + name);
  if (!cond) failures++;
};
const eq = (got, want, name) => {
  ok(Object.is(got, want), name);
  if (!Object.is(got, want)) console.log(`       got  ${got}\n       want ${want}`);
};

const REGISTRY = read("registry.json");
const stageOf = (model, name) => REGISTRY[model].stages.find(s => s.stage === name);
const rowFor = stage => ({stage: stage.stage, tokens: stage.tokens, stageEntry: stage});

// ------------------------------------------------------- a byte composition ---
// The Pile publishes raw sizes, not token counts. Every child here used to come
// back with `v: undefined`, which the layout filters out — so the card rendered
// one flat pretraining box and no sources, saying by omission that a 22-source
// corpus has no composition on file.
const pile = stageOf("pythia-12b-deduped", "pretrain");
eq(pile.composition_unit, "bytes", "the Pile's composition is declared in bytes");
const pileKids = T.childrenOf(rowFor(pile), {});
eq(pileKids.length, 22, "all 22 Pile components become children");
ok(pileKids.every(c => Number.isFinite(c.v) && c.v > 0),
   "every child has a positive area value");
ok(pileKids.every(c => Number.isFinite(c.tokens) && c.tokens > 0),
   "every child has a token figure to label the box with");
eq(T.treemapLayout(pileKids.map(c => ({...c, v: c.v})), 400, 200).length, 22,
   "the layout keeps all 22 rather than filtering them out");

// Bytes are never in the box's unit, so what crosses over is the share. The
// scaled token figures have to add back up to the stage they sit inside.
const pileTokens = pileKids.reduce((a, c) => a + c.tokens, 0);
ok(Math.abs(pileTokens - pile.tokens) <= 1e-6 * pile.tokens,
   "the scaled children sum to the stage's published token total");
// The largest component keeps its rank after scaling — a share carried across
// is still a share.
eq(pileKids[0].name, "Pile-CC (web)", "the composition keeps its published order");
ok(pileKids[0].v === Math.max(...pileKids.map(c => c.v)), "and Pile-CC is the largest");
// The tooltip states the measured quantity in its own unit and marks the token
// figure as derived, so nothing reads as a token count EleutherAI published.
ok(/GB|TB/.test(pileKids[0].detail), "the tooltip reports the source's own byte size");
ok(pileKids[0].detail.includes("≈"), "and marks the token figure as approximate");

// ------------------------------------------------------ a token composition ---
// The other unit, unchanged: Dolma 3 mixes publish token counts and a stage
// that trains on its whole recipe shows the mix's own numbers rather than
// scaled ones.
const dolma = stageOf("olmo-3-7b-instruct", "pretrain");
ok(!dolma.composition_unit, "a Dolma 3 mix declares no unit, meaning tokens");
const dolmaKids = T.childrenOf(rowFor(dolma), {});
ok(dolmaKids.length > 0, "the token composition still produces children");
ok(dolmaKids.every((c, i) => c.v === dolma.composition[i].tokens),
   "and a token composition's area is its published token count");

// ---------------------------------------------------------- no composition ---
// A stage with neither a published composition nor a shard listing has no
// children to invent, and the caller draws it as one solid box.
eq(T.childrenOf({stage: "x", tokens: 1e9, stageEntry: {stage: "x"}}, {}), null,
   "a stage with nothing to break down gets no children");

console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
