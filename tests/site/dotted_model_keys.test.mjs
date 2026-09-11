import assert from "node:assert/strict";
import { loadPage, read } from "./page.mjs";

const { questionFiles, splitModelKey } = loadPage();
const registry = read("registry.json");
const model = "olmo-3.1-32b-instruct";
assert.ok(registry[model], "exercise the actual dotted registry target");
const slug = "question.with.dots", stanceOnly = "direction-only";
// No paid question run is needed: these are the filenames export would add.
const manifest = [...read("manifest.json"),
  `${model}.sft.ask-${slug}.json`, `${model}.dpo.ask-${slug}.json`,
  `${model}.sft.stance-${slug}.json`, `${model}.dpo.stance-${stanceOnly}.json`,
  `${model}.budget-${slug}.json`, `${model}.budget-${stanceOnly}.json`,
  // A sibling target sharing a prefix must not contribute cards to this one.
  `${model}.variant.sft.ask-sibling.json`,
];
const asks = questionFiles(model, manifest, "ask");
const stances = questionFiles(model, manifest, "stance");
assert.deepEqual(asks[slug], [
  ["sft", `${model}.sft.ask-${slug}.json`], ["dpo", `${model}.dpo.ask-${slug}.json`],
]);
assert.deepEqual(stances[slug], [["sft", `${model}.sft.stance-${slug}.json`]]);
assert.deepEqual(stances[stanceOnly], [["dpo", `${model}.dpo.stance-${stanceOnly}.json`]]);
assert.equal(asks.sibling, undefined);
// These are the two paths into budgetCard: an ask slug, or a stance-only slug.
const budgetSlugs = new Set([...Object.keys(asks), ...Object.keys(stances)]);
assert.deepEqual([...budgetSlugs].sort(), [slug, stanceOnly].sort());
for (const key of budgetSlugs) assert.ok(manifest.includes(`${model}.budget-${key}.json`));

// Follow the key from a comparison deep link back to a real context record.
const context = read(`${model}.dpo.context.json`);
const row = context.records[0].row;
for (const key of ["helpfulness", `ask-${slug}`]){
  const link = `#compare/dpo/${model}.${key}/row-${row}`;
  const [, stage, combined] = link.slice(1).split("/");
  const parsed = splitModelKey(combined, registry);
  assert.deepEqual(parsed, {model, key});
  assert.ok(registry[parsed.model].stages.some(s => s.stage === stage));
  assert.ok(read(`${parsed.model}.${stage}.context.json`).records.some(r => r.row === row));
}
assert.deepEqual(splitModelKey(`${model}.dpo`, registry), {model, key: "dpo"}, "search context link");
assert.deepEqual(splitModelKey("olmo-3-7b-think.honesty", registry),
  {model: "olmo-3-7b-think", key: "honesty"}, "undotted links remain valid");
const overlapping = {"a": {}, "a.b": {}, "a.b.c": {}};
assert.deepEqual(splitModelKey("a.b.c.ask-topic.with.dots", overlapping),
  {model: "a.b.c", key: "ask-topic.with.dots"}, "longest registered model prefix wins");
assert.equal(splitModelKey("unknown.helpfulness", registry), null);
assert.equal(splitModelKey(model, registry), null);
assert.equal(splitModelKey(undefined, registry), null);
console.log("dotted models retain question discovery and comparison/search context links");
