import assert from "node:assert/strict";
import { loadPage, read } from "./page.mjs";

const { sameDraw, sameRevision, resolverFor, pairingEvidence, crossRows, promptKey } = loadPage();
const labels = Array.from({length: 20}, (_, row) => ({row, prompt: `original ${row}`, label: "honesty"}));
const profile = {dataset: "x/y", sample: 20, seed: 0, revisions: {context: "rev1"},
  records: labels.map(r => ({row: r.row, k: promptKey(r.prompt), m: {source: "original source"}}))};
const run = {dataset: "x/y", sample: 20, seed: 0, revision: "rev1", records: labels};
const cross = (p, r) => crossRows(r.records, resolverFor(p, r, "source"), item => item.label);
const evidence = (p, r, c) => pairingEvidence(p, r, r.records.length - c.unmatched, r.records.length);
assert.equal(sameRevision(profile, run), true);
assert.equal(evidence(profile, run, cross(profile, run)).refuse, undefined);

for (const missingSide of ["profile", "run"]){
  const p = structuredClone(profile), r = structuredClone(run);
  if (missingSide === "profile") p.revisions.context = null;
  else delete r.revision;
  assert.equal(sameRevision(p, r), false);
  assert.equal(sameDraw(p, r), null, "unknown provenance permits checking prompt evidence");
  // Identical positions from a republished dataset are not identical prompts.
  p.records = p.records.map(item => ({...item, k: promptKey(`replacement ${item.row}`)}));
  const allChanged = cross(p, r);
  assert.equal(allChanged.unmatched, 20);
  assert.ok(evidence(p, r, allChanged).refuse, "equal numeric row sets cannot override missing prompt matches");

  // A 95% overall match can keep the grid, but the changed row must never be
  // attached to its replacement's source metadata.
  p.records = structuredClone(profile.records);
  p.records[0] = {row: 0, k: promptKey("replacement"), m: {source: "wrong source"}};
  const oneChanged = cross(p, r);
  assert.equal(oneChanged.unmatched, 1);
  assert.equal(oneChanged.rows.reduce((n, row) => n + row.total, 0), 19);
  assert.equal(oneChanged.rows.some(row => row.name === "wrong source"), false);
  assert.equal(evidence(p, r, oneChanged).refuse, undefined);
  assert.ok(evidence(p, r, oneChanged).note);
}

// Unknown revisions retain the legacy ambiguity guard, even with row IDs.
const sharedOpening = "shared opening ".repeat(40);
const ambiguous = {...profile, revisions: {}, records: [
  {row: 0, k: promptKey(sharedOpening), m: {source: "one"}},
  {row: 1, k: promptKey(sharedOpening), m: {source: "two"}},
]};
const repeated = {...run, records: [
  {row: 0, prompt: sharedOpening + "first", label: "honesty"},
  {row: 1, prompt: sharedOpening + "second", label: "helpfulness"},
]};
assert.equal(cross(ambiguous, repeated).undecidable, 2);
assert.equal(cross(ambiguous, repeated).rows.length, 0);
// Known matching revisions resolve the same opening by its actual row.
const pinned = {...ambiguous, revisions: {context: "rev1"}};
assert.deepEqual(cross(pinned, repeated).rows.map(row => row.name), ["one", "two"]);
assert.equal(sameRevision(pinned, {...repeated, revision: "rev2"}), false);
assert.ok(sameDraw(pinned, {...repeated, revision: "rev2"}));
assert.equal(sameRevision({...pinned, revisions: {context: "rev1", moved: ["rev2"]}}, repeated), false);
assert.equal(sameRevision(pinned, {...repeated, revision_moved_to: "rev2"}), false);

// Existing unstamped DPO context remains usable through its prompt evidence.
const legacy = read("olmo-3.1-32b-instruct.dpo.profile.json");
for (const kind of ["labels", "languages"]){
  const current = read(`olmo-3.1-32b-instruct.dpo.${kind}.json`);
  current.records = current.records.filter(r => r.label);
  const c = crossRows(current.records, resolverFor(legacy, current, "preference_type"), r => r.label);
  assert.equal(sameRevision(legacy, current), false);
  assert.equal(sameDraw(legacy, current), null);
  assert.equal(evidence(legacy, current, c).refuse, undefined, `${kind} remains displayable`);
  assert.ok(c.rows.length);
}
console.log("crosstab row identity requires revision proof or prompt corroboration");
