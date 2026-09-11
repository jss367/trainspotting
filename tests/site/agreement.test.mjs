import assert from "node:assert/strict";
import { loadPage, read, dataFiles } from "./page.mjs";

const { stabilityNote } = loadPage();
const labels = {dataset: "a/b", revision: "rev1", system_sha: "rubric", classifier: "c",
  sample: 6, seed: 0, generated: "2026-09-11T12:00:00"};
const comparison = {labels_run: {...labels}, replicate: {...labels, n: 6, accuracy: 5/6,
  kappa: 0.75, same_draw: true, comparison_issues: []}};
assert.match(stabilityNote(comparison, labels), /same 6 prompts/);
assert.equal(stabilityNote(null, labels), null);
assert.equal(stabilityNote({...comparison, replicate: {...comparison.replicate, n: 0}}, labels), null);
for (const same_draw of [false, undefined]){
  assert.equal(stabilityNote({...comparison, replicate: {...comparison.replicate, same_draw}}, labels), null);
}
assert.equal(stabilityNote({...comparison, replicate: {...comparison.replicate,
  comparison_issues: ["different revision"]}}, labels), null);
for (const key of ["dataset", "revision", "system_sha", "classifier", "sample", "seed", "generated"]){
  assert.equal(stabilityNote(comparison, {...labels, [key]: "changed"}), null, key);
  assert.equal(stabilityNote(comparison, {...labels, [key]: undefined}), null, key);
}
for (const run of ["labels_run", "replicate"]){
  assert.equal(stabilityNote({...comparison, [run]: {...comparison[run], revision_moved_to: "rev2"}}, labels), null);
}
assert.equal(stabilityNote(comparison, {...labels, revision_moved_to: "rev2"}), null);
// Every saved check must still qualify its actual displayed run after export.
for (const name of dataFiles().filter(n => n.endsWith(".agreement.json"))){
  const a = read(name), current = read(name.replace(".agreement.json", ".labels.json"));
  if (a.replicate.n) assert.match(stabilityNote(a, current), /Stability check/, name);
  else assert.equal(stabilityNote(a, current), null);
}
console.log("agreement provenance and stability notes passed");
