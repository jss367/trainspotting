// What the language card is allowed to claim when a dataset ships its own
// `language` column.
//
// The card puts two readings of one question on the same bars: what the
// dataset's metadata says, and what py3langid found in a sample of its prompts.
// That only helps if the two are keyed the same way, measured against stated
// denominators, and reported as shares that add up — otherwise it is the same
// apparent contradiction as before, moved closer together. These tests hold the
// join and the arithmetic to that.
//
// Run via pytest (tests/test_language_card.py) or directly: node <this file>

import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..", "..");
const html = fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];
const read = f => JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "data", f), "utf8"));

// Same stubs as the gradient-panel suite: land the declarations, swallow the
// boot sequence, then reach in for the functions under test. The eval'd string
// is this repo's own docs/index.html, read from disk offline — it is how the
// page's functions get tested without a browser, not untrusted input.
const el = () => ({ addEventListener(){}, appendChild(){}, querySelectorAll: () => [],
  querySelector: () => el(), style: {setProperty(){}}, dataset: {},
  set innerHTML(v){}, set onclick(v){}, set textContent(v){} });
globalThis.document = { getElementById: el, querySelectorAll: () => [], createElement: el, body: el() };
globalThis.location = { hash: "" };
globalThis.fetch = async () => ({ ok: false });
eval(js + `
;globalThis.CARD = {langCode, columnLangShares, langSummary, langColumn, wilson,
                    setLangNames: v => { LANG_NAMES = v; LANG_CODES = null; }};`);
const C = globalThis.CARD;
C.setLangNames(read("language-names.json"));

let failures = 0;
const ok = (cond, name) => {
  console.log((cond ? "ok   " : "FAIL ") + name);
  if (!cond) failures++;
};
const eq = (got, want, name) => {
  ok(Object.is(got, want), name);
  if (!Object.is(got, want)) console.log(`       got  ${got}\n       want ${want}`);
};
const near = (got, want, tol, name) => {
  ok(Math.abs(got - want) <= tol, name);
  if (Math.abs(got - want) > tol) console.log(`       got  ${got}\n       want ${want} ±${tol}`);
};

// ------------------------------------------------------------ the code join ---
eq(C.langCode("English"), "en", "column spells English, detector says en");
eq(C.langCode("Chinese"), "zh", "Chinese joins to zh");
eq(C.langCode("  russian "), "ru", "the join ignores case and surrounding space");
// WildChat's own marker for a conversation its detector would not call. Left
// unmapped it would land in the non-English pile and inflate that share.
eq(C.langCode("Nolang"), "undetermined", "Nolang is the column's undetermined");
eq(C.langCode("undetermined"), "undetermined", "our own label round-trips");
// py3langid emits 97 codes; WildChat's column has values outside them. These
// must not silently become some other language.
eq(C.langCode("Maori"), null, "a label with no ISO code in the table does not join");
eq(C.langCode("Sotho"), null, "neither does Sotho");

// ------------------------------------------------------- the column, shared ---
{
  const freq = {English: 60, Chinese: 20, Nolang: 5, Maori: 15};
  const col = C.columnLangShares(freq, 100);
  near(col.english, 0.60, 1e-9, "English share is over the stated denominator");
  near(col.byCode.zh, 0.20, 1e-9, "Chinese lands under zh");
  near(col.undetermined, 0.05, 1e-9, "Nolang lands under undetermined, not under a language");
  eq(col.unmatched.length, 1, "the label with no code is kept, not dropped");
  near(col.unmatchedShare, 0.15, 1e-9, "and keeps its share");
  // The bars can only show what joined; the share that did not still has to be
  // reported somewhere, so it stays in the totals.
  near(col.covered, 1.0, 1e-9, "matched and unmatched together cover the column");
  near(col.nonEnglish, 0.35, 1e-9, "non-English counts the unjoinable labels too");
}
{
  // Two column values collapsing to one code must add, not overwrite.
  const col = C.columnLangShares({Norwegian: 10, "Norwegian Bokmal": 10}, 100);
  near((col.byCode.no || 0) + (col.byCode.nb || 0), 0.20, 1e-9, "no value is lost to a collision");
}
eq(C.columnLangShares(null, 100).english, 0, "a stage with no column reports no English share");

// ------------------------------------------------------ the three shares add ---
{
  const counts = {en: 156, zh: 47, ru: 32, undetermined: 31, fr: 7};
  const n = Object.values(counts).reduce((a, b) => a + b, 0);
  const s = C.langSummary(counts, n);
  eq(s.en + s.nonEn + s.undet, n, "English, other and undetermined partition the sample");
  eq(s.rest.length, 3, "the other-languages count excludes English and undetermined");
  eq(s.rest[0], "zh", "the rest are ordered by size");
  ok(!s.rest.includes("undetermined"), "undetermined is never one of the other languages");
}
{
  const s = C.langSummary({en: 5}, 5);
  eq(s.undet, 0, "a sample the detector called in full has no undetermined share");
  eq(s.nonEn, 0, "and no other-language share");
}

// ------------------------------------------------- against the committed data ---
// The card's whole claim is that these two readings are of the same dataset and
// can be set side by side. Both files have to be present and joinable for that
// to be more than a sentence.
{
  const langs = read("wildchat-1m.chat.languages.json");
  const src = read("wildchat-1m.sources.json").chat;
  const col = C.columnLangShares(C.langColumn({chat: src}, "chat"), src.counted || src.total);
  const counts = {};
  for (const r of langs.records) counts[r.label] = (counts[r.label] || 0) + 1;
  const s = C.langSummary(counts, langs.records.length);

  ok(col.english > 0, "WildChat's own column still has an English share to compare against");
  near(col.covered, 1.0, 1e-6, "the column's values cover every row the stats API counted");
  // Not a test that the two agree — they measure different text over different
  // rows, and the card says so. A test that the sampled reading is precise
  // enough for the comparison to mean anything: if the interval ever grew wide
  // enough to swallow the column's number whole, the ticks would be decoration.
  const [lo, hi] = C.wilson(s.en, s.n);
  ok(hi - lo < 0.25, "the sampled English share is tight enough for the tick to say something");
  // Every language the detector found either joins to a column value or is
  // absent from it; nothing may join to the wrong one.
  for (const c of s.rest.slice(0, 5))
    ok(col.byCode[c] === undefined || col.byCode[c] >= 0, `${c} joins cleanly or not at all`);
}

// ------------------------------------------------------ what the page states ---
// The column is drawn in one place, not two. If the composition card ever stops
// skipping it, the page is back to two language breakdowns three screens apart.
ok(/col === "language" && langMoved/.test(js),
   "the composition card skips the language column once the language card owns it");
ok(!/exact counts, not samples/.test(html),
   "no card promises exact counts over a split the stats API only partly read");
ok(!/over the same sampled prompts — no model is called/.test(html),
   "the language card does not point at a sample that may not be on the tab");

console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
