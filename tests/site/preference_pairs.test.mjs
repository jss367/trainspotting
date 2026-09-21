// What the preference-pair card prints, held to what the file underneath says.
//
// The card's two headline numbers are a mean and a median of the same signed
// quantity, printed side by side precisely because they can differ — on
// Dolci-Instruct-DPO they differ in *sign*. Two formatting choices would erase
// that, and neither raises: dropping the sign, and rounding to the two
// significant figures every other length on this page uses, which renders a
// mean of 2,067 and a median of 1,609 as "2k" and "2k".
//
// Run via pytest (tests/test_site_suites.py) or directly: node <this file>

import { loadPage, read, dataFiles } from "./page.mjs";

const P = loadPage();

let failures = 0;
const ok = (cond, name) => {
  console.log((cond ? "ok   " : "FAIL ") + name);
  if (!cond) failures++;
};
const eq = (got, want, name) => {
  ok(got === want, name);
  if (got !== want) console.log(`       got  ${JSON.stringify(got)}\n       want ${JSON.stringify(want)}`);
};

// ------------------------------------------------------------ the formatter ---
eq(P.fmtSigned(185), "+185", "a positive gap keeps its sign");
eq(P.fmtSigned(-23), "−23", "a negative gap keeps its sign");
eq(P.fmtSigned(0), "+0", "no gap reads as no gap either way");
ok(P.fmtSigned(2067) !== P.fmtSigned(1609),
   "a mean and a median 458 characters apart do not print the same");
eq(P.fmtSigned(2067), "+2,067", "the full figure, not two significant ones");
eq(P.fmtSigned(-1234567), "−1,234,567", "and at any magnitude");

// ------------------------------------------------- the committed pairs files ---
const files = dataFiles().filter(f => f.endsWith(".pairs.json"));
ok(files.length > 0, "the site ships at least one preference-pair summary");

for (const name of files){
  const d = read(name);
  const tag = name.replace(".pairs.json", "");
  // Every pair lands in exactly one of the three buckets the card draws: the
  // chosen side longer, the rejected side longer, or a tie. A pair that fell
  // out of all three would shorten a bar rather than raise anything.
  const chosen = d.delta.hist_chosen.reduce((a, b) => a + b, 0);
  const rejected = d.delta.hist_rejected.reduce((a, b) => a + b, 0);
  eq(chosen + rejected + d.ties, d.n, `${tag}: every pair is in one direction or a tie`);
  eq(d.length_rule.n, d.n - d.ties, `${tag}: the length rule's denominator drops the ties`);
  // The card draws the rate's interval as whiskers; an interval that does not
  // bracket its own rate would draw them on the wrong side of the fill.
  ok(d.length_rule.lo <= d.length_rule.rate && d.length_rule.rate <= d.length_rule.hi,
     `${tag}: the interval brackets the rate`);
  // The generator rule is fitted in-sample, so the card prints it without
  // whiskers. If a file ever grew an interval for it, the card would be
  // claiming sampling error the number does not have.
  if (d.models.rule){
    ok(d.models.rule.fitted === true, `${tag}: the generator rule is marked as fitted`);
    ok(d.models.rule.lo === undefined, `${tag}: the generator rule carries no interval`);
  }
  // The card prints the split as a decomposition of the mean above it.
  if (d.split)
    ok(Math.abs(d.split.reasoning.mean + d.split.answer.mean - d.delta.mean) < 1e-6,
       `${tag}: thinking plus answer is the whole gap`);
}

console.log(failures ? `\n${failures} failure(s)` : "\nall ok");
process.exit(failures ? 1 : 0);
