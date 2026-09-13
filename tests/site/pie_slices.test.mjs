// The geometry behind the bar/pie toggle on a part-to-whole strip.
//
// A pie is the one chart on this page whose errors are invisible: a slice drawn
// with the wrong large-arc flag is still a slice, a slice that swallows the
// circle still looks like a pie, and nothing about the picture says the angles
// do not add up. So the angles are asserted here rather than eyeballed there.
//
// Run via pytest (tests/test_site_suites.py) or directly: node <this file>

import { loadPage, read, dataFiles, pageHtml, pageSource } from "./page.mjs";

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
const near = (got, want, tol, name) => {
  const good = Math.abs(got - want) <= tol;
  ok(good, name);
  if (!good) console.log(`       got  ${got}\n       want ${want} ±${tol}`);
};

const R = 100;
const seg = (k, v) => ({k, v, c: "#000", detail: k});

// --------------------------------------------------------- the whole circle ---
// Every value in the chart is a share of one total, so the fractions have to be
// exactly that and nothing else.
{
  const s = T.pieSlices([seg("a", 3), seg("b", 1)], R);
  eq(s.length, 2, "two positive values give two slices");
  near(s.reduce((a, x) => a + x.frac, 0), 1, 1e-12, "the fractions sum to one");
  near(s[0].frac, 0.75, 1e-12, "a value of 3 in 4 is three quarters of the circle");
  eq(s[0].k, "a", "slices come back in the order given, not sorted by size");
}

// ------------------------------------------------------------- the arc flags ---
// The large-arc flag is the one input SVG will not infer. Set wrong, a 60%
// slice renders as the 40% slice beside it and the picture inverts.
{
  const [big, small] = T.pieSlices([seg("big", 60), seg("small", 40)], R);
  ok(/A 100 100 0 1 1 /.test(big.d), "a slice over half the circle takes the large arc");
  ok(/A 100 100 0 0 1 /.test(small.d), "a slice under half the circle takes the small arc");
  ok(big.d.startsWith("M 0 0 L") && big.d.endsWith("Z"),
     "a slice is closed back through the centre");
}

// A slice at exactly half is the boundary the flag is chosen on, and either
// arc draws the same semicircle — what must not happen is the sweep flipping.
for (const s of T.pieSlices([seg("a", 1), seg("b", 1)], R))
  ok(/A 100 100 0 [01] 1 /.test(s.d), `a half slice sweeps clockwise (${s.k})`);

// ----------------------------------------------------------- twelve o'clock ---
// Reading a pie means starting somewhere agreed. The first slice starts at the
// top and the stages run clockwise from there, matching the strip's left-to-right.
{
  const [first, second] = T.pieSlices([seg("a", 1), seg("b", 3)], R);
  ok(first.d.includes("L 0.000 -100.000"), "the first slice starts at twelve o'clock");
  near(first.mid, 0.125, 1e-12, "a quarter slice from the top has its midpoint at 45°");
  near(second.mid, 0.625, 1e-12, "the next slice's midpoint follows the first's end");
}

// -------------------------------------------------------------- one big slice ---
// The reason this file exists. `A` between two identical points draws nothing,
// so a lone slice used to be an empty card — exactly the case a reader asking
// "how much of it was pretraining" is most likely to land on.
{
  const [only] = T.pieSlices([seg("pretrain", 6e12)], R);
  eq(only.frac, 1, "a lone value is the whole circle");
  eq((only.d.match(/A /g) || []).length, 2, "the full circle is drawn as two arcs");
  ok(!/M 0 0 L/.test(only.d), "the full circle has no wedge edge to the centre");
}

// A slice that rounds to the whole circle still leaves the others in the list,
// drawn to scale. Dropping them would make the pie say the pipeline has one
// stage, which is the opposite of what the card is for.
{
  const s = T.pieSlices([seg("pretrain", 5.9e12), seg("sft", 1e9), seg("dpo", 1e8)], R);
  eq(s.length, 3, "a 0.002% stage is still a slice");
  ok(s[1].frac > 0 && s[1].frac < 1 / 360, "and it is thinner than a degree");
  ok(/A 100 100 0 0 1 /.test(s[1].d), "drawn to scale, not to a minimum");
}

// ------------------------------------------------------------- absent values ---
// A stage with no committed size must not take a wedge — an unsized stage
// showing as a zero-width slice is fine, but one taking angle is a wrong chart.
{
  const s = T.pieSlices([seg("a", 5), seg("unsized", 0), seg("negative", -3)], R);
  eq(s.length, 1, "zero and negative values get no slice");
  eq(s[0].frac, 1, "and they take none of the circle either");
  eq(T.pieSlices([seg("a", 0)], R).length, 0, "nothing positive at all draws nothing");
}

// ------------------------------------------------------------- the real card ---
// The pipeline the toggle was added for. Olmo's post-training is a sliver, and
// the point of asserting it here is that the note under the pie says so.
{
  const model = "olmo-3-32b-think";
  const stages = read("registry.json")[model].stages;
  // The card's own two sources of size: the paper's token counts for the corpus
  // stages, and the sampled estimate on each post-training profile for the rest.
  const rows = stages.map(s => {
    if (s.tokens) return seg(s.stage, s.tokens);
    const f = `${model}.${s.stage}.profile.json`;
    const p = dataFiles().includes(f) ? read(f) : null;
    return seg(s.stage, p && p.tokens ? p.tokens.tokens : 0);
  });
  const slices = T.pieSlices(rows, R);
  ok(slices.length > 3, "the pipeline has more sized stages than just the corpus ones");
  const pretrain = slices.find(s => s.k === "pretrain");
  ok(pretrain && pretrain.frac > 0.9, "pretraining is over 90% of the tokens");
  const post = slices.filter(s => ["sft", "dpo", "rlvr"].includes(s.k));
  eq(post.length, 3, "all three post-training stages are sized");
  ok(post.every(s => s.frac < 0.005), "no post-training stage reaches half a percent of the circle");
  // SFT is the largest of them and lands at about one degree; DPO and the RL
  // stage are two and three orders of magnitude below that. The note under the
  // pie is the only place those two are accounted for.
  ok(post.filter(s => s.frac < 1 / 360).length >= 2,
     "at least two post-training stages are thinner than a degree, so the note has to fire");
  // What the toggle is for: the same reader, the same card, the other question.
  // Sample counts are within an order of magnitude of each other, so this one
  // is a pie a reader can actually read off.
  const sampled = T.pieSlices(post.map(s => seg(s.k, read(`${model}.${s.k}.profile.json`).chars.n)), R);
  ok(sampled.every(s => s.frac > 1 / 360),
     "the sampled-examples strip has no invisible slice, so its pie carries no note");
}

// ------------------------------------------------------------------ the page ---
// The toggle is markup and CSS, not geometry, and it is the half a reader
// touches. These hold the parts that would fail silently.
{
  const src = pageSource(), css = pageHtml();
  ok(/aria-pressed="true">bar/.test(src), "the strip loads as a bar, not a pie");
  ok(/data-view="pie"/.test(src), "there is a pie button to switch to");
  ok(/\.viewpick button\[aria-pressed="true"\]/.test(css),
     "the pressed button is styled, so the current form is visible without a hover");
  ok(/\.pie path \{[^}]*stroke: var\(--surface-1\)/.test(css),
     "slices are separated by a surface stroke, the pie's version of the strip's gap");
  // A label's ink is resolved against its fill once, at draw time. The fills
  // are CSS variables that flip ends of the ramp with the colour scheme, so a
  // pie left on screen through a theme change would keep black text on what is
  // now a dark slice. Dropping the cache only fixes the next pie drawn.
  ok(/data-ink="/.test(src), "marks wearing computed ink record the fill they were resolved against");
  ok(/INK = \{\};\s*\n\s*repaintInk\(\);/.test(src),
     "a colour-scheme change repaints existing ink, not just the cache");
  ok(/el\.ownerSVGElement/.test(src),
     "the repaint knows an SVG label wears `fill` and an HTML tile wears `color`");
}

console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
