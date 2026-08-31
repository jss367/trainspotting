// What the DPO "how this pair updates the model" panel is allowed to claim.
//
// The panel says two things that are true or false rather than approximate: a
// span is a shared *opening* (its two log-probability terms cancel one for
// one), and a pair carries *no gradient at all*. Both rest on the two responses
// being byte-identical up to some point in the sequence DPO actually scored,
// and the stored context row is not always a faithful copy of that sequence —
// splitting the thinking span out drops the <think> markers and the whitespace
// around them, and every field is cut at 4,000 characters. These tests hold the
// panel to claiming exactness only where the row can establish it.
//
// Run via pytest (tests/test_gradient_panel.py) or directly: node <this file>

import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..", "..");
const html = fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];

// The page's top level touches the DOM and fetches its data; stub enough of both
// that the function declarations land, and swallow the boot sequence.
const el = () => ({ addEventListener(){}, appendChild(){}, querySelectorAll: () => [],
  querySelector: () => el(), style: {setProperty(){}}, dataset: {},
  set innerHTML(v){}, set onclick(v){}, set textContent(v){} });
globalThis.document = { getElementById: el, querySelectorAll: () => [], createElement: el, body: el() };
globalThis.location = { hash: "" };
globalThis.fetch = async () => ({ ok: false });
eval(js + `
;globalThis.PANEL = {diffPair, opChars, uniqueChars, sideText, sideCut, demotePrefix,
                     gradientSection, rawResponseStored, renderDPO, sharedTurns,
                     candidateTurns};`);
const P = globalThis.PANEL;

let failures = 0;
const ok = (cond, name) => {
  console.log((cond ? "ok   " : "FAIL ") + name);
  if (!cond) failures++;
};
const eq = (got, want, name) => {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  ok(g === w, name);
  if (g !== w) console.log(`       got  ${g}\n       want ${w}`);
};

// ---------------------------------------------------------------- the diff ---
const LONG = "the quick brown fox jumps over the lazy dog and keeps running";
const sideOf = (ops, f) => ops.map(o => o[f]).join("");

eq(P.diffPair("", ""), [], "empty pair diffs to nothing");
eq(P.diffPair(LONG, LONG), [{t: "prefix", a: LONG, b: LONG}], "identical text is all opening");
eq(P.diffPair("", "x y"), [{t: "rejected", a: "", b: "x y"}], "an empty side is all rejected");
// The opening stops before the delimiter: tokenizers usually give that space to
// the word after it, which differs, so it goes into the divergence instead.
eq(P.diffPair(LONG + " alpha", LONG + " beta"),
  [{t: "prefix", a: LONG, b: LONG}, {t: "chosen", a: " ", b: ""}, {t: "rejected", a: "", b: " "},
   {t: "chosen", a: "alpha", b: ""}, {t: "rejected", a: "", b: "beta"}],
  "a shared head, then the delimiter and the differing word");
eq(P.diffPair("prefix " + LONG, "other " + LONG).map(o => o.t), ["chosen", "rejected", "same"],
  "a shared tail is matched wording, not an opening");
// "world" matches on both sides but is far too short to mean anything, so it
// folds back into each side; the byte-exact "hello " opening is kept, since
// however short, its two terms really do cancel.
eq(P.diffPair("hello there world", "hello brave world").map(o => o.a).join(""), "hello there world",
  "coincidental matches fold back into both sides without losing text");
eq(P.opChars(P.diffPair("hello there world", "hello brave world"), "prefix", "a"), "hello".length,
  "a short opening is kept, minus the delimiter");

// The opening is a claim about token sequences, so whitespace counts.
const CODE_A = "def f(x):\n    return x  # a long enough matched run of code here\n";
const CODE_B = "def f(x):\n\t\treturn x  # a long enough matched run of code here\n";
eq(P.diffPair(CODE_A, CODE_B).map(o => o.t), ["prefix", "chosen", "rejected", "same"],
  "indentation that differs ends the opening");
eq(P.opChars(P.diffPair(CODE_A, CODE_B), "prefix", "a"), "def f(x):".length,
  "the opening stops before the whitespace ahead of the first differing byte");

// Neither panel may show the other side's text: 165 of the committed rows have a
// matched span whose two sides differ in whitespace alone.
for (const [a, b] of [[CODE_A, CODE_B], ["one two three four", "one two zzz four"],
                      ["a\nb\nc", "a\nb"], ["  leading ws " + LONG, "\tleading ws " + LONG]]){
  const ops = P.diffPair(a, b);
  ok(sideOf(ops, "a") === a, `chosen side rejoins byte for byte: ${JSON.stringify(a.slice(0, 24))}`);
  ok(sideOf(ops, "b") === b, `rejected side rejoins byte for byte: ${JSON.stringify(b.slice(0, 24))}`);
}

// Python's len() counts code points; every length here must agree with it.
eq(P.opChars([{t: "same", a: "a\u{1F600}b", b: "a\u{1F600}b"}], "same", "a"), 3,
  "an emoji counts as one character, as the exported chars do");
ok(P.gradientSection({prompt_full: {text: "p", chars: 1}, row: 1, meta: {},
     chosen: {model: "b", turns: [{role: "assistant", text: "hi \u{1F600} there friend", chars: 17}]},
     rejected: {model: "s", turns: [{role: "assistant", text: "hi \u{1F600} there pal", chars: 14}]}})
   .replace(/\s+/g, " ").includes("shared opening <em>"),
  "a pair with an emoji is not mistaken for a truncated one");

eq(P.demotePrefix([{t: "prefix", a: "abc", b: "abc"}, {t: "same", a: "def", b: "def"},
                   {t: "chosen", a: "x", b: ""}]),
  [{t: "same", a: "abcdef", b: "abcdef"}, {t: "chosen", a: "x", b: ""}],
  "demoting an opening merges it into the overlap that follows");

// ------------------------------------------------------------- the claims ---
const turn = (text, {chars, reasoning} = {}) => ({role: "assistant", text,
  chars: chars ?? text.length,
  ...(reasoning ? {reasoning: {text: reasoning, chars: reasoning.length}} : {})});
const pair = (c, r) => ({prompt_full: {text: "p", chars: 1}, row: 1, meta: {},
  chosen: {model: "big", turns: [turn(...[].concat(c))]},
  rejected: {model: "small", turns: [turn(...[].concat(r))]}});
const panel = rec => P.gradientSection(rec).replace(/\s+/g, " ");
const claimsOpening = out => /shared opening <em>/.test(out);
const claimsZero = out => out.includes("gradient of exactly zero");

let out = panel(pair([LONG + " alpha"], [LONG + " beta"]));
ok(claimsOpening(out), "an untruncated pair with no thinking span can show an opening");

out = panel(pair([LONG + " alpha", {chars: 9000}], [LONG + " beta"]));
ok(!claimsOpening(out), "a truncated side shows no opening");
ok(out.includes("one side is cut at 4,000 characters, so whether the responses stay identical"),
  "and names truncation as the reason");

out = panel(pair([LONG + " alpha", {reasoning: "same trace"}],
                 [LONG + " beta", {reasoning: "same trace"}]));
ok(!claimsOpening(out), "thinking spans that match as stored still show no opening");
ok(out.includes("drops the"), "and name the <think> normalization as the reason");

out = panel(pair([LONG + " alpha", {reasoning: "one trace"}],
                 [LONG + " beta", {reasoning: "a different trace"}]));
ok(out.includes("two thinking spans in front of these answers differ"),
  "thinking spans that differ are named as the reason");

out = panel(pair(["identical answer text here"], ["identical answer text here"]));
ok(claimsZero(out), "a byte-identical untruncated pair carries no gradient");

// A word diff matches words whose whitespace differs, so "the diff found nothing
// unique" is not identity. Every separator differs here, so the matched run is
// too long for the coincidence filter to break up.
const WS_A = "one two three four five six seven eight nine ten ";
const WS_B = WS_A.replace(/ /g, "\t");
ok(P.uniqueChars(P.diffPair(WS_A, WS_B)) === 0, "the diff finds nothing unique in a tab-for-space pair");
ok(!claimsZero(panel(pair([WS_A], [WS_B]))), "but that pair is not called zero-gradient");

ok(!claimsZero(panel(pair(["identical text", {chars: 9000}], ["identical text", {chars: 9000}]))),
  "nor is an identical-looking pair that was cut");

// ------------------------------------------------------ the branch point ---
// A multi-turn pair shares its leading turns with itself; those are the
// conversation both candidates answer in, not either candidate.
const anyTurn = (role, text) => ({role, text, chars: text.length});
const convo = [
  anyTurn("user", "how do I bake bread?"),
  anyTurn("assistant", "Mix flour, water, salt and yeast, then prove it twice."),
  anyTurn("user", "and if I have no yeast?"),
];
const multi = {prompt_full: {text: "how do I bake bread?", chars: 20}, row: 2, meta: {},
  chosen: {model: "big", turns: [...convo, anyTurn("assistant", "Use a sourdough starter instead " + LONG)]},
  rejected: {model: "small", turns: [...convo, anyTurn("assistant", "Use a sourdough starter instead and hope")]}};
eq(P.sharedTurns(multi.chosen, multi.rejected), 3, "the branch point is after the shared turns");
eq(P.candidateTurns(multi.chosen, 3).length, 1, "one candidate turn follows the branch");
eq(P.sideText(multi.chosen, "answer", 3), "Use a sourdough starter instead " + LONG,
  "only the continuation counts as the chosen response");
ok(!P.sideText(multi.chosen, "answer", 3).includes("Mix flour"),
  "the shared assistant turn is not part of either response");
ok(P.renderDPO(multi, "ds").includes("the conversation both answers continue"),
  "and it is rendered once, as the conversation the pair branches from");
// with the shared history excluded the pair is one untruncated turn a side, so
// the opening it does share is claimable
ok(/shared opening <em>/.test(P.gradientSection(multi, 3).replace(/\s+/g, " ")),
  "a multi-turn pair can still show an opening in its continuation");
eq(P.sharedTurns({turns: [{role: "assistant", text: "same", chars: 4}]},
                 {turns: [{role: "assistant", text: "same", chars: 4}]}), 0,
  "the last turn is the candidate answer even when the two lists agree entirely");

// --------------------------------------------------- against the real rows ---
const dataDir = path.join(ROOT, "docs", "data");
const files = fs.readdirSync(dataDir).filter(f => f.includes(".dpo.context."));
ok(files.length > 0, "committed DPO context runs are present to check against");

let rows = 0, openings = 0, zeros = 0, demotions = 0, slowest = 0;
const reasons = {};
for (const f of files){
  for (const rec of JSON.parse(fs.readFileSync(path.join(dataDir, f), "utf8")).records){
    rows++;
    const t0 = process.hrtime.bigint();
    const raw = P.renderDPO(rec, "ds");
    slowest = Math.max(slowest, Number(process.hrtime.bigint() - t0) / 1e6);
    const flat = raw.replace(/\s+/g, " ");

    // no placeholder leaking into the markup (the response text itself may
    // legitimately contain any of these words, so strip it first)
    const markup = raw.replace(/<span class="(pre|ovl)">[^<]*<\/span>/g, "")
                      .replace(/<mark class="\w+">[^<]*<\/mark>/g, "")
                      .replace(/<div class="ctxtext">[^<]*<\/div>/g, "");
    ok(!/NaN|Infinity|undefined/.test(markup) || rows < 0,
       `row ${rec.row} in ${f}: markup carries no bad numbers`);

    const shared = P.sharedTurns(rec.chosen, rec.rejected);
    const stored = P.rawResponseStored(rec.chosen, shared) && P.rawResponseStored(rec.rejected, shared);
    if (claimsOpening(flat)){
      openings++;
      ok(stored, `row ${rec.row}: an opening is claimed only with the raw response stored`);
    }
    if (claimsZero(flat)){
      zeros++;
      ok(stored && !P.sideCut(rec.chosen, shared) && !P.sideCut(rec.rejected, shared) &&
         P.sideText(rec.chosen, "answer", shared) === P.sideText(rec.rejected, "answer", shared) &&
         P.sideText(rec.chosen, "reasoning", shared) === P.sideText(rec.rejected, "reasoning", shared),
         `row ${rec.row}: zero gradient is claimed only on byte-exact untruncated equality`);
    }
    if (flat.includes("that is not a shared opening")){
      demotions++;
      let matched = false;
      for (const [k, pat] of [["multi-turn after the branch", "more than one turn a side after it branches"],
                              ["shared history cut", "the pair branches from is cut at 4,000"],
                              ["thinking differs", "two thinking spans in front of these"],
                              ["think markers normalized", "match as stored, but the dataset drops"],
                              ["candidate cut", "whether the responses stay identical past the stored text"]])
        if (flat.includes(pat)){ reasons[k] = (reasons[k] || 0) + 1; matched = true; break; }
      ok(matched, `row ${rec.row}: a withheld opening names one of the known reasons`);
    }
    // each panel shows its own side's text
    const ops = P.diffPair(P.sideText(rec.chosen, "answer", shared),
                           P.sideText(rec.rejected, "answer", shared));
    ok(sideOf(ops, "a") === P.sideText(rec.chosen, "answer", shared) &&
       sideOf(ops, "b") === P.sideText(rec.rejected, "answer", shared),
       `row ${rec.row}: neither side's text is rewritten`);
  }
}

console.log(`\n${rows} committed DPO rows: ${openings} show a shared opening, ${zeros} carry no gradient, ` +
            `${demotions} explain why a matching opening is not one`);
console.log(`demotion reasons: ${JSON.stringify(reasons)}`);
console.log(`slowest single render: ${slowest.toFixed(1)}ms`);
console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
