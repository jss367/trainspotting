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

import { loadPage, read, dataFiles } from "./page.mjs";

const P = loadPage();

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
     chosen: {model: "b", turns: [{role: "assistant", text: "hi \u{1F600} there friend", chars: 17, raw: true}]},
     rejected: {model: "s", turns: [{role: "assistant", text: "hi \u{1F600} there pal", chars: 14, raw: true}]}})
   .replace(/\s+/g, " ").includes("shared opening <em>"),
  "a pair with an emoji is not mistaken for a truncated one");

eq(P.demotePrefix([{t: "prefix", a: "abc", b: "abc"}, {t: "same", a: "def", b: "def"},
                   {t: "chosen", a: "x", b: ""}]),
  [{t: "same", a: "abcdef", b: "abcdef"}, {t: "chosen", a: "x", b: ""}],
  "demoting an opening merges it into the overlap that follows");

// ------------------------------------------------------------- the claims ---
// `raw: true` is what the exporter writes when a turn's stored text is the
// content itself; pass raw: false to model a turn that was cut or normalized.
const turn = (text, {chars, reasoning, raw = true} = {}) => ({role: "assistant", text,
  chars: chars ?? text.length, ...(raw ? {raw: true} : {}),
  ...(reasoning ? {reasoning: {text: reasoning, chars: reasoning.length}} : {})});
const pair = (c, r) => ({prompt_full: {text: "p", chars: 1}, row: 1, meta: {},
  chosen: {model: "big", turns: [turn(...[].concat(c))]},
  rejected: {model: "small", turns: [turn(...[].concat(r))]}});
const panel = rec => P.gradientSection(rec).replace(/\s+/g, " ");
const claimsOpening = out => /shared opening <em>/.test(out);
const claimsZero = out => out.includes("gradient of exactly zero");

let out = panel(pair([LONG + " alpha"], [LONG + " beta"]));
ok(claimsOpening(out), "an untruncated pair with no thinking span can show an opening");

out = panel(pair([LONG + " alpha", {chars: 9000, raw: false}], [LONG + " beta"]));
ok(!claimsOpening(out), "a truncated side shows no opening");
ok(out.includes("one side is cut at 4,000 characters, so whether the responses stay identical"),
  "and names truncation as the reason");

out = panel(pair([LONG + " alpha", {reasoning: "same trace", raw: false}],
                 [LONG + " beta", {reasoning: "same trace", raw: false}]));
ok(!claimsOpening(out), "thinking spans that match as stored still show no opening");
ok(out.includes("drops the"), "and name the <think> normalization as the reason");

out = panel(pair([LONG + " alpha", {reasoning: "one trace", raw: false}],
                 [LONG + " beta", {reasoning: "a different trace", raw: false}]));
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

ok(!claimsZero(panel(pair(["identical text", {chars: 9000, raw: false}],
                          ["identical text", {chars: 9000, raw: false}]))),
  "nor is an identical-looking pair that was cut");

// The exporter's flag is the only proof the stored text is what was scored.
// "<think></think>same answer" and "same answer" both arrive with no reasoning
// field and the same text; only the flag separates them.
ok(!claimsZero(panel(pair(["same answer", {raw: false}], ["same answer"]))),
  "a normalized response is not called zero-gradient against an unmodified one");
ok(!/shared opening <em>/.test(panel(pair([LONG + " alpha", {raw: false}], [LONG + " beta"]))),
  "nor does a normalized response show a shared opening");

// ------------------------------------------------------ the branch point ---
// A multi-turn pair shares its leading turns with itself; those are the
// conversation both candidates answer in, not either candidate.
const anyTurn = (role, text) => ({role, text, chars: text.length, raw: true});
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

// A pair can branch on a turn the model did not write. The answers behind it are
// then replies to different text, however alike they look, and the turn that
// made them differ has to be visible.
const ANSWER = "Use a sourdough starter instead " + LONG;
const forked = {prompt_full: {text: "q", chars: 1}, row: 3, meta: {},
  chosen: {model: "big", turns: [convo[0], anyTurn("user", "no yeast, what now?"), anyTurn("assistant", ANSWER)]},
  rejected: {model: "small", turns: [convo[0], anyTurn("user", "no flour, what now?"), anyTurn("assistant", ANSWER)]}};
eq(P.sharedTurns(forked.chosen, forked.rejected), 1, "the branch is at the differing user turn");
eq(P.postBranchContext(forked.chosen, 1).map(t => t.text), ["no yeast, what now?"],
  "the differing user turn is post-branch context, not part of the response");
const forkedOut = P.gradientSection(forked, 1).replace(/\s+/g, " ");
ok(!/shared opening <em>/.test(forkedOut),
  "identical answers to different questions claim no shared opening");
ok(!forkedOut.includes("gradient of exactly zero"),
  "nor zero gradient, though the two answers are byte-identical");
ok(forkedOut.includes("branches on a turn the model did not write"),
  "and the reason names the branch turn");
ok(P.renderDPO(forked, "ds").includes("no yeast, what now?"),
  "the turn that made the answers differ is rendered, not filtered away");

// A turn whose whole output is a tool call is stored empty and never raw, so two
// completions differing only there cannot be read as the same conversation.
const toolTurn = {role: "assistant", text: "", chars: 0, omitted: ["tool_calls"]};
const tooled = {prompt_full: {text: "q", chars: 1}, row: 4, meta: {},
  chosen: {model: "big", turns: [convo[0], toolTurn, anyTurn("assistant", "the same answer " + LONG)]},
  rejected: {model: "small", turns: [convo[0], toolTurn, anyTurn("assistant", "the same answer " + LONG)]}};
const tooledOut = P.gradientSection(tooled, P.sharedTurns(tooled.chosen, tooled.rejected))
  .replace(/\s+/g, " ");
eq(P.sharedTurns(tooled.chosen, tooled.rejected), 1,
  "a turn carrying output the record does not keep ends the provable history");
ok(!/shared opening <em>/.test(tooledOut),
  "a tool-call turn blocks the shared-opening claim");
ok(!tooledOut.includes("gradient of exactly zero"),
  "and blocks zero gradient, though every stored byte matches");
ok(P.renderDPO(tooled, "ds").includes("tool_calls"),
  "and the view says which field the turn carried but the record does not keep");

// The reason given has to be the true one. A sole candidate carrying a tool call
// alongside its text is not truncated, and neither is one whose empty thinking
// span was normalized away.
const auxAnswer = {prompt_full: {text: "q", chars: 1}, row: 5, meta: {},
  chosen: {model: "big", turns: [convo[0],
    {role: "assistant", text: LONG + " alpha", chars: (LONG + " alpha").length, omitted: ["tool_calls"]}]},
  rejected: {model: "small", turns: [convo[0], anyTurn("assistant", LONG + " beta")]}};
const auxOut = P.gradientSection(auxAnswer, 1).replace(/\s+/g, " ");
ok(!auxOut.includes("one side is cut at 4,000 characters, so whether"),
  "an answer carrying a tool call is not reported as truncated");
ok(auxOut.includes("beside its text, which this record does not keep"),
  "it is reported as carrying output the record does not keep");

const normalized = {prompt_full: {text: "q", chars: 1}, row: 6, meta: {},
  chosen: {model: "big", turns: [convo[0],
    {role: "assistant", text: LONG + " alpha", chars: (LONG + " alpha").length}]},
  rejected: {model: "small", turns: [convo[0], anyTurn("assistant", LONG + " beta")]}};
const normOut = P.gradientSection(normalized, 1).replace(/\s+/g, " ");
ok(!normOut.includes("one side is cut at 4,000 characters, so whether"),
  "a normalized answer is not reported as truncated either");
ok(normOut.includes("does not hold an answer as written"),
  "it is reported as normalized");

// An empty turn on one side only shifts everything after it out of alignment, so
// the two completions cannot read as the same conversation.
const emptyTurn = {role: "assistant", text: "", chars: 0, raw: true};
const lopsided = {prompt_full: {text: "q", chars: 1}, row: 7, meta: {},
  chosen: {model: "big", turns: [convo[0], emptyTurn, anyTurn("assistant", "the same answer " + LONG)]},
  rejected: {model: "small", turns: [convo[0], anyTurn("assistant", "the same answer " + LONG)]}};
eq(P.sharedTurns(lopsided.chosen, lopsided.rejected), 1,
  "an empty turn on one side ends the provable history");
ok(!P.gradientSection(lopsided, 1).replace(/\s+/g, " ").includes("gradient of exactly zero"),
  "and the answers behind it are not called a zero-gradient pair");

// Shared context the prompt section cannot show has to be rendered somewhere.
const sysShared = {role: "system", text: "You are terse.", chars: 14, raw: true};
const systemPair = {prompt_full: {text: "q", chars: 1}, row: 8, meta: {},
  chosen: {model: "big", turns: [sysShared, anyTurn("user", "q"), anyTurn("assistant", "A " + LONG)]},
  rejected: {model: "small", turns: [sysShared, anyTurn("user", "q"), anyTurn("assistant", "B " + LONG)]}};
const sysOut = P.renderDPO(systemPair, "ds");
ok(sysOut.includes("the conversation both answers continue"),
  "a shared system turn is shown even with no shared assistant turn");
ok(sysOut.includes("You are terse."),
  "and its text appears, since prompt extraction keeps only the first user message");

// `prompt_full` stands for the opening user turn and no other, so a later turn
// that repeats it still has to be shown — with its own role and position.
const echoed = {prompt_full: {text: "say it twice", chars: 12}, row: 10, meta: {},
  chosen: {model: "big", turns: [anyTurn("user", "say it twice"), anyTurn("system", "say it"),
                                 anyTurn("user", "go"), anyTurn("assistant", "A " + LONG)]},
  rejected: {model: "small", turns: [anyTurn("user", "say it twice"), anyTurn("system", "say it"),
                                     anyTurn("user", "go"), anyTurn("assistant", "B " + LONG)]}};
ok(P.renderDPO(echoed, "ds").includes("the conversation both answers continue"),
  "a shared turn whose text reads as part of the prompt is still shown");

// A side with no stored thinking span may be carrying its reasoning in a field
// the record does not keep, which is not the same as not reasoning.
const hiddenReasoning = {prompt_full: {text: "q", chars: 1}, row: 11, meta: {},
  chosen: {model: "big", turns: [convo[0], turn("the answer", {reasoning: "weighing it up", raw: false})]},
  rejected: {model: "small", turns: [convo[0],
    {role: "assistant", text: "another answer", chars: 14, omitted: ["reasoning_content"]}]}};
const hiddenOut = P.gradientSection(hiddenReasoning, 1).replace(/\s+/g, " ");
ok(hiddenOut.includes("carries its reasoning in a field this record does not keep"),
  "a side whose reasoning is stored elsewhere is not called reasoning-free");
ok(!hiddenOut.includes("with nothing opposite it"),
  "and the span is not described as facing nothing");

// A whitespace-only <think> span leaves no reasoning field and no trace but the
// missing raw flag, so the other side cannot be called reasoning-free either.
const whitespaceThink = {prompt_full: {text: "q", chars: 1}, row: 12, meta: {},
  chosen: {model: "big", turns: [convo[0], turn("the answer", {reasoning: "weighing it up", raw: false})]},
  rejected: {model: "small", turns: [convo[0], turn("another answer", {raw: false})]}};
const wsOut = P.gradientSection(whitespaceThink, 1).replace(/\s+/g, " ");
ok(wsOut.includes("not held as written"),
  "a normalized quiet side is reported as unreadable, not as silent");
ok(!wsOut.includes("with nothing opposite it"),
  "and the span is not described as facing nothing");

// A thinking span on one side only is described as such.
const oneSided = {prompt_full: {text: "q", chars: 1}, row: 9, meta: {},
  chosen: {model: "big", turns: [convo[0], turn("the answer", {reasoning: "weighing it up", raw: false})]},
  rejected: {model: "small", turns: [convo[0], anyTurn("assistant", "another answer")]}};
const oneSidedOut = P.gradientSection(oneSided, 1).replace(/\s+/g, " ");
ok(oneSidedOut.includes("Only the chosen response carries a thinking span"),
  "a one-sided thinking span is not described as two");
ok(!oneSidedOut.includes("Both responses carry a thinking span"),
  "and the both-sided sentence does not appear");

// A matched span holds each side's own text, so its two lengths can differ; the
// legend must not print the chosen side's number as though it covered both.
const WIDE = WS_A.replace(/ /g, "  ");   // same words, wider gaps: a longer span
const wsPair = {prompt_full: {text: "q", chars: 1}, row: 13, meta: {},
  chosen: {model: "big", turns: [convo[0], anyTurn("assistant", WS_A + "alpha")]},
  rejected: {model: "small", turns: [convo[0], anyTurn("assistant", WIDE + "beta")]}};
const wsLegend = P.gradientSection(wsPair, 1).replace(/\s+/g, " ");
ok(/matched later <em>\d+ \/ \d+ ch<\/em>/.test(wsLegend),
  "a matched span of differing length reports both sides' counts");
ok(!/matched later <em>\d+ ch<\/em>/.test(wsLegend),
  "and does not report one as though it covered both");

// The summary sits under the answers diff and must not speak for the response as
// a whole: a think row's reasoning is diffed separately and counted nowhere here.
const thinkSummary = {prompt_full: {text: "q", chars: 1}, row: 14, meta: {},
  chosen: {model: "big", turns: [convo[0], turn("A " + LONG, {reasoning: "a long trace of reasoning here", raw: false})]},
  rejected: {model: "small", turns: [convo[0], turn("B " + LONG, {reasoning: "a different trace entirely", raw: false})]}};
const thinkOut = P.gradientSection(thinkSummary, 1).replace(/\s+/g, " ");
ok(thinkOut.includes("none of these counts include it") ||
   thinkOut.includes("none of this counts"),
  "a summary over the answers says the thinking span is not in it");
ok(!/the two responses have no wording in common/i.test(thinkOut),
  "and does not claim the responses share nothing when only the answers were compared");
ok(!/chosen response's \d/.test(thinkOut),
  "nor reports an answer-only total as the response's length");

// --------------------------------------------------- against the real rows ---
const files = dataFiles().filter(f => f.includes(".dpo.context."));
ok(files.length > 0, "committed DPO context runs are present to check against");

let rows = 0, openings = 0, zeros = 0, demotions = 0, slowest = 0;
const reasons = {};
for (const f of files){
  for (const rec of read(f).records){
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
                              ["branch point not provable", "reaches a turn this record does not hold as written"],
                              ["answer carries omitted output", "beside its text, which this record does not keep"],
                              ["answer normalized", "does not hold an answer as written"],
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
