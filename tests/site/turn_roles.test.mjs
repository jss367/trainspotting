// What the turn-roles card reads, held to what the file underneath says.
//
// The card's one hard rule is that a stage with no stored roles reads "not
// recorded", never 0% — a flat prompt string says nothing about whether a
// system turn wrapped it. The rest is bookkeeping a bar would hide if it broke:
// the listed system-prompt openings must not claim more rows than have a system
// turn, and the remainder the card names must be what is left over.
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

// ----------------------------------------------------------- the summary ---
const flat = P.rolesSummary({total_rows: 500, columns: []});
eq(flat.recorded, false, "a mix with no role column is not recorded");
eq(flat.share, undefined, "and carries no share to draw as 0%");

const s = P.rolesSummary({
  total_rows: 1000, rows_with_turns: 1000, columns: ["messages"],
  rows_by_role: {user: 1000, assistant: 1000, system: 120},
  system: {rows: 120, distinct_texts: 80, groups: 3, prompts: [
    {preview: "You are a helpful function-calling AI assistant.", rows: 100, distinct_texts: 79, sources: {tools: 100}},
    {preview: "You are Olmo.", rows: 15, distinct_texts: 1, sources: {hardcoded: 15}},
  ]},
});
eq(s.recorded, true, "a mix with messages is recorded");
eq(s.share, 0.12, "the system share is over the rows with stored turns");
eq(s.unlisted, 5, "rows under openings the file did not list are named");
eq(s.unlistedGroups, 1, "and so are those openings");
eq(s.roles.find(r => r.role === "system").frac, 0.12, "a role's bar is its rows over the rows with turns");

// An RL mix stores turns for a few thousand rows and a flat string for the
// rest. The rest are not rows without a system turn.
const partial = P.rolesSummary({
  total_rows: 170000, rows_with_turns: 7000, columns: ["source_prompt"],
  rows_by_role: {user: 7000, system: 700}, system: {rows: 700, prompts: []},
});
eq(partial.share, 0.1, "a share is over the rows that store turns, not the stage");

eq(P.rolesSummary(null), null, "no file, no card");

// --------------------------------------------------- the committed files ---
const files = dataFiles().filter(f => f.endsWith(".roles.json"));
ok(files.length > 0, "the site ships at least one turn-roles file");

for (const name of files){
  const d = read(name);
  const tag = name.replace(".roles.json", "");
  const r = P.rolesSummary(d);
  ok(Array.isArray(d.columns), `${tag}: says which columns it read, even when none`);
  if (!r.recorded){
    ok(d.system === undefined, `${tag}: a not-recorded stage carries no system count to misread as zero`);
    continue;
  }
  ok(r.base <= r.total, `${tag}: rows with turns are a subset of all rows`);
  for (const x of r.roles)
    ok(x.n <= r.base, `${tag}: no role is in more rows than have turns (${x.role})`);
  eq(d.rows_by_role.system || 0, r.system, `${tag}: the system role count and the system block agree`);
  const bySource = Object.values(d.system.by_source).reduce((a, b) => a + b, 0);
  eq(bySource, r.system, `${tag}: system rows by source add up to the system rows`);
  ok(r.unlisted >= 0, `${tag}: the listed openings do not claim more rows than have a system turn`);
  ok(r.distinct >= r.groups, `${tag}: there are at least as many distinct texts as openings`);
  for (const p of r.prompts){
    ok(p.preview.length <= d.preview_chars, `${tag}: an opening is cut at the stated length`);
    const fromSources = Object.values(p.sources).reduce((a, b) => a + b, 0);
    eq(fromSources, p.rows, `${tag}: an opening's rows by source add up`);
  }
}

if (failures){
  console.log(`\n${failures} failure(s)`);
  process.exit(1);
}
