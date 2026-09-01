// One name, one definition, across the whole site script.
//
// docs/index.html is a single <script>, so every top-level `function` shares a
// namespace and the last declaration silently wins — hoisting means the earlier
// one is never called, even from code written above it. That is how
// `budgetCard` broke: the per-question card added later took the name of the
// token-budget card, so `renderModel` called the wrong one, got a Promise where
// it wanted a node, and threw before mounting any section on the page. Nothing
// in the file looked wrong at either site.
//
// Run via pytest (tests/test_site_declarations.py) or directly: node <this file>

import { pageSource } from "./page.mjs";

const js = pageSource();

// Top level only: a nested helper is scoped to its function and may reuse a name.
const declared = new Map();
js.split("\n").forEach((line, i) => {
  const m = /^(?:async )?function ([A-Za-z0-9_$]+)|^(?:const|let|var) ([A-Za-z0-9_$]+)\b/.exec(line);
  if (!m) return;
  const name = m[1] || m[2];
  if (!declared.has(name)) declared.set(name, []);
  declared.get(name).push(i + 1);
});

let failures = 0;
for (const [name, lines] of declared){
  if (lines.length === 1) continue;
  failures++;
  console.log(`FAIL ${name} is declared ${lines.length} times, at lines ${lines.join(", ")}`);
}
if (!failures) console.log(`ok   ${declared.size} top-level names, each declared once`);
process.exit(failures ? 1 : 0);
