// Boot docs/index.html's script in node, and hand back the functions it defines.
//
// The page is one file: markup, styles and every function the browser runs, in
// one <script>. That is a deliberate choice — it is a static site served off
// GitHub Pages with no build step — and it means a test can only reach a
// function by evaluating the page.
//
// This is the one place that knows how. It was written four times before, once
// per suite plus a variant in test_searchindex.py that lifted a single function
// out by brace-matching its source; that variant broke the moment the function
// it lifted started calling a helper defined elsewhere in the file, which is
// exactly the refactor the page needed. A function pulled out of its file is a
// copy, and copies are the thing these tests exist to catch.
//
// `eval` here runs the repository's own committed script, read from disk. It is
// not input, and there is nothing to inject: the same code is what the browser
// executes when someone opens the page.
import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..", "..");

// Enough of a DOM that the function declarations land. The page's top level
// touches document and fetches its data; neither is served here, so the boot
// fails — expected, and swallowed, because a suite's exit code should be about
// its assertions.
const el = () => ({ addEventListener(){}, appendChild(){}, querySelectorAll: () => [],
  querySelector: () => el(), style: {setProperty(){}}, dataset: {},
  set innerHTML(v){}, set onclick(v){}, set textContent(v){} });

// Every name a suite asks for, collected in one place so a rename fails loudly
// here rather than as `undefined is not a function` halfway through a suite.
const EXPORTS = [
  // the DPO gradient panel
  "diffPair", "opChars", "uniqueChars", "sideText", "sideCut", "demotePrefix",
  "gradientSection", "rawResponseStored", "renderDPO", "sharedTurns",
  "candidateTurns", "postBranchContext",
  // the language card
  "langCode", "columnLangShares", "langSummary", "langColumn", "wilson",
  // the pipeline treemap
  "childrenOf", "treemapLayout",
  // the search box
  "searchFields", "scanRecords", "branchPoint", "matchIndex",
];

// The committed data the page serves, by filename. Three suites read it and
// each had written this line for itself.
export const DATA = path.join(ROOT, "docs", "data");
export const read = f => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8"));
export const dataFiles = () => fs.readdirSync(DATA);


// The page as text, for the handful of assertions that are about the source
// rather than about what a function returns — a claim in the prose that the
// data no longer supports is a bug the same way a wrong number is.
export const pageHtml = () => fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
export const pageSource = () => pageHtml().match(/<script>([\s\S]*)<\/script>/)[1];


export function loadPage(){
  const js = pageSource();
  globalThis.document = { getElementById: el, querySelectorAll: () => [],
                          createElement: el, body: el() };
  globalThis.location = { hash: "" };
  globalThis.fetch = async () => ({ ok: false });
  // Swallow the boot's own fetch failures and nothing else.
  //
  // The page boots by fetching its data files; nothing serves them here, so
  // `fetchJSON` throws `new Error(<filename>)` and that rejection is expected.
  // A blanket no-op listener also swallows every *other* unhandled rejection
  // for the rest of the process, though — which under Node's default
  // `--unhandled-rejections=throw` is the one thing that would have failed a
  // suite that broke asynchronously. A suite going quietly green is worse than
  // one that crashes, so anything that is not a data filename is rethrown and
  // takes the process down as it would have anyway.
  const bootFailure = e => e instanceof Error && /^[\w.-]+\.json$/.test(e.message);
  process.on("unhandledRejection", e => { if (!bootFailure(e)) throw e; });
  const missing = [];
  eval(js + `
;globalThis.__PAGE = {
  // Not a function the page defines: the language card reads its display names
  // out of a module-scope cache the page fills at boot, and nothing serves that
  // file here. This is the one hook a suite needs into page internals.
  setLangNames: v => { LANG_NAMES = v; LANG_CODES = null; },
};
for (const name of ${JSON.stringify(EXPORTS)}) {
  try { globalThis.__PAGE[name] = eval(name); } catch { globalThis.__MISSING = (globalThis.__MISSING || []).concat(name); }
}`);
  missing.push(...(globalThis.__MISSING || []));
  if (missing.length)
    throw new Error("docs/index.html no longer defines: " + missing.join(", "));
  // Present, but not necessarily the one the author meant: these names resolve
  // after the page has run, so a name declared twice hands back whichever
  // declaration won. `declarations.test.mjs` is what rules that out, and it has
  // to, because a suite testing the loser would pass while the browser ran the
  // winner. That is how `budgetCard` shipped broken.
  return globalThis.__PAGE;
}

