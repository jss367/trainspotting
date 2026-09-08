// The site's script, importable from a test.
//
// docs/js/app.js is an ES module whose top level touches neither the DOM nor
// the network — everything that does lives in its exported boot() — so a suite
// imports it like any other module and calls the functions the browser runs.
// Nothing is eval'd and nothing is lifted out of a file by name: a function
// pulled out of its file is a copy, and copies are the thing these tests exist
// to catch.
//
// It was not always so. The page used to be one <script> in index.html, and
// this file booted it under `eval` with a DOM stub, once per suite before it
// was consolidated here. The module split is what made the stub unnecessary.
import fs from "fs";
import path from "path";
import * as app from "../../docs/js/app.js";

const ROOT = path.resolve(import.meta.dirname, "..", "..");

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
  // the one hook into page internals: the language card reads its display
  // names from a module-scope cache boot() fills, and nothing serves that
  // file here.
  "setLangNames",
];

// The committed data the page serves, by filename. Three suites read it and
// each had written this line for itself.
export const DATA = path.join(ROOT, "docs", "data");
export const read = f => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8"));
export const dataFiles = () => fs.readdirSync(DATA);

// The page and its script as text, for the handful of assertions that are about
// the source rather than about what a function returns — a claim in the prose
// that the data no longer supports is a bug the same way a wrong number is.
export const pageHtml = () => fs.readFileSync(path.join(ROOT, "docs", "index.html"), "utf8");
export const pageSource = () => fs.readFileSync(path.join(ROOT, "docs", "js", "app.js"), "utf8");

export function loadPage(){
  const missing = EXPORTS.filter(name => typeof app[name] !== "function");
  if (missing.length)
    throw new Error("docs/js/app.js no longer exports: " + missing.join(", "));
  return app;
}
