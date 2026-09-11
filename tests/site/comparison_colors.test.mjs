import assert from "node:assert/strict";
import { loadPage, read, pageHtml } from "./page.mjs";

const { comparisonColors } = loadPage();
const registry = read("registry.json");
const models = Object.keys(registry).filter(m => registry[m].is_model !== false
  && registry[m].stages.some(s => s.hf_dataset));
assert.ok(models.length >= 9, "exercise the expanded model registry");
const colors = comparisonColors(models);
const slots = models.map(m => colors.get(m));
assert.equal(new Set(slots).size, models.length, "every model gets its own slot");
assert.ok(slots.every(Boolean), "the palette covers every registered comparison model");
assert.deepEqual(slots.slice(0, 3), ["var(--series-1)", "var(--series-2)", "var(--series-3)"]);

// Different custom-property names are insufficient if their theme definitions
// resolve to the same color (or if the added properties were never defined).
const themes = [...pageHtml().matchAll(/:root\s*\{([^}]+)\}/g)];
assert.equal(themes.length, 2, "validate light and dark palettes");
for (const [, theme] of themes){
  const css = new Map([...theme.matchAll(/(--series-\d+):\s*(#[0-9a-f]{6});/gi)]
    .map(([, key, value]) => [key, value.toLowerCase()]));
  const resolved = slots.map(slot => css.get(slot.slice(4, -1)));
  assert.ok(resolved.every(Boolean), "every slot is defined in this theme");
  assert.equal(new Set(resolved).size, models.length, "no two models resolve to the same color");
}
// The renderer retains this full-model map for subset legends and bars.
const subset = [models[1], models[5], models[8]];
assert.deepEqual(subset.map(m => colors.get(m)), [slots[1], slots[5], slots[8]]);
console.log("all compared models have distinct colors in both themes");
