// Training measurements have their own provenance and never inherit sample-label shares.
const esc = value => String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const finite = value => typeof value === "number" && Number.isFinite(value);
const number = value => !finite(value) ? "Not measured" : value === 0 ? "0" : value < 0.001 ? value.toExponential(2) : value.toLocaleString(undefined, {maximumSignificantDigits: 3});
const percent = value => finite(value) ? `${number(value * 100)}%` : "Undefined (zero starting norm)";
const names = {pretrain: "Pretraining", midtrain: "Midtraining", "long-context": "Long-context training", sft: "Instruction tuning", dpo: "Preference training", rlvr: "Reinforcement learning"};
const stageName = stage => names[stage] || stage;
const sourceLink = (url, label) => /^https:\/\//.test(url || "") ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>` : esc(label);
const checkpointLink = point => sourceLink(`https://huggingface.co/${point.repo}/tree/${point.resolved_revision}`, point.label || point.revision);

export function checkpointChart(points){
  if (!points || points.length < 2) return "";
  const maximum = Math.max(...points.flatMap(p => [p.net_rms, p.observed_path_rms]).filter(finite), 0);
  if (!points.every(p => finite(p.net_rms) && finite(p.observed_path_rms))) return "";
  const top = maximum || 1;
  const x = i => 65 + i * 590 / (points.length - 1);
  const y = value => 155 - value / top * 125;
  const line = field => points.map((p, i) => `${x(i)},${y(p[field])}`).join(" ");
  return `<div class="change-legend"><span>● Net change from first checkpoint</span><span>┄ Movement between saved checkpoints</span></div>
    <p class="change-scroll-hint">Scroll horizontally to see all checkpoints →</p>
    <div class="change-chart-wrap" tabindex="0" role="region" aria-label="Scrollable checkpoint chart"><svg class="change-chart" viewBox="0 0 720 205" role="img" aria-label="Weight movement across saved checkpoints. Horizontal positions are equally spaced checkpoints, not elapsed training time.">
      ${[0, 0.5, 1].map(f => `<line x1="65" y1="${y(top * f)}" x2="655" y2="${y(top * f)}" class="change-grid"/><text x="55" y="${y(top * f) + 4}" text-anchor="end">${number(maximum * f)}</text>`).join("")}
      <polyline points="${line("observed_path_rms")}" class="change-path"/>
      <polyline points="${line("net_rms")}" class="change-net"/>
      ${points.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.net_rms)}" r="4" class="change-dot"><title>${esc(p.label || p.revision)}: net ${number(p.net_rms)}, observed movement ${number(p.observed_path_rms)}</title></circle><text x="${x(i)}" y="178" text-anchor="middle">${esc(p.label || p.revision)}</text>`).join("")}
      <text x="360" y="201" text-anchor="middle">Checkpoint order · equally spaced · root-mean-square weight units</text>
    </svg></div>`;
}

function weightDetail(stage){
  if (!stage) return `<div class="change-empty"><strong>Checkpoint weights have not been compared for this phase.</strong><p>Measuring net change needs its starting and ending weights. Dataset sizes and learning rates cannot supply this number.</p></div>`;
  const points = stage.checkpoints;
  const first = points[0], last = points.at(-1);
  return `<div class="change-stats">
      <div><span>Net weight change</span><strong>${number(stage.net.rms)}</strong><small>root-mean-square distance</small></div>
      <div><span>Relative to starting weights</span><strong>${percent(stage.net.relative_l2)}</strong><small>weight distance ÷ starting weight norm</small></div>
      <div><span>Observed movement</span><strong>≥ ${number(stage.observed_path_rms)}</strong><small>lower bound from ${points.length} checkpoints</small></div>
    </div>
    ${checkpointChart(points)}
    <p class="change-note">${stage.coverage === "full_stage" ? "Phase boundaries included." : "Partial phase: these checkpoints do not cover the whole stage."}
      Updates between saved checkpoints are unobserved. Movement is not a measure of knowledge gained.</p>
    <details class="change-details"><summary>Which layers changed?</summary>
      <div class="change-table-wrap"><table class="change-table"><thead><tr><th>Parameter group</th><th>Net change</th><th>Relative change</th></tr></thead><tbody>
      ${stage.net.layers.map(l => `<tr><th>${esc(l.name)}</th><td>${number(l.rms)}</td><td>${percent(l.relative_l2)}</td></tr>`).join("")}</tbody></table></div>
    </details>
    <p class="change-note">${checkpointLink(first)} → ${checkpointLink(last)} · ${sourceLink(stage.lineage_source, "Checkpoint ancestry")}</p>`;
}

export function probeHeatmap(stages, measurements){
  const topics = [...new Set((measurements?.stages || []).flatMap(s => (s.behavior?.topics || []).map(t => t.name)))];
  if (!topics.length) return `<div class="change-empty"><strong>No fixed-prompt comparisons yet.</strong><p>Each checkpoint must answer the same inputs before we can measure how its predictions changed.</p></div>`;
  return `<p class="change-note">Prediction differences by topic, from each phase’s first to last measured checkpoint. Darker cells mean more change, not better performance. The scale is 0–1 bit of Jensen–Shannon divergence.</p>
    <div class="change-table-wrap"><table class="change-table change-heatmap"><thead><tr><th>Probe topic</th>${stages.map(s => `<th>${esc(stageName(s.stage))}</th>`).join("")}</tr></thead><tbody>
    ${topics.map(topic => `<tr><th>${esc(topic)}</th>${stages.map(s => {
      const measured = measurements.stages.find(m => m.stage === s.stage);
      const value = measured?.behavior?.topics.find(t => t.name === topic)?.js_bits;
      return finite(value) ? `<td style="--intensity:${Math.min(100, Math.max(0, value * 45)).toFixed(2)}%">${number(value)}${measured.coverage !== "full_stage" ? "*" : ""}</td>` : `<td class="change-missing">Not measured</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="change-note">${measurements.probes.length} fixed prompts; one next-token distribution per prompt. These topic labels do not measure refusal rates, honesty, or accuracy. * Partial phase.</p>`;
}

function examples(stage){
  if (!stage) return "";
  const first = stage.checkpoints[0], last = stage.checkpoints.at(-1);
  if (!first.examples?.length) return "";
  return `<details class="change-details"><summary>Read before-and-after continuations</summary>
    <p class="change-note">Greedy continuations, at most 16 new tokens, using identical raw prompts without chat templates. Illustrations only; instruction-tuned models may expect a chat template.</p>
    ${first.examples.map((e, i) => `<div class="change-example"><strong>${esc(e.topic)}</strong><pre>${esc(e.prompt)}</pre><div><p><span>Before</span>${esc(e.continuation) || "(No visible text)"}</p><p><span>After</span>${esc(last.examples?.[i]?.continuation) || "(No visible text)"}</p></div></div>`).join("")}</details>`;
}

export function evaluationTable(stages, evaluations){
  if (!evaluations?.benchmarks?.length) return `<div class="change-empty"><strong>No published phase comparison has been imported for this model.</strong><p>Checkpoint probes and published benchmark scores are separate measurements.</p></div>`;
  return `<p class="change-note">Publisher-reported benchmark scores, on a 0–100 scale. Signed changes compare adjacent phases in this table, in percentage points. Each benchmark keeps its own evaluation protocol.</p>
    <div class="change-table-wrap"><table class="change-table change-heatmap"><thead><tr><th>Benchmark</th>${stages.map(s => `<th>${esc(stageName(s.stage))}</th>`).join("")}</tr></thead><tbody>
    ${evaluations.benchmarks.map(b => `<tr><th>${esc(b.name)}<small>${esc(b.category)}</small></th>${stages.map((s, i) => {
      const value = b.scores[s.stage];
      const previous = i > 0 ? b.scores[stages[i - 1].stage] : null;
      const delta = finite(value) && finite(previous) ? value - previous : null;
      return finite(value) ? `<td style="--intensity:${Math.min(100, Math.max(0, value * 0.45))}%">${value.toFixed(1)}${delta !== null ? `<small class="${delta < 0 ? "change-decrease" : ""}">${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp</small>` : "<small>No earlier score</small>"}</td>` : `<td class="change-missing">Not reported</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="change-note">${esc(evaluations.note)} ${sourceLink(evaluations.source, "Read the source table")}</p>`;
}

export function trainingChangeHTML(model, measurements, evaluations, state){
  const stages = model.stages;
  const measured = measurements?.stages || [];
  const current = measured.find(s => s.stage === state.stage);
  const maxRms = Math.max(...measured.map(s => s.net.rms), 0);
  return `<div class="change-heading"><div><div class="change-kicker">Model development</div><h2>How training changed the model</h2></div><span class="change-coverage">${measured.length} of ${stages.length} phases with weight measurements</span></div>
    <p class="sub">Follow the model from pretraining to its final form. Compare changes in its weights, predictions, and published benchmark scores.</p>
    ${!measured.length ? `<p class="change-note"><a href="#pythia-70m-deduped">Explore a measured training run: Pythia 70M →</a></p>` : ""}
    <div class="change-phases" aria-label="Training phase">${stages.map((s, i) => {
      const entry = measured.find(m => m.stage === s.stage);
      return `<button type="button" data-change-stage="${esc(s.stage)}" aria-pressed="${state.stage === s.stage}"><span class="change-step">${String(i + 1).padStart(2, "0")}</span><strong>${esc(stageName(s.stage))}</strong><span>${s.tokens ? `${number(s.tokens / 1e9)}B training tokens` : "Post-training"}</span><div class="change-mini-track">${entry ? `<i style="width:${maxRms ? entry.net.rms / maxRms * 100 : 0}%"></i>` : ""}</div><small>${entry ? `Net ${number(entry.net.rms)}${entry.coverage === "partial" ? " · partial phase" : ""}` : "Weights not measured"}</small></button>`;
    }).join("")}</div>
    <div class="change-modes" role="group" aria-label="Change measurement">${[["weights", "Weight movement"], ["probes", "Prediction changes"], ["evaluations", "Published evaluations"]].map(([key, label]) => `<button type="button" data-change-mode="${key}" aria-pressed="${state.mode === key}">${label}</button>`).join("")}</div>
    <div class="change-content">${state.mode === "weights" ? `<h3>${esc(stageName(state.stage))}</h3>${weightDetail(current)}` : state.mode === "probes" ? `${probeHeatmap(stages, measurements)}<h3>${esc(stageName(state.stage))}</h3>${current ? examples(current) : "<p class=\"change-note\">No before-and-after examples measured for this phase.</p>"}` : evaluationTable(stages, evaluations)}</div>
    <details class="change-details change-method"><summary>How to read these measurements</summary>
      <p><b>Net change</b> is the root mean square of the ending weight minus the starting weight across all unique parameters. Relative change divides their Euclidean distance by the starting weight norm. Neither is a percentage of knowledge learned.</p>
      <p><b>Observed movement</b> sums root-mean-square distances between saved checkpoints. It is a lower bound on the full training path. Denser checkpoint coverage can increase it; compare only matching checkpoint schedules. Parameter distances require aligned weights from the same model lineage.</p>
      <p><b>Prediction changes</b> compare next-token probabilities on identical fixed prompts, using mean Jensen–Shannon divergence in bits. Zero means identical predictions; larger values mean more change. This small exploratory probe set is not a capability or safety benchmark.</p>
      <p><b>Missing values</b> mean no measurement is available. They never mean zero change. Phase widths show order, not elapsed time; mini bars compare measured net distances within this model. Merged checkpoints must be identified in the ancestry source.</p>
      ${measurements ? `<p>Measured ${esc(measurements.measured_at.slice(0, 10))} · ${measurements.probes.length} prompts · ${esc(measurements.dtype)} · ${esc(measurements.device)}. Probe fingerprint: <code>${esc(measurements.probe_sha256)}</code></p>` : ""}
      <p><a href="training-change.md">Measurement methods and reproduction</a></p>
    </details>`;
}

export function trainingChangeCard(model, measurements, evaluations){
  if (model.is_model === false) return null;
  const card = document.createElement("section");
  card.className = "card training-change";
  card.setAttribute("aria-label", "How training changed the model");
  const state = {stage: measurements?.stages?.[0]?.stage || model.stages[0].stage,
                 mode: measurements?.stages?.length ? "weights" : evaluations ? "evaluations" : "weights"};
  const render = () => { card.innerHTML = trainingChangeHTML(model, measurements, evaluations, state); };
  card.addEventListener("click", event => {
    const button = event.target.closest("button[data-change-stage],button[data-change-mode]");
    if (!button || !card.contains(button)) return;
    const attribute = button.hasAttribute("data-change-stage") ? "data-change-stage" : "data-change-mode";
    const value = button.getAttribute(attribute);
    state[attribute === "data-change-stage" ? "stage" : "mode"] = value;
    render();
    // Replacing innerHTML must not drop keyboard focus back to the document.
    [...card.querySelectorAll(`button[${attribute}]`)].find(b => b.getAttribute(attribute) === value)?.focus({preventScroll: true});
  });
  render();
  return card;
}
