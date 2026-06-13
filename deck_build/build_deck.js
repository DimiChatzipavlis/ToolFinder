const pptxgen = require("pptxgenjs");
const path = require("path");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaWindowMaximize, FaRandom, FaExclamationTriangle } = require("react-icons/fa");

async function iconPng(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(IconComponent, { color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

const FIG = path.resolve(__dirname, "..", "experiments", "results", "figures");
const fig = (name) => path.join(FIG, name);

// ---- palette (drawn from the report figures) ----
const NAVY = "1B2A4A";      // dark backgrounds
const BLUE = "4878A8";      // primary accent (matches fig bars)
const ORANGE = "E1812C";    // sharp accent (matches fig orange series)
const INK = "1E293B";       // body text on light
const MUTED = "64748B";     // captions / secondary
const LIGHT = "F4F6F9";     // content background
const WHITE = "FFFFFF";
const CARDLINE = "DCE3EC";

const HEAD = "Georgia";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";       // 10 x 5.625 in
pres.author = "ToolFinder";
pres.title = "ToolFinder";
const W = 10, H = 5.625;

const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.16 });

// aspect-ratio-preserving image box: fit within (maxW,maxH), return centered coords
function fitImage(slide, file, origW, origH, boxX, boxY, maxW, maxH, opts = {}) {
  const ar = origW / origH;
  let w = maxW, h = maxW / ar;
  if (h > maxH) { h = maxH; w = maxH * ar; }
  const x = boxX + (maxW - w) / 2;
  const y = boxY + (maxH - h) / 2;
  if (opts.frame) {
    slide.addShape(pres.shapes.RECTANGLE, { x: x - 0.06, y: y - 0.06, w: w + 0.12, h: h + 0.12, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  }
  slide.addImage({ path: file, x, y, w, h });
  return { x, y, w, h };
}

function footer(slide, n) {
  slide.addText("ToolFinder — Dense Retrieval for Open-Set MCP Tool Routing", { x: 0.55, y: H - 0.4, w: 7.5, h: 0.3, fontFace: BODY, fontSize: 8, color: MUTED, margin: 0 });
  slide.addText(String(n), { x: W - 0.9, y: H - 0.4, w: 0.4, h: 0.3, fontFace: BODY, fontSize: 9, color: MUTED, align: "right", margin: 0 });
}

// content-slide scaffold: left accent bar + section chip + title
function contentHeader(slide, chip, title) {
  slide.background = { color: LIGHT };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: BLUE } });
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.55, y: 0.42, w: 1.0 + chip.length * 0.085, h: 0.34, fill: { color: NAVY }, rectRadius: 0.05 });
  slide.addText(chip.toUpperCase(), { x: 0.55, y: 0.42, w: 1.0 + chip.length * 0.085, h: 0.34, fontFace: BODY, fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle", charSpacing: 1, margin: 0 });
  slide.addText(title, { x: 0.52, y: 0.84, w: 9.0, h: 0.62, fontFace: HEAD, fontSize: 26, bold: true, color: NAVY, margin: 0 });
}

function statCard(slide, x, y, w, big, label, color) {
  const h = 1.35;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h, fill: { color } });
  slide.addText(big, { x: x + 0.18, y: y + 0.12, w: w - 0.3, h: 0.62, fontFace: HEAD, fontSize: 30, bold: true, color, valign: "middle", margin: 0 });
  slide.addText(label, { x: x + 0.18, y: y + 0.74, w: w - 0.3, h: 0.5, fontFace: BODY, fontSize: 11, color: INK, valign: "top", margin: 0 });
}

async function main() {
const ICONS = {
  bloat: await iconPng(FaWindowMaximize, "#FFFFFF"),
  middle: await iconPng(FaRandom, "#FFFFFF"),
  small: await iconPng(FaExclamationTriangle, "#FFFFFF"),
};

// ============================================================= SLIDE 1 — TITLE
let s = pres.addSlide();
s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.13, fill: { color: ORANGE } });
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: H - 0.13, w: W, h: 0.13, fill: { color: BLUE } });
s.addText("ToolFinder", { x: 0.8, y: 1.55, w: 8.4, h: 1.0, fontFace: HEAD, fontSize: 54, bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("Dense Retrieval for Open-Set MCP Tool Routing", { x: 0.8, y: 2.62, w: 8.4, h: 0.5, fontFace: BODY, fontSize: 20, color: "CADCFC", align: "center", margin: 0 });
s.addText("A leakage-controlled benchmark and a trained two-model comparison", { x: 0.8, y: 3.12, w: 8.4, h: 0.4, fontFace: BODY, fontSize: 13, italic: true, color: "9FB3D1", align: "center", margin: 0 });
// --- credentials placeholder (user fills) ---
s.addText("[ Your name · Student ID · Course · Supervisor · Date ]", { x: 0.8, y: 4.15, w: 8.4, h: 0.4, fontFace: BODY, fontSize: 14, color: "FFFFFF", align: "center", margin: 0 });
s.addText("github.com/DimiChatzipavlis/ToolFinder", { x: 0.8, y: 4.6, w: 8.4, h: 0.3, fontFace: "Consolas", fontSize: 11, color: ORANGE, align: "center", margin: 0 });

// ===================================================== SLIDE 2 — PROBLEM
s = pres.addSlide();
contentHeader(s, "Problem", "Tool selection is the bottleneck — and a safety boundary");
const probs = [
  [ICONS.bloat, "Context bloat", "Binding every MCP schema fills the window before reasoning begins."],
  [ICONS.middle, "Lost in the middle", "Similar APIs collide in-context, raising tool-selection errors."],
  [ICONS.small, "Small models break", "3B local models emit malformed calls under long-prompt pressure."],
];
let py = 1.7;
probs.forEach(([icon, h, d]) => {
  s.addShape(pres.shapes.OVAL, { x: 0.7, y: py, w: 0.42, h: 0.42, fill: { color: BLUE } });
  s.addImage({ data: icon, x: 0.81, y: py + 0.11, w: 0.2, h: 0.2 });
  s.addText(h, { x: 1.3, y: py - 0.02, w: 4.3, h: 0.34, fontFace: BODY, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText(d, { x: 1.3, y: py + 0.32, w: 4.4, h: 0.5, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  py += 0.92;
});
s.addShape(pres.shapes.RECTANGLE, { x: 6.0, y: 1.7, w: 3.5, h: 2.55, fill: { color: NAVY }, shadow: shadow() });
s.addText("Selection decides whether a local model can use a 500-tool catalog at all — and a wrong route to a destructive tool turns a retrieval error into a harmful action.",
  { x: 6.25, y: 1.95, w: 3.0, h: 1.6, fontFace: BODY, fontSize: 14, color: WHITE, italic: true, valign: "middle", margin: 0 });
s.addText("We treat tool selection as open-set retrieval: unseen phrasings, unseen tools, new servers, out-of-scope requests, and adversarial descriptions.",
  { x: 0.7, y: 4.5, w: 8.7, h: 0.6, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0 });
footer(s, 2);

// ===================================================== SLIDE 3 — PRIOR WORK
s = pres.addSlide();
contentHeader(s, "Related work", "We add evaluation rigor, not a new architecture");
function column(x, title, color, rows) {
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 4.2, h: 2.5, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 4.2, h: 0.5, fill: { color } });
  s.addText(title, { x: x + 0.15, y: 1.65, w: 3.9, h: 0.5, fontFace: BODY, fontSize: 14, bold: true, color: WHITE, valign: "middle", margin: 0 });
  s.addText(rows.map((r, i) => ({ text: r, options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } })),
    { x: x + 0.2, y: 2.3, w: 3.85, h: 1.75, fontFace: BODY, fontSize: 12, color: INK, valign: "top" });
}
column(0.7, "Tool USE / generation", BLUE, ["MRKL — modular router framing", "Gorilla — API-call gen + retriever", "ToolLLM — 16k API tool use", "Toolformer — self-supervised calls"]);
column(5.1, "Tool SELECTION (shipped)", MUTED, ["semantic-router (Aurelio Labs)", "LangChain tool retriever", "LlamaIndex object retrievers", "OpenAI function-retrieval cookbook"]);
s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.35, w: 8.8, h: 0.7, fill: { color: ORANGE } });
s.addText("Our contribution is a leakage-controlled benchmark + a controlled comparison of what these ship as defaults — not architectural novelty.",
  { x: 0.9, y: 4.35, w: 8.4, h: 0.7, fontFace: BODY, fontSize: 13, bold: true, color: WHITE, valign: "middle", margin: 0 });
footer(s, 3);

// ===================================================== SLIDE 4 — BENCHMARK + LEAK
s = pres.addSlide();
contentHeader(s, "Benchmark", "The benchmark — and why naive splits lie");
fitImage(s, fig("fig_leakage_audit.png"), 913, 586, 4.9, 1.6, 4.7, 3.0, { frame: true });
s.addText([
  { text: "1,695", options: { fontSize: 30, bold: true, color: BLUE, fontFace: HEAD, breakLine: true } },
  { text: "real intents", options: { fontSize: 12, color: INK, breakLine: true } },
], { x: 0.7, y: 1.65, w: 1.85, h: 0.9, margin: 0, valign: "top" });
s.addText([
  { text: "574", options: { fontSize: 30, bold: true, color: BLUE, fontFace: HEAD, breakLine: true } },
  { text: "real tool schemas", options: { fontSize: 12, color: INK, breakLine: true } },
], { x: 2.7, y: 1.65, w: 1.9, h: 0.9, margin: 0, valign: "top" });
s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 2.75, w: 3.9, h: 1.9, fill: { color: NAVY }, shadow: shadow() });
s.addText([
  { text: "96% Recall@1", options: { fontSize: 22, bold: true, color: ORANGE, fontFace: HEAD, breakLine: true } },
  { text: "from a 1-NN lookup over training queries alone — no model, no schema.", options: { fontSize: 13, color: WHITE, breakLine: true } },
  { text: "The original random split answered itself.", options: { fontSize: 12, italic: true, color: "CADCFC" } },
], { x: 0.9, y: 2.95, w: 3.5, h: 1.5, margin: 0, valign: "top" });
s.addText("Queries follow a scenario×template grammar; random row splits scatter paraphrases across train/test.", { x: 4.9, y: 4.7, w: 4.7, h: 0.5, fontFace: BODY, fontSize: 10, italic: true, color: MUTED, align: "center", margin: 0 });
footer(s, 4);

// ===================================================== SLIDE 5 — REGIMES
s = pres.addSlide();
contentHeader(s, "Method", "Leakage-controlled evaluation, enforced in CI");
const regimes = [
  ["Regime 1", "Unseen queries", "Scenario-grouped split (in-grammar upper bound)", BLUE],
  ["Regime 1b", "Template-disjoint", "Templates AND scenarios held out — the real control", ORANGE],
  ["Regime 2", "Unseen tools", "15 tools never trained on; ranked vs full catalog", BLUE],
  ["Regime 3", "Unseen servers", "195 queries, 574-tool corpus; training stays GitHub-only", BLUE],
];
let rx = 0.7, ry = 1.7;
regimes.forEach((r, i) => {
  const x = rx + (i % 2) * 4.5, y = ry + Math.floor(i / 2) * 1.45;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.2, h: 1.25, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.08, h: 1.25, fill: { color: r[3] } });
  s.addText([
    { text: r[0] + "  ", options: { bold: true, color: r[3], fontSize: 15, fontFace: HEAD } },
    { text: r[1], options: { bold: true, color: NAVY, fontSize: 14 } },
  ], { x: x + 0.2, y: y + 0.12, w: 3.9, h: 0.4, margin: 0 });
  s.addText(r[2], { x: x + 0.2, y: y + 0.55, w: 3.85, h: 0.6, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
});
s.addText("Split hygiene is a failing CI test (tests/test_split_hygiene.py) — no scenario, template, or tool may cross buckets.",
  { x: 0.7, y: 4.7, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0 });
footer(s, 5);

// ===================================================== SLIDE 6 — TWO MODELS
s = pres.addSlide();
contentHeader(s, "Deep learning", "Two trained architectures, compared");
function modelCard(x, tag, name, color, rows) {
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 4.2, h: 2.95, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 4.2, h: 0.62, fill: { color } });
  s.addText([{ text: tag + "  ", options: { bold: true, color: WHITE, fontSize: 12 } }, { text: name, options: { bold: true, color: WHITE, fontSize: 15, fontFace: HEAD } }],
    { x: x + 0.2, y: 1.65, w: 3.9, h: 0.62, valign: "middle", margin: 0 });
  s.addText(rows.map((r) => ({ text: r, options: { bullet: true, breakLine: true, paraSpaceAfter: 7 } })),
    { x: x + 0.22, y: 2.45, w: 3.8, h: 2.05, fontFace: BODY, fontSize: 12, color: INK, valign: "top" });
}
modelCard(0.7, "MODEL A", "Bi-encoder", BLUE, ["MiniLM-L6 (22M), MNRL loss", "No-duplicates batch sampler (fixes in-batch false negatives)", "Encodes query + schema independently", "BGE / MPNet trained as capacity ablation"]);
modelCard(5.1, "MODEL B", "Cross-encoder reranker", ORANGE, ["ms-marco-MiniLM-L6 (22M), BCE", "1 positive : 4 mined hard negatives", "Joint cross-attention over (query, schema)", "Reranks bi-encoder top-10 — 10 passes/query"]);
s.addText("3 seeds {13, 42, 1337} · selection on validation MRR@10 · same evaluation code path for every system.",
  { x: 0.7, y: 4.7, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0 });
footer(s, 6);

// ===================================================== SLIDE 7 — MAIN RESULTS
s = pres.addSlide();
contentHeader(s, "Results", "Fine-tuning dominates every regime");
fitImage(s, fig("fig_main_results.png"), 1896, 835, 0.7, 1.62, 5.55, 3.0, { frame: true });
statCard(s, 6.5, 1.62, 3.0, "+0.42 R@1", "FT MiniLM over BM25, regime 1 (p ≤ 0.0001)", ORANGE);
statCard(s, 6.5, 3.12, 3.0, "0.33 → 0.99", "frozen → fine-tuned MiniLM R@1 (capacity ≪ training)", BLUE);
s.addText("0.99 / 0.91 / 0.67 R@1 across regimes 1/2/3 vs 0.57 / 0.76 / 0.49 for BM25. Frozen MiniLM (0.33) loses to BM25.",
  { x: 0.7, y: 4.72, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 11.5, color: MUTED, italic: true, margin: 0 });
footer(s, 7);

// ===================================================== SLIDE 8 — CONTROL
s = pres.addSlide();
contentHeader(s, "Rigor", "The result survives the leakage control");
fitImage(s, fig("fig_loss_curves.png"), 1514, 644, 4.7, 1.95, 5.0, 2.5, { frame: true });
statCard(s, 0.7, 1.7, 3.6, "0.958 vs 0.650", "FT MiniLM vs 1-NN leakage probe on the template-disjoint split", BLUE);
s.addText([
  { text: "Regime 1's 1.000 was an in-grammar upper bound. ", options: { breakLine: true } },
  { text: "Retraining on regime 1b (templates + scenarios both unseen) still gives 0.958 — the win is real, not leaked.", options: {} },
], { x: 0.7, y: 3.2, w: 3.7, h: 1.5, fontFace: BODY, fontSize: 13, color: INK, valign: "top", margin: 0 });
s.addText("Loss converges in 3–5 epochs; validation MRR tracks it — no overfitting.", { x: 4.7, y: 4.55, w: 5.0, h: 0.4, fontFace: BODY, fontSize: 10, italic: true, color: MUTED, align: "center", margin: 0 });
footer(s, 8);

// ===================================================== SLIDE 9 — SAFETY & SCALING
s = pres.addSlide();
contentHeader(s, "Safety & systems", "Rejection, robustness, and the right index");
fitImage(s, fig("fig_scaling.png"), 1566, 649, 4.75, 1.95, 4.95, 2.45, { frame: true });
statCard(s, 0.7, 1.65, 3.7, "0.99 ROC-AUC", "out-of-scope rejection (fine-tuned encoder)", BLUE);
statCard(s, 0.7, 3.12, 3.7, "0% vs 83%", "poisoning hijack: fine-tuned vs BM25 (fine-tuning is the defense)", ORANGE);
s.addText("Query encoding (~86 ms CPU) dominates routing latency, so exact Flat is the right default; HNSW only cuts search time beyond ~10³ tools, at a recall cost.",
  { x: 4.75, y: 4.5, w: 4.95, h: 0.6, fontFace: BODY, fontSize: 10, italic: true, color: MUTED, align: "center", margin: 0 });
footer(s, 9);

// ===================================================== SLIDE 10 — NEGATIVE RESULTS
s = pres.addSlide();
contentHeader(s, "Honest findings", "What did not work — reported, not hidden");
const negs = [
  ["Reranking hurts", "Model B degrades Model A (0.944 vs 0.988) at ~140× the per-query cost — a stronger retriever + weaker reranker."],
  ["Default ≠ best", "The shipped router loads a zero-shot encoder; the fine-tuned weights that win are regenerable but not committed."],
  ["Synthetic data", "All queries follow a known grammar; no production traffic, single domain — disclosed as the data's ceiling."],
];
let ny = 1.7;
negs.forEach(([h, d]) => {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: ny, w: 5.55, h: 0.95, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: ny, w: 0.07, h: 0.95, fill: { color: ORANGE } });
  s.addText(h, { x: 0.85, y: ny + 0.1, w: 5.3, h: 0.32, fontFace: BODY, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  s.addText(d, { x: 0.85, y: ny + 0.42, w: 5.3, h: 0.5, fontFace: BODY, fontSize: 10.5, color: INK, margin: 0 });
  ny += 1.05;
});
// Native bar chart: Model A vs Model B R@1 by regime — shows reranking degrades A.
s.addShape(pres.shapes.RECTANGLE, { x: 6.4, y: 1.7, w: 3.15, h: 2.95, fill: { color: WHITE }, line: { color: CARDLINE, width: 1 }, shadow: shadow() });
s.addChart(pres.charts.BAR, [
  { name: "Model A (bi-encoder)", labels: ["Regime 1", "Regime 2", "Regime 3"], values: [0.988, 0.909, 0.667] },
  { name: "Model B (+ rerank)", labels: ["Regime 1", "Regime 2", "Regime 3"], values: [0.944, 0.867, 0.632] },
], {
  x: 6.5, y: 1.78, w: 2.95, h: 2.78, barDir: "col",
  chartColors: [BLUE, ORANGE],
  showTitle: false, showValue: true, dataLabelFormatCode: "0.00", dataLabelFontSize: 7, dataLabelColor: INK,
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisHidden: true,
  catAxisLabelColor: MUTED, catAxisLabelFontSize: 8,
  valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 8, legendColor: INK,
});
s.addText("Reranking (B) sits below the bi-encoder (A) in every regime.", { x: 6.4, y: 4.68, w: 3.15, h: 0.34, fontFace: BODY, fontSize: 9.5, italic: true, color: MUTED, align: "center", margin: 0 });
footer(s, 10);

// ===================================================== SLIDE 11 — CONCLUSION
s = pres.addSlide();
s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.13, fill: { color: ORANGE } });
s.addText("Conclusions", { x: 0.8, y: 0.55, w: 8.4, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, margin: 0 });
const concl = [
  "A small fine-tuned bi-encoder solves MCP tool selection — and is itself the best poisoning defense and OOD rejector.",
  "Fine-tuning, not model size, is the lever: 22M beats a frozen 109M encoder.",
  "Negative results reported: reranking does not help at this scale; exact search beats HNSW at realistic sizes.",
  "Fully reproducible — pinned seeds, CI-enforced split hygiene, run_all.py.",
];
s.addText(concl.map((c) => ({ text: c, options: { bullet: { code: "2022" }, color: "E8EEF7", fontSize: 15, breakLine: true, paraSpaceAfter: 12 } })),
  { x: 0.85, y: 1.5, w: 8.5, h: 2.3, fontFace: BODY, valign: "top" });
s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 4.05, w: 8.4, h: 0.95, fill: { color: "12203D" }, line: { color: BLUE, width: 1 } });
s.addText("Limitations: synthetic queries (known grammar), single training domain, LLM-in-context arm not run (environment-blocked). All disclosed; none inflate the reported numbers.",
  { x: 1.0, y: 4.05, w: 8.0, h: 0.95, fontFace: BODY, fontSize: 12, italic: true, color: "9FB3D1", valign: "middle", margin: 0 });
s.addText("github.com/DimiChatzipavlis/ToolFinder", { x: 0.8, y: H - 0.42, w: 8.4, h: 0.3, fontFace: "Consolas", fontSize: 10, color: ORANGE, align: "center", margin: 0 });

const out = await pres.writeFile({ fileName: path.resolve(__dirname, "..", "reports", "ToolFinder_presentation.pptx") });
console.log("wrote", out);
}
main().catch((e) => { console.error(e); process.exit(1); });
