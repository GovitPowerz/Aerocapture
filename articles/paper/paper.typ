// =============================================================================
// NN aerocapture guidance, revisited -- follow-up to Gelly & Vernis 2009.
// Compile: make -C articles/paper pdf (runs typst from the repo ROOT so figure paths resolve,
// and passes the git head the colophon prints).
// Data: articles/paper/data/ (results.json + the eval JSONs). Every main-body table and every
// quoted number of Sections 5-10 and Appendix C is an accessor call through results.typ or is
// asserted against the bundle at compile time (the ou_*/v4_* blocks below); Sections 4-5 and
// Appendix B are development-regime studies and quote their transcribed numbers under that label.
// Rebuild everything: make -C articles/paper paper (see the Makefile).
// Section order: methodology-first (the spine). One noise regime in the main body (per-scenario
// density noise, ADR-0006); the shared noise path is the development regime of Sections 4-5.
// dense_515 carried as a full efficiency-reference row throughout.
// =============================================================================

// arXiv-preprint layout matched to the Mamba-3 PDF (arXiv:2603.15569v1): Linux
// Libertine body (Typst's bundled Libertinus Serif) over a wide ~6.5in text block,
// Computer Modern math (standing in for newtxmath), booktabs-style tables with the
// caption ABOVE, hyperref-style colors (maroon citations, blue internal refs/links).
#set document(title: "Seventeen years later: stateful neural guidance and the tail that sizes a Mars aerocapture mission", author: "Grégory Gelly")
#set page(paper: "us-letter", margin: (x: 1in, top: 1.1in, bottom: 1.15in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(font: "DejaVu Sans Mono", size: 0.82em)  // cmtt stand-in for literals/URLs
// Mamba-3 body metrics: 10pt over ~12.4pt leading, flush-left paragraphs separated
// by vertical space (no indent).
#set par(justify: true, leading: 0.56em, spacing: 0.95em)
#set enum(indent: 1.2em, body-indent: 0.5em)
#set list(indent: 1.2em, body-indent: 0.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
// Headings: section \Large-bold (~14pt), subsection \large (~12pt),
// subsubsection \normalsize (~10.5pt), with roomier LaTeX-article skips.
#show heading: set text(weight: "bold")
#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 2): set text(size: 12pt)
#show heading.where(level: 3): set text(size: 10.5pt)
#show heading.where(level: 1): set block(above: 1.7em, below: 1.0em)
#show heading.where(level: 2): set block(above: 1.5em, below: 0.9em)
#show heading.where(level: 3): set block(above: 1.3em, below: 0.8em)
// hyperref-style colors from the Mamba-3 PDF: citations in a crimson/maroon,
// internal refs (sections, tables, equations) and URLs in blue.
#let citecolor = rgb("#b83a5c")
#let linkcolor = rgb("#2b5ba8")
#show cite: set text(fill: citecolor)
#show ref: set text(fill: linkcolor)
#show link: set text(fill: linkcolor)
// Booktabs tables: no vertical rules, no cell grid; each table adds its own
// top/mid/bottom hlines. Table captions sit ABOVE the table (LaTeX convention).
#set table(stroke: none, inset: (x: 7pt, y: 4.5pt))
#show figure.where(kind: table): set figure.caption(position: top)
#show figure: set block(above: 1.5em, below: 1.5em)
#set figure(gap: 0.8em)
#show grid: set block(above: 1.5em, below: 1.5em)

// Figure helper: include from figures/, attach the caption and the label.
#let fig(path, cap, lbl) = [#figure(image("figures/" + path, width: 100%), caption: cap)#lbl]

// Bundle accessors (results.json, confirmatory_eval.json, quant/finalists_results.json,
// confirmatory_marginal.json, quote_marginal.json).
#import "results.typ" as R
// ---------------------------------------------------------------------------------------------
// The per-scenario far-tail cells (data/confirmatory_marginal.json, per_draw, 10 x 100 000 scenarios
// per cell): the nine networks trained from scratch at the headline allocation (three Mamba-962
// seeds, three dense-515 seeds, one GRU / dense-972 / LSTM), the retuned classical laws, the
// mechanism controls. Seed 1 of every cell is the deployed one by the rule fixed before any 10^6
// score existed; seeds 2 and 3 give the mean and the standard deviation. The asserts pin every
// count and ordering the prose states.
// ---------------------------------------------------------------------------------------------
#assert(R.marginal.n_replicates == 10 and R.marginal.n_per_replicate == 100000, message: "the per-scenario quotes state a 10 x 100 000 pool")
#let hl(label) = R.marg("ou_marginal/" + label)
#let mamba_seeds = ("hl_mamba_p962", "hl_mamba_p962_s2", "hl_mamba_p962_s3").map(hl)
#let dense_seeds = ("hl_dense_p515", "hl_dense_p515_s2", "hl_dense_p515_s3").map(hl)
#let champ = mamba_seeds.first()
#let dense = dense_seeds.first()
#let gru = hl("hl_gru_p1014")
#let lstm = hl("hl_lstm_p1082")
#let dense972 = hl("hl_dense_p972")
#let nets9 = mamba_seeds + dense_seeds + (gru, dense972, lstm)
#let jftc = hl("classical/ftc_joint")
#let fnpag = hl("classical/fnpag")
#let law_labels = ("classical/ftc_joint", "classical/fnpag", "classical/pred_guid_joint", "classical/energy_controller_joint",
  "classical/pred_guid", "classical/ftc", "classical/energy_controller", "classical_ungated/equilibrium_glide", "classical_ungated/piecewise_constant")
#let laws = law_labels.map(hl)
#let mamba3 = R.mean_sd(mamba_seeds.map(c => c.cvar999))
#let dense3 = R.mean_sd(dense_seeds.map(c => c.cvar999))
#let mamba3_95 = R.mean_sd(mamba_seeds.map(c => c.cvar95))
#let dense3_95 = R.mean_sd(dense_seeds.map(c => c.cvar95))
// Joint-FTC is the best classical law at CVaR99.9 and FNPAG at CVaR95; the prose names them so.
#assert(laws.all(c => c.cvar999 >= jftc.cvar999) and laws.all(c => c.cvar95 >= fnpag.cvar95),
  message: "the prose names joint-FTC the best classical law at CVaR99.9 and FNPAG at CVaR95")
#assert(jftc.lost == 0 and jftc.viol_pct == 0 and jftc.hl_viol_n == 0, message: "the prose states that the retuned joint-FTC captures every scenario and violates nothing")
#assert(fnpag.lost > 0, message: "the prose states that FNPAG loses scenarios at 10^6")
// Capture and feasibility of the networks: the deployed seed and seed 3 capture every scenario,
// seed 2 loses a handful; every network exceeds the heat-load ceiling on some scenarios per 10^6,
// the deployed seed on the most of its three.
#assert(champ.lost == 0 and mamba_seeds.at(2).lost == 0 and mamba_seeds.at(1).lost > 0 and mamba_seeds.at(1).lost < 10,
  message: "the prose states two Mamba seeds capture 10^6 of 10^6 and the third loses a handful")
#assert(nets9.all(c => c.hl_viol_n != none and c.hl_viol_n > 0) and champ.hl_viol_n == calc.max(..mamba_seeds.map(c => c.hl_viol_n)),
  message: "the prose states every network exceeds the heat-load ceiling on some scenarios, the deployed seed on the most of its three")
#let net_hl_span = R.span(nets9.map(c => c.hl_viol_n))
// The margins the headline quotes: the Mamba three-seed mean against the best law and the best
// dense network; the deployed seed against joint-FTC and FNPAG.
#let margin3_jftc = jftc.cvar999 - mamba3.mean
#let margin3_fnpag = fnpag.cvar999 - mamba3.mean
#let margin_champ_jftc = jftc.cvar999 - champ.cvar999
#let margin_champ_fnpag = fnpag.cvar999 - champ.cvar999
#assert(margin3_jftc > 0 and margin_champ_jftc > 0 and margin_champ_jftc < mamba3.sd, message: "the prose states the deployed seed's margin to joint-FTC is inside the seed spread")
// The architecture ordering of the development regime does not survive: the Mamba and dense
// three-seed means sit within either seed standard deviation of each other; the deployed seed is
// the worst of its three at CVaR99.9 and the best at CVaR95.
#assert(calc.abs(mamba3.mean - dense3.mean) < calc.min(mamba3.sd, dense3.sd), message: "the prose states the Mamba and dense three-seed means sit inside either seed spread")
#assert(champ.cvar999 == calc.max(..mamba_seeds.map(c => c.cvar999)) and champ.cvar95 == calc.min(..nets9.map(c => c.cvar95)),
  message: "the prose states the deployed seed has the worst CVaR99.9 of its three and the best CVaR95 of the nine")
#let seeds_below_jftc = nets9.filter(c => c.cvar999 < jftc.cvar999).len()
#assert(seeds_below_jftc == nets9.len() - 1 and nets9.all(c => c.cvar999 < fnpag.cvar999), message: "the prose states eight of nine network seeds below joint-FTC and all nine below FNPAG at CVaR99.9")
#let net_span999 = R.span(nets9.map(c => c.cvar999))
// The mechanism controls (Section 6.3): the deployed seed with its state zeroed every tick, and
// the two retrained families (matched five-tick window, no predicted-DV inputs) at three seeds.
#let reset = hl("v4_reset_state")
#let window_seeds = ("ctrl_window_p970", "ctrl_window_p970_s2", "ctrl_window_p970_s3").map(hl)
#let nodv_seeds = ("ctrl_mamba_p962_nodv", "ctrl_mamba_p962_nodv_s2", "ctrl_mamba_p962_nodv_s3").map(hl)
#let window3 = R.mean_sd(window_seeds.map(c => c.cvar999))
#let nodv3 = R.mean_sd(nodv_seeds.map(c => c.cvar999))
#let welch_t(a, b) = (a.mean - b.mean) / calc.sqrt((a.sd * a.sd + b.sd * b.sd) / 3)
#assert(calc.abs(welch_t(window3, mamba3)) < 1 and calc.abs(welch_t(nodv3, mamba3)) < 1,
  message: "the prose states neither control's three-seed CVaR99.9 difference to the champion family is resolved (Welch t below 1)")
#assert(reset.cvar999 > 4 * champ.cvar999 and reset.capture_pct > 99.9, message: "the prose states the reset-state tail collapses by more than a factor of four at capture above 99.9%")
// The n = 1000 paired final-evaluation pool (data/results.json, per_draw): the deployed seed against
// every law and network; the laws against each other; the PPO cells against their population twins.
#let pr(key) = R.paired(key)
#assert(pr("v4_nn_vs_jointftc").win_rate_a > 0.99 and pr("v4_nn_vs_fnpag").win_rate_a > 0.99 and pr("v4_nn_vs_ftc").win_rate_a == 1,
  message: "the prose states the deployed network wins more than 99% of paired scenarios against joint-FTC and FNPAG and every one against fixed-reference FTC")
#assert(("v4_headline_vs_dense515", "v4_headline_vs_dense972", "v4_headline_vs_gru", "v4_headline_vs_lstm").all(k => pr(k).delta_cvar95_ci.at(1) < 0),
  message: "the prose states the deployed seed beats every other network at CVaR95 on the paired pool with intervals excluding zero")
#assert(pr("v4_jointftc_vs_fnpag").delta_cvar95_ci.at(0) < 0 and pr("v4_jointftc_vs_fnpag").delta_cvar95_ci.at(1) > 0,
  message: "the prose states joint-FTC and FNPAG are tied at CVaR95 on the paired pool")
// Off-nominal stress at depth (data/stress_depth.json): the champion captures least of the five and
// every paired capture deficit has an interval excluding zero; FNPAG alone has the worse
// conditional tail.
#let st = R.stress_cell
#let st_laws = ("joint-FTC", "FTC-fixed", "PredGuid", "FNPAG")
#assert(st_laws.all(l => R.stress_pair(l).delta_capture_pts_ci.at(1) < 0), message: "the prose states the champion captures less than every law under stress, intervals excluding zero")
#assert(st_laws.filter(l => R.stress_pair(l).delta_cvar95_ci.at(1) < 0) == ("FNPAG",), message: "the prose states FNPAG is the one law with a worse conditional stress tail than the champion")
#let st_cap_deficit = R.span(st_laws.map(l => -R.stress_pair(l).delta_capture_pts))
// Objective centering under per-scenario noise (Section 7.3): the three centered seeds against the
// medium-deployed and the high-regime-retuned joint-FTC on the 9M stress pool at n = 10 000.
#let cd_seeds = ("mamba_centered_s1", "mamba_centered_s2", "mamba_centered_s3")
#let cd = R.centered4
#let cdp = R.centered4_pair
#let cd_span(labels, field, d: 0) = R.span(labels.map(l => cd(l).at(field)), d: d)
#let cd_viol_span(labels, which) = R.span(labels.map(l => cd(l).violation_pct.at(which)), d: 1)
#assert(cd_seeds.all(s => calc.abs(cdp(s, "jointFTC-medium").delta_capture_pts) < 0.5 and cdp(s, "jointFTC-medium").delta_cvar95_ci.at(1) < 0),
  message: "the prose states every centered seed sits at capture parity with the medium-deployed joint-FTC and beats its conditional tail")
#assert(cd_seeds.all(s => cdp(s, "jointFTC-high").delta_capture_pts_ci.at(0) > 0 and cdp(s, "jointFTC-high").delta_cvar95_ci.at(1) < 0),
  message: "the prose states every centered seed out-captures and out-tails the high-regime-retuned joint-FTC")
#assert(cd("mamba_centered_s1").violation_pct.heat_flux > cd("jointFTC-medium").violation_pct.heat_flux
  and cd("mamba_centered_s3").violation_pct.heat_flux < cd("jointFTC-medium").violation_pct.heat_flux
  and cd_seeds.all(s => cd(s).violation_pct.heat_load > cd("jointFTC-medium").violation_pct.heat_load),
  message: "the prose states seed 1 exceeds heat flux more often than the medium joint-FTC, seed 3 less often, and every seed exceeds heat load more often")
#let lv = R.lever4
#let lever_labels = ("stacked", "plus_sims", "plus_bucket", "plus_transform", "centered")
#assert(lever_labels.slice(1).all(l => lv(l).dv_cvar95 < lv("stacked").dv_cvar95) and lv("centered").dv_cvar95 == calc.min(..lever_labels.map(l => lv(l).dv_cvar95)),
  message: "the prose states every lever lowers the stacked tail and the centered cell has the lowest")
#assert(lv("plus_bucket").violation_pct.heat_load == calc.max(..lever_labels.map(l => lv(l).violation_pct.heat_load)),
  message: "the prose states the middle bucket alone carries the worst heat-load exceedance of the five")
// The single-core compute quotes (data/compute_benchmark.json): ms per simulation, the network's
// factor below FNPAG and over FTC, its cost of state over the dense network, and at the 100x
// flight-processor scaling FNPAG's per-replan cost and the network's cost per 1 s update.
#let ms(label) = R.bench(label).ms_per_sim
#let ms_vs_fnpag = int(calc.round(ms("FNPAG") / ms("NN-mamba")))
#let ms_vs_ftc = ms("NN-mamba") / ms("FTC")
#let ms_state = ms("NN-mamba") / ms("NN-dense")
#let fl_replan_ms = 100 * R.bench("FNPAG").ms_per_replan_derived
#let fl_update_ms = 100 * R.bench("NN-mamba").us_per_update_incl_sim / 1000
#assert(calc.abs((1000 / fl_update_ms) / (2000 / fl_replan_ms) - 30) < 5, message: "the Compute paragraph states the two deadline margins differ by a factor of thirty")
#let bench_rustc = R.benchmark.meta.rustc.split(" ").at(1).split(".").slice(0, 2).join(".")
#let bench_spread = ("NN-mamba", "NN-dense", "FTC", "FNPAG").map(l => 100 * R.mean_sd(R.bench(l).ms_per_sim_repeats).sd / ms(l))
// Noise conditioning (Section 6.4, data/quote_marginal.json, paired n = 1000): the five shared-path
// champions and three laws scored under both regimes (ou_shift = per-scenario minus shared-path
// CVaR95), and the nine headline-allocation cells, trained under per-scenario noise, scored the
// same way.
#assert(R.quotes.n_sims == 1000, message: "the conditioning quotes state a paired n = 1000 pool")
#let ou_nets = ("mamba_p962", "lstm_p1082", "gru_p1014", "dense_p972", "dense_p515")
#let ou_laws = ("fnpag", "pred_guid", "ftc")
#let ou_shift(label) = R.ou(label).cvar95 - R.ou(label, regime: "frozen").cvar95
#let ou_law_shift = ou_laws.map(ou_shift).sum() / ou_laws.len()
#assert(ou_nets.all(l => ou_shift(l) >= 2 * ou_law_shift and ou_shift(l) <= 4 * ou_law_shift),
  message: "the conditioning paragraph states that the shared-path networks give up 2--4 times the classical laws' tail")
#assert(ou_nets.all(l => R.ou(l).viol_pct > R.ou(l, regime: "frozen").viol_pct) and ou_laws.all(l => R.ou(l).viol_pct == 0),
  message: "the conditioning paragraph states every shared-path network adds heat-load violations under per-scenario noise and no law incurs any")
#let hl_labels = ("hl_mamba_p962", "hl_mamba_p962_s2", "hl_mamba_p962_s3", "hl_dense_p515", "hl_dense_p515_s2", "hl_dense_p515_s3", "hl_gru_p1014", "hl_dense_p972", "hl_lstm_p1082")
#let hl_shifts = hl_labels.map(l => ou_shift("ou_" + l))
#assert(hl_shifts.all(s => calc.abs(s) < 5), message: "the conditioning paragraph states every per-scenario-trained cell scores the two regimes within 5 m/s")
// Section 8: the deployed seed's closed-loop input sensitivity, and its fresh-pool re-quote.
#let abl_top = R.ablation_ranked.slice(0, 3).map(r => r.name)
#assert(abl_top == ("orbital_energy", "predicted_dv1", "predicted_dv2"), message: "Section 8 names orbital energy and the first two predicted-DV inputs as the three largest sensitivities")
#assert(R.ablation_live.len() == 17 and R.ablation_live.all(r => r.delta > 0), message: "Section 8 states all seventeen masked inputs cost something when zeroed")
#let champ_run = R.per_draw_regime("ou_marginal/hl_mamba_p962")
#assert(calc.abs(R.fresh4.dv_cvar95 - champ_run.dv_cvar95) < 5 and R.fresh4.capture_pct == 100, message: "Section 6 states the fresh-pool re-quote sits within 5 m/s of the final-evaluation CVaR95 at full capture")
// The loss-versus-tail correlation of Section 6.3 is drawn in the figure (scripts/fig_loss_vs_tail.py)
// and read as descriptive; the prose quotes its sign only, which the champion row pins: the lowest
// validation loss of its family and its worst far tail.
#let mamba_rms = ("hl_mamba_p962", "hl_mamba_p962_s2", "hl_mamba_p962_s3").map(l => R.per_draw_regime("ou_marginal/" + l).best_val_rms_within_transform_only)
#assert(mamba_rms.first() == calc.min(..mamba_rms), message: "Section 6.3 states the deployed seed has the lowest validation loss of its three")
// Appendix C: the v4 champion's quantization (data/quant_v4/).
#let q4 = R.quant4_marg
#let q4p = R.quant4_paired
#let q4_cells = ("ou_marginal/hl_mamba_p962", "quant_v4/ptq4_verdict", "quant_v4/qat4_finetune", "quant_v4/qat4_scratch").map(q4)
#let q4_grid8 = R.quant4_grid.variants.filter(v => v.bits == 8)
#assert(q4_grid8.len() == 4 and q4_grid8.all(v => v.capture_rate == 1 and calc.abs(v.delta_dv_cvar95) < 0.5), message: "Appendix C states every 8-bit cell captures everything within 0.5 m/s of full precision")
#let q4_verdict = R.quant4_grid.verdict
#assert(q4_verdict.bits == 4 and q4_verdict.granularity == "per_tensor" and q4_verdict.tensor_policy == "proj_only", message: "Appendix C names the 4-bit per-tensor projections-only cell as the verdict")
#let q4_grid4 = R.quant4_grid.variants.filter(v => v.bits == 4)
#assert(q4_grid4.filter(v => v.capture_rate == 1).len() == 1, message: "Appendix C states the verdict is the only 4-bit cell at full capture")
#assert(q4p("quant_v4/qat4_finetune").delta_cvar999.ci95.at(1) < 0 and q4p("quant_v4/qat4_scratch").delta_cvar999.ci95.at(0) < 0 and q4p("quant_v4/qat4_scratch").delta_cvar999.ci95.at(1) > 0
  and q4p("quant_v4/ptq4_verdict").delta_cvar999.ci95.at(0) > 0,
  message: "Appendix C states the fine-tune's far-tail delta is negative, the scratch arm's straddles zero and the PTQ cell's is positive")
#let q4_loo_top = R.quant4_grid.loo.sorted(key: v => -v.delta_dv_cvar95).first().tensor
#assert(q4_loo_top == "layer_1.x_proj_w", message: "Appendix C names the SSM input projection as the most sensitive tensor")
// The nine headline-allocation cells on the n = 1000 final-evaluation pool (data/results.json):
// the flat median the architecture section quotes.
#let net_runs = hl_labels.map(l => R.per_draw_regime("ou_marginal/" + l))
#let net_p50_span = R.span(net_runs.map(r => r.dv_p50))
// Section 6.3's loss-versus-tail reading: the lowest-loss dense seed has the widest tail of the nine.
#let dense_rms = ("hl_dense_p515", "hl_dense_p515_s2", "hl_dense_p515_s3").map(l => R.per_draw_regime("ou_marginal/" + l).best_val_rms_within_transform_only)
#let dense_lowest_loss = range(3).sorted(key: i => dense_rms.at(i)).first()
#assert(dense_seeds.at(dense_lowest_loss).cvar999 == calc.max(..nets9.map(c => c.cvar999)),
  message: "Section 6.3 states the lowest-loss dense seed has the widest far tail of the nine runs")
// Section 6.2's heat-load reading: the deployed seed's worst heat load on its appendix card (n = 1000,
// per_draw) is within 1% of the limit; the relaxed-ceiling policies re-flown under the nominal limit
// exceed it on a tenth of scenarios.
#let apx_mamba = json("figures/appendix/nn_mamba/stats.json")
#assert(apx_mamba.noise_seeding == "per_draw" and apx_mamba.constraints.heat_load.max > apx_mamba.constraints.heat_load.limit
  and apx_mamba.constraints.heat_load.max < 1.01 * apx_mamba.constraints.heat_load.limit,
  message: "Section 6.2 states the deployed seed's worst heat load on its appendix card sits a fraction of a percent over the limit")
#let relaxed_v4_viol = R.span((("mamba_p962", "q27"), ("mamba_p962", "q30"), ("dense_p515", "q27"), ("dense_p515", "q30")).map(((c, q)) => R.slope(c, q, eval: "v4").viol_pct), d: 1)
// Section 7.3's two further readings: the centered and middle-bucket dense tails overlap at n = 1000,
// and the high-regime-retuned joint-FTC loses to the medium-deployed one on the same pool.
#assert(lv("centered").dv_cvar95_ci.at(1) > lv("plus_bucket").dv_cvar95_ci.at(0), message: "the prose states the centered and middle-bucket dense tails overlap")
#assert(cd("jointFTC-high").capture_pct < cd("jointFTC-medium").capture_pct and cd("jointFTC-high").dv_cvar95 > cd("jointFTC-medium").dv_cvar95,
  message: "the prose states the high-regime-retuned joint-FTC loses to the medium-deployed one on capture and tail")
#assert(lv("plus_sims").violation_pct.heat_flux > 0.9 * lv("stacked").violation_pct.heat_flux and lv("plus_transform").violation_pct.heat_flux < 0.5 * lv("stacked").violation_pct.heat_flux,
  message: "the prose states sixteen scenarios leave the heat-flux exceedance where the stacked objective put it and the linear transform cuts it")

#v(0.15in)
#align(center)[
  #text(size: 16pt, weight: "bold", hyphenate: false)[Seventeen years later: stateful neural guidance\
  and the tail that sizes a Mars aerocapture mission]
  #v(16pt)
  #text(size: 12pt)[Grégory Gelly]
  #v(1pt)
  #text(size: 10pt)[#link("mailto:gregory.gelly@gmail.com")[gregory.gelly\@gmail.com]]
  #v(3pt)
  #text(size: 9pt, style: "italic")[Preprint, 2026]
]
#v(18pt)

#align(center)[#text(size: 11pt, weight: "bold")[Abstract]]
#v(2pt)
#pad(x: 0.45in)[
  #set par(justify: true, leading: 0.44em)
  #text(size: 10pt)[In 2009 we showed that a single-hidden-layer feed-forward network, trained by a
  genetic algorithm, could fly the aerocapture of a Mars Sample Return vehicle more efficiently than
  a Cerimele--Gamble feedback law, and we closed that work by asking for the obvious next step: a
  comparison against predictor--corrector guidance. This paper answers it. We train stateful neural
  guidance policies and benchmark them, on identical Monte Carlo scenarios drawn from a bit-validated
  simulator with an independent density-noise realization per scenario, against six classical
  schemes retuned on the same objective, including a numerical predictor--corrector (FNPAG) and a
  reference-tracking feedback law (FTC). Because the mission's correction propellant is sized off the
  worst-case $Delta v$, we lead every comparison with the tail of its distribution, not the mean. On
  $10^6$ pre-registered confirmatory scenarios a #box[$962$-parameter] recurrent (Mamba) policy
  trained from scratch holds a far-tail $"CVaR"_(99.9)$ of #box[$#R.fixed(mamba3.mean) plus.minus #R.fixed(mamba3.sd)$ m/s]
  over three training seeds, $#R.fixed(margin3_jftc, d: 0)$ m/s below the best classical scheme (a
  well-referenced FTC, $#R.fixed(jftc.cvar999)$) and $#R.fixed(margin3_fnpag, d: 0)$ below FNPAG, at
  $#R.fixed(ms("NN-mamba"))$ ms per simulation, $#ms_vs_fnpag times$ faster than FNPAG; the deployed
  seed captures every scenario and exceeds the heat-load ceiling on $#R.fixed(champ.viol_pct, d: 2)%$
  of them, where the classical law exceeds nothing. The seed spread is the finding's limit: a
  #box[$515$-parameter] dense network reaches $#R.fixed(dense3.mean) plus.minus #R.fixed(dense3.sd)$ on the
  same pool, so the networks beat the classical schemes by a margin the spread does not erase, and
  beat one another by a margin it does. The result rests on a training methodology that is itself a
  contribution: a non-stationary, adaptive-seed Monte Carlo environment turns the genetic algorithm
  from the *worst* optimizer under fixed scenarios into the *best*, and a policy trained on one
  density history fits that history -- the shared-path champions of our development regime lose
  $2$--$4 times$ more tail than the classical laws when each scenario gets its own realization. The
  deployed policy computes with its state (zeroing it every tick multiplies the median cost by six),
  but matched controls without learned state or without the engineered cost-to-go inputs reach the
  same tail within seed variance. The main deployment caveat: under a deliberately harsher
  off-nominal regime the analytic law generalizes better than the network, a gap that a
  regime-matched training objective closes at capture parity, at the price of more frequent
  heat-flux exceedance.]
]
#v(18pt)

= Introduction

Aerocapture replaces a propulsive orbit-insertion burn with a single pass through a planet's
atmosphere: the vehicle enters on a hyperbolic trajectory and bleeds its excess energy through
aerodynamic drag, modulating only the bank angle to steer the lift vector and arrive at a bound
target orbit. The maneuver is attractive precisely because it is propellant-free, but it is also
unforgiving. The control authority is proportional to dynamic pressure, the entry corridor is
narrow, and the dispersions -- atmospheric density first among them -- are large. There is no
closed-form optimal bank-angle law, so the classical practice is to make strong simplifying
assumptions (a decoupling of in-plane and out-of-plane motion, a reference trajectory built on
apoapsis control alone) in order to obtain a tractable feedback or predictor--corrector law. These
simplifications are exactly what costs correction propellant downstream.

Neural networks are especially valuable for guidance problems that admit no analytic solution, and
aerocapture is a textbook case. In 2009 we showed that a feed-forward network with a single hidden
layer, trained by a genetic algorithm without any reference trajectory or input--output pairs, could
perform the aerocapture of a Mars Sample Return (MSR) vehicle at a mean correction cost of
$116.7$ m/s against $144.8$ m/s for a Cerimele--Gamble-derived feedback law -- a $19%$ reduction,
approaching the $113$ m/s periapsis-raise floor @gelly2009neural, itself building on earlier
aerocapture-corridor and soft-landing guidance studies @vernis2004tradeoff @gelly2007guidance. The
mechanism was that the network
learned the in-plane and out-of-plane logic *jointly*, where the classical law treats them
separately. We closed that paper with an explicit hook: "the next step would be to extend our work
on the aerocapture to skip entry missions and evaluate the performance of neural guidance compared
to classic algorithms such as the predictor-corrector schemes." This paper is that next step,
seventeen years later.

In the interval we carried the same evolutionary-training philosophy to recurrent networks for
speech, where a coordinated-gate LSTM trained by quantum-behaved particle-swarm optimization
@sun2004qpso, with a divide-and-conquer initialization and task-aligned differentiable losses, became
the working method @gelly2015mwe @gelly2016divide @gelly2017angular. Stateful cells, swarm training, smart
initialization, and custom losses are precisely the machinery this paper brings back to where it
started. With their innate ability to exploit the trajectory history, recurrent and selective
state-space policies are natural candidates for a guidance law that must anticipate the atmospheric
pass rather than merely react to the current state.

We make three contributions:

+ *A training methodology, not just a policy.* We show that a genetic algorithm is the *worst*
  optimizer for this problem under a fixed set of Monte Carlo scenarios -- it overfits the repeated
  cases -- and the *best* under a non-stationary, adaptive-seed environment that keeps moving the
  scenarios beneath it. Switching the seed schedule from fixed to moving takes the mean correction
  cost from $154$ to $120$ m/s in three-seed means (first seeds $160.3 arrow.r 118.0$), the single
  largest effect in the campaign. The moving
  environment, a tail-weighting cost transform, and hardest-case seed curation form one matched
  system, and the noise regime the policy trains on is part of the match: a network trained on one
  shared density history fits it, and only training under per-scenario noise produces a policy that
  scores the same under both.

+ *An architecture finding, with its limit measured.* Across dense, gated-recurrent (GRU, LSTM),
  windowed, attention (Transformer), and selective state-space (Mamba) cells, the engineered,
  cost-aligned inputs flatten the *median* correction cost to a common #net_p50_span m/s for every
  architecture trained to convergence. At the far-tail depth where the tanks are sized, every
  network family's seed mean sits below the best classical scheme ($#R.fixed(margin3_jftc, d: 0)$ m/s for
  the Mamba's three seeds), but the families sit inside one another's seed spread, and three matched
  controls -- state reset, an explicit observation window, input removal -- show that the deployed
  policy uses its state without showing that it needs it. Training loss does not pick the tail
  winner: the deployed seed has the lowest validation loss of its family and the worst far tail.

+ *A systematic head-to-head of neural versus predictor--corrector aerocapture guidance* -- to our
  knowledge the first for an MSR-class Mars aerocapture to compare an end-to-end learned policy
  against classical baselines co-tuned on the same objective, on paired dispersed scenarios, under
  a far-tail correction-$Delta v$ risk metric. On
  identical dispersions the deployed network beats the best classical scheme by $#R.fixed(-pr("v4_nn_vs_jointftc").delta_mean)$ m/s in mean
  and $#R.fixed(-pr("v4_nn_vs_jointftc").delta_cvar95)$ m/s at $"CVaR"_95$, winning $#R.fixed(100 * pr("v4_nn_vs_jointftc").win_rate_a)%$ of paired scenarios, at a
  per-simulation compute cost $#ms_vs_fnpag times$ below the numerical
  predictor--corrector -- with the caveat that the analytic law is more robust off-nominal, under a
  deliberately harsh stress regime we return to in Section 7.3.

This answer lands in a field that has moved since 2009. On the classical side, FNPAG @lu2015fnpag
set the numerical predictor--corrector standard and was carried across Mars aerocapture mission and
vehicle design maps @matz2017mars; more recent work replans under uncertainty explicitly --
two-stage stochastic and robust formulations that beat a deterministic predictor--corrector under
density perturbations @zucchelli2021twostage -- solves the full constrained replan by convexification
@rataczak2025cpag, or augments the bank channel with angle-of-attack modulation through optimal
control @sonandres2025abamguid. Machine learning has so far entered the loop mostly as a *component*
inside a classical scheme: LSTM density estimation feeding FNPAG @sonandres2025density, generative
failure-mode indicators steering a predictor--corrector @calkins2025riskaware. End-to-end learned
aerocapture guidance evaluated against classical baselines on common dispersed scenarios -- with the
classical side given the same objective and tuning freedom, and the comparison read at the far-tail
depth that sizes the propellant -- is, to our knowledge, the gap this paper fills.

All of this rests on a high-fidelity simulator -- $J_2$/$J_3$/$J_4$ gravity, Gauss--Markov density
perturbations, thermal limits, pilot dynamics -- regression-validated against a legacy reference
implementation
across all $725$ time steps of a guided trajectory, $22$ of $24$ output channels bit-identical (the
two mismatches trace to uninitialized variables in the reference; Appendix A lists the independent
physics checks and the simulator features this campaign does not exercise). The next
section formalizes the aerocapture problem and the sizing-tail objective; Section 3 describes the
guidance schemes; Section 4 presents the training methodology; Sections 5--7 give the optimizer,
architecture, and classical-versus-neural results; and Section 8 reports what the deployed network
actually uses.

= Problem and objective

== Dynamics and the aerocapture corridor

We study the same robotic MSR mission concept as the 2009 work. The vehicle reaches the atmospheric
entry interface at an altitude of $130$ km with a relative velocity of $5687$ m/s, a flight-path
angle of $-10.81 degree$, and an azimuth of $38.04 degree$ (the 2009 study placed the interface at
$120$ km with a $-10.24 degree$ flight-path angle; vehicle and target are unchanged); it carries a mass of $1089$ kg on a $14.7$ m#super[2]
reference area and is statically trimmed at a fixed angle of attack, so the only control is the bank
angle $mu$, slewed at up to $15 degree$/s. The guidance targets a $500 times 11$ km orbit at
atmosphere exit (apoapsis $times$ periapsis altitude), at $50 degree$ inclination, subsequently corrected to a $500$ km circular parking
orbit. Energy is dissipated by drag during a single atmospheric pass; the bank
angle rotates the lift vector to manage how much energy is shed and in which plane.

The natural state space for the maneuver is the (orbital energy, dynamic pressure) plane
(@fig-corridor). The vehicle enters with positive orbital energy (hyperbolic, $E approx +4.9$ MJ/kg)
and must exit with negative energy (bound, $E approx -5.9$ MJ/kg) without either skipping back out --
a hyperbolic escape -- or descending so deep that it crashes or violates a thermal limit. The set of
trajectories that capture is bounded by two constant-bank profiles: an overshoot boundary (full
lift-up, the limit against escape) and an undershoot boundary (full lift-down, the limit against
crash). A practical *restricted corridor* tightens these to the bank profiles that land within a
bounded apoapsis error of the target. Control authority is proportional to dynamic pressure, so the
guidance has the most leverage deep in the pass and almost none on the thin entry and exit legs.
The vehicle must respect a peak heat-flux limit of $200$ kW/m#super[2], a $4$ g load limit, and an
integrated heat-load limit of $25$ MJ/m#super[2].

#fig("fig_corridor.svg", [Empirical trajectory-occupancy envelope of the aerocapture corridor in the
(orbital energy, dynamic pressure) plane, traced by a $1\,000\,000$-run dispersed Monte Carlo of
randomized signed piecewise-constant bank profiles (roll reversals included), with the undispersed
full-lift-up and full-lift-down constant-bank boundary traces overlaid (dashed). The shaded band
spans the occupied corridor: the upper edge is the $p_(99.9)$ dynamic pressure of all capturing
trajectories (the crash-side limit), the lower edge the $p_(0.5)$ of trajectories capturing below a
$5000$ km apoapsis (the escape-side limit); as quantiles of sampled profiles these edges are
empirical, not a formal reachable set. The vehicle enters hyperbolic ($E > 0$, right) and bleeds
energy into a bound orbit ($E < 0$, left); the $200$-run Monte Carlo ensemble of the development-regime Mamba
champion and its undispersed nominal (heavy line) fly well inside the envelope.], <fig-corridor>)

== The objective: the tail that sizes the mission

Performance is measured by the *correction $Delta v$* -- the propellant cost to carry the captured
orbit to the $500$ km circular parking orbit, summed over the periapsis-raising, circularization,
and plane-change burns. At atmosphere exit the periapsis always sits inside the atmosphere and must be raised, so the
correction cost has a floor of roughly $105$ m/s at this entry interface, set by the nominal
periapsis raise (dispersed cases spread a few m/s around it; the 2009 study's interface put the
same floor near $113$ m/s); a
guidance law cannot do better than deliver the vehicle to that floor across all dispersions. We
define a run as a *capture* when it terminates in a bound orbit (the simulator's terminal flag
$i_"final" = 3$ with eccentricity $< 1$) and compute $Delta v$ over captured runs only.

The point we want to make sharply, because it governs every comparison in this paper, is that the
*mean* correction cost is operationally almost irrelevant. Aerocapture propellant tanks are sized
for the worst credible case, conventionally the $3 sigma$ design point ($approx$ the $99.87$th
percentile), not the median. Two guidance laws with the same mean but different tails are not equally
good: the one with the heavier tail forces larger tanks and a heavier, more expensive mission. We
therefore treat the *tail* of the $Delta v$ distribution as the objective, and report the conditional
value-at-risk -- the mean cost in the worst $(1-alpha)$ fraction of cases, $"CVaR"_alpha$
@rockafellar2000cvar -- at $alpha = 95%$ and, for sizing decisions, at the far-tail depth
$"CVaR"_(99.9)$, together with the $95$th and $99$th percentiles and the sample maximum (a
descriptive bound). Empirically $"CVaR"_alpha$ is the mean of
the worst $max(1, "round"((1-alpha) n))$ captured observations, without interpolation; every tail
statistic is reported with the number of observations it averages. The far tail
cannot be estimated from a #box[$1000$-case] ensemble (a single sample beyond $p_(99.9)$), so every
sizing number in this paper is computed on a pre-registered pool of $10^6$ scenarios,
training-disjoint by construction, and the shallower $n = 1000$ and $n = 10\,000$ pools carry the
$"CVaR"_95$ statistics. We lead with the tail; the mean is reported for continuity with the 2009 work.

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, center, left, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Domain*], [*Dims*], [*Distribution*], [*Dispersion (medium regime)*],
    ),
    table.hline(stroke: 0.35pt),
    [Entry state], [6], [Gaussian], [altitude $plus.minus 0.3$ km, velocity $plus.minus 3$ m/s, flight-path/azimuth $plus.minus 0.15$--$0.3degree$ ($3sigma$; medium)],
    [Atmospheric density], [1], [Uniform], [$plus.minus 50%$ multiplicative bias (medium)],
    [Aerodynamics], [3], [Uniform], [drag $plus.minus 5%$, lift $plus.minus 10%$, angle of attack $plus.minus 1degree$ (medium)],
    [Navigation errors], [7], [Gaussian], [altitude $plus.minus 2$ km, horizontal $approx plus.minus 9$ km per axis, velocity $plus.minus 1.2$ m/s, drag-accel $plus.minus 0.3$ m/s#super[2] ($3sigma$; medium)],
    [Mass], [1], [Uniform], [$plus.minus 1%$ (medium)],
    [Vehicle], [2], [Uniform], [reference area $plus.minus 2%$, max bank rate $plus.minus 10%$ (medium)],
    [Pilot dynamics], [3], [Uniform], [time constant / damping / frequency $plus.minus 10%$ (medium)],
    [Nav-filter gain], [1], [Gaussian], [$plus.minus 0.3$ absolute ($3sigma$; medium)],
    [Winds], [2], [Uniform], [speed $times [0.7, 1.3]$, direction $plus.minus 5degree$ (low; model disabled -- draws inert)],
    [Density perturbation], [process], [Gauss--Markov], [correlation time $120$ s, $5%$ RMS, time-varying, one realization per scenario (low)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The dispersion model. Twenty-six static draws across nine domains, plus a time-varying
  Gauss--Markov density perturbation (the tenth row). The medium regime uses medium presets
  except for the winds and density perturbation (low); the wind model itself is disabled in this
  campaign, so its two draws are inert. Gaussian dispersions are quoted at $3sigma$; uniform
  dispersions at their full range.],
) <tbl-dispersions>

The mission is flown under $26$ dispersed parameters across nine domains, summarized in
@tbl-dispersions, plus a time-varying Gauss--Markov (Ornstein--Uhlenbeck) density perturbation that
evolves during each run. The medium regime uses medium presets for the initial-state,
atmospheric, aerodynamic, navigation, mass, vehicle, pilot, and navigation-filter domains and low
presets for the winds and the density perturbation. The atmospheric density bias alone spans
$plus.minus 50%$ -- it is the dominant driver of apoapsis error, and a guidance law blind to it
cannot reject it. The simulator's wind model follows a parametric Mars profile @forget1999mars but is disabled for
this campaign (its two draw dimensions are carried but inert); the density perturbation is an
Ornstein--Uhlenbeck process layered on the static bias. Within a multi-scenario batch, draws are
generated by Latin-hypercube sampling @mckay1979lhs for space-filling coverage; the evaluation
pools draw one scenario per seed and are therefore plain independent samples -- which is what the
replicate- and bootstrap-based intervals of Sections 6--7 assume.

The noise model is the point to fix before any number. A scenario is its $26$ static draws *and*
its own realization of the Ornstein--Uhlenbeck process: the process's stream is seeded from a hash
of the scenario's dispersion draw, so identical draws reproduce and distinct scenarios receive
distinct density histories, and every pool of this paper -- training, validation, evaluation, the
$10^6$ confirmatory -- marginalizes over the perturbation. One exception is named once and kept
out of every table that mixes with it: the *development regime*, in which all scenarios of a pool
share a single realization of the perturbation (time-varying within each run, constant across
runs). The methodology and optimizer studies of Sections 4--5 and the architecture probes of
Appendix B were run in it; they compare like with like under one conditioning, and their numbers
are development quantities. Section 6.4 measures what the conditioning does to a trained policy.

= Guidance schemes

We benchmark the neural policies against six classical schemes that span the spectrum of aerocapture
guidance, from analytic feedback through numerical prediction. All schemes share the same simulator,
the same navigation chain, and -- where they need one -- the same kind of reference trajectory, so
the comparison isolates the guidance law itself (@tbl-schemes).

== Classical schemes

*Feedback Trajectory Control (FTC)* is the Cerimele--Gamble-derived scheme from the 2009 study
@cerimele1985simplified @gelly2009neural and our continuity baseline. It builds a virtual reference
trajectory on apoapsis control alone -- a constant-bank profile that reaches the target apoapsis
without inclination control, tabulated as $cos mu_"ref"$, $dot(h)_"ref"$, and $q_"ref"$ versus orbital
energy -- and enslaves the commanded bank to it through a proportional law on the altitude-rate and
dynamic-pressure errors,
$ cos mu_"com" = cos mu_"ref" + G_(dot(h)) (dot(h) - dot(h)_"ref") / q + G_q (q - q_"ref") / q, $ <eq-ftc>
where $q$ is the dynamic pressure and $G_(dot(h))$, $G_q$ are the altitude-rate and dynamic-pressure
feedback gains; the command is clamped to $0degree$ or $180degree$ when $|cos mu_"com"| > 1$. It decouples in-plane (apoapsis) from
out-of-plane (inclination) motion, the latter handled by a roll-reversal logic that flips the bank
sign whenever the inclination error, projected a few seconds ahead, exceeds a threshold inside an
energy-gated activation window. FTC is analytic,
fast, and -- as we will show -- only as good as its reference.

*FNPAG* is Lu's fully numerical predictor--corrector @lu2015fnpag @lu2014predictor, subsequently
carried across Mars aerocapture mission and vehicle design maps @matz2017mars. Every two-second
replan cycle it integrates the equations of motion forward to atmosphere exit and bisects the
constant capture-bank magnitude until the predicted osculating exit apoapsis matches the target,
scaling the onboard atmosphere by the navigation-estimated density factor so the predictor tracks
the measured atmosphere rather than a nominal model; the command is held between replans. It is the most accurate classical scheme and, at roughly eleven forward
integrations per replan, by far the most expensive.

*PredGuid* is the Apollo/Shuttle-heritage drag-tracking law @bairstow2006reentry @harpold1979shuttle:
it tracks a drag-versus-energy reference profile with negative feedback.

*Energy controller* tracks an energy-dissipation reference through dynamic-pressure and altitude-rate feedback.

*Equilibrium glide* holds the equilibrium-glide condition with altitude-rate damping and a velocity bias, using the
navigation-filtered density rather than a static table.

*Piecewise constant* flies an $N$-segment constant-bank profile; it is the simplest scheme, produces the reference trajectory and corridor the
other schemes consume, and -- like the full-neural network -- emits a *signed* bank, so it bypasses
the shared roll-reversal, exit, and thermal-limiter logic.

== Neural guidance

The neural policy maps an observation vector to a bank command. We generalize the 2009
single-hidden-layer feed-forward network -- five hand-picked inputs (orbital energy, eccentricity,
inclination, velocity, non-gravitational acceleration) feeding a two-output bank decoder -- along
three independent axes: the *inputs* the policy reads, the *bank decoder* it writes through, and the
*cell type* that carries state across the pass. We take them in turn.

=== Inputs

The modern policy draws from a #box[$35$-element] candidate vector; a learned input mask selects
the subset that actually reaches the network (the deployed atan2 policies use $17$). Sixteen entries
are instantaneous orbital, aerodynamic, and thermal state variables. The other nineteen are
*engineered* signals that hand the policy temporal context and a cost-aligned summary of the flown
trajectory, so it need not integrate the history itself. A few are genuinely autoregressive --
seam-free $(sin, cos)$ encodings of the recent bank history and the roll-reversal telemetry -- but
most are instantaneous functions of the current osculating orbit: reference-trajectory
interpolations ($dot(h)$ and $q$ at the current energy), a closed-form exit-bank teacher signal, the
periapsis altitude, and -- most important, as the ablation of Section 8 shows -- the three
*predicted correction-$Delta v$* components (the energy-closing, periapsis-correction, and
plane-change burns). Those last three are smooth, causal, and cost-aligned: they tell the network
what the maneuver would cost if it stopped now.

=== Bank decoder

The 2009 paper read the bank from a two-element output,
$ mu = "atan2"(o_1, o_2), $ <eq-atan2>
which we keep as the default (#raw("atan2_signed")). It wastes half its range when only the magnitude
is needed, so for magnitude-only policies the #raw("acos_tanh") decoder maps a single output smoothly
onto $[0, pi]$,
$ mu = arccos(tanh(o_1)). $
Two further single-output decoders attack the $plus.minus pi$ wrap seam a raw angle output suffers
near $mu = pi$. The #raw("scaled_pi") decoder pushes the seam out of the operating region,
$ mu = "wrap"_pi(n pi tanh(o_1)), $
while the #raw("delta") decoder applies a bounded increment on the previous realized bank,
$ mu = "wrap"_pi(mu_"prev" + Delta_max tanh(o_1)). $

=== Cell type

Where 2009 had a single hidden layer, we span six architecture families behind one
common runtime, each carrying a different kind of internal state across the atmospheric pass:

- *Dense* feed-forward -- memoryless; maps the current observation straight to a command. The
  2009-style baseline.
- *GRU* @cho2014gru -- a gated recurrent cell carrying one hidden-state vector, updated each tick
  through reset and update gates.
- *LSTM* @hochreiter1997lstm -- a recurrent cell that adds a separate long-term cell state alongside
  the hidden state, regulated by input, forget, and output gates.
- *Windowed* -- a zero-parameter FIFO buffer of the last $N$ observations flattened into a dense
  stack: explicit recent history rather than a learned state.
- *Transformer* @vaswani2017attention -- a causal-attention block attending over a fixed-length
  key/value window of recent steps.
- *Mamba* @gu2023mamba -- a selective state-space core: a linear recurrence whose gates depend on the
  input, carrying a compact SSM state.

We use *recurrent* loosely for any cell that carries internal state across the pass. All six train
and deploy through the same bit-validated Rust runtime, and all are sized so the comparison across
cell types holds the parameter budget roughly fixed. Appendix B probes three further recent
recurrent families -- closed-form continuous-time cells @hasani2022cfc, the exponential-gated and
matrix-memory xLSTM variants @beck2024xlstm, and the discretization and complex-state axes of
Mamba-3 @lahoti2026mamba3 -- against these cells at matched budget; none improves on them.

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, left, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Scheme*], [*Bank command*], [*Reference*], [*Compute*], [*Heritage / note*],
    ),
    table.hline(stroke: 0.35pt),
    [FTC], [magnitude + roll reversal], [yes], [fast], [Cerimele--Gamble apoapsis enslavement],
    [FNPAG], [magnitude + roll reversal], [no], [slow], [onboard forward integration, bisection corrector],
    [PredGuid], [magnitude + roll reversal], [yes], [fast], [Apollo/Shuttle drag tracking],
    [Energy controller], [magnitude + roll reversal], [yes], [fast], [energy-dissipation tracking],
    [Equilibrium glide], [magnitude + roll reversal], [no], [fast], [equilibrium-glide condition, nav density],
    [Piecewise constant], [signed ($N$ segments)], [no], [fast], [produces reference + corridor],
    [Neural network], [signed or magnitude], [inputs], [fast], [35-input candidate vector, stateful cells],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The benchmarked schemes. "Reference" marks dependence on a tabulated reference trajectory
  ("inputs": the network reads two reference interpolations as observations but does not enslave to
  them); "Compute" classes are quantified in @sec-deployability (fast: $#int(calc.round(ms("FTC")))$--$#int(calc.round(ms("NN-mamba")))$ ms/sim; slow: $#int(calc.round(ms("FNPAG")))$ ms/sim).
  Signed-bank schemes (full-neural, piecewise-constant) bypass the shared roll-reversal, exit-phase,
  and thermal-limiter logic.],
) <tbl-schemes>

= Training methodology

The training methodology is the load-bearing contribution. The policies are trained without
input--output pairs or any reference-tracking objective (two observation inputs interpolate the
reference table; nothing enslaves to it): each candidate network is simulated on a batch of dispersed
Monte Carlo scenarios, and its fitness is the resulting correction-$Delta v$ cost (with soft
constraint penalties), so the optimizer searches directly on mission performance. The same recipe
trained the 2009 networks. What changed -- and what makes the modern policies work -- is the
realization that *how the scenario batch is chosen, generation to generation, matters more than the
optimizer or the architecture*. The right environment is non-stationary: it keeps moving the
scenarios beneath the population so that no individual can memorize a fixed set. Four design choices
-- a genetic algorithm, an adaptive (moving) seed schedule, a tail-weighting cost transform, and a
hardest-case seed curation -- form one matched system. We take them in turn. The seed-strategy,
cost-transform, curation and allocation studies of this section and the optimizer grids of the next
were run in the development regime of Section 2.2 (one density-noise realization shared across the
scenarios of a pool); they compare like with like under that conditioning and their numbers are
development quantities. The reinforcement-learning baseline of Section 5 and everything from
Section 6 on are per-scenario.

== Seed strategy: the genetic algorithm needs a non-stationary objective

The single largest effect in the entire campaign is the seed strategy. We compared three: *fixed*
(a deterministic scenario batch, identical every generation), *rotating* (a fresh random batch every
generation), and *adaptive* (a curated batch, refreshed on a validated improvement or periodically,
drawn from the cost distribution of the best individuals). The decisive cells carry three
independent training seeds each. Under fixed seeds the genetic algorithm
is the *worst* optimizer we tested and an unstable one -- three trainings land at $160.3$, $166.9$,
and $133.5$ m/s mean ($"CVaR"_95$ $215.8$ on the first seed) -- because
it overfits the repeated scenarios, evolving a policy that is excellent on those particular draws and
mediocre elsewhere. Rotating the seeds rescues it outright and *reliably*: $119.4$ m/s three-seed
mean with a $3.5$ m/s seed range (first seed $120.0$ mean, $144.5$ $"CVaR"_95$) -- a $34$ m/s drop
in three-seed means, a $71$ m/s drop on the tail on the first seeds, and the two distributions do
not touch: the fixed schedule's luckiest seed sits $12.5$ m/s above the moving schedule's worst.
Adaptive curation is indistinguishable from rotating on the *mean* at this allocation ($119.6$
versus $119.4$, seed ranges overlapping; the first-seed $118.0$ was a favorable draw); its
contribution is the hardest-seed tail shaping of Section 4.2, which a plain rotating schedule
cannot express.

The control that makes this a statement about the *objective* and not about compute is CMA-ES. Run
on the identical fixed-versus-rotating change, CMA-ES does not move -- $125.7$ versus $126.1$ m/s
in three-seed means, seed ranges within $plus.minus 1$ (first seeds $126.9 arrow.r 127.3$)
(@fig-seed). Under fixed seeds the genetic algorithm's selection converges onto the quirks of those
particular draws; CMA-ES neither suffers under the fixed batch nor benefits from the moving one. We
state that asymmetry empirically rather than mechanistically: a plausible reading is that CMA-ES's
continuous re-estimation of its parameter-space search distribution already decorrelates successive
generations, while scenario noise chiefly perturbs its rank-based updates and step-size control --
consistent with its self-termination on the noisy objective (Section 5) -- but our experiments were
not designed to isolate the mechanism.
Rotating costs marginally more compute, but CMA-ES given the same marginal compute gains nothing, so
the lever is the non-stationarity of the objective, not the extra evaluations. The practical
consequence is striking: under a fixed objective one would deploy CMA-ES and discard the genetic
algorithm; under a moving objective the genetic algorithm becomes the best optimizer in the study.
The moving environment does not make the genetic algorithm *robust* to a moving objective -- it makes
the genetic algorithm *need* one.

#fig("fig_seed_strategy.svg", [Fixed versus rotating seeds, per optimizer. The genetic algorithm is
the worst optimizer under fixed seeds and is rescued by rotating them ($-34$ m/s in three-seed
means, $-71$ m/s at $"CVaR"_95$ on the first seeds); CMA-ES is essentially unchanged (candidate
mechanisms are discussed in the text). Black dots: three independent training seeds for the
repeated GA and CMA-ES cells -- the fixed-seed pathology is itself high-variance. The
lever is the non-stationary objective, not the extra compute.], <fig-seed>)

== Cost transform, curation, and allocation

The same worst-case-leaning logic that makes us report the tail also shapes how we train. Because the
mission is sized off the far tail, we apply a monotonic *cost transform* to each per-simulation cost
before aggregating, so that the optimizer feels expensive scenarios more sharply. Applied per
scenario before the root-mean-square aggregation over the individual's batch (Appendix A), the
cubed transform makes the per-individual objective
$ J = ( 1/n sum_(i=1)^n C_i^6 )^(1\/2) , $ <eq-objective>
a monotone function of the $L_6$ norm of the per-scenario cost vector -- a high-moment objective
that weights an individual's worst scenarios far more heavily than its typical ones. (For a single
scenario a monotone transform is ranking-neutral; across a batch it deliberately is not.) We prefer
this smooth high-moment proxy over optimizing an empirical tail quantile directly because the
deployed allocation evaluates as few as two scenarios per individual per generation -- far too few
to estimate a quantile -- while the moving batch supplies tail coverage across generations.
Evaluated at the depth that matters -- a far-tail
$n = 10\,000$ pool -- the cubed transform compresses the extreme tail best ($"CVaR"_(99.9)$ $153.0$ m/s,
sample max $160.1$) against linear ($156.7$/$162.2$), square-root ($158.4$/$167.1$), squared
($162.7$/$180.9$), and logarithmic ($162.3$/$180.6$). The logarithm is worst across the shallow and
mid tail because it over-compresses and starves the gradient between captures; only at the very
extreme is the squared transform worse still. These are single training runs, and the cubed-versus-linear
margin ($3$--$5$ m/s) is within the run-to-run scatter we measure in Section 6, so we read the
direction -- deeper tail-weighting pays at deeper sizing depth -- rather than the exact ranking as
the finding. A shallower metric ($"CVaR"_95$) would have mildly favored
square-root; the deeper we look into the tail, the more the tail-weighting pays, which is exactly why
the sizing depth must decide (@fig-cost).

Seed *curation* is the same mechanism applied to the scenarios rather than the cost. At each
refresh the adaptive strategy bins the cost distribution of the best individuals into quantiles and
picks one representative seed per bin; the choice of representative is the lever. Picking the
*hardest* seed per bin (the "max" bucket) dominates the far tail ($"CVaR"_(99.9)$ $153.0$ m/s) against
the bin median ($193.9$), a random pick ($173.1$), and the *easiest* seed ($225.9$). The easiest-seed
bucket is the cautionary tale: the best mean, $117.8$ m/s, and a catastrophic worst case -- it drops
captures and reaches $245$ m/s, optimizing the average at the expense of the worst case. Trimming
the cost distribution helped nothing (@fig-curation). These too are single runs, but the min- and
middle-bucket gaps ($40$--$70$ m/s) sit far outside run scatter. The cost transform
and the curation bucket are therefore one idea -- force the policy onto the hard cases -- expressed
twice.

Finally, *allocation*: given a compute budget, is it better spent on more scenarios per generation or
more generations? Under rotating seeds with a fixed generation count the sweet spot is moderate
($n_"sims" = 10$). But under the adaptive strategy with the budget spent on generations, the answer
flips hard: just $n_"sims" = 2$ per generation, run for many generations, gives the best mean
($109.9$ m/s) and $"CVaR"_95$ ($117.5$), beating $5$, $20$, and $100$ (@fig-nsims). The few-sample
noise per generation is bought back, and then some, by the far greater scenario diversity the moving
environment sees over the run. The deployed headline policy uses this allocation.

#grid(columns: 2, gutter: 6pt,
  fig("fig_cost_transform.svg", [Cost transform versus the sizing tail, at the sizing depth
  ($n = 10\,000$ pool). The cubed transform minimizes both the far-tail $"CVaR"_(99.9)$ and the
  sample maximum, even though a shallow $"CVaR"_95$ read would mildly favor square-root.], <fig-cost>),
  fig("fig_training_n_sims.svg", [Allocation of the compute budget ($"CVaR"_95$; the mean orders both
  series identically). Under the adaptive schedule, two scenarios per generation over many
  generations dominates larger per-generation batches.], <fig-nsims>))
#fig("fig_curation.svg", [Seed curation. Left: far-tail $"CVaR"_(99.9)$ ($n = 10\,000$) per curation
  bucket -- selecting the hardest seed per cost-CDF bin (the "max" bucket) compresses the sizing
  tail, and the easiest-seed bucket blows it up (worst case $245$ m/s). Right: mean ($n = 1000$) --
  the easiest bucket wins the mean, the optimize-the-average trap; the trim variants match the max
  bucket and were not carried to the far-tail pool.], <fig-curation>)

That the optimal allocation pours the budget into generations points to the last methodological fact:
training here is *compute-bound, not overfitting-bound*. A stationary objective eventually overfits,
and one stops early. The moving objective never converges to a fixed landscape, so the validation
error -- the RMS on the reserved selection pool of @tbl-pools; we keep the conventional name for
the loss curve -- keeps falling for nearly the whole twenty-thousand-generation run and then
plateaus rather than degrading, i.e. the policy does not memorize scenarios (@fig-plateau). The plateau also exposes a counter-intuitive dimensionality effect that recurs in the
architecture results: the #box[$972$-parameter] dense network learns *faster* early -- more plasticity --
but the #box[$515$-parameter] network overtakes it and plateaus *lower* (validation RMS $1.326 times 10^6$
versus $1.433 times 10^6$). For the gradient-free genetic algorithm, the extra dense parameters are
more search burden than added capacity; the "more parameters, learn faster" intuition does not
transfer.

#fig("fig_plateau.svg", [Best validation RMS versus generation for the two dense reference networks.
Both keep improving until late in the twenty-thousand-generation run and then plateau -- under the
non-stationary objective the policy generalizes across the dispersion distribution rather than
memorizing scenarios. The #box[$515$-parameter] network plateaus below the
#box[$972$-parameter] one: beyond a few hundred parameters, extra dense capacity hurts the
gradient-free search.], <fig-plateau>)

#figure({
  set text(size: 9pt)
  table(
    columns: (auto, auto, auto, auto, 2.1in),
    align: (left, center, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header([*Pool*], [*n*], [*Noise*], [*Queries*], [*Decisions taken on it*]),
    table.hline(stroke: 0.35pt),
    [Training batches], [$2$/gen], [per-scenario], [every generation], [weight updates (moving, curated)],
    [Selection (offset 1M)], [$1000$], [per-scenario], [$approx 13\,000$ per run], [in-training argmin promotion],
    [Development far tail (offset 2M)], [$10\,000$], [shared path], [tens], [cost transform, curation bucket, allocation (Section 4)],
    [Final evaluation (offset 2M)], [$1000$], [per-scenario], [once per cell], [none -- the paired $"CVaR"_95$ tables],
    [Fresh re-quote (offset 8M)], [$1000$], [per-scenario], [once; the grid], [quantization cell choice (Appendix C)],
    [Confirmatory sizing (App. A)], [$10 times 100\,000$], [per-scenario], [once, post-freeze], [none -- every quoted sizing number],
    [Off-nominal stress (offset 9M)], [$10\,000$], [per-scenario], [once per policy], [none (robustness probe, Section 7)],
    [Architecture probes (offset 10M)], [$1000$], [shared path], [once per arm-repeat], [none (Appendix B verdicts)],
    table.hline(stroke: 0.7pt),
  )},
  caption: [Scenario-pool roles, the noise regime each was run in, and the decisions each pool
  influenced. The selection pool is adaptively reused by the promotion gate and the development
  far-tail pool informed the methodology choices, so neither is an unbiased test set. The
  confirmatory pool was generated from a seed range disjoint from every earlier draw, after all
  methodology, architecture, and checkpoint choices were frozen, and each cell was evaluated on it
  exactly once; the deployed seed of every cell was fixed by rule before the pool was queried.],
) <tbl-pools>

The studies of this section and the next are exploratory -- single training runs in the
development regime unless stated otherwise; the confirmatory statements of this paper are the
per-scenario frozen-pool quantities of Sections 6--7.

= Optimizer and dimensionality

Every optimizer in this study is a gradient-free population method, a deliberate choice rather than
an omission. The training objective is the correction $Delta v$ of a complete atmospheric pass --
produced by a simulator with discrete capture, crash, and atmosphere-exit termination,
threshold-triggered phase transitions, and hard constraint limits -- so it is a black-box function
of the network weights with no usable gradient. Policy-gradient reinforcement learning
(PPO @schulman2017ppo, SAC @haarnoja2018sac) does not require a differentiable simulator or a
differentiable reward -- it estimates gradients from sampled rollouts and can in principle optimize
a terminal-only objective. We trained PPO under the protocol of the population cells it is
compared against: the same #box[$17$-input] observation contract, per-input normalization and
`atan2` decoder, the same architecture and co-tuned navigation and command-shaping scaffolding as
the #box[$515$-weight] dense and #box[$1014$-weight] GRU cells of Section 6 (the GRU trained by PPO
with truncated backpropagation through time), per-scenario noise, a reserved training pool
(offset $3 times 10^6$), promotion on the reserved selection pool and the quote on the reserved
final-evaluation pool, paired against the population cell (@tbl-rl). The reward is potential-based
per-step shaping aligned to the predicted correction cost plus the true terminal cost, the budget
is $30 times 10^6$ environment steps per cell, the deployed artifact is the best validation
checkpoint, and the environment flies exactly what the deployed network flies (a bit-identity gate
holds the two loops together). From scratch, PPO lands $#R.fixed(calc.min(..("v4_ppo_scratch_vs_pop_dense515", "v4_ppo_scratch_s2_vs_pop_dense515", "v4_ppo_scratch_s3_vs_pop_dense515").map(k => pr(k).delta_mean)), d: 0)$--$#R.fixed(calc.max(..("v4_ppo_scratch_vs_pop_dense515", "v4_ppo_scratch_s2_vs_pop_dense515", "v4_ppo_scratch_s3_vs_pop_dense515").map(k => pr(k).delta_mean)), d: 0)$ m/s above the dense population cell
in paired mean over three torch seeds and $#R.fixed(calc.min(..("v4_ppo_scratch_vs_pop_gru1014", "v4_ppo_scratch_s2_vs_pop_gru1014", "v4_ppo_scratch_s3_vs_pop_gru1014").map(k => pr(k).delta_mean)), d: 0)$--$#R.fixed(calc.max(..("v4_ppo_scratch_vs_pop_gru1014", "v4_ppo_scratch_s2_vs_pop_gru1014", "v4_ppo_scratch_s3_vs_pop_gru1014").map(k => pr(k).delta_mean)), d: 0)$ above the GRU one, the seed-to-seed spread as large as the gap, and
every scratch cell still improving at the budget. Warm-started from the population champion, PPO
deploys the champion (the first validation checkpoint; paired mean $#R.signed(pr("v4_ppo_warm_vs_pop_dense515").delta_mean)$ and
$#R.signed(pr("v4_ppo_warm_vs_pop_gru1014").delta_mean)$ m/s) and then walks off it, consistent with the stochastic shaped return optimizing a
different quantity than the deterministic mission cost. Population search on the mission cost
itself was simply the stronger tool here, so throughout we optimize the mission cost directly.

// One tbl-rl row: the PPO cell's n = 1000 statistics (results.json, per_draw), its any-constraint
// violation share on the 10^6 pool, and its paired deltas against the population cell of the same
// architecture (negative favors PPO).
#let ci8(iv, d: 1) = text(size: 8pt)[\[#R.ci(iv, d: d)\]]
#let rl_row(name, key, pair) = {
  let r = R.per_draw_regime(key)
  let c = R.marg("paper/" + key)
  let p = pr(pair)
  (name, [#R.fixed(r.capture_pct)], [#R.fixed(c.viol_pct, d: 2)], [#R.fixed(r.dv_mean)], [#R.fixed(r.dv_cvar95)],
    [$#R.signed(p.delta_mean)$ #ci8(p.delta_mean_ci)], [$#R.signed(p.delta_cvar95)$], [#R.fixed(p.win_rate_a * 100)])
}
#figure({
  set text(size: 9pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([*PPO cell*], [*Capture %*], [*Viol. %*], [*Mean*], [$bold("CVaR"_95)$], [$bold(Delta"mean")$ vs pop.], [$bold(Delta"CVaR"_95)$], [*PPO-win %*]),
    table.hline(stroke: 0.35pt),
    ..rl_row([Dense, scratch, s1], "rl/hl_dense_p515_ppo_scratch", "v4_ppo_scratch_vs_pop_dense515"),
    ..rl_row([Dense, scratch, s2], "rl/hl_dense_p515_ppo_scratch_s2", "v4_ppo_scratch_s2_vs_pop_dense515"),
    ..rl_row([Dense, scratch, s3], "rl/hl_dense_p515_ppo_scratch_s3", "v4_ppo_scratch_s3_vs_pop_dense515"),
    ..rl_row([Dense, warm start], "rl/hl_dense_p515_ppo_warm", "v4_ppo_warm_vs_pop_dense515"),
    table.hline(stroke: 0.35pt),
    ..rl_row([GRU, scratch, s1], "rl/hl_gru_p1014_ppo_scratch", "v4_ppo_scratch_vs_pop_gru1014"),
    ..rl_row([GRU, scratch, s2], "rl/hl_gru_p1014_ppo_scratch_s2", "v4_ppo_scratch_s2_vs_pop_gru1014"),
    ..rl_row([GRU, scratch, s3], "rl/hl_gru_p1014_ppo_scratch_s3", "v4_ppo_scratch_s3_vs_pop_gru1014"),
    ..rl_row([GRU, warm start], "rl/hl_gru_p1014_ppo_warm", "v4_ppo_warm_vs_pop_gru1014"),
    table.hline(stroke: 0.7pt),
  )},
  caption: [PPO against the population-trained cell of the same architecture (the #box[$515$-weight]
  dense and #box[$1014$-weight] GRU cells), correction $Delta v$ in m/s. Capture, mean and $"CVaR"_95$ on the $n = 1000$ final-evaluation pool; Viol. is the share
  of the $10^6$ confirmatory pool exceeding any constraint (the population trainer's feasibility
  gate has no counterpart in the RL loop: the unseeded dense scratch run bought its capture with
  heat flux); the deltas are paired against the population cell (negative favors PPO) with a
  $10\,000$-resample bootstrap $95%$ interval on the mean. Seed 1 of each scratch cell ran before
  the torch seed was pinned; seeds 2 and 3 are repeats.],
) <tbl-rl>

Having established that the genetic algorithm is the right optimizer under a moving objective, two
questions remain: does it need a population that scales with the search dimension, and does the
optimizer choice even matter for the low-dimensional classical-gain problems? We compared the genetic
algorithm @goldberg1989genetic against CMA-ES @hansen2001cmaes, particle-swarm optimization
@kennedy1995pso and its quantum-behaved variant @sun2004qpso, differential evolution @storn1997de, and a
three-island heterogeneous model, on an optimizer-by-budget grid at the largest dense network
($3998$ weights) and an optimizer-by-dimension grid spanning the #box[$26$-parameter] FTC-gain problem and
the #box[$515$-parameter] dense network.

At $3998$ weights the population size is decisive (@fig-optimizer). The genetic algorithm is best at a
population of $150$ ($118.0$ m/s mean; three-seed $119.6$, range $3.0$) and $300$ ($120.5$ m/s) -- a gap smaller than the run-to-run
scatter we measure on retrained cells in Section 6, so we report them as indistinguishable -- but at
a population of $60$ it *collapses* to
$166.3$ m/s. Sixty individuals cannot cover a four-thousand-dimensional weight space; selection
drifts. So the tempting generalization that the genetic algorithm dominates at any budget is wrong:
at a starved population it is no better than a single restart. The corrective is simple and worth
stating plainly -- the population must scale with the search dimension. CMA-ES improves smoothly with
budget ($133.3 arrow.r 126.3 arrow.r 121.8$ m/s) but never reaches the genetic algorithm's optimum and
self-terminates on the noisy objective before exhausting a generous generation count, an asymmetry to
keep in mind for compute-matched comparisons. The three-island heterogeneous trainer (particle swarm,
genetic, and differential evolution, with periodic migration) is the most budget-robust of all,
$120$--$124$ m/s across every budget, but the well-populated single genetic algorithm edges it.

The dimensionality grid carries the more useful lesson. On the #box[$26$-parameter] FTC-gain problem every
gradient-free optimizer we ran lands in a tight band, roughly $170$--$178$ m/s -- islands $169.8$,
particle swarm $170.9$, CMA-ES $171.7$, quantum-behaved swarm $172.9$, differential evolution
$178.2$ -- an $8$ m/s spread we do not attempt to rank (Section 9). Optimizer choice barely matters when there are only twenty-six gains
to tune. It is at neural-network dimensionality that the optimizers separate: at $515$ weights the
genetic algorithm is best ($117.4$ m/s), the islands next ($118.6$), and particle swarm worst
($129.8$), a $12$ m/s spread. The confound is that the #box[$26$-parameter] cell is a different
guidance scheme (FTC), not a narrowed version of the same network, so the comparison conflates
dimension with law; but the direction is clear -- the optimizer earns its keep on the high-dimensional
weight search, not on low-dimensional gain tuning.

#fig("fig_optimizer.svg", [Optimizer by population budget at $3998$ weights. The genetic algorithm is
best at populations of $150$--$300$ but collapses at $60$ (the population must scale with the search
dimension); the heterogeneous island model is the most budget-robust; CMA-ES improves with budget but
trails the genetic optimum.], <fig-optimizer>)

The island model -- the most budget-robust optimizer above -- was built for exactly that robustness
(@fig-islands). It splits one population budget across three islands running complementary operators
(particle swarm, genetic, and differential evolution) on the same scenarios, and every $k$
generations migrates each island's best $n$ individuals into the others, overwriting their worst.
Two properties follow. The heterogeneous operators explore the weight space differently and the
migration cross-pollinates their discoveries, so a sub-population trapped in a local optimum is
pulled out by a migrant from another island instead of having to escape on its own -- the diversity
a single homogeneous method lacks. And because the three strategies share one run's evaluations
through migration, one need not run each optimizer separately and keep the best. The result never
collapses the way a starved genetic algorithm does (mean $120$--$124$ m/s across every budget); the price
is that at a well-chosen budget a single large genetic algorithm still edges it, so the islands buy
robustness to the budget choice rather than a lower optimum.

#fig("fig_islands.svg", [The three-island heterogeneous optimizer. Three islands run complementary
gradient-free operators (particle swarm, genetic, differential evolution) on the same scenarios;
periodic migration exchanges each island's best individuals for the others' worst. The heterogeneity
plus migration is what escapes local optima, and one population budget covers all three searches.], <fig-islands>)

= Architecture: the headline result

We now sweep the cell type. The finding comes in two halves: a flat median across cell types, and
a separation that appears only on the *tail* -- between the networks and the classical laws at
every depth, and between the networks themselves only inside their run-to-run spread.

== Parameter budget and the capability floor

A parameter-budget sweep across six families (dense, GRU, LSTM, windowed, Transformer, Mamba), each
trained identically at two scenarios per generation for five thousand generations in the
development regime, makes the first
half of the thesis concrete (@fig-pareto). Every cell in the sweep captures $100%$ of the
time -- there is no capability collapse, and a dense network with as few as $102$ parameters still
guides the maneuver at $100%$ capture and $120.8$ m/s mean. Within the dense family the cost is flat
from a few hundred to a few thousand parameters ($515 arrow.r 972 arrow.r 1957$ weights give
$117.4 arrow.r 116.9 arrow.r 116.8$ m/s) and a #box[$3998$-weight] network gains nothing further -- it is
over-parameterized for the genetic search, consistent with the plateau result of the training
methodology (Section 4).
At a matched budget the recurrent and state-space families edge the best dense cell (LSTM $112.0$,
GRU $112.8$, Mamba $114.9$ versus the best dense $116.8$), while the Transformer pays an
attention-overhead penalty at small budgets -- its worst cell, at $762$ parameters, is the worst in
the whole sweep ($121.9$ m/s), and not for lack of parameters ($762 > 515$): a small budget forces the
attention block to a tiny model width and two dimensions per head, which the genetic algorithm trains
poorly. The Pareto front is therefore shallow on the mean; the interesting structure is in the tail,
which a five-thousand-generation sweep cannot resolve.

#fig("fig_pareto.svg", [Parameter budget versus sizing tail per architecture family (left) and the
dense-family capability floor (right), in the development regime. Every cell captures $100%$; the dense cost is flat above a few
hundred parameters; recurrent and state-space families edge dense at matched budget; the Transformer
is penalized at small budgets. The left panel plots the $99$th-percentile cost and the right panel
$"CVaR"_95$; the body quotes means, under which the flat cells are indistinguishable.], <fig-pareto>)

== The tail at sizing depth

To resolve the tail we trained the strongest recurrent and state-space candidates plus the dense
references from scratch at one allocation -- two scenarios per generation, a population of $512$,
$20\,000$ generations, per-scenario noise, the full methodology of Section 4 -- and evaluated
each once on the frozen confirmatory pool ($10 times 100\,000$ scenarios). The Mamba and the
#box[$515$-parameter] dense network carry three independent training seeds each; the GRU, the LSTM
and the #box[$972$-parameter] dense network one. Seed 1 of every cell is the deployed one, fixed by
rule before the pool was queried; seeds 2 and 3 report the mean and the spread (@tbl-arch,
@fig-archtail).

// One tbl-arch row: capture (4 decimals: a seed losing 4 of 10^6 reads 99.9996), heat-load
// exceedances per 10^6, CVaR95, CVaR99.9 +- replicate s.e., worst case; the mean rows carry the
// three-seed mean +- sd of the two CVaRs.
#let arch_row(name, c, cap_d: 4) = (name, [$#R.fixed(c.capture_pct, d: cap_d)$], [$#c.hl_viol_n$], [$#R.fixed(c.cvar95)$],
  [$#R.fixed(c.cvar999) plus.minus #R.fixed(c.cvar999_se)$], [$#R.fixed(c.max, d: 0)$])
#let arch_mean_row(name, seeds) = {
  let m95 = R.mean_sd(seeds.map(c => c.cvar95))
  let m999 = R.mean_sd(seeds.map(c => c.cvar999))
  (name, [], [], [$#R.fixed(m95.mean) plus.minus #R.fixed(m95.sd)$], [$#R.fixed(m999.mean) plus.minus #R.fixed(m999.sd)$], [])
}
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([*Cell*], [*Capture %*], [*Heat load $>$ limit (per $10^6$)*], [$bold("CVaR"_95)$], [$bold("CVaR"_(99.9) plus.minus "s.e.")$], [*Max*]),
    table.hline(stroke: 0.35pt),
    ..arch_row([Mamba $962$, seed 1 (deployed)], champ),
    ..arch_row([Mamba $962$, seed 2], mamba_seeds.at(1)),
    ..arch_row([Mamba $962$, seed 3], mamba_seeds.at(2)),
    ..arch_mean_row([_Mamba $962$, three-seed mean $plus.minus$ sd_], mamba_seeds),
    table.hline(stroke: 0.35pt),
    ..arch_row([Dense $515$, seed 1 (efficiency reference)], dense),
    ..arch_row([Dense $515$, seed 2], dense_seeds.at(1)),
    ..arch_row([Dense $515$, seed 3], dense_seeds.at(2)),
    ..arch_mean_row([_Dense $515$, three-seed mean $plus.minus$ sd_], dense_seeds),
    table.hline(stroke: 0.35pt),
    ..arch_row([GRU $1014$], gru),
    ..arch_row([Dense $972$], dense972),
    ..arch_row([LSTM $1082$], lstm),
    table.hline(stroke: 0.35pt),
    ..arch_row([FTC, joint reference (retuned)], jftc),
    ..arch_row([FNPAG (retuned)], fnpag),
    table.hline(stroke: 0.7pt),
  ),
  caption: [The headline-allocation cells on the confirmatory pool ($10 times 100\,000$ scenarios
  per cell, per-scenario noise): correction $Delta v$ in m/s over captured scenarios, $"CVaR"_(99.9)$
  with its standard error over the ten replicates, the sample maximum as a descriptive bound. The
  heat-load column counts the scenarios on which the integrated heat load exceeds the $25$ MJ/m#super[2]
  limit; no cell exceeds the heat-flux or g-load limits. The two retuned classical references of
  Section 7 close the table.],
) <tbl-arch>

Read family by family, the networks beat the classical laws at sizing depth. The Mamba's three
seeds average $"CVaR"_(99.9) = #R.fixed(mamba3.mean) plus.minus #R.fixed(mamba3.sd)$ m/s, the dense
reference's $#R.fixed(dense3.mean) plus.minus #R.fixed(dense3.sd)$, and the single-seed GRU, dense-$972$ and LSTM
cells land in the same range (#net_span999 across the nine runs), against $#R.fixed(jftc.cvar999)$
for the best classical scheme and $#R.fixed(fnpag.cvar999)$ for FNPAG: eight of the nine network
seeds sit below joint-FTC and all nine below FNPAG. Read seed by seed, the ordering between
architectures does not survive. The two three-seed means sit within either family's seed
standard deviation of each other, the single-seed cells sit inside both spreads, and the deployed
Mamba seed -- the lowest validation loss of its three and the best $"CVaR"_95$ of the nine
($#R.fixed(champ.cvar95)$) -- has the worst far tail of its family ($#R.fixed(champ.cvar999)$, $#R.fixed(margin_champ_jftc)$ m/s under
joint-FTC). The development-regime ordering we reported earlier (Mamba below LSTM below dense,
with seed ranges that barely touched) was a property of the shared density history (Section 6.4);
under per-scenario noise the seed spread is the architecture effect's size, and a sizing engineer
should quote the family mean with its spread, not one seed. The sample maximum, as before, retires
as a comparison statistic at this depth: at $10^6$ scenarios it is one draw from the extreme tail
(seed 2 of the Mamba, the second-best $"CVaR"_(99.9)$ of its family, logged a $#R.fixed(mamba_seeds.at(1).max, d: 0)$ m/s excursion).

Feasibility is reported beside every tail number because it is no longer free. Every network
exceeds the integrated heat-load limit on some scenarios of the $10^6$ pool -- #net_hl_span per
million across the nine cells, $#champ.hl_viol_n$ for the deployed seed -- where the retuned
joint-FTC exceeds nothing and FNPAG exceeds it on $#fnpag.hl_viol_n$; no cell exceeds the heat-flux
or g-load limits. The training gate holds every candidate to zero violations on its $n = 1000$
selection pool, so these are exceedances the gate cannot see, at the $10^(-4)$--$10^(-3)$ level,
and the appendix cards (Appendix D) put the deployed seed's worst heat load a fraction of a
percent over the limit. We therefore report the share rather than exclude the cell, and we read
the ceiling itself as a design margin: fine-tuning the deployed seed and the dense reference under
ceilings relaxed by $10%$ and $20%$ moves their $"CVaR"_95$ by $#R.signed(R.slope_summary("mamba_p962").cvar95_q30_minus_q25)$ and
$#R.signed(R.slope_summary("dense_p515").cvar95_q30_minus_q25)$ m/s, inside each cell's seed spread
($plus.minus #R.fixed(R.slope_summary("mamba_p962").source_seed_sd)$ and $plus.minus #R.fixed(R.slope_summary("dense_p515").source_seed_sd)$), so the heat-load ceiling
is not what sets the tail, and a policy carries its training ceiling (re-flown under the nominal
limit the relaxed policies exceed it on #relaxed_v4_viol$%$ of scenarios). Capture is the other half of the
feasibility check: two of the Mamba seeds capture $10^6$ of $10^6$ scenarios, the third loses
$#mamba_seeds.at(1).lost$; the dense seeds lose $#dense_seeds.at(1).lost$ and $#dense_seeds.at(2).lost$; FNPAG loses $#fnpag.lost$ (all physical crashes on
individual re-runs without the evaluation timeout).

#fig("fig_arch_tail.svg", [Per-seed $"CVaR"_95$ (left) and $"CVaR"_(99.9)$ (right) on the
confirmatory pool ($10 times 100\,000$ scenarios per seed, per-scenario noise) for the
headline-allocation cells; bars are seed means, whiskers on the right are $plus.minus 1.96$
replicate standard errors (hidden under the marker where shorter), the retuned joint-FTC and FNPAG
are dashed. The deployed cell is seed 1 of the Mamba (starred): the best point on the left panel
and the worst of its three on the right. Every family mean sits below both classical references;
the families overlap one another.], <fig-archtail>)

== What the controls say about state

The mechanism we claimed from the development regime -- that genuine internal state buys the tail
-- is the claim the per-scenario campaign tests most directly, with three controls evaluated once
each on the confirmatory pool. *State reset*: the deployed Mamba evaluated with its recurrent state
zeroed at every guidance tick captures $#R.fixed(reset.capture_pct, d: 2)%$ of scenarios but at a
$"CVaR"_(99.9)$ of $#R.fixed(reset.cvar999, d: 0)$ m/s against $#R.fixed(champ.cvar999, d: 0)$, and the whole distribution moves,
not only the far tail: $"CVaR"_95$ goes from $#R.fixed(champ.cvar95, d: 0)$ to $#R.fixed(reset.cvar95, d: 0)$. The trained policy computes with its state;
it is not a feedforward law in disguise. *Matched history*: a dense cell fed an explicit five-tick
observation window ($970$ parameters, no learned state, identical allocation and regime) retrained
over three seeds reaches $"CVaR"_(99.9) = #R.fixed(window3.mean) plus.minus #R.fixed(window3.sd)$ m/s. *No
predicted-$Delta v$*: the same Mamba architecture retrained without the three cost-aligned inputs
($914$ parameters, three seeds) reaches $#R.fixed(nodv3.mean) plus.minus #R.fixed(nodv3.sd)$. Against the champion
family's $#R.fixed(mamba3.mean) plus.minus #R.fixed(mamba3.sd)$, the window control sits
$#R.signed(window3.mean - mamba3.mean)$ m/s and the stripped network $#R.signed(nodv3.mean - mamba3.mean)$ (Welch $t = #R.fixed(welch_t(window3, mamba3), d: 2)$ and
$#R.fixed(welch_t(nodv3, mamba3), d: 2)$ on three seeds a side); the window's mean rests on one seed
($#R.fixed(calc.max(..window_seeds.map(c => c.cvar999)))$), its other two sit inside the champion's range, and its seed spread is the widest
of the three families. Neither difference is resolved. The honest statement is therefore
narrower than the one we set out to make: the deployed policy depends on its own state (the reset
control), but at this allocation and budget neither learned state nor the engineered cost-to-go
inputs buys a far-tail advantage that three seeds can see. What the architecture buys is not
settled by this campaign; what it costs is (Section 7.2).

Training loss does not pick the tail winner, and the per-scenario runs make the point more
sharply than the development regime did. Across the nine headline-allocation runs the best
validation RMS -- the training objective on the reserved selection pool -- is not positively
correlated with the far tail (@fig-losstail): the deployed Mamba seed has the lowest loss of its
family and its worst $"CVaR"_(99.9)$, and the dense seed with the lowest loss of its family has the
widest tail of the nine. The selection gate promotes on that loss, so a deployed seed is the seed
the gate liked, and the far tail is a quantity the gate cannot see; this is why every network
cell carries its seed spread and why the deployed seed was fixed by rule rather than selected.
The reason the median stays flat is visible from the inside (Section 8): the engineered,
cost-aligned inputs already encode most of what a memory cell could recover, so every architecture
trained to convergence reaches the same #net_p50_span m/s typical cost. What remains open is
whether more sophisticated memory would help: Appendix B probes three further recurrent families
(CfC, the xLSTM cells, Mamba-3's axes) at matched budget in the development regime, and none beats
the plain cells -- two arms are significantly worse.

#fig("fig_loss_vs_tail.svg", [Best validation RMS versus far-tail $"CVaR"_(99.9)$ on the
confirmatory pool for the nine headline-allocation runs (filled) and the four bundled fine-tune runs
of the other recipe (hollow; Section 6.4). The Spearman rank correlation over the nine
same-allocation runs is printed on the figure and read as descriptive (the runs are not
exchangeable across families): the loss the gate selects on does not order the sizing tail.], <fig-losstail>)

The deployed headline policy is seed 1 of the Mamba cell: a #box[$962$-parameter]
dense-to-selective-state-space-to-dense stack (a #box[$17$-input] dense encoder, a Mamba core of inner
width $16$ and state size $12$, a two-output dense decoder with the atan2 bank decoder), trained under
the full methodology of Section 4 and deployed by the rule stated before any confirmatory score
existed. It captures $100%$ of the time at $#R.fixed(R.fresh4.dv_mean)$ m/s mean and $#R.fixed(R.fresh4.dv_cvar95)$ m/s
$"CVaR"_95$ on a fresh, never-trained-or-selected-on pool, against $#R.fixed(champ_run.dv_mean)$ and
$#R.fixed(champ_run.dv_cvar95)$ on the final-evaluation pool of @tbl-perf, and on the confirmatory pool
captures $10^6$ of $10^6$ scenarios at $"CVaR"_95 = #R.fixed(champ.cvar95)$: there is no selection
optimism in the deployed seed's shallow tail, and its far tail is the one @tbl-arch reports. The
#box[$515$-parameter] dense network is the *efficiency reference*: half the parameters, no internal state,
a competitive median and, on this evidence, a far tail inside the Mamba's spread; if compute or
simplicity is the binding constraint it is the better pick.

== Noise conditioning: a policy with state fits the history it sees

The regime matters more than the architecture, and the development regime shows why. There every
scenario of a pool shared one realization of the density perturbation, and the policies trained
under it were the headline of an earlier version of this work. Scored on a paired $n = 1000$ pool
under both regimes (@tbl-conditioning), the classical laws lose #R.span(ou_laws.map(ou_shift)) m/s of
$"CVaR"_95$ when each scenario receives its own realization -- ordinary distribution widening --
while the five shared-path-trained networks lose #R.span(ou_nets.map(ou_shift)), $2$--$4 times$ more, and every
one adds heat-load violations, which no law incurs. A policy with internal state can fit the one
density history it ever sees, and did. The nine cells of @tbl-arch, trained under per-scenario
noise, score the two regimes within #R.span(hl_shifts.map(calc.abs), d: 1) m/s of each other: training on the
marginal distribution is what makes a policy's number a marginal number, and it is the one change
between that earlier headline and this one.

// One tbl-conditioning row from data/quote_marginal.json: shared-path CVaR95, per-scenario CVaR95,
// each with what the cell gives up under that regime (capture below 100%, any heat-load violation;
// the same rule for every row), and their difference.
#let ou_gives_up(c, who: none) = {
  let parts = ()
  if c.capture_pct < 100 { parts.push([$#R.fixed(c.capture_pct)%$ capture]) }
  if c.viol_pct > 0 { parts.push([$#R.fixed(c.viol_pct)%$ heat-load viol.]) }
  if parts.len() == 0 { none } else { text(size: 8pt)[(#if who != none [#who ]#parts.join([, ]))] }
}
#let ou_value(label, regime) = {
  let c = R.ou(label, regime: regime)
  [$#R.fixed(c.cvar95)$ #ou_gives_up(c)]
}
#let ou_regime_row(policy, label) = {
  let shift = ou_shift(label)
  assert(shift > 0, message: label + " does not lose tail under per-scenario noise: the table sets every delta with a math plus")
  (policy, ou_value(label, "frozen"), ou_value(label, "marginal"), [$+#R.fixed(shift)$])
}
#let ou_pgftc = ("pred_guid", "ftc")
#let ou_pgftc_span(regime) = {
  let notes = ou_pgftc.zip(([PredGuid], [FTC])).map(((l, who)) => ou_gives_up(R.ou(l, regime: regime), who: who)).filter(n => n != none)
  [#R.span(ou_pgftc.map(l => R.ou(l, regime: regime).cvar95)) #notes.join([ ])]
}
#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([Shared-path-trained policy], [Shared path], [Per-scenario], [$Delta$]),
    table.hline(stroke: 0.4pt),
    ..ou_regime_row([Mamba $962$], "mamba_p962"),
    ..ou_regime_row([LSTM $1082$], "lstm_p1082"),
    ..ou_regime_row([GRU $1014$], "gru_p1014"),
    ..ou_regime_row([Dense $972$], "dense_p972"),
    ..ou_regime_row([Dense $515$], "dense_p515"),
    table.hline(stroke: 0.4pt),
    ..ou_regime_row([FNPAG], "fnpag"),
    [PredGuid / FTC], ou_pgftc_span("frozen"), ou_pgftc_span("marginal"),
    R.span(ou_pgftc.map(ou_shift), f: R.signed),
    table.hline(stroke: 0.7pt),
  ),
  caption: [Shared-path versus per-scenario $"CVaR"_95$ (m/s, paired $n = 1000$) for the
  development-regime champions and the classical laws tuned in that regime. Beside each value, a
  capture rate below $100%$ and any heat-load violation are noted, and nothing else is. The
  networks give up $2$--$4 times$ more tail than the classical schemes; on the per-scenario tail
  the development-regime FNPAG beats every shared-path-trained network.],
) <tbl-conditioning>

A final architectural detail concerns the bank decoder. Among the single-output decoders that attack
the $plus.minus pi$ wrap seam, the classical two-output atan2 decoder of @eq-atan2 still
wins: it reaches $117.4$ m/s mean and $128.7$ m/s $"CVaR"_95$ against the delta decoder ($119.9$/$141.6$)
and the scaled-$pi$ decoder ($122.2$/$140.4$), with the edge concentrated on the tail (roughly $12$--$13$
m/s at $"CVaR"_95$, paired) (@fig-outparam). The decoder we inherited from 2009 is still the right one.

#fig("fig_output_param.svg", [Bank-decoder variants on the #box[$515$-parameter] dense network. The
two-output atan2 decoder inherited from the 2009 work wins, with most of its advantage on the tail.], <fig-outparam>)

= Classical versus neural network

We now place the neural policy against the classical schemes on identical Monte Carlo scenarios --
the same seed pools, the same dispersions, the same per-scenario density noise. Two things
have to be established: that the classical baselines are tuned to their best, and that the comparison
is read at sizing depth.

== A co-optimized reference recovers the predictor--correctors

Before comparing against the classical schemes we must give them their best shot. Every classical
scheme was retuned by the same genetic algorithm on the same cost under per-scenario noise
(Appendix A). The reference-tracking laws (FTC, the energy controller, PredGuid) are only as good
as the reference trajectory they enslave to, and the legacy constant-bank reference is not
optimal. We therefore let the genetic algorithm co-optimize the reference: a single extra gene
sets the constant bank angle that generates the reference table, regenerated per individual, so
the law and the trajectory it tracks adapt together. The effect is large (@fig-joint). FTC falls
from $#R.fixed(R.per_draw_regime("ou_marginal/classical/ftc").dv_mean)$ to $#R.fixed(R.per_draw_regime("ou_marginal/classical/ftc_joint").dv_mean)$ m/s mean and from
$#R.fixed(R.per_draw_regime("ou_marginal/classical/ftc").dv_cvar95)$ to $#R.fixed(R.per_draw_regime("ou_marginal/classical/ftc_joint").dv_cvar95)$ at $"CVaR"_95$ (paired
$#R.signed(pr("v4_joint_vs_fixed_ftc").delta_mean)$ and $#R.signed(pr("v4_joint_vs_fixed_ftc").delta_cvar95)$, @tbl-paired), and at the far tail
from $#R.fixed(hl("classical/ftc").cvar999)$ to $#R.fixed(jftc.cvar999)$. The energy controller recovers
$#R.fixed(R.per_draw_regime("ou_marginal/classical/energy_controller").dv_mean - R.per_draw_regime("ou_marginal/classical/energy_controller_joint").dv_mean, d: 0)$ m/s of mean and PredGuid
$#R.fixed(R.per_draw_regime("ou_marginal/classical/pred_guid").dv_mean - R.per_draw_regime("ou_marginal/classical/pred_guid_joint").dv_mean, d: 0)$. The reference *was* FTC's weakness: a feedback law tracking a poor target cannot
out-perform the target.

With its reference repaired, FTC becomes the best classical scheme at sizing depth: it ties FNPAG
on the paired pool ($#R.signed(pr("v4_jointftc_vs_fnpag").delta_mean)$ m/s in mean, $"CVaR"_95$ interval straddling zero) and
pulls ahead at the far tail ($#R.fixed(jftc.cvar999)$ against $#R.fixed(fnpag.cvar999)$, Section 7.2) -- and it is
analytic and fast. The aerocapture story we wrote in 2009, in which FTC was the baseline to beat,
becomes: a *well-referenced* FTC is the classical state of the art, and it costs a fraction of a
numerical predictor--corrector to run.

#fig("fig_joint_reference.svg", [Fixed versus co-optimized reference for the three reference-tracking
schemes, retuned under per-scenario noise and scored on the $n = 1000$ final-evaluation pool.
Co-optimizing the constant-bank reference recovers most of FTC's deficit and lifts it to the
best classical scheme.], <fig-joint>)

== The deployability triangle <sec-deployability>

Three schemes define the deployability frontier: the neural network, the well-referenced FTC, and
FNPAG. They trade off along three axes -- accuracy, compute, and robustness -- and no single scheme
dominates all three.

*Accuracy.* On the paired $n = 1000$ pool the deployed network wins at every statistic: against
joint-FTC by $#R.fixed(-pr("v4_nn_vs_jointftc").delta_mean)$ m/s in mean, $#R.fixed(-pr("v4_nn_vs_jointftc").delta_p95)$ at $p_95$ and
$#R.fixed(-pr("v4_nn_vs_jointftc").delta_cvar95)$ at $"CVaR"_95$, winning $#R.fixed(100 * pr("v4_nn_vs_jointftc").win_rate_a)%$ of scenarios; against FNPAG the
margins are $#R.fixed(-pr("v4_nn_vs_fnpag").delta_mean)$ / $#R.fixed(-pr("v4_nn_vs_fnpag").delta_p95)$ / $#R.fixed(-pr("v4_nn_vs_fnpag").delta_cvar95)$ m/s, winning
$#R.fixed(100 * pr("v4_nn_vs_fnpag").win_rate_a)%$ (@tbl-paired). The tail margin is consistently *larger* than the mean margin --
the network's advantage is precisely where the mission is sized. At the sizing depth itself the
margin narrows to the seed spread of Section 6.2: the deployed seed's $"CVaR"_(99.9)$ of
$#R.fixed(champ.cvar999) plus.minus #R.fixed(champ.cvar999_se)$ sits $#R.fixed(margin_champ_jftc)$ m/s below joint-FTC ($#R.fixed(jftc.cvar999) plus.minus #R.fixed(jftc.cvar999_se)$) and
$#R.fixed(margin_champ_fnpag)$ below FNPAG ($#R.fixed(fnpag.cvar999) plus.minus #R.fixed(fnpag.cvar999_se)$); the Mamba family mean sits
$#R.fixed(margin3_jftc)$ and $#R.fixed(margin3_fnpag)$ below. FNPAG's extreme tail fattens with depth in a way its shallow
statistics hide -- it ties joint-FTC at $"CVaR"_95$ and loses $#R.fixed(fnpag.cvar999 - jftc.cvar999)$ m/s at $"CVaR"_(99.9)$, with
$#fnpag.lost$ physical crashes in $10^6$ (surface impacts on individual re-runs without the evaluation
timeout, not censoring) -- so the repaired FTC is the better classical scheme at sizing depth.

*Compute.* On a single idle core, the dense network runs at $#R.fixed(ms("NN-dense"), d: 2)$ ms per simulation and the stateful
Mamba at $#R.fixed(ms("NN-mamba"), d: 2)$ ms, against $#R.fixed(ms("FTC"), d: 2)$ ms for FTC and $#R.fixed(ms("FNPAG"))$ ms for FNPAG (@fig-classical). The network is
about $#R.fixed(ms_vs_ftc) times$ FTC -- the same fast class -- and $#ms_vs_fnpag times$ faster than the numerical
predictor--corrector. On accuracy and compute FNPAG is dominated -- joint-FTC matches its accuracy
and the network beats it, both at a small fraction of its cost -- though the off-nominal stress
below keeps robustness a separate axis. The selective-state-space core costs about $#R.fixed(ms_state) times$ the
dense network, the price of a state whose tail benefit Section 6.3 could not resolve. Per control action rather than per trajectory: the
network's forward pass costs $approx #int(calc.round(R.bench("NN-mamba").us_per_update_incl_sim))$ µs per $1$ s guidance update (whole-simulation cost
divided by the $approx #R.bench("NN-mamba").n_guidance_updates$ updates flown -- an upper bound on pure inference), while FNPAG's
predictor work amounts to $approx #R.fixed(R.bench("FNPAG").ms_per_replan_derived, d: 2)$ ms per $2$ s replan cycle. Flight processors run one to
two orders of magnitude slower than the laptop core measured here; at a conservative $100 times$
scaling neither scheme breaches its own deadline ($approx #int(calc.round(fl_replan_ms))$ ms per replan against $2$ s;
$approx #R.fixed(fl_update_ms)$ ms per update against $1$ s), but the deadline margins differ by a factor of thirty,
and the relative cost ordering is host-independent. None of these measurements establish worst-case
execution time or memory on qualified hardware (Section 9).

*Robustness -- the caveat.* We trained the network on the medium dispersion regime. Under a
deliberately harsher off-nominal regime (atmosphere, density perturbation, navigation, and filter all
set high; Appendix A), scored at $n = 10\,000$ with bootstrap intervals, the picture inverts, and
we report it plainly because it is the one place the network loses (@fig-robust). All stress-regime
tail statistics are conditional on capture -- $"CVaR"_95 (Delta v | "capture")$ -- and a conditional
tail can improve by failing the hardest scenarios, so we read every stress comparison
lexicographically: capture probability first (a difference within half a point read as parity),
conditional tail cost second, and no tail win is claimed across a larger capture-rate deficit.
The deployed network captures $#R.fixed(st("NN").capture_pct)%$ $[#R.fixed(st("NN").capture_pct_ci.at(0)), #R.fixed(st("NN").capture_pct_ci.at(1))]$ of the stress
pool, the lowest of the five schemes: #st_cap_deficit points below the classical laws, every paired
deficit with an interval excluding zero (joint-FTC $#R.fixed(st("joint-FTC").capture_pct)%$, FNPAG
$#R.fixed(st("FNPAG").capture_pct)%$, fixed-reference FTC $#R.fixed(st("FTC-fixed").capture_pct)%$, PredGuid $#R.fixed(st("PredGuid").capture_pct)%$). Its conditional
tail, $#R.fixed(st("NN").dv_cvar95, d: 0)$ m/s, is worse than every law's but FNPAG's ($#R.fixed(st("joint-FTC").dv_cvar95, d: 0)$ for joint-FTC,
$#R.fixed(st("FNPAG").dv_cvar95, d: 0)$ for FNPAG). The lesson is not that the network is fragile -- it captures
$#R.fixed(st("NN").capture_pct, d: 0)%$ far outside its training regime -- but that a training-free analytic law
extrapolates better than a policy trained on a narrower distribution. The network wins the nominal
sizing tail it was trained for; Section 7.3 asks whether the gap is intrinsic.

#grid(columns: 2, gutter: 6pt,
  fig("fig_classical_vs_nn.svg", [Deployability: per-simulation compute versus far-tail
  $"CVaR"_(99.9)$ on the confirmatory pool (network points are three-seed means). Both
  network points sit lower-left -- a tighter tail than every classical scheme and far cheaper than
  FNPAG; the seed spread of @tbl-arch is the width of that margin.], <fig-classical>),
  fig("fig_robustness.svg", [Off-nominal stress at $n = 10\,000$: capture-rate drop and $"CVaR"_95$
  inflation from each scheme's nominal confirmatory numbers, with bootstrap $95%$ whiskers. The
  medium-trained network loses the most capture of the five; the analytic laws extrapolate
  better.], <fig-robust>))

// One tbl-perf row: capture / mean / p95 / CVaR95 from the n = 1000 final-evaluation run
// (per_draw asserted by the accessor), Viol. and CVaR99.9 from the cell's 10^6 confirmatory row.
#let perf_row(scheme, key, bold: false) = {
  let r = R.per_draw_regime(key)
  let c = R.marg(key)
  let hi(s) = if bold { strong(s) } else { s }
  (scheme, [#R.fixed(r.capture_pct)], [#R.fixed(c.viol_pct, d: 2)], [#R.fixed(r.dv_mean)], [#R.fixed(r.dv_p95)],
    [#hi(R.fixed(r.dv_cvar95))], [#hi(R.fixed(c.cvar999))])
}

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Scheme*], [*Capture %*], [*Viol. %*†], [*Mean*], [$bold(p_95)$], [$bold("CVaR"_95)$], [$bold("CVaR"_(99.9))$†],
    ),
    table.hline(stroke: 0.35pt),
    ..perf_row([NN -- Mamba $962$ (deployed)], "ou_marginal/hl_mamba_p962", bold: true),
    ..perf_row([NN -- GRU $1014$], "ou_marginal/hl_gru_p1014"),
    ..perf_row([NN -- dense $515$ (efficiency ref.)], "ou_marginal/hl_dense_p515"),
    ..perf_row([NN -- dense $972$], "ou_marginal/hl_dense_p972"),
    ..perf_row([NN -- LSTM $1082$], "ou_marginal/hl_lstm_p1082"),
    ..perf_row([FNPAG], "ou_marginal/classical/fnpag"),
    ..perf_row([FTC (joint reference)], "ou_marginal/classical/ftc_joint"),
    ..perf_row([Energy controller (joint reference)], "ou_marginal/classical/energy_controller_joint"),
    ..perf_row([PredGuid (joint reference)], "ou_marginal/classical/pred_guid_joint"),
    ..perf_row([FTC (fixed reference)], "ou_marginal/classical/ftc"),
    ..perf_row([PredGuid (fixed reference)], "ou_marginal/classical/pred_guid"),
    ..perf_row([Energy controller (fixed reference)], "ou_marginal/classical/energy_controller"),
    ..perf_row([Equilibrium glide#super[‡]], "ou_marginal/classical_ungated/equilibrium_glide"),
    ..perf_row([Piecewise constant#super[‡]], "ou_marginal/classical_ungated/piecewise_constant"),
    table.hline(stroke: 0.7pt),
  ),
  caption: [Final Monte Carlo performance under per-scenario noise, correction $Delta v$ in m/s,
  ordered by $"CVaR"_95$. Capture / mean / $p_95$ / $"CVaR"_95$ are on the $n = 1000$
  final-evaluation pool; †Viol. and $"CVaR"_(99.9)$ are on the frozen confirmatory pool
  ($10 times 100\,000$ scenarios per scheme; each $"CVaR"_(99.9)$ averages $1000$ tail observations,
  replicate standard errors in @tbl-arch). Network rows are the deployed seed of each cell (seed 1,
  fixed by rule); the Mamba and dense-$515$ three-seed means are in @tbl-arch. Viol. is the share of
  confirmatory draws exceeding any of the heat-flux, g-load, or heat-load limits: the networks'
  exceedances are heat load at the $10^(-4)$--$10^(-3)$ level (Section 6.2), the fixed-reference
  FTC's and the two signed-bank schemes' are heat flux, and the joint-reference trackers and FNPAG
  exceed nothing or next to it. Every classical scheme was retuned under per-scenario noise at one
  allocation (Appendix A); #super[‡]these two were selected without the strict feasibility ceiling
  the other seven met, because no candidate passed it. A $100%$ capture rate means "no failures
  observed": with zero failures in $n$ independent scenarios the one-sided $95%$ upper bound on the
  failure probability is $approx 3\/n$. The mean is reported for continuity with the 2009 work but is
  operationally secondary to the tail.],
) <tbl-perf>

// One tbl-paired row from a results.json paired comparison (A = a, B = b of the record).
#let paired_row(label, key, mark: none) = {
  let p = pr(key)
  (label, [$#R.signed(p.delta_mean)$], [$[#R.ci(p.delta_mean_ci)]$], [$#R.signed(p.delta_p95)$],
    [$#R.signed(p.delta_cvar95)$], [$[#R.ci(p.delta_cvar95_ci)]$], [#R.fixed(p.win_rate_a * 100)#mark], [#R.pval(p.wilcoxon_p)])
}

#figure({
  set text(size: 9pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    table.hline(stroke: 0.7pt),
    table.header(
      [*A vs B*], [$bold(Delta"mean")$], [*95% CI*], [$bold(Delta p_95)$], [$bold(Delta"CVaR"_95)$], [*95% CI*], [*A-win %*], [*p*],
    ),
    table.hline(stroke: 0.35pt),
    ..paired_row([Mamba vs FTC, fixed ref.], "v4_nn_vs_ftc"),
    ..paired_row([Mamba vs FTC, joint ref.], "v4_nn_vs_jointftc"),
    ..paired_row([Mamba vs FNPAG], "v4_nn_vs_fnpag"),
    ..paired_row([Mamba vs dense $515$], "v4_headline_vs_dense515", mark: super[‡]),
    ..paired_row([Mamba vs GRU $1014$], "v4_headline_vs_gru", mark: super[‡]),
    ..paired_row([Mamba vs dense $972$], "v4_headline_vs_dense972", mark: super[‡]),
    ..paired_row([Mamba vs LSTM $1082$], "v4_headline_vs_lstm", mark: super[‡]),
    ..paired_row([FTC: joint vs fixed ref.], "v4_joint_vs_fixed_ftc"),
    ..paired_row([FTC, joint vs FNPAG], "v4_jointftc_vs_fnpag"),
    table.hline(stroke: 0.7pt),
  )},
  caption: [Paired comparisons on the $n = 1000$ final-evaluation pool under per-scenario noise,
  correction $Delta v$ in m/s; negative $Delta$ favors A. The intervals are $10\,000$-resample
  paired bootstraps. Win-rate and $p$ (Wilcoxon signed-rank) are computed on the per-scenario cost;
  $p$ is truncated at $10^(-15)$ -- the normal-approximation statistic saturates at or near sign
  unanimity at $n = 1000$, certifying a (near-)unanimous direction, not a resolved tail probability.
  Deltas are computed on unrounded per-scenario values, so a delta may differ from the difference of
  the rounded marginals in @tbl-perf by $0.1$ m/s. #super[‡]For the intra-network rows the deployed
  seed wins the shallow tail with intervals excluding zero; at sizing depth the families overlap
  (@tbl-arch).],
) <tbl-paired>

#fig("fig_survival.svg", [Empirical survival curves of the correction $Delta v$ on the confirmatory
pool ($10 times 100\,000$ scenarios per scheme, per-scenario noise; log scale, curves subsampled
every $approx 100$th order statistic) for the deployed Mamba and dense seeds, the LSTM cell, and the
two retuned classical references. The network-to-classical separation holds at every depth; the
network curves cross one another.], <fig-survival>)

== Matching the objective to the regime <sec-objcenter>

The robustness caveat of @sec-deployability invites a question: is the off-nominal gap intrinsic to
neural guidance, or an artifact of a training objective tuned for the wrong regime? Before answering
it, the regime itself deserves scrutiny. The high-dispersion stress pool is not a realistic operating
point -- no mission would be flown on an entry profile that fails roughly one pass in twenty under
*any* guidance scheme. In practice the entry interface would simply be re-targeted so that the $3sigma$
dispersion envelope still captures, restoring a near-certain margin by construction. We therefore read
this regime not as a deployment scenario but as a deliberate *stress probe of the training method*: it
pushes the optimizer into a regime where roughly $5%$ of scenarios are catastrophic, and asks whether
the genetic algorithm can still make progress.

Under that probe, the deployed objective cannot. Training on the high regime with the cubed
transform, the hardest-seed curation and two scenarios per individual (the *stacked* cell of
@fig-objcenter) reaches $#R.fixed(lv("stacked").capture_pct)%$ capture with a $"CVaR"_95$ of
$#R.fixed(lv("stacked").dv_cvar95, d: 0)$ m/s on the stress pool, while exceeding the heat-flux limit on
$#R.fixed(lv("stacked").violation_pct.heat_flux, d: 0)%$ of draws. The mechanism is the one the medium regime hides: when failures dominate, cubing
the cost makes the objective a spiky near-discrete failure count, max-bucket curation feeds the
optimizer only the hardest seeds, and a per-individual batch of two cannot estimate any of it -- so
selection has no gradient to climb. The worst-case shaping that is a virtue in the medium regime
becomes a liability once the noise outruns the sample budget.

*Centering* the objective -- more scenarios per individual (a real cost estimate), a central
curation bucket, and a milder transform -- restores the gradient. On the dense vehicle a
one-lever-at-a-time sweep (@tbl-levers, @fig-objcenter) attributes the recovery, with the
constraint exceedance read beside every tail number because the levers trade them. Every lever
lowers the stacked tail; the middle bucket does the most alone ($"CVaR"_95$
$#R.fixed(lv("stacked").dv_cvar95, d: 0) arrow.r #R.fixed(lv("plus_bucket").dv_cvar95, d: 0)$) and carries the worst heat-load exceedance of
the five ($#R.fixed(lv("plus_bucket").violation_pct.heat_load)%$); sixteen scenarios per individual add
$#R.fixed(lv("plus_sims").capture_pct - lv("stacked").capture_pct)$ capture points and leave the heat-flux exceedance where the stacked
objective put it ($#R.fixed(lv("plus_sims").violation_pct.heat_flux, d: 0)%$ of draws), the linear transform adds
$#R.fixed(lv("plus_transform").capture_pct - lv("stacked").capture_pct)$ points and cuts it to $#R.fixed(lv("plus_transform").violation_pct.heat_flux)%$; only all three together reach the lowest tail
($#R.fixed(lv("centered").dv_cvar95, d: 0)$ m/s) with every exceedance low
($#R.fixed(lv("centered").violation_pct.heat_flux)$ / $#R.fixed(lv("centered").violation_pct.g_load)$ / $#R.fixed(lv("centered").violation_pct.heat_load)%$
for heat flux, g-load and heat load). At $n = 1000$ a capture rate carries about $1.4$ points of
binomial half-width, so the capture column is read as a direction, and the centered and
middle-bucket tails overlap. The three knobs are a coupled system, exactly as in the medium regime,
but now the coupling has teeth.

// One tbl-levers row (data/objective_centering_v4.json): capture, mean, CVaR95 with its bootstrap
// 95% CI, and the three exceedance shares.
#let ci_s(iv, d: 0) = text(size: 8pt)[\[$#R.fixed(iv.at(0), d: d)$, $#R.fixed(iv.at(1), d: d)$\]]
#let lever_row(name, label) = {
  let c = lv(label)
  (name, [$#R.fixed(c.capture_pct)$], [$#R.fixed(c.dv_mean, d: 0)$], [$#R.fixed(c.dv_cvar95, d: 0)$ #ci_s(c.dv_cvar95_ci)],
    [$#R.fixed(c.violation_pct.heat_flux)$], [$#R.fixed(c.violation_pct.g_load)$], [$#R.fixed(c.violation_pct.heat_load)$])
}
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([Dense $515$ cell], [Capture (%)], [Mean], [$"CVaR"_95$ (m/s)], [Heat flux (%)], [g-load (%)], [Heat load (%)]),
    table.hline(stroke: 0.4pt),
    ..lever_row([Stacked: cubed, max bucket, $n = 2$], "stacked"),
    ..lever_row([$+$ $16$ scenarios per individual], "plus_sims"),
    ..lever_row([$+$ middle bucket], "plus_bucket"),
    ..lever_row([$+$ linear transform], "plus_transform"),
    ..lever_row([Centered: all three], "centered"),
    table.hline(stroke: 0.7pt),
  ),
  caption: [The objective levers on the high-regime stress pool ($n = 1000$, per-scenario noise),
  one at a time from the stacked objective: capture, mean and $"CVaR"_95$ of the correction
  $Delta v$ over captured scenarios (bootstrap $95%$ interval in brackets), and the share of draws
  exceeding each constraint limit.],
) <tbl-levers>

The effect transfers to the deployed architecture, and the comparison is read at sizing depth:
three independent centered-Mamba retrainings and two joint-FTC baselines -- the medium-deployed
retune of @tbl-perf and a joint-FTC retuned on the high regime -- scored on $n = 10\,000$
scenarios of the same stress pool, paired on scenario, with bootstrap $95%$ intervals
(@tbl-centered). The three seeds hold capture at #cd_span(cd_seeds, "capture_pct", d: 1)$%$ against
$#R.fixed(cd("jointFTC-medium").capture_pct)%$ for the medium-deployed joint-FTC, within the half-point parity band, and
every seed wins the conditional tail: $"CVaR"_95 (Delta v | "capture")$ of #cd_span(cd_seeds, "dv_cvar95") m/s
against $#R.fixed(cd("jointFTC-medium").dv_cvar95, d: 0)$, every paired delta negative with an interval excluding
zero. Against the high-regime retune ($#R.fixed(cd("jointFTC-high").capture_pct)%$, $#R.fixed(cd("jointFTC-high").dv_cvar95, d: 0)$ m/s) every seed wins
both capture and tail, but that baseline is degraded -- no candidate of its run was gate-feasible
and it loses to the medium retune on the same pool -- so the medium comparison is the binding one.
On that evidence the off-nominal gap is not intrinsic to neural guidance: the analytic law's edge in
@sec-deployability was a property of the *mismatched* training objective.

The price is stated in the same table. No policy of either family trained on the high regime
passes the strict feasibility gate; the comparison stands at the exceedance each policy shows, and
the centered seeds exceed the heat-flux limit on #cd_viol_span(cd_seeds, "heat_flux")$%$ of draws
against $#R.fixed(cd("jointFTC-medium").violation_pct.heat_flux)%$ for the medium joint-FTC, and the heat-load limit on
#cd_viol_span(cd_seeds, "heat_load")$%$ against $#R.fixed(cd("jointFTC-medium").violation_pct.heat_load)%$. Seed 1 pays for part of its
tail in heat flux; seed 3 wins its tail with less heat-flux exceedance than the baseline; across
the three seeds the tail ranking is the reverse of the heat-flux ranking, which three points make
an observation, not a measured trade-off. The stress regime itself remains one a real mission
would design away. The durable lesson is methodological and reinforces Section 4: the optimal
worst-case weighting is matched to the environment's noise and the per-individual sample budget,
not fixed once -- and the constraint exceedance is part of what the match has to hold.

// One tbl-centered row (data/centered_depth_v4.json): capture and conditional CVaR95 with their
// bootstrap 95% CIs, the three exceedance shares, and the paired CVaR95 delta against the
// medium-deployed joint-FTC with its CI (negative: the seed's tail is better; a dash for the baselines).
#let cd_row(name, label) = {
  let c = cd(label)
  let delta = if cd_seeds.contains(label) {
    let p = cdp(label, "jointFTC-medium")
    [$#R.signed(p.delta_cvar95, d: 0)$ #text(size: 8pt)[\[#R.ci(p.delta_cvar95_ci, d: 0)\]]]
  } else { [--] }
  (name, box[$#R.fixed(c.capture_pct)$ #ci_s(c.capture_pct_ci, d: 1)], box[$#R.fixed(c.dv_cvar95, d: 0)$ #ci_s(c.dv_cvar95_ci)],
    [$#R.fixed(c.violation_pct.heat_flux)$], [$#R.fixed(c.violation_pct.g_load)$], [$#R.fixed(c.violation_pct.heat_load)$], box(delta))
}
#figure({
  set text(size: 9pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    table.hline(stroke: 0.7pt),
    table.header([Cell], [Capture (%)], [$"CVaR"_95$ (m/s)], [Heat flux (%)], [g-load (%)], [Heat load (%)], [$Delta$ vs medium]),
    table.hline(stroke: 0.4pt),
    ..cd_row([Centered Mamba, s1], "mamba_centered_s1"),
    ..cd_row([Centered Mamba, s2], "mamba_centered_s2"),
    ..cd_row([Centered Mamba, s3], "mamba_centered_s3"),
    table.hline(stroke: 0.4pt),
    ..cd_row([Joint-FTC, medium], "jointFTC-medium"),
    ..cd_row([Joint-FTC, high regime], "jointFTC-high"),
    table.hline(stroke: 0.7pt),
  )},
  caption: [The centered high-regime cells at sizing depth: $n = 10\,000$ scenarios of the 9M stress
  pool under per-scenario noise; capture and the conditional $"CVaR"_95$ of the correction $Delta v$
  with bootstrap $95%$ intervals, the share of draws exceeding each constraint limit, and each seed's
  paired $"CVaR"_95$ delta against the medium-deployed joint-FTC (negative: the seed's tail is
  better). Every seed sits at capture parity with the medium baseline and beats its tail; no cell
  of either family is feasible under the strict gate.],
) <tbl-centered>

#fig("fig_objective_centering.svg", [Objective centering under the high-dispersion stress probe
(9M pool, per-scenario noise), three panels over the same ten cells: capture (top), conditional
$"CVaR"_95$ (middle), and the heat-flux (solid) and heat-load (hatched) exceedance shares (bottom).
The five dense lever cells are the $n = 1000$ rows of @tbl-levers; the three centered-Mamba seeds and
the two joint-FTC baselines are the $n = 10\,000$ rows of @tbl-centered, with bootstrap $95%$
whiskers. Centering recovers the tail and transfers to the Mamba; the exceedance panel is read with
the tail panel, because the levers and the seeds trade them.], <fig-objcenter>)

= What the network uses

To close the architecture argument we ask which inputs the deployed Mamba policy actually relies
on, through a closed-loop input-sensitivity analysis: each input is zeroed in turn and the
resulting cost increase measured on $n = 1000$ scenarios under per-scenario noise (@fig-ablation).
Zeroing a normalized input perturbs every subsequent state of the closed loop, so the deltas
measure closed-loop dependence, not isolated feature importance. All seventeen masked inputs cost
something when zeroed. The largest degradation by far comes from the orbital energy -- the
coordinate of the corridor itself -- and immediately behind it two of the engineered, cost-aligned
*predicted correction-$Delta v$* components: the energy-closing burn (#raw("predicted_dv1")) and
the periapsis-correction burn (#raw("predicted_dv2")), each evaluated on the current osculating
orbit. The reference-trajectory altitude rate, the radial velocity and the dynamic-pressure error
-- the quantities a reference-tracking law would feed back on -- follow; the bank-history encodings
contribute least.

This is the flat median of Section 6 seen from the inside. The network's strongest dependencies
after the energy are signals that tell it, causally and smoothly, what the maneuver would cost if
it stopped now -- a cost-to-go surrogate handed to the policy as an input. Because the osculating
orbit is itself an integral of the flown trajectory, those signals already summarize most of the
history a memory cell could recover, and a dense network without any memory matches the recurrent
ones on the median: the engineered inputs do the work that recurrence would otherwise do. The
no-predicted-$Delta v$ control of Section 6.3 is the same finding at the tail: retrained without
these three inputs the policy reaches the same far tail within seed variance, so the inputs are a
convenience the search exploits rather than information the policy cannot recover from the rest of
the state. The network is using the physics we handed it; whether it reserves its memory for the
tail is the question the controls left open.

#fig("fig_ablation.svg", [Closed-loop input sensitivity: per-input cost increase when each input is
zeroed, for the deployed Mamba policy ($n = 1000$, per-scenario noise, log cost transform). The
orbital energy dominates, followed by the engineered, cost-aligned predicted-$Delta v$ components
and the reference-tracking signals.], <fig-ablation>)

= Discussion and limitations

The clearest limitation is the off-nominal robustness gap of @sec-deployability -- the deployed
network wins the nominal sizing tail it was trained for and loses, off-nominal, to a training-free
analytic law, capturing #st_cap_deficit points less than the classical schemes under the stress
regime. @sec-objcenter traces that gap to the training objective rather than to neural guidance:
at sizing depth the centered retrain beats the medium-deployed joint-FTC on the conditional tail
at capture parity, across three seeds, but it buys part of that tail with heat-flux exceedance,
no high-regime policy of either family is feasible under the strict gate, and the stress regime is
one a real mission would design away. What remains open is a centered retrain whose constraint
exceedance matches the analytic law's.

The second limitation is the one the per-scenario campaign measured on our own earlier claim. The
networks beat the classical schemes at sizing depth by a margin the seed spread does not erase, but
they beat one another by a margin it does: the Mamba's three seeds span $#R.fixed(calc.max(..mamba_seeds.map(c => c.cvar999)) - calc.min(..mamba_seeds.map(c => c.cvar999)))$ m/s of
$"CVaR"_(99.9)$ and the dense reference's $#R.fixed(calc.max(..dense_seeds.map(c => c.cvar999)) - calc.min(..dense_seeds.map(c => c.cvar999)))$, the controls without learned state
or without the cost-to-go inputs land inside those spans, and the deployed seed is the worst of its
family at the depth that sizes the tanks. A sizing engineer should quote a family mean with its
spread, repeat seeds before quoting any architecture ordering, and expect the gate -- which
promotes on validation loss -- to pick a seed the far tail does not favor. The broader pattern,
seen three times -- the shared density history of Section 6.4, the off-nominal regime, the heat-load
ceiling a policy carries -- is that the network is exactly as good as the distribution it trains
on, and widening the training environment recovers what the narrow one gave away.

A third tradeoff is the cost of state. The deployed Mamba runs at $#R.fixed(ms("NN-mamba"), d: 2)$ ms per simulation against
$#R.fixed(ms("NN-dense"), d: 2)$ ms for the dense network -- about $#R.fixed(ms_state) times$ for the selective-state-space core. The head
itself accounts for $approx 1.2$ ms of that cost; single-precision arithmetic would roughly halve
it, while integer quantization buys memory rather than speed at this scale (Appendix C). Both
remain in the fast compute class, an order of magnitude below the numerical predictor--corrector,
so the choice is between the network and FTC, not between the network and FNPAG. If on-board
compute or implementation simplicity is the binding constraint, the memoryless
#box[$515$-parameter] dense network is the efficiency reference: a competitive median at half the
parameters, and a far tail this campaign cannot separate from the recurrent one.

A flight-qualification path for a stateful network remains open, and the deployability triangle
suggests its shape: a simplex arrangement in which the analytic joint-FTC -- whose off-nominal
robustness Section 7.2 established -- runs in parallel as an onboard monitor with authority to take
over on envelope violation, while the network flies the nominal corridor it wins on. The deployed
policy is a fixed forward pass of $962$ parameters in double precision with no data-dependent
control flow, so worst-case execution time and memory bound trivially; verifying the *decision*
behavior across the dispersion envelope, rather than the code, is the open problem. Appendix C
quantizes the deployed head (weight-only): $8$-bit is free, and a $4$-bit projections-only head
reached by quantization-aware fine-tuning holds the sizing tail at a $#R.fixed(R.quant4_memory(4, "per_tensor", "proj_only").f64_baseline_bytes / R.quant4_memory(4, "per_tensor", "proj_only").total_bytes)
times$ memory reduction -- pruning remains open. Run-to-run variance is calibrated at the tail
through the three-seed repeats of every deciding cell and, for the decisive seed-strategy and
CMA-ES cells of the development regime, at the mean through dedicated repeats (Section 4.1); the
tight optimizer differences of Section 5 stay reported as indistinguishable rather than ranked.

Finally, a methodological note for anyone reproducing this. The training is not bit-reproducible from
a seed alone -- it never was -- because the non-stationary objective and the operator randomness make
each run a fresh draw; the deployed policy is reproduced from its saved weights and checkpoints, not
re-derived. This is the same property that forced us to measure run-to-run variance directly rather
than assume it away.

= Conclusion

Seventeen years ago we showed that a feed-forward network trained by a genetic algorithm could fly an
MSR aerocapture more efficiently than a Cerimele--Gamble feedback law, and we asked for a comparison
against predictor--correctors. This paper delivers it, and the answer is favorable to neural guidance
on the metric that matters, with its limit measured. Under independent per-scenario density noise,
on $10^6$ pre-registered confirmatory scenarios, a #box[$962$-parameter] recurrent (Mamba) policy
trained from scratch holds a far-tail $"CVaR"_(99.9)$ of $#R.fixed(mamba3.mean) plus.minus #R.fixed(mamba3.sd)$ m/s
over three training seeds, $#R.fixed(margin3_jftc, d: 0)$ m/s below a well-referenced FTC and
$#R.fixed(margin3_fnpag, d: 0)$ below the numerical predictor--corrector, which it out-runs
$#ms_vs_fnpag times$; on the paired pool it beats both at every statistic on more than $99%$ of
scenarios. A memoryless #box[$515$-parameter] dense network lands inside that seed spread, and so do
the matched controls without learned state or without the engineered cost-to-go inputs: the networks
beat the classical schemes by a margin the spread does not erase, and beat one another by a margin
it does.

Two findings carry beyond the headline number. The first is methodological: a genetic algorithm is the
wrong optimizer for a fixed objective and the right one for a moving one, and the moving
Monte Carlo environment -- adaptive seeds, a tail-weighting cost transform, and hardest-case curation
-- is a matched system that converts it from the worst optimizer we tested ($154$ m/s three-seed
mean) to the best
($120$); and the noise regime the policy trains on is part of the match, because a policy with state
fits the one density history it sees. The second is about what a policy has to be: engineered,
cost-aligned inputs flatten the typical cost across every cell type we tried, the deployed policy
computes with its state, and whether that state is needed for the tail is a question three seeds
could not answer at this budget. Training loss did not pick the tail winner; the selection gate
promotes on it, so the deployed seed is fixed by rule and quoted with its family's spread.

The honest drawback that closed the 2009 paper -- that the training is too heavy to run on board --
remains, but its sting is gone: the deployed policy is a fixed forward pass that costs a few
milliseconds, and the heavy optimization happens once, on the ground -- ground compute buying
in-flight margin. The next step would be to widen
the training environment -- annealing the tail-weighting and the dispersion envelope over the run, or
stratifying the curated batch across cost quantiles -- until the network's off-nominal robustness
matches its nominal accuracy at the analytic law's constraint margin, and
to carry these stateful policies beyond the single capture maneuver -- to skip-entry and Earth-return
legs, and to on-line adaptation of the deployed policy in flight.

#pagebreak()
#bibliography("refs.bib", title: "References", style: "harvard-cite-them-right")

#pagebreak()
#set heading(numbering: none)
= Appendix A: reproduction details

One compact reference for the settings and systems behind every number.

*Simulator.* All schemes fly through one native (Rust) simulator: fixed-step fourth-order
Runge--Kutta integration (Gill variant), $J_2$--$J_4$ zonal gravity, a tabulated Mars atmosphere
carrying the static Monte Carlo bias and the Ornstein--Uhlenbeck perturbation, pilot dynamics,
thermal tracking, and the navigation--guidance--control chain sequenced on its own cadences; the
bias-filter navigation recovers density through lift-corrected inverse dynamics. The implementation is
regression-validated against the 2009 study's legacy code -- across all $725$ time steps of a guided
trajectory, $22$ of $24$ output channels are bit-identical (the two mismatches trace to
uninitialized variables in the reference) -- which establishes equivalence with the flight-heritage
implementation, not physical validation. Independent physics checks run in the test suite: an
analytic potential-gradient oracle for the $J_2$--$J_4$ gravity expansion, integrator convergence
on analytic systems, vacuum energy and angular-momentum conservation, and cross-language
forward-pass parity for the network runtime at machine epsilon.

*Seed pools.* Every Monte Carlo pool derives from the mission base seed through disjoint reserved
streams: the selection pool (offset $10^6$, $n = 1000$, the in-training promotion gate -- queried
$#int(calc.round(champ_run.actual_sims.validation / 1000))$ times over the deployed run, so it is adaptively reused and is not an unbiased estimate
of generalization), final evaluation (offset $2 times 10^6$: the $n = 1000$ paired pool of
@tbl-perf and @tbl-paired, and, in the development regime, the $n = 10\,000$ far-tail pool of
Section 4), the fresh re-quote pool (offset $8 times 10^6$), the off-nominal stress pool (offset
$9 times 10^6$), and the architecture-probe pool (offset $10^7$, Appendix B). Training scenarios
are drawn outside every reserved stream; evaluation pools draw one scenario per seed (independent
samples), and multi-scenario batches use Latin-hypercube draws. @tbl-pools states which decisions
each pool influenced and the noise regime each was run in.

*Noise seeding.* A scenario's density-perturbation stream is seeded from a hash of its own
dispersion draw (`noise_seeding = "per_draw"`, the simulator's default): identical draw, identical
realization -- runs stay reproducible -- and distinct scenarios receive distinct paths, so every
one-scenario-per-seed pool marginalizes over the process. The development regime seeds the stream
from a constant plus the within-batch scenario index, which is zero for every one-scenario batch,
so all scenarios of such a pool share one realization. The repository's `arxiv-v2` and `arxiv-v3`
drafts of this paper quoted the development regime in their main body; this version is
per-scenario throughout, and Section 6.4 is the only place the two are set side by side.

*Confirmatory sizing pool.* All methodology, architecture, and checkpoint choices were frozen at a
recorded revision before the pool was generated, and the deployed seed of every cell (seed 1) was
fixed by rule before any cell was scored on it. Seeds are drawn without duplicates from
$[2^31, 2^32)$ -- structurally disjoint from every earlier draw, since all earlier pools,
training batches, and curation probes live in $[0, 2^31)$ -- arranged as ten replicate pools of
$100\,000$ scenarios sharing seeds across schemes (so per-replicate differences are paired). Every
cell is evaluated on it exactly once; tail statistics use the estimator of Section 2.2
($"CVaR"_(99.9)$ averages $100$ observations per replicate, $1000$ pooled), and replicate-level
dispersion gives the quoted standard errors and difference intervals with no distributional
assumptions. Non-captures are re-flown individually without the evaluation wall clock and
classified; every non-capture quoted in this paper is a physical crash.

*Off-nominal stress regime.* The Section 7 stress pool raises four domains to their high presets:
the atmospheric density bias to $plus.minus 100%$ (from $plus.minus 50%$); the Gauss--Markov
density perturbation to a $30$ s correlation time at $20%$ RMS (from $120$ s at $5%$); navigation
errors to $3sigma$ altitude $plus.minus 3$ km, horizontal $approx plus.minus 18$ km per axis,
velocity $plus.minus 3$ m/s, drag acceleration $plus.minus 0.6$ m/s#super[2]\; and the
navigation-filter gain draw to $plus.minus 0.45$ absolute ($3sigma$).

*Dispersion rationale.* The $plus.minus 50%$ nominal density span is a deliberately conservative
envelope for Mars seasonal and dust-loading variability about a tabulated mean profile
@forget1999mars; domains are drawn independently as a modeling choice (no cross-correlations are
claimed). The static bias and the Ornstein--Uhlenbeck process model different frequencies --
profile-scale uncertainty versus along-track variability -- and do not double-count; the process
is zero-initialized at entry and reaches its stationary variance within roughly one correlation
time. The two wind draw dimensions are retained inert to keep the $26$-dimensional draw layout
stable across configurations with and without the wind model.

*Classical tuning parity.* Every classical scheme was tuned by the same genetic algorithm on the
same cost $C$ below -- identical penalties, virtual costs, and transform -- under per-scenario
noise, co-optimizing its full parameter set including the shared navigation-filter and actuator
gains the network also tunes ($26$ parameters for FTC against the network's weights plus $3$
actuator-side parameters), at $2000$ generations $times 300$ individuals $times 10$ scenarios per
generation, one allocation for all nine. Seven schemes were selected under the strict feasibility
ceiling of the training gate (zero constraint violations on the selection pool); for the
equilibrium-glide and piecewise-constant schemes no candidate passed it in $2000$ generations, so
their quoted cells were selected without the ceiling and carry their exceedance in @tbl-perf. The
classical searches plateau far inside their budget, so the network's longer run buys the comparison
nothing. The joint-reference co-optimization of Section 7.1 is the classical counterpart of the
network's co-tuned actuator parameters, and the information asymmetry is tested directly: retrained
without the three predicted-$Delta v$ observations the classical schemes cannot consume, the
network reaches the same far tail within seed variance and still beats the co-tuned joint-FTC on
every reported statistic (Section 6.3, third control).

*Correction burns.* The reported $Delta v$ is the three-burn plan from the captured orbit (apsis
radii $r_a$, $r_p$) to the $500$ km circular parking orbit (radius $r_c$), using the elliptical
apsis speed $v(r_1, r_2) = sqrt(2 mu r_2 \/ (r_1 (r_1 + r_2)))$: a periapsis correction at
apoapsis, $Delta v_1 = v(r_a, r_c) - v(r_a, r_p)$; circularization at the new periapsis,
$Delta v_2 = sqrt(mu \/ r_c) - v(r_c, r_a)$; and a plane change $Delta v_3 = 2 v_n sin(Delta i \/ 2)$
at the cheaper node, $v_n$ the smaller of the target-orbit speeds at the two nodes. The total is
$|Delta v_1| + |Delta v_2| + |Delta v_3|$.

*Cost and objective.* Per simulation the cost is
$ C = Delta v + s(Delta v - T) + s(Delta v - T)^2 / (2 S) + sum_j w_j thin s((x_j - L_j) \/ L_j), $
with $s$ a softplus (a $C^oo$ knee), $T = 1000$ m/s, $S$ the quadratic scale, and the penalty sum
over the peak heat-flux, peak-load, and integrated heat-load exceedances normalized by their limits
$L_j$, weights $w_j = 1$. Non-captures receive a virtual cost above any captured correction:
$10\,000 + v_oo$ m/s for hyperbolic escape ($v_oo$ the excess speed), and
$3000 + 1000 min(|E - E_"target"|, 50) - 500 thin t\/t_max$ for crash and timeout outcomes
(energies in MJ/kg) -- so captures are always preferred while the optimizer keeps a gradient
toward the capture boundary. The per-simulation cost is cubed (the deployed transform) and
aggregated across an individual's scenario batch by root-mean-square -- the $L_6$-norm-equivalent
objective of @eq-objective.

*Optimizer (deployed headline).* pymoo genetic algorithm (SBX crossover $eta = 3$, polynomial
mutation $eta = 5$ at rate $0.15$), population $512$, two scenarios per individual per generation,
run for $20\,000$ generations. The adaptive seed curation, precisely: every
second generation or on a promotion, (1) draw $1000$ fresh probe scenarios outside every reserved
stream; (2) score them with the current best individual; (3) sort the per-scenario costs into
$n_"sims"$ equal-count quantile bins; (4) take the hardest scenario of each bin as the next
training batch -- scenarios are replaced as a set, never accumulated. The selection gate re-runs
each new argmin on the reserved $n = 1000$ selection pool and promotes on strict RMS improvement.

*Training harness.* Population evaluation is batched through in-process Python bindings to the
native core: each generation's individuals are simulated scenario-parallel across CPU cores with the
interpreter lock released, network weights pass in memory rather than through files, and the
dispersion-independent tables (atmosphere, winds, reference trajectories) are shared read-only
across the batch. At the measured single-core cost of Section 7.2, the deployed run's
#R.sci(champ_run.actual_sims.training) training passes (#R.sci(champ_run.actual_sims.total) with the selection, curation and
re-evaluation passes) amount to roughly $#int(calc.round(champ_run.actual_sims.total * ms("NN-mamba") / 3.6e6))$ core-hours -- two nights of training
on a laptop.

*Deployed architecture.* Dense($17 -> 16$, swish) $->$ Mamba($d_"inner" = 16$, $d_"state" = 12$)
$->$ Dense($16 -> 2$, asinh), atan2 bank decoder; $962$ trainable parameters. The input mask selects
$17$ of the $35$ candidate observations (indices 0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27--30, 32--34:
instantaneous orbital/aerodynamic state, two reference-trajectory interpolations, the seam-free
bank-history pairs, and the three predicted correction-$Delta v$ components). Each input is
normalized by a calibrated per-input affine or asinh transform mapping its $[p_5, p_95]$ span to
$[-1, 1]$. Three actuator-side parameters -- the navigation density-filter gain and its rate limit,
and the command-shaping acceleration limit -- are co-optimized with the network weights.

*FNPAG.* Two-second replan cycle; bisection corrector on the constant capture-bank magnitude (about
eleven forward integrations per replan); forward-predictor integration step GA-tuned at $3.8$ s; the
onboard atmosphere is scaled by the navigation-estimated density factor so the predictor tracks the
measured atmosphere.

*One runtime, training to flight.* The policy that flies is the artifact that trained: candidates
are evaluated, selected, and deployed through the same native forward pass, so no export or
re-implementation step separates the training loop from the flight code. An independent PyTorch
implementation of every cell type (used by the supervised warm-start path, not by the deployed
runs) is held to numerical agreement with the native runtime by cross-language tests -- maximum
absolute forward-pass difference near machine epsilon ($10^(-16)$--$10^(-14)$) over $100$-step
stateful sequences.

*Timing.* Wall-clock per complete simulation over $200$ sequential runs of each deployed scheme on
one idle core of an Apple M4 Pro laptop (native code compiled with rustc #bench_rustc at the release
profile with link-time optimization; $64$-bit floats throughout; single-threaded). Quoted values
are the median of five timed batch repeats after one warm-up; the standard deviation over the
repeats is #R.span(bench_spread, d: 1)% of the median. A flown simulation spans $approx 480$--$770$ guidance updates depending on the scheme,
giving the per-update and per-replan shares quoted in Section 7.2 (whole-simulation cost divided by
update count -- a loose upper bound on pure guidance inference). These are
implementation-and-host measurements supporting the relative comparison; flight-processor
worst-case execution time and memory footprint are not established here (Section 9).

*Artifacts.* The simulator, training harness, analysis code, every configuration, the deployed
network weights, the committed per-run evaluation records behind each table, and the scripts that
regenerate every figure and number are released publicly under an MIT license at
#link("https://github.com/GovitPowerz/Aerocapture"). Every number in the tables regenerates,
without retraining, from those records.

#pagebreak(weak: true)
= Appendix B: architecture probes -- CfC, xLSTM, and the Mamba-3 axes

A controlled negative result supporting the Section 6 headline: does any recent recurrent family
beat the plain selective SSM on the sizing tail? Three probes, each anchored on a Section 6 cell at
matched parameter budget: a closed-form continuous-time cell (CfC @hasani2022cfc, hypothesis:
input-dependent time constants suit the fast-near-periapsis, static-in-vacuum phase structure)
against the GRU anchor; the exponential-gated sLSTM and matrix-memory mLSTM of xLSTM @beck2024xlstm
(hypothesis: sharp revision of a stored estimate at the bounce or a density shock) against the LSTM
anchor; and a $2 times 2$ over Mamba-3's @lahoti2026mamba3 two axes -- exponential-trapezoidal
discretization and complex (rotational) state -- at the deployed Mamba anchor, whose Euler-real arm
is bit-identical to the deployed Mamba layer.

This appendix is a development-regime study (Section 2.2): every arm and its baseline trained and
scored under the shared density history, so its verdicts are within-regime comparisons. Every arm
shares one training protocol (the genetic algorithm at population $300$ for $5000$
generations, two scenarios per individual, adaptive hardest-seed curation, live actuator
scaffolding) and one reserved evaluation pool, the probe pool (offset $10^7$, $n = 1000$; each arm
scored with its co-trained scaffolding), with three seed-repeats per arm; $sigma_"run"$ is the
standard deviation over repeats. Capture is $99.97$--$100%$ everywhere and every arm passes the
feasibility check of Section 6.2 -- zero heat-flux, g-load, and heat-load violations on every
repeat -- so the comparison is a pure tail-$Delta v$ story. Two caveats apply throughout: these are $p_95$/$"CVaR"_95$ statistics
at $n = 1000$, not the far-tail $"CVaR"_(99.9)$ sizing metric, and the probe budget is deliberately
sub-headline ($1.5$M evaluations versus the deployed $512 times 20\,000$), so the absolute values
sit above the headline numbers and are not mission figures.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Arm*], [*Params*], [*Capture %*], [*Viol. %*], [$bold(p_50)$], [$bold(p_95 plus.minus sigma_"run")$], [$bold("CVaR"_95 plus.minus sigma_"run")$],
    ),
    table.hline(stroke: 0.35pt),
    [Mamba (baseline)], [962], [99.97], [0.0], [114.0], [$121.6 plus.minus 0.5$], [$124.1 plus.minus 0.3$],
    [Mamba-3 trapezoidal], [978], [100.0], [0.0], [115.5], [$124.9 plus.minus 2.1$], [$128.8 plus.minus 2.3$],
    [Mamba-3 complex], [1154], [100.0], [0.0], [113.7], [$121.1 plus.minus 1.8$], [$123.8 plus.minus 2.1$],
    [Mamba-3 both], [1170], [100.0], [0.0], [114.0], [$121.6 plus.minus 1.3$], [$124.2 plus.minus 1.2$],
    [GRU (baseline)], [1014], [100.0], [0.0], [114.7], [$123.7 plus.minus 1.5$], [$126.7 plus.minus 1.3$],
    [CfC], [1003], [100.0], [0.0], [116.5], [$126.5 plus.minus 0.7$], [$130.4 plus.minus 0.1$],
    [LSTM (baseline)], [1082], [100.0], [0.0], [115.4], [$124.3 plus.minus 1.4$], [$127.3 plus.minus 1.4$],
    [sLSTM], [1042], [100.0], [0.0], [115.7], [$124.7 plus.minus 2.6$], [$127.6 plus.minus 3.1$],
    [mLSTM], [1078], [100.0], [0.0], [118.1], [$127.4 plus.minus 3.5$], [$130.8 plus.minus 4.8$],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The nine probe arms: three-seed means, correction $Delta v$ in m/s on the shared
  $n = 1000$ probe pool. Baselines are retrained in-regime (not the higher-budget reference runs --
  see the budget caveat below). Viol. is the any-constraint violation rate; every arm is feasible
  on every repeat.],
) <tbl-probes>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Treatment vs baseline*], [*Metric*], [*Gap*], [$bold(sigma_"comb")$], [*Verdict*],
    ),
    table.hline(stroke: 0.35pt),
    [CfC vs GRU], [$p_95$ / $"CVaR"_95$], [$+2.8$ / $+3.7$], [$1.6$ / $1.3$], [*significantly worse*],
    [Trapezoidal vs Mamba], [$p_95$ / $"CVaR"_95$], [$+3.3$ / $+4.8$], [$2.2$ / $2.3$], [*significantly worse*],
    [Complex vs Mamba], [$p_95$ / $"CVaR"_95$], [$-0.5$ / $-0.3$], [$1.9$ / $2.1$], [within run variance],
    [Both vs Mamba], [$p_95$ / $"CVaR"_95$], [$+0.0$ / $+0.1$], [$1.4$ / $1.2$], [within run variance],
    [sLSTM vs LSTM], [$p_95$ / $"CVaR"_95$], [$+0.4$ / $+0.3$], [$2.9$ / $3.4$], [within run variance],
    [mLSTM vs LSTM], [$p_95$ / $"CVaR"_95$], [$+3.1$ / $+3.5$], [$3.8$ / $5.0$], [within run variance (high $sigma_"run"$)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Within-family significance: gap = treatment minus baseline (positive = worse), cleared
  when $|"gap"| > sigma_"comb" = sqrt(sigma_"base"^2 + sigma_"arm"^2)$, with the per-arm
  $sigma_"run"$ taken from @tbl-probes.],
) <tbl-probes-sig>

The within-family rows are the rigorous claims. The CfC is significantly worse than the GRU on both
tail statistics, and its tiny $sigma_"run"$ ($0.1$ on $"CVaR"_95$) makes the loss stable and
reproducible, not a bad-luck draw: continuous-time time constants add a harder optimization
landscape for a timescale adaptation the fixed-cadence gates already learn. Mamba-3's trapezoidal
discretization is significantly worse than the plain cell -- the seed-repeats upgrade an earlier
single-run "no benefit" to a measured degradation. Complex state, both axes combined, the sLSTM, and
the mLSTM all sit within run variance of their baselines; the mLSTM is notably high-variance
($sigma_"run"$ up to $4.8$) and trends worse without clearing the bar -- matrix memory buys
instability, not tail robustness. Cross-family, all nine arms share the regime and pool but the
anchors differ slightly ($962$/$1014$/$1082$ parameters), so the ranking is suggestive rather than
matched: the plain Mamba tops the field (tied with its own complex arms), about $2$ m/s ahead of the
GRU and $2.7$ ahead of the LSTM at $p_95$ -- consistent with the deployed headline.

// The budget caveat's reference rows (data/probes/): each probe's higher-budget reference against
// its retrained in-regime baseline, p95 in m/s. The Mamba and LSTM references beat their
// baselines; the GRU sweep cell does not.
#let pr_mamba = R.probe_ref("mamba3", "962_baseline", "baseline")
#let pr_lstm = R.probe_ref("xlstm", "lstm_p1082_sweep", "lstm")
#let pr_gru = R.probe_ref("cfc", "gru_p1014_sweep", "gru")
#assert(pr_mamba.ref < pr_mamba.base and pr_lstm.ref < pr_lstm.base and pr_gru.ref > pr_gru.base,
  message: "the budget caveat states that the Mamba and LSTM references beat their baselines at p95 and the GRU sweep cell does not")
One caveat is load-bearing. Each probe also scored a higher-budget reference row: the Section 6
sweep cell for the GRU and the LSTM (population $512$ for the same $5000$ generations,
$1.7 times$ the probe evaluations) and a full-budget plain-Mamba run ($512 times 10\,000$,
$3.4 times$) for the Mamba. The Mamba and LSTM references sit #R.span((pr_mamba, pr_lstm).map(r => r.base - r.ref)) m/s better at
$p_95$ than their retrained in-regime baselines (Mamba $#R.fixed(pr_mamba.base)$ versus $#R.fixed(pr_mamba.ref)$; LSTM
$#R.fixed(pr_lstm.base)$ versus $#R.fixed(pr_lstm.ref)$), a training-budget effect, not architecture. The GRU sweep cell
sits $#R.fixed(pr_gru.ref - pr_gru.base)$ m/s worse than its baseline ($#R.fixed(pr_gru.base) plus.minus #R.fixed(pr_gru.sd)$ versus
$#R.fixed(pr_gru.ref)$), so one larger-budget run can land on either side. Every treatment therefore compares
against its retrained in-regime baseline and never against a higher-budget reference. Two
smaller caveats: the probe cells are trained through the gradient-free path only (no warm-start),
and the sLSTM's $40$-parameter deficit against the LSTM is a cell-definition cost (single bias),
not a budget mismatch -- it does not explain its null.

The consistent null across three independent families points at the task, not the cells. A single
atmospheric pass is a few hundred guidance ticks whose latent state worth remembering is a handful
of slowly-varying dispersion parameters -- density bias, the Ornstein--Uhlenbeck perturbation state,
aerodynamic dispersions -- and the engineered inputs already carry most of the temporal signal
(Section 8). A diagonal selective-SSM state of dimension $12$--$16$ saturates what little internal
memory the problem rewards; the CfC's timescale adaptation, the xLSTM's revision and associative
recall, and Mamba-3's long-context and state-tracking axes all target capacity this smooth,
low-bandwidth control signal never exercises. The deployed cell wins not by more sophisticated
memory but by just enough memory, cheaply. Framed positively, the probes validate the methodology:
the adaptive-seed, tail-led, matched-anchor protocol distinguishes between architectures rather than
rubber-stamping the newest one -- three 2024--2026 recurrent families tried, none better, and two
arms (the CfC and the trapezoidal discretization) significantly worse.

#pagebreak(weak: true)
= Appendix C: quantizing the deployed Mamba head

Can the deployed #box[$962$-parameter] policy shrink for flight memory without giving back the
tail it was selected for? This appendix quantizes the deployed head: weight-only, symmetric,
fake-quantized. Rounded values are stored back as $64$-bit floats, so the runtime, the regression
goldens, and the simulation are untouched by construction. A post-training quantization (PTQ)
grid crosses bit width ($8$/$6$/$4$/$3$/$2$), scale granularity (per-channel, per-tensor), and
tensor policy (all tensors, or projections only, keeping the SSM dynamics and bias scalars in
floating point); two quantization-aware training (QAT) arms then retrain at the best $4$-bit cell
of the grid. Selection ran on the fresh re-quote pool of Appendix A ($n = 1000$ per grid cell,
per-scenario noise); the finalists below are quoted once on the frozen confirmatory pool of
Section 7 ($10 times 100\,000$ scenarios each), paired by replicate against the deployed policy,
preserving the paper's selection-versus-quote discipline.

The grid's shape is simple (@fig-quant-sweep). Eight-bit quantization is free: every $8$-bit cell
captures every scenario within $0.5$ m/s of full precision at $"CVaR"_95$, with every tensor
quantized or not. Degradation is visible from $6$ bits and catastrophic at $2$. At $4$ bits -- the
target compression -- the selection rule (highest capture rate, then lowest $"CVaR"_95$) picks
per-tensor scales on the projection tensors only: it is the one $4$-bit cell at full capture
($"CVaR"_95$ $#R.fixed(q4_verdict.dv_cvar95, d: 0)$ m/s against $#R.fixed(R.quant4_grid.baseline.dv_cvar95, d: 0)$ in full precision), where per-tensor
all-tensors captures $#R.fixed(100 * R.quant4_cell(4, "per_tensor", "all").capture_rate)%$ and the two per-channel cells
$#R.fixed(100 * R.quant4_cell(4, "per_channel", "proj_only").capture_rate)%$ and $#R.fixed(100 * R.quant4_cell(4, "per_channel", "all").capture_rate)%$; at $3$ bits only the same cell stays near full capture
($#R.fixed(100 * R.quant4_cell(3, "per_tensor", "proj_only").capture_rate)%$). The ranking between granularities is a rounding draw rather than a
granularity effect: per-channel scales have the lower weight error, and a jitter sweep in the
repository that enlarges every quantization step by a few percent moves the $4$-bit cells across
one another. The policy split is the robust finding; the granularity verdict is one draw.

#fig("fig_quantization.svg", [The post-training quantization grid on the re-quote pool
($n = 1000$ per cell, per-scenario noise): capture rate (left) and $"CVaR"_95$ on a log axis (right)
versus bit width, for the four granularity $times$ policy series; the verdict cell is starred.], <fig-quant-sweep>)

// One tbl-quant-finalists row: capture / CVaR95 / CVaR99.9 (+- replicate SE) / heat-load exceedances
// from the pooled confirmatory cell, p50 from the finalists' re-quote pool (n = 10 000; the
// confirmatory file has no p50), and the paired replicate delta of CVaR99.9 against the deployed
// policy with its t(9) 95% interval (a dash on the deployed row).
#let quant_row(label, conf_label, finalist_label) = {
  let c = q4(conf_label)
  let delta = if conf_label in R.quant4.paired {
    let p = q4p(conf_label).delta_cvar999
    box[$#R.signed(p.mean)$ #text(size: 8pt)[\[#R.ci(p.ci95)\]]]
  } else { [--] }
  (label, [#R.fixed(c.capture_pct, d: 4)], [#R.fixed(R.quant4_finalist(finalist_label).dv_p50)],
    [#R.fixed(c.cvar95)], [$#R.fixed(c.cvar999) plus.minus #R.fixed(c.cvar999_se)$], delta, [#c.hl_viol_n])
}
#figure({
  set text(size: 9pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Variant*], [*Capture %*], [$bold(p_50)$], [$bold("CVaR"_95)$], [$bold("CVaR"_(99.9) plus.minus "SE")$], [$bold(Delta"CVaR"_(99.9))$ paired], [*Heat load*],
    ),
    table.hline(stroke: 0.35pt),
    ..quant_row([Deployed (64-bit)], "ou_marginal/hl_mamba_p962", "champion_fp"),
    ..quant_row([PTQ, best 4-bit], "quant_v4/ptq4_verdict", "ptq4_verdict"),
    ..quant_row([QAT fine-tune, 4-bit], "quant_v4/qat4_finetune", "qat4_finetune"),
    ..quant_row([QAT scratch, 4-bit], "quant_v4/qat4_scratch", "qat4_scratch"),
    table.hline(stroke: 0.7pt),
  )},
  caption: [Quantization finalists: capture, $"CVaR"_95$, $"CVaR"_(99.9)$ and the count of heat-load
  exceedances on the frozen confirmatory pool ($10 times 100\,000$ scenarios each, per-scenario noise;
  correction $Delta v$ in m/s; replicate standard errors), $p_50$ on the re-quote pool
  ($n = 10\,000$), and the paired replicate delta of $"CVaR"_(99.9)$ against the deployed policy
  with its $95%$ interval over the ten replicates. The 4-bit cells use per-tensor scales on the
  projection tensors only. No cell exceeds the heat-flux or g-load limits beyond one scenario.],
) <tbl-quant-finalists>

Post-training rounding is not deployable: the best PTQ cell costs $#R.signed(q4p("quant_v4/ptq4_verdict").delta_cvar95.mean, d: 0)$ m/s at $"CVaR"_95$
and $#R.signed(q4p("quant_v4/ptq4_verdict").delta_cvar999.mean, d: 0)$ at $"CVaR"_(99.9)$, and loses $#q4("quant_v4/ptq4_verdict").lost$ of $10^6$ scenarios.
Quantization-aware training recovers the tail by either route. The champion checkpoint resumed for
$3000$ generations with every candidate's weights fake-quantized before each fitness evaluation,
so the genetic search optimizes the quantized policy directly, pays $#R.signed(q4p("quant_v4/qat4_finetune").delta_cvar95.mean)$ m/s at $"CVaR"_95$ and
gains $#R.fixed(-q4p("quant_v4/qat4_finetune").delta_cvar999.mean)$ at $"CVaR"_(99.9)$ (interval excluding zero) at full capture; the same
$4$-bit constraint trained from scratch at the champion's full $512 times 20\,000$ budget matches the
champion's far tail ($#R.signed(q4p("quant_v4/qat4_scratch").delta_cvar999.mean)$, interval across zero) at $#R.signed(q4p("quant_v4/qat4_scratch").delta_cvar95.mean)$
m/s of $"CVaR"_95$. Both intervals condition on this champion, the worst far tail of its three
seeds (Section 6.2), and both QAT rows sit inside that family's range: a $4$-bit projections-only
head is tail-equivalent to the full-precision family at sizing depth, by fine-tune or from scratch,
and not better than it. The fine-tune's deployed head is the validation gate's last promotion, and
its final population was wholly gate-infeasible on heat load -- the gate kept the feasible head.

Which parts of a selective SSM tolerate $4$ bits? A one-tensor-at-a-time probe quantizes a single
tensor group at $4$ bits per-tensor and keeps everything else in floating point (re-quote pool,
$n = 1000$ per cell):

// One tbl-quant-loo row: the tensor group, its parameter count (architecture constants), capture
// and the CVaR95 delta of the one-tensor probe.
#let loo_row(name, tensor, params) = {
  let v = R.quant4_loo(tensor)
  (name, [#params], [#R.fixed(100 * v.capture_rate)], [$#R.signed(v.delta_dv_cvar95)$])
}
#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([*Tensor group*], [*Params*], [*Capture %*], [$bold(Delta"CVaR"_95)$]),
    table.hline(stroke: 0.35pt),
    ..loo_row([input dense $W$], "layer_0.w", 272),
    ..loo_row([SSM input projection $W_(x,"proj")$], "layer_1.x_proj_w", 400),
    ..loo_row([SSM $Delta$-projection $W_(Delta)$], "layer_1.dt_proj_w", 16),
    ..loo_row([SSM dynamics $a_log$], "layer_1.a_log", 192),
    ..loo_row([SSM residual gains $d_"skip"$], "layer_1.d_skip", 16),
    ..loo_row([output dense $W$], "layer_2.w", 32),
    table.hline(stroke: 0.7pt),
  ),
  caption: [One-tensor-at-a-time sensitivity at $4$ bits per-tensor ($n = 1000$ per cell,
  re-quote pool, per-scenario noise). The SSM input projection and the input dense layer are the
  sensitive groups on this champion. The deltas do not add up to the all-tensors cell ($Delta v$
  statistics are conditional on capture) -- quantization errors interact.],
) <tbl-quant-loo>

The SSM input projection is the bottleneck on this champion ($#R.signed(R.quant4_loo("layer_1.x_proj_w").delta_dv_cvar95, d: 0)$ m/s from one tensor),
with the input dense layer next; the dynamics parameters $a_log$ and the residual gains, which an
earlier champion trained on the shared density path had put first, cost
$#R.signed(R.quant4_loo("layer_1.a_log").delta_dv_cvar95, d: 0)$ and $#R.signed(R.quant4_loo("layer_1.d_skip").delta_dv_cvar95, d: 0)$ here. The sensitivity map is a property of
the trained weights, not of the architecture, which is why the policy split (quantize the $720$
projection weights, keep the $242$ dynamics and bias scalars in floating point) is selected on the
grid per champion rather than assumed. The $Delta$-projection costs $#R.signed(R.quant4_loo("layer_1.dt_proj_w").delta_dv_cvar95, d: 0)$ under
one per-tensor scale (per-channel scaling of its single column would be lossless by construction).

The deployment benefit, stated plainly, is memory -- not compute:

// One tbl-quant-memory row from the analytic footprint of quantization_results.json.
#let mem_row(name, bits, granularity, policy, bold: false) = {
  let m = R.quant4_memory(bits, granularity, policy)
  let hi(s) = if bold { strong(s) } else { s }
  (hi(name), [#m.quant_bytes B], [#m.scale_bytes B], [#m.fp_bytes B], hi([#m.total_bytes B]), hi([$#R.fixed(m.f64_baseline_bytes / m.total_bytes) times$]))
}
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([*Cell*], [*Packed*], [*Scales*], [*Float*], [*Total*], [*vs 64-bit*]),
    table.hline(stroke: 0.35pt),
    ..mem_row([8-bit per-channel, projections], 8, "per_channel", "proj_only"),
    ..mem_row([4-bit per-channel, projections], 4, "per_channel", "proj_only"),
    ..mem_row([4-bit per-tensor, projections (deployed QAT cell)], 4, "per_tensor", "proj_only", bold: true),
    ..mem_row([4-bit per-tensor, all tensors], 4, "per_tensor", "all"),
    table.hline(stroke: 0.7pt),
  ),
  caption: [Analytic memory footprint (packed $b$-bit weights + $32$-bit scales + floating-point
  remainder; $64$-bit baseline $#R.quant4_memory(4, "per_tensor", "proj_only").f64_baseline_bytes$ B). The all-tensors cell is shown for
  contrast only -- it loses capture at $4$ bits. The headline is the deployed cell, bounded below
  by the $#R.quant4_memory(4, "per_tensor", "proj_only").fp_bytes$ B of dynamics and bias parameters kept in floating point.],
) <tbl-quant-memory>

On compute, the head accounts for $approx 1.22$ ms of the deployed policy's $#R.fixed(ms("NN-mamba"), d: 2)$ ms
whole-simulation cost (Section 7.2), a measurement of the architecture that the champion's weights
do not change:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header([*Forward kernel*], [*ns / tick*], [*vs deployed*], [*ms / simulation*]),
    table.hline(stroke: 0.35pt),
    [64-bit (deployed runtime)], [1888], [--], [1.22],
    [64-bit hand-rolled], [1670], [$-11.6%$], [1.08],
    [32-bit hand-rolled], [1290], [$-31.7%$], [0.83],
    [int8 weights, int8 activations], [1735], [$-8.1%$], [1.12],
    [packed int4 weights, int8 activations], [1747], [$-7.5%$], [1.13],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Forward-pass micro-benchmark (median ns per guidance tick, laptop-class core, scalar
  code, the SSM recurrence in floating point for every variant; ms per simulation over the
  measured $644$ forward passes of a flown trajectory). Quantization is not a compute win at this
  scale.],
) <tbl-quant-bench>

Single precision is the compute sweet spot ($-32%$ per tick); the integer kernels beat the
deployed runtime by only $approx 8%$, because dynamic activation quantization plus the
floating-point SSM recurrence ($192$ exponentials per tick) dominate a #box[$962$-parameter]
workload, and $4$-bit nibble unpacking cancels its bandwidth advantage. Simulation throughput in
the accuracy study is unchanged by construction, so no simulation-time claim is made. The
deployment case for the $4$-bit head is the $#R.fixed(R.quant4_memory(4, "per_tensor", "proj_only").f64_baseline_bytes / R.quant4_memory(4, "per_tensor", "proj_only").total_bytes) times$ memory footprint and
fixed-point-capable projection arithmetic, not speed.

Four caveats bound these claims. The accuracy study is weight-only: activations, the hidden state,
and the input normalization stay in $64$-bit floats (the integer kernels quantize activations for
the compute measurement only). Both QAT arms are single runs. The grid
cells are single-model $n = 1000$ evaluations. And the kernel ratios are laptop-class scalar
measurements that do not transfer to flight processors -- the memory table does.

#pagebreak(weak: true)
= Appendix D: per-scheme mission reports

Three two-page mission-performance cards on the final-evaluation Monte Carlo pool ($n = 1000$,
per-scenario noise), each pinned to its deployed policy so the statistics reproduce the
corresponding row of @tbl-perf: the deployed Mamba seed, the retuned joint-FTC (its
co-optimized-reference variant, Section 7.1) and the retuned FNPAG. The first page shows the corridor behaviour -- the classified
trajectory ensemble in the (energy, dynamic pressure), (energy, inclination), and
(energy, bank) planes, with the undispersed nominal overlaid; the dynamic-pressure
panel sets the ensemble against the shared occupancy envelope of @fig-corridor,
so one can read at a glance whether each scheme flies inside it. The second page shows
the correction-$Delta v$ distribution (total and its three burns), the thermal and
load-constraint margins, and the full statistics. Panels reuse the training
pipeline's own report charts; captured trajectories are blue, constraint violations
orange, failures red. The repository's generated reports carry the same card for every other
cell of this paper.

#include "appendix.typ"


// Colophon: what this PDF was built from (data/provenance.json, written by
// scripts/write_provenance.py; git_head is passed by `make pdf`).
#let prov = json("data/provenance.json")
#v(1em)
#block(width: 100%, stroke: (top: 0.35pt), inset: (top: 6pt))[
  #set text(size: 8.5pt)
  #set par(justify: false)
  *Provenance.* Every cell of @tbl-rl, @tbl-arch, @tbl-conditioning, @tbl-perf, @tbl-paired,
  @tbl-levers, @tbl-centered, @tbl-quant-finalists, @tbl-quant-loo and @tbl-quant-memory, every
  quoted number of Sections 5--10 and Appendix C, the compute quotes, the heat-load ceiling
  sentence of Section 6.2 and the budget caveat of Appendix B are read at compile time from
  `data/results.json`, `data/confirmatory_marginal.json`, `data/quote_marginal.json`,
  `data/heat_load_slope.json`, `data/stress_depth.json`, `data/centered_depth_v4.json`,
  `data/objective_centering_v4.json`, `data/quant_v4/`, `data/compute_benchmark.json`,
  `data/probes/` and the deployed seed's bundled run directory through `results.typ`, and the
  counts and orderings the prose states are asserted against the same files. Sections 4--5 and
  Appendix B quote development-regime numbers under that label. Paper inputs digest (SHA-256 over every
  tracked paper input, `data/provenance.json`): #raw(prov.paper_inputs_sha256). Run logs: Release
  #raw(prov.release_tag). Simulator crate #prov.simulator_crate_version, Typst #prov.typst_version,
  matplotlib #prov.matplotlib_version. Noise regime: #prov.noise_regime. Compiled at git
  #raw(sys.inputs.at("git_head", default: "(not passed: compile with make -C articles/paper pdf)")).
]
