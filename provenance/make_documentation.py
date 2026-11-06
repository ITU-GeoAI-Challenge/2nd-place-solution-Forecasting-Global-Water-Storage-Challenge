"""Generate DOCUMENTATION.docx — the detailed solution document.

Follows Zindi's documentation guideline (Overview, Architecture, ETL, Data Modeling,
Inference, Run Time, Performance Metrics, Error Handling and Logging, Maintenance and
Monitoring) and the code-review submission checklist.

    python provenance/make_documentation.py
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = Path(__file__).resolve().parent.parent
d = Document()
for st, sz in (("Normal", 10), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.5)):
    s = d.styles[st]; s.font.name = "Calibri"; s.font.size = Pt(sz)
    if st.startswith("Heading"): s.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

def H(t, lvl=1): d.add_heading(t, level=lvl)
def P(t, bold=False, italic=False, size=None):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    return p
def B(items):
    for i in items: d.add_paragraph(str(i), style="List Bullet")
def CODE(t):
    p = d.add_paragraph(); r = p.add_run(t); r.font.name = "Consolas"; r.font.size = Pt(8.5)
    p.paragraph_format.left_indent = Inches(0.3); return p
def TBL(head, rows, widths=None):
    t = d.add_table(rows=1, cols=len(head)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(head):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            rr = cells[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(9)
    d.add_paragraph()
    return t

# ============================== title =====================================
ti = d.add_paragraph(); ti.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = ti.add_run("A Step Ahead of Drought — Forecasting Global Water Storage")
r.bold = True; r.font.size = Pt(19)
st = d.add_paragraph(); st.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = st.add_run("Solution documentation\npublic leaderboard RMSE 0.662803625")
r.font.size = Pt(11.5); r.italic = True
d.add_paragraph()

# ============================== 1 =========================================
H("1. Overview and Objectives")
P("The task is to forecast monthly terrestrial water storage (TWS) anomalies on a global "
  "1° land grid of 15,715 cells. Training covers 2002-05 to 2015-08 (138 months) and the "
  "test period 2015-09 to 2018-12 (18 months); the two are strictly sequential with no "
  "overlap, so this is forward extrapolation rather than interpolation. The target for a "
  "row at month t is TWS at month t+1, verified empirically: target[t] equals TWS_t[t+1] "
  "for 100.00% of 1,977,398 joinable training rows.")
P("Two properties of the test set shape the whole design. First, 66.5% of test rows have "
  "their TWS_t masked, so for most rows the model must forecast from an anchor several "
  "months old rather than from last month's observation. Lead times therefore range from "
  "1 to 7 months and are handled by a single model with lead time as a feature. Second, "
  "the GRACE to GRACE-FO mission gap (2017-07 to 2018-06) removes observations entirely "
  "from part of the window.")
P("Objective and outcome. The objective was the competition metric, RMSE, minimised "
  "without violating the date-discipline and external-data rules. The delivered solution "
  "is a minimum-variance blend of six independently trained models, shrunk toward a "
  "persistence anchor:")
CODE("submission = persistence + γ · ( Σ wᵢ · modelᵢ  −  persistence )")
P("It scores 0.662803625 on the public leaderboard against 0.886420231 for persistence, a "
  "25.2% reduction in RMSE. Neither the weights nor γ is tuned by trial and error against "
  "the leaderboard; both are solved in closed form (§4.4). The method predicted this "
  "submission's score as 0.662660 before it was submitted, an error of +0.000144.")

# ============================== 2 =========================================
H("2. Architecture Diagram")
P("Data flow from raw sources through ETL, the shared feature panel, the six members, the "
  "blend and the verification steps.")
d.add_picture(str(HERE / "provenance" / "architecture.png"), width=Inches(6.6))
d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
P("Regenerate with provenance/make_architecture_diagram.py.", italic=True, size=8.5)

# ============================== 3 =========================================
H("3. ETL Process")
H("3.1 Extract", 2)
TBL(["Source", "Format / volume", "Method", "Coverage"], [
 ["Competition files (Train, Test, SampleSubmission)", "CSV, ~315 MB, 2,154,021 train and 280,961 test rows",
  "Provided by the host; read directly", "2002-05 – 2018-12"],
 ["ERA5-Land monthly means (Copernicus CDS, DOI 10.24381/cds.68d2bb30)", "netCDF, ~750 MB at 0.25°",
  "data_prep/era5_fetch.py via the cdsapi client", "2002–2018, one request per year"],
 ["ESA CCI satellite soil moisture (CDS satellite-soil-moisture)", "netCDF",
  "data_prep/cci_fetch.py", "2002–2018, monthly average"],
 ["CDS drought indices (derived-drought-historical-monthly)", "netCDF",
  "data_prep/drought_fetch.py", "SPEI at 12/24/48-month accumulation"],
])
P("Frequency. The competition data is static, so extraction is a one-time operation. In an "
  "operational setting the three external sources publish monthly and the same scripts "
  "would run on that cadence; ERA5-Land carries roughly a two-to-three month latency for "
  "the final product, which is why the guidelines permit ERA5T operationally with final "
  "ERA5-Land as the retrospective proxy. This solution uses the final product throughout "
  "and states so.")
P("Volume note. ERA5-Land native resolution is 0.1°, which is 1401×3600 cells per field "
  "and roughly 100× more data than the target grid needs. Requests are made at 0.25° "
  "server-side, which contains the 1° cell centres exactly and cuts the download by about "
  "two orders of magnitude. Snow water equivalent is the one exception and is taken at "
  "native 0.1°, because 505 glaciated Arctic cells (3.2%) differed measurably between the "
  "two resolutions.")

H("3.2 Transform", 2)
P("Regridding. Every external source is reduced to the competition's own 1° / 15,715-cell "
  "grid by land-area averaging: each native point is assigned to the 1° box containing it "
  "and averaged over the non-missing members of that box. ERA5-Land masks ocean as NaN, so "
  "a nanmean is already the land-area mean. Coastal boxes therefore average over fewer "
  "points, and boxes with no land at all are dropped.")
B(["Unit trap handled explicitly: in the ERA5-Land monthly product, precipitation, "
   "evaporation, runoff, snowmelt, snowfall and potential evaporation are stored as daily "
   "mean RATES (m/day), not monthly totals, and are multiplied by days-in-month. "
   "Instantaneous fields (snow depth, soil water layers, temperatures, snow cover) must "
   "not be, and are not.",
   "Grid alignment trap handled explicitly: the CDS 1° grid sits on whole degrees while the "
   "competition cell centres sit on half degrees, so a 1° request would sample half a cell "
   "off in both axes. data_prep/era5_fetch.py refuses --grid 1.0 at runtime.",
   "Soil-moisture fallback: the ESA CCI root-zone retrieval (0–100 cm) has 72.8% coverage "
   "against the surface retrieval's 77.0%. The panel takes the root zone where observed and "
   "falls back to the surface product where not, so the block is a strict superset of "
   "either alone rather than stripping the feature from ~4% of rows."])
P("Feature construction. All members share a (cell × month) panel built once per run. TWS "
  "is filled from its three consistent sources — train TWS_t at month t, train target at "
  "t+1 (the target is next month's TWS), and test TWS_t where unmasked — after which a "
  "forward-running index gives, for every (cell, month), the most recent month at or before "
  "it carrying an observation. That index defines the anchor and the lead time.")
P("Derived features are all as-of: per-cell climatology, variance and trend are cumulative "
  "along the month axis and sliced at the anchor, so no statistic can include a month at or "
  "after the target. The reference model uses 132 features spanning the cell's own TWS "
  "history, the supplied covariates at the anchor and at t, neighbourhood aggregates at "
  "ring 1/2/3, per-cell response coefficients, and ERA5-derived water-balance terms. The "
  "storage member adds four columns for 136.")
H("3.3 Load", 2)
P("Regridded sources are written as three Parquet files keyed on (time, lat, lon): "
  "era5_grid.parquet (191 MB), drought_grid.parquet (114 MB) and cci_grid.parquet (38 MB). "
  "Parquet is used for columnar reads — a member loads only the handful of variables it "
  "needs rather than the whole table.")
P("In memory the panel is held as dense float32 arrays of shape (15,715 × 201) per "
  "variable, indexed by integer cell and month codes rather than by lat/lon lookups. "
  "Per-cell statistics are computed once for all cells with cumulative sums instead of "
  "per-row, which is the single largest efficiency decision in the pipeline. Model inputs "
  "are built in chunks of 400,000 rows to bound peak memory.")

# ============================== 4 =========================================
H("4. Data Modeling")
H("4.1 Problem framing", 2)
P("Five of the six members predict the CHANGE from the anchor rather than the level: "
  "label = y − TWS[cell, anchor], prediction = anchor + f(X). This removes the per-cell "
  "mean, which is large and static, and leaves the model to predict only the part that "
  "moves. The sixth member (v85) predicts the level directly and is kept precisely because "
  "it makes a different error.")
P("Training rows are (cell, t, anchor) triples generated across all lead times, weighted by "
  "share[h−1], the lead-time mix of the actual test set, so that validation is measured "
  "under the test distribution rather than the training one.")
H("4.2 The six members", 2)
TBL(["Member", "Public LB", "Weight", "Description"], [
 ["xgb_v60_recurse", "0.685562412", "0.2154", "Recursive: three stages, each retrained on the previous stage's predictions as pseudo-labels; inference steps month-by-month, refilling the TWS panel between months"],
 ["xgb_v85_level_hl96", "0.672036642", "0.1955", "Predicts the target level rather than the change from the anchor"],
 ["xgb_v87_storage_hl96", "0.672415764", "0.1830", "132 features plus four modelled-storage columns: ERA5-Land soil column and snow, as anomalies against an as-of climatology and as anchor→t changes"],
 ["realmlp_v72_e15", "0.695636003", "0.1779", "RealMLP-TD neural network (pytabkit 1.5.0), 15 epochs pinned, early stopping off"],
 ["xgb_v81_recency_hl96", "0.670554680", "0.1611", "Reference model: 132 features, 5 seeds × 802 rounds, refit weighted by recency with a 96-month half-life"],
 ["xgb_v32_era5", "0.677796328", "0.0672", "Earlier ERA5-Land feature generation"],
])
P("Members are selected for decorrelation, not accuracy. The two weakest models by RMSE — "
  "the recursive model and the neural network — carry the largest and fourth-largest "
  "weights, both above the best single model's 0.161. A blend gains from members whose "
  "errors differ; a more accurate model predicting nearly the same thing contributes "
  "nothing. Adding the second-best blend variant to this set gives it exactly zero weight.")
P("Where each member was trained. Each is shipped as the artefact that actually produced "
  "it, not a later re-run of the same configuration.")
TBL(["Member", "Platform", "Producing artefact"], [
 ["xgb_v81_recency_hl96", "Kaggle CPU", "members/kaggle_v81_recency_deep.ipynb"],
 ["xgb_v85_level_hl96", "Kaggle CPU", "members/kaggle_v85_level_deep.ipynb"],
 ["xgb_v87_storage_hl96", "Kaggle CPU", "members/kaggle_v87_storage_deep.ipynb"],
 ["xgb_v60_recurse", "Kaggle CPU", "members/kaggle_v60_recurse_deep.ipynb"],
 ["realmlp_v72_e15", "Kaggle CPU", "members/kaggle_v72_realmlp_e15.ipynb — training recorded at version 17: https://www.kaggle.com/code/lastopt/fgws25?scriptVersionId=347420311"],
 ["xgb_v32_era5", "LOCAL (not Kaggle)", "members/sub_xgb_v32_era5.py"],
])
P("Five of the six ran on Kaggle. xgb_v32_era5 is the exception and was trained locally; its "
  "run log is at provenance/emissions/runlog_v32_era5.log and is the only log in the project "
  "that writes a blend member. Supporting experiment record: "
  "https://www.kaggle.com/code/lastopt/fgws25 . The RealMLP member's training run is "
  "recorded at a pinned version of that notebook, "
  "https://www.kaggle.com/code/lastopt/fgws25?scriptVersionId=347420311 (version 17). "
  "That matters more for this member than the others: it is the only one trained with "
  "torch rather than XGBoost, and torch on CPU is not bit-identical across thread "
  "counts, so a recorded execution is the strongest evidence available for it.")
P("One consequence for anyone re-running this: v32 is the member whose reproduction on "
  "Kaggle is not a like-for-like repeat of its original environment. It is CPU XGBoost with "
  "fixed seeds so it should still match, but if any member returns agreement rather than "
  "equality, that is the one with a legitimate reason.")
H("4.3 Algorithm and hyperparameters", 2)
P("Five members are XGBoost with an identical configuration; only features, target or "
  "training procedure differ.")
CODE("objective        reg:squarederror        grow_policy      lossguide\n"
     "eval_metric      rmse                    max_leaves       255\n"
     "learning_rate    0.05                    max_depth        0 (unbounded, leaf-wise)\n"
     "min_child_weight 30.0                    tree_method      hist\n"
     "colsample_bytree 0.8                     max_bin          127\n"
     "subsample        0.8                     nthread          16\n"
     "reg_lambda       1.0                     seed             42 (+1000·i per seed)\n\n"
     "FRAC 0.20 of available row-lead pairs    N_SEEDS 5        refit rounds 802 (fixed)\n"
     "VAL_MONTHS 24                            recency half-life 96 months")
P("min_child_weight is 30 rather than a row count because it is a sum of Hessians and the "
  "rows carry lead-time weights averaging about 1/7; 200 rows corresponds to roughly 29 "
  "Hessian. The refit round count is FIXED at 802 rather than derived from early stopping, "
  "so that a change in validation behaviour cannot silently change model capacity.")
P("Reproducibility of training. Every seed is set explicitly (seed = 42 + 1000·i for the "
  "i-th of five). Training is CPU-only and the XGBoost version is pinned to 3.2.0 inside "
  "each notebook, checked through importlib.metadata BEFORE the library is imported — "
  "pip-installing a different version after the C extension has loaded does not swap it, so "
  "an install-then-reload would silently keep the old build. GPU training was rejected "
  "deliberately: it scored slightly better and emits less, but CPU and GPU XGBoost produce "
  "different models from identical inputs.")
H("4.4 Blending and calibration", 2)
P("Two submissions differ exactly by the difference of their errors, pᵢ − pⱼ = eᵢ − eⱼ. The "
  "full error covariance is therefore recoverable from the submission files and the "
  "published scores alone, with no held-out data:")
CODE("cov(eᵢ, eⱼ) = ( sᵢ² + sⱼ² − mean((pᵢ − pⱼ)²) ) / 2\n"
     "w           = Σ⁻¹1 / (1ᵀ Σ⁻¹ 1),  clipped non-negative by active-set iteration")
P("The same identity prices a candidate before it is submitted: a new model improves the "
  "blend if and only if sⱼ < √(s_b² + d²) with d² = mean((p_blend − pⱼ)²). This follows "
  "without assuming anything about how the errors correlate, and every member was admitted "
  "on that test.")
P("Shrinkage toward the persistence anchor is solved rather than searched. With "
  "A = s²_persistence, C = mean((blend − anchor)²) and B pinned by MSE(γ=1) = s²_blend:")
CODE("MSE(γ) = A + 2γB + γ²C        γ* = −B/C        γ* = 0.9669873714")
P("Back-solving γ from the measured score of the submitted file afterwards returns 0.9667 "
  "against the 0.9670 used — a difference worth 0.000000, so the shrinkage sits at its "
  "optimum rather than near it.")
H("4.5 Validation approach", 2)
P("Validation is the last 24 months of the training window, with rows weighted to the test "
  "lead-time mix. This ranks candidates but does NOT predict leaderboard movement reliably, "
  "and that was measured rather than assumed: one feature block improved weighted "
  "validation by 0.00738 — placebo-controlled, 7.5× the measured noise floor, passing every "
  "row-level check — and scored 0.00186 WORSE on the leaderboard.")
P("The diagnosis is that a conventional split measures the wrong kind of generalisation. "
  "Validation covers the 24 months immediately after training, while the test period runs 1 "
  "to 40 months beyond it. A forward split — fit on months ≤ cut, evaluate strictly after — "
  "was therefore adopted for late-stage screening, and it reversed the ranking of candidate "
  "post-processing schemes: one worth −0.00540 on a random split came out at +0.00044 "
  "forward and was rejected. Two guards followed from this: redundancy against existing "
  "columns is treated as the only screen with demonstrated leaderboard validity, and "
  "candidates are priced with the closed-form bar in §4.4 rather than by local gain.")

# ============================== 5 =========================================
H("5. Inference")
P("Inference is offline batch scoring, not a hosted service. The full test set is predicted "
  "in one pass at the end of each member's training notebook.")
B(["Input: the same (cell × month) panel used in training, indexed at each test row's "
   "(cell, t, anchor). Feature construction code is shared with training verbatim, so no "
   "train/serve skew is possible.",
   "The recursive member is the exception: it predicts month by month, writing each month's "
   "prediction back into the TWS panel before advancing, so later months condition on "
   "earlier predictions rather than on a stale anchor.",
   "Output: one CSV of (ID, Target) per member, aligned on ID against SampleSubmission and "
   "never on row order.",
   "Interpretation: Target is the predicted TWS anomaly for that cell at month t+1, in the "
   "same normalised units as the supplied TWS_t. It is a point forecast with no uncertainty "
   "band — see §10."])
P("Combination. blend/make_persistence.py rebuilds the anchor and blend/make_final_blends.py "
  "produces the submission. Together they take under a minute and train nothing.")
P("Versioning and retraining. Each member is identified by its notebook and its output CSV; "
  "member scores are recorded alongside the weights, and the blend is fully determined by "
  "those two things, so any change is auditable. Retraining for a new month means rerunning "
  "the fetch scripts, rebuilding the panel and rerunning the notebooks; the blend weights "
  "would need re-solving only if member scores changed materially, since the weight solution "
  "depends on scores and pairwise prediction distances alone.")

# ============================== 6 =========================================
H("6. Run Time")
P("Measured wall clock. Training ran on Kaggle CPU instances (4 physical cores, "
  "NTHREAD=16, no GPU, no internet at train time).")
TBL(["Stage", "Script / notebook", "Wall clock"], [
 ["Fetch ERA5-Land", "data_prep/era5_fetch.py", "~4 h (CDS queue-dependent)"],
 ["Fetch ESA CCI", "data_prep/cci_fetch.py", "~1 h"],
 ["Fetch CDS SPEI", "data_prep/drought_fetch.py", "~2 h"],
 ["Regrid, all three", "data_prep/*_grid.py", "~25 min"],
 ["Train xgb_v60_recurse", "members/kaggle_v60_recurse_deep.ipynb", "246.7 min"],
 ["Train xgb_v81_recency_hl96", "members/kaggle_v81_recency_deep.ipynb", "167.7 min"],
 ["Train xgb_v87_storage_hl96", "members/kaggle_v87_storage_deep.ipynb", "84.7 min"],
 ["Train realmlp_v72_e15", "members/kaggle_v72_realmlp_e15.ipynb", "59.4 min"],
 ["Train xgb_v85_level_hl96", "members/kaggle_v85_level_deep.ipynb", "58.2 min"],
 ["Train xgb_v32_era5 (local, not Kaggle)", "members/sub_xgb_v32_era5.py", "117.8 min"],
 ["Rebuild the anchor", "blend/make_persistence.py", "~40 s"],
 ["Build the submission", "blend/make_final_blends.py", "~15 s"],
 ["Date-discipline proof", "provenance/leak_test.py", "~80 s"],
])
P("Provenance was checked per member by comparing each prediction CSV against the file that wrote it. Five of six are clean: the CSV postdates its notebook by roughly the recorded runtime. The exception is xgb_v32_era5, trained locally rather than on Kaggle, whose script carries a modification date two days after the run that produced the shipped CSV — it is the right script, writing that exact filename with a matching run log, but byte-level correspondence with what executed cannot be asserted without re-running it.")
P("Training total is about 12.3 hours across the six members. Everything after training — "
  "the blend, the shrinkage and both verification steps — completes in under three minutes.")

# ============================== 7 =========================================
H("7. Performance Metrics")
H("7.1 Leaderboard", 2)
TBL(["Submission", "Public RMSE", "Note"], [
 ["persistence (baseline)", "0.886420231", "last observed TWS carried forward"],
 ["best single model (xgb_v81_recency_hl96)", "0.670554680", "—"],
 ["blend of six, before shrinkage", "≈0.663", "min-variance weights"],
 ["submission.csv (blend + shrinkage)", "0.662803625", "the submitted file"],
])
P("Private leaderboard score: to be released by the host. The forecast made before "
  "submission was 0.662660, an error of +0.000144 against the realised public score.")
P("Public-score precision. With 280,961 test rows and leptokurtic errors (measured kurtosis "
  "≈ 7–8 from two independent estimates), the sampling standard deviation of a public RMSE "
  "on a 20–30% split is roughly 0.003–0.004. Differences below about 0.001 between "
  "submissions are therefore not meaningfully distinguishable, which is why members are "
  "admitted on the closed-form bar rather than on small score differences.")
H("7.2 Validation and efficiency", 2)
TBL(["Metric", "Value"], [
 ["Weighted validation RMSE, reference model", "0.6048 (24-month holdout, test lead-time weights)"],
 ["Validation rows", "300,000–400,000 sampled from the holdout"],
 ["ETL output size", "343 MB of Parquet from ~1.2 GB of netCDF"],
 ["Peak training memory", "≈5 GB (panel ≈4.4 GB, freed before fitting)"],
 ["Inference throughput", "280,961 rows scored per member in the training run"],
 ["Blend cost", "no training; two file passes"],
 ["Reproduction fidelity", "max |diff| 1.776e-15 rebuilding submission.csv"],
])

# ============================== 8 =========================================
H("8. Error Handling and Logging")
P("The pipeline is a batch research pipeline, so the design principle is to fail loudly and "
  "early rather than to recover silently. A wrong number that completes is far more "
  "expensive here than a crash.")
B(["Assertions on shape and alignment. Feature construction asserts the built matrix has "
   "exactly len(NAMES) columns. Every submission writer asserts 280,961 rows, all values "
   "finite, and IDs identical to SampleSubmission in its own order. Members are joined on "
   "ID and a missing ID raises rather than producing a NaN.",
   "Guarded external requests. data_prep/era5_fetch.py refuses any request whose dataset "
   "name or variable name matches a GRACE-derived product, and refuses --grid 1.0. Both "
   "raise SystemExit with an explanation rather than warning.",
   "Provenance logging. Every CDS download appends a JSON line to "
   "provenance/era5_provenance.jsonl recording the request, output file, byte count, "
   "SHA-256 and UTC timestamp.",
   "Run logging. Each notebook writes runinfo_<tag>.json (library versions, thread counts, "
   "CPU count, wall clock, refit depth) and emissions_<tag>.csv from CodeCarbon.",
   "Resident-memory tracing. Training logs RSS at each major allocation so that an "
   "out-of-memory failure can be attributed to a stage rather than guessed at.",
   "Known reporting defect, disclosed: runinfo records best_iter as null for members whose "
   "refit depth is fixed rather than early-stopped, and records the number of output "
   "variants rather than the number of recursive stages. Neither affects the model; both "
   "are logging bugs and are recorded as such rather than quietly corrected in the outputs."])
P("Verification as a first-class step. Two checks are shipped as runnable code rather than "
  "as claims. provenance/leak_test.py corrupts every input at every month after a cutoff and "
  "asserts the features do not move. For an end-to-end check, retrain the members from "
  "members/, compare each fresh output against the corresponding file in predictions/, and "
  "rebuild the blend from those RETRAINED outputs rather than from the shipped CSVs — so a "
  "member that fails to reproduce cannot be hidden by re-blending files that were already "
  "correct.")

# ============================== 9 =========================================
H("9. Maintenance and Monitoring")
P("What would need watching if this were run operationally each month.")
B(["Input drift and latency. The three external sources publish monthly with different "
   "latencies; ERA5-Land's final product lags by two to three months, so an operational "
   "system would run on ERA5T and re-run on the final product when it lands. The difference "
   "must be logged, not assumed to be zero.",
   "Instrument changes. The target is a satellite product with a mission change (GRACE to "
   "GRACE-FO) and a 12-month gap inside the test window. The change-from-anchor "
   "parameterisation is partly protective, because a constant offset cancels when anchor and "
   "target come from the same instrument, but a new mission would require revalidation.",
   "Anchor staleness. Prediction quality depends on how old the anchor is. Lead time is "
   "already a feature and RMSE degrades from 0.573 at one month to 0.638 at seven, so "
   "monitoring should be sliced by lead time rather than pooled.",
   "Blend weights. The weights depend on member scores. If members are retrained the weights "
   "should be re-solved, not carried forward — the solution is closed-form and costs seconds.",
   "Scaling. Cost is linear in cells × months and the per-cell statistics are vectorised "
   "across the whole grid, so a finer grid is a memory question rather than an algorithmic "
   "one. The panel is float32 and dominates the footprint; a 0.5° grid would be roughly 4× "
   "the memory and would need chunking by latitude band.",
   "Retraining cadence. Monthly retraining is affordable (about 11 hours of CPU) but "
   "probably unnecessary — the learning curve is flat, and seven-fold more training data "
   "changed validation RMSE by 0.00011. Quarterly retraining with monthly inference is the "
   "better trade."])

# ============================== 10 ========================================
H("10. Model behaviour, bias and interpretability")
H("10.1 Where the error is", 2)
P("Error is uneven and is reported sliced rather than pooled "
  "(provenance/bias_slices_v81.csv).")
TBL(["Slice", "Best", "Worst", "Note"], [
 ["Latitude band", "N midlatitudes 0.561", "S extratropics 0.671", "the worst band also carries a systematic mean error of −0.215 and is the thinnest slice: 10,807 validation rows against 107,424"],
 ["Calendar month", "≈0.494", "March 0.799, July 0.737", "these two months carry 35% of total loss; snowmelt and monsoon onset"],
 ["Event size", "0.347 smallest changes", "1.530 largest changes", "the model is weakest on the largest storage changes"],
 ["Lead time", "h=1 0.573", "h=7 0.638", "h=1 carries 38.7% of the weighted loss"],
])
P("The regional imbalance follows from how little land exists at southern extratropical "
  "latitudes and has not been corrected. A fairness toolkit such as AI Fairness 360 was not "
  "used: those target protected-attribute classification, and this is a gridded regression "
  "with no protected attributes. The meaningful axes here are geographic, seasonal and "
  "magnitude, and all are measured above.")
H("10.2 What drives the predictions", 2)
P("TreeSHAP — exact for tree ensembles rather than sampled — over 30,000 validation rows of "
  "the reference model, the architecture four of the six members share. Figure: "
  "provenance/shap_summary_v81.png; full ranking: provenance/shap_values_v81.csv.")
TBL(["Feature", "Share of mean |SHAP|", "What it is"], [
 ["tws_anchor", "7.9%", "the last observed storage value"],
 ["anchor_minus_roll12", "6.3%", "anchor relative to its own 12-month rolling mean"],
 ["nbr3_spei06_t", "4.7%", "6-month SPEI averaged over the ring-3 neighbourhood"],
 ["d_spei24", "4.1%", "change in 24-month SPEI between anchor and t"],
 ["ecm_dec_spei12", "3.9%", "error-correction decay term on 12-month SPEI"],
 ["nbr3_d_spei12", "3.7%", "neighbourhood change in 12-month SPEI"],
])
P("No single feature dominates: the strongest is 7.9% and the top twelve together are 41%. "
  "The result we did not expect is that SPEI CHANGES outrank SPEI LEVELS — d_spei24 and "
  "d_spei48 both beat the raw indices, and a neighbour-cell index ranks third overall. A "
  "plausible reading is that with 66.5% of test rows masked the model is bridging a window "
  "it cannot observe, and what carries a forecast across that window is how the drought "
  "state moved rather than where it currently sits. This describes the model's behaviour, "
  "not the hydrology.")
H("10.3 Reusability", 2)
P("Lead time is a feature rather than a separate model, so one fit covers h=1…7 and "
  "extending the horizon needs more rows, not new code. Location never enters as "
  "coordinates — only each cell's own climatology, variance and trend — so the pipeline "
  "transfers to any gridded monthly variable, such as soil moisture, groundwater or "
  "streamflow, by swapping the panel source. External covariates all pass through a single "
  "regrid step, so adding a dataset is one fetch script plus a column. The binding "
  "assumption is a regular monthly grid with a usable observation history per cell; "
  "ungauged basins or irregular in-situ networks would need a different anchor definition.")
H("10.4 Sustainability", 2)
P("CodeCarbon wrapped every training run (provenance/emissions/): 0.275 kg CO₂e across "
  "about 12.3 hours, CPU only. Efficiency came from float32 panels, "
  "cumulative-sum statistics computed once per cell rather than per row, max_bin=127, and "
  "training on 20% of available row-lead pairs — a measured optimum, not a shortcut. The "
  "blend trains nothing, so the final 0.0007 of score cost no compute at all.")
TBL(["Member", "Wall clock", "kg CO₂e"], [
 ["xgb_v60_recurse", "246.7 min", "0.0202"], ["xgb_v81_recency_hl96", "167.7 min", "0.0138"],
 ["xgb_v87_storage_hl96", "84.7 min", "0.0225"], ["realmlp_v72_e15", "59.4 min", "0.0101"],
 ["xgb_v32_era5 (local)", "117.8 min", "0.0520"],
 ["xgb_v85_level_hl96", "58.2 min", "0.1566"], ["total", "~12.3 h", "0.275"],
])
P("One anomaly is flagged rather than smoothed over: xgb_v85's 0.157 kg is a sevenfold "
  "outlier for a shorter run than several others, which we attribute to the grid carbon "
  "intensity of whichever region the hosted runtime used rather than to the model. "
  "xgb_v32_era5 is the one member trained locally, so its 0.052 kg is measured against "
  "this machine's grid rather than a data centre's and is not directly comparable.")

# ============================== 11 ========================================
H("11. Compliance and data governance")
B(["No GRACE-derived external data at any date. No TWSA product, mascon solution or "
   "GRACE-trained reconstruction (GTWS-MLrec, GRAiCE) is used. data_prep/era5_fetch.py "
   "refuses such requests by dataset name and by variable name at runtime; the CDS "
   "gravimetry product sits in the same category as ERA5 and is explicitly blocked.",
   "No feature reads month t+1 or later. Proven empirically: provenance/leak_test.py "
   "corrupts every input — both competition CSVs and all three Parquet files — at every "
   "month after a cutoff, first to NaN and then to a large constant, rebuilds the feature "
   "panel and asserts the features are bit-identical. BOTH architectures are covered: the "
   "gradient-boosted member passes 136/136 and the RealMLP member 132/132, under both "
   "corruptions. Covering both matters because the RealMLP member builds its panel with "
   "its own class P and train_rows — only its features() is byte-identical to the "
   "gradient-boosted reference — so a proof on one does not imply the other.",
   "Coordinates are a lookup key only, used to sample external grids at each cell, never a "
   "model feature. The leak test also asserts that no feature name is coordinate-derived.",
   "ERA5-Land is used as the final retrospective product, which the guidelines permit as the "
   "proxy for ERA5T in historical validation. This is stated rather than left implicit.",
   "A host clarification obtained from the competition forum on 2026-09-04, permitting "
   "non-GRACE land-surface-model variables and their deterministic combinations with source "
   "dates at or before t, is reproduced verbatim in "
   "provenance/host_ruling_2026-09-04_gldas.md because it does not appear in the written "
   "guidelines."])

# ============================== 12 ========================================
H("12. Code review checklist")
TBL(["Requirement", "Status"], [
 ["Random seeds set for reproducible output", "Yes — seed = 42 + 1000·i per seed; all five members use it"],
 ["Code reproduces the submitted leaderboard score", "The blend arithmetic rebuilds submission.csv from the member predictions to max |diff| 1.776e-15. Full end-to-end reproduction — retraining every member from source and blending the fresh outputs — takes about 12.3 h of training and therefore spans more than one Kaggle session"],
 ["Libraries imported at the top of each script", "Yes"],
 ["Only libraries actually used are included", "Yes"],
 ["requirements.txt with specific versions", "Yes — requirements.txt, installable as one file, covering every third-party import: xgboost 3.2.0, pytabkit 1.5.0, torch 2.10.0, numpy, pandas, pyarrow, codecarbon, cdsapi, xarray, netCDF4, matplotlib, python-docx. Versions are taken from the recorded run metadata in provenance/emissions/runinfo_*.json, not from a freeze of this machine"],
 ["Coding environment documented", "Yes — training on Kaggle CPU (4 cores, NTHREAD=16, no GPU); blend and verification on Python 3.12.3, Ubuntu, CPU"],
 ["All datasets and their sources documented", "Yes — §3.1, with DOI and per-download checksums in provenance/era5_provenance.jsonl"],
 ["All processing performed in code", "Yes — no manual or spreadsheet steps anywhere in the chain"],
 ["Only freely available packages", "Yes — all open source; Copernicus CDS requires free registration"],
 ["Reusable methods rather than duplicated blocks", "Yes — feature construction is shared verbatim across members; the blend solver is a single function"],
 ["Meaningful variable and function names", "Yes"],
 ["Comments on complex functions", "Yes — including the unit and grid traps in §3.2, which are the least obvious parts"],
])

# ============================== appendix ==================================
H("Appendix — repository contents")
CODE("submission.csv          the scored submission (public LB 0.662803625)\n"
     "README.md               concise version of this document\n"
     "requirements.txt        every dependency, pinned, installable as one file\n"
     "blend/                  anchor, blend, and the offline reproduction check\n"
     "members/                the artefact that produced each member: five Kaggle notebooks\n"
     "                        plus sub_xgb_v32_era5.py, the one member trained locally\n"
     "predictions/            the six member CSVs and the persistence anchor\n"
     "data_prep/              fetch and regrid for ERA5-Land, ESA CCI, CDS SPEI\n"
     "provenance/             leak test, SHAP, bias slices, checksums, host ruling, carbon")
P("Three commands are runnable as shipped:", size=9.5)
CODE("python blend/make_persistence.py  --data <competition CSVs>\n"
     "python blend/make_final_blends.py --data <competition CSVs>\n"
     "python provenance/leak_test.py --data-root <project root>      # 132/132")

out = HERE / "DOCUMENTATION.docx"
d.save(out)
print(f"wrote {out}")
