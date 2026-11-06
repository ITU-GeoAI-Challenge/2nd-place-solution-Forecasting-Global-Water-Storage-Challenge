import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11.5, 8.8)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
C = {"src": "#dbe7f3", "etl": "#d6ecd9", "mdl": "#fbe6cf", "out": "#f5d7d7", "chk": "#ece5f5"}
def box(x, y, w, h, t, c, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45",
                 fc=C[c], ec="#5b6b7a", lw=0.9))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.5)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=10, lw=0.95, color="#5b6b7a"))

ax.text(50, 97, "Forecasting Global Water Storage — data flow", ha="center", fontsize=12.5, weight="bold")
ax.text(50, 93.8, "submission.csv  ·  public LB RMSE 0.662803625", ha="center",
        fontsize=8.8, style="italic", color="#444")

box(2,  83, 21, 8, "Competition files\nTrain / Test /\nSampleSubmission", "src", 7.8)
box(26, 83, 22, 8, "ERA5-Land monthly\nCopernicus CDS\n(reanalysis)", "src", 7.8)
box(51, 83, 22, 8, "ESA CCI soil moisture\nCopernicus CDS\n(satellite retrieval)", "src", 7.8)
box(76, 83, 22, 8, "CDS drought indices\nSPEI 12 / 24 / 48\n(Copernicus CDS)", "src", 7.8)

box(26, 71, 72, 7, "data_prep/  —  land-area regrid onto the competition's 1° grid, 15,715 cells\n"
                   "era5_grid.parquet  ·  cci_grid.parquet  ·  drought_grid.parquet", "etl", 7.9)
for x in (37, 62, 87): arr(x, 83, x, 78.4)
arr(12.5, 83, 12.5, 63.4)

box(14, 56, 72, 7, "(cell × month) panel  —  TWS from three consistent sources, as-of statistics,\n"
                   "per-cell climatology / variance / trend.  Every index at or before month t.", "etl", 7.9)
arr(62, 71, 62, 63.4)

ax.text(50, 51.5, "six members, trained independently  —  notebooks/ , CPU only, versions pinned at runtime",
        ha="center", fontsize=7.6, style="italic", color="#444")
lab = [("v81\nrecency", 3), ("v85\nlevel", 18), ("v87\nstorage", 33),
       ("v32\nERA5", 48), ("v60\nrecursive", 63), ("v72\nRealMLP", 78)]
for t, x in lab:
    box(x, 39, 14, 8, t, "mdl", 7.9)
    arr(x+7, 56, x+7, 47.4)

box(14, 25, 72, 8, "blend/make_final_blends.py  —  minimum-variance weights recovered from the member\n"
                   "predictions and their public scores alone:\n"
                   "cov(eᵢ,eⱼ) = ( sᵢ² + sⱼ² − mean((pᵢ−pⱼ)²) ) / 2", "out", 7.9)
for t, x in lab: arr(x+7, 39, 50, 33.4)

box(3, 11, 30, 8, "blend/make_persistence.py\npersistence anchor\n(last observed TWS ≤ t)", "out", 7.8)
box(39, 11, 47, 8, "shrink toward the anchor,  γ = 0.9670, solved in closed form\n→  submission.csv", "out", 8.0, bold=True)
arr(50, 25, 62, 19.4); arr(33, 15, 39, 15)

ax.add_patch(FancyBboxPatch((3, 1.5), 95, 5.5, boxstyle="round,pad=0.4",
             fc=C["chk"], ec="#5b6b7a", lw=0.9))
ax.text(50, 4.2, "verification:   blend/verify_reproduction.py rebuilds submission.csv to max|diff| 1.8e-15\n"
                 "provenance/leak_test.py — 136/136 features unchanged under corrupted future inputs",
        ha="center", va="center", fontsize=7.4, linespacing=1.6)
fig.savefig("provenance/architecture.png", dpi=155, bbox_inches="tight")
print("ok")
