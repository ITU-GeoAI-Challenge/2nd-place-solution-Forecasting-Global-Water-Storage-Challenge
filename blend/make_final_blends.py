"""Build the submission: minimum-variance blend of six members, shrunk toward persistence.

    python blend/make_final_blends.py [--data DIR] [--out FILE]

--data must hold SampleSubmission.csv (default: ./data/raw, or $GWS_DATA). Members and the
anchor are read from predictions/. Writes submission.csv at the package root.

Weights and the shrinkage gamma are solved in closed form from the member CSVs plus their
published scores -- nothing is tuned by trial and error on the leaderboard. Method in
README §1. `blend/verify_reproduction.py` checks the output against the shipped file.

⚠️ S_PERSISTENCE is the persistence submission's own LB score, 0.886420231. A wrong value
here does not raise -- it silently shifts gamma. The verifier is what catches it.
"""
import argparse, os
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
S = {"xgb_v81_recency_hl96": 0.670554680, "xgb_v60_recurse":      0.685562412,
     "xgb_v32_era5":         0.677796328, "realmlp_v72_e15":      0.695636003,
     "xgb_v85_level_hl96":   0.672036642, "xgb_v87_storage_hl96": 0.672415764}
S_PERSISTENCE = 0.886420231

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("GWS_DATA", "data/raw"))
ap.add_argument("--out", default=str(HERE / "submission.csv"))
a = ap.parse_args()

ids = pd.read_csv(Path(a.data) / "SampleSubmission.csv")["ID"]
def col(name):
    v = pd.read_csv(HERE / "predictions" / f"{name}.csv").set_index("ID")["Target"].reindex(ids)
    if v.isna().any():
        raise SystemExit(f"{name}: {int(v.isna().sum())} IDs missing vs SampleSubmission")
    return v.to_numpy(np.float64)

names = list(S)
P = {k: col(k) for k in names}
ANCH = col("persistence")

C = np.array([[S[x]**2 if x == y else (S[x]**2 + S[y]**2 - np.mean((P[x]-P[y])**2)) / 2
               for y in names] for x in names])
sub = list(range(len(names)))
for _ in range(len(names)):                       # non-negativity by active-set clipping
    w = np.linalg.solve(C[np.ix_(sub, sub)], np.ones(len(sub))); w /= w.sum()
    if (w >= -1e-9).all(): break
    sub = [sub[i] for i in range(len(sub)) if w[i] > 1e-9]
W = np.zeros(len(names)); W[sub] = w

blend = sum(wi * P[k] for wi, k in zip(W, names))
f0 = np.sqrt(max(W @ C @ W, 0))
A = S_PERSISTENCE ** 2
Cv = np.mean((blend - ANCH) ** 2)
B = (f0 ** 2 - A - Cv) / 2
gamma = -B / Cv
pred = ANCH + gamma * (blend - ANCH)

for k, wi in zip(names, W): print(f"  {wi:.10f}  {k}")
print(f"  gamma {gamma:.10f};  blend variance forecast {f0:.6f}")
out = pd.DataFrame({"ID": ids, "Target": pred})
assert len(out) == 280961 and out["Target"].notna().all()
out.to_csv(a.out, index=False)
print(f"wrote {a.out}  ({len(out):,} rows)")
