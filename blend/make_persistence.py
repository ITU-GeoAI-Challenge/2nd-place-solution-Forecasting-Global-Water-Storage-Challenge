"""Build the shrinkage anchor: each test row's last observed TWS at or before month t.

    python blend/make_persistence.py [--data DIR] [--out FILE]

--data must hold Train.csv and Test.csv (default: ./data/raw, or $GWS_DATA).
Writes predictions/persistence.csv. Also the naive baseline, public LB 0.886420231.

Self-contained by design: it rebuilds the (cell x month) TWS panel from the competition
files alone, so reproducing the submission needs no model source. The panel takes TWS from
its three consistent places -- train `TWS_t` at month t, train `target` at t+1 (the target
IS next month's TWS), and test `TWS_t` where it is not masked -- then walks forward taking
the most recent observed month. Every value is therefore at or before t.
"""
import argparse, os
import numpy as np, pandas as pd
from pathlib import Path

MONTH0 = 2002 * 12 + 4
N_MONTHS = (2019 * 12 + 0) - MONTH0 + 1          # 201

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("GWS_DATA", "data/raw"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "predictions" / "persistence.csv"))
a = ap.parse_args()
D = Path(a.data)

tr = pd.read_csv(D / "Train.csv", usecols=["time", "lat", "lon", "TWS_t", "target"])
te = pd.read_csv(D / "Test.csv",  usecols=["ID", "time", "lat", "lon", "TWS_t"])
ss = pd.read_csv(D / "SampleSubmission.csv")["ID"]

gid = lambda la, lo: (np.rint(la + 55.5).astype(np.int64) * 360
                      + np.rint(lo + 179.5).astype(np.int64))
mid = lambda s: (pd.to_datetime(s).dt.year.to_numpy() * 12
                 + pd.to_datetime(s).dt.month.to_numpy() - 1 - MONTH0).astype(np.int64)
g_tr, g_te = gid(tr.lat.to_numpy(), tr.lon.to_numpy()), gid(te.lat.to_numpy(), te.lon.to_numpy())
codes, uniq = pd.factorize(np.concatenate([g_tr, g_te]), sort=True)
c_tr, c_te = codes[:len(g_tr)], codes[len(g_tr):]
m_tr, m_te = mid(tr.time), mid(te.time)

t = np.full((len(uniq), N_MONTHS), np.nan, np.float32)
t[c_tr, m_tr] = tr["TWS_t"].to_numpy()
nx = m_tr + 1; ok = nx < N_MONTHS
t[c_tr[ok], nx[ok]] = tr["target"].to_numpy()[ok]      # target is next month's TWS
v = te["TWS_t"].to_numpy(); s = ~np.isnan(v)
t[c_te[s], m_te[s]] = v[s]

obs = ~np.isnan(t)
idx = np.where(obs, np.arange(N_MONTHS, dtype=np.int32)[None, :], np.int32(-1))
last = np.maximum.accumulate(idx, axis=1)              # most recent observed month <= m
anchor_month = last[c_te, m_te]
assert (anchor_month >= 0).all(), "a test cell has no prior observation"
vals = t[c_te, anchor_month].astype(np.float64)
assert np.isfinite(vals).all()

out = pd.Series(vals, index=te["ID"].to_numpy()).reindex(ss)
assert out.notna().all(), "test IDs do not cover SampleSubmission"
df = pd.DataFrame({"ID": ss, "Target": out.to_numpy()})

dst = Path(a.out)
if dst.exists():
    old = pd.read_csv(dst)
    same = (old["ID"].tolist() == df["ID"].tolist()
            and np.allclose(old["Target"].to_numpy(), df["Target"].to_numpy(), atol=1e-9))
    print(f"reproduces existing {dst.name} exactly: {same}")
    if not same:
        raise SystemExit(f"MISMATCH max|diff| {np.abs(old.Target - df.Target).max():.3e}")
else:
    dst.parent.mkdir(parents=True, exist_ok=True); df.to_csv(dst, index=False)
    print(f"wrote {dst}")
print(f"rows {len(df):,}  mean {df.Target.mean():+.4f}  sd {df.Target.std():.4f}")
