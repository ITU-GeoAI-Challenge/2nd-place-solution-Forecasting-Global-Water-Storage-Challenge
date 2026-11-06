"""EMPIRICAL date-discipline proof for any sub_xgb_*.py.

Guidelines section 3: external covariates are permitted "provided every source date is at
or before t", and "any feature encoding month t+1 or later, FROM ANY SOURCE, remains
grounds for disqualification".

Reasoning about index arithmetic is not proof. This corrupts EVERY input -- Train.csv,
Test.csv and all three parquets -- at every month strictly after a cutoff M, rebuilds the
panel from the corrupted tree, and asserts the features for rows with t <= M are
BIT-IDENTICAL. If any feature reads the future, it moves, and the assert fires.

    python provenance/leak_test.py --data-root=/path/to/project

--data-root (or $GWS_ROOT) must contain data/raw/*.csv and the three parquets under
data/. Defaults to models/sub_xgb_v87_storage.py as the module under test.

Usage note: corrupts to NaN *and* to a large constant, because a NaN can be silently
swallowed by a nan-aware reduction while a finite wrong value would still leak.
"""
import os, sys, shutil, time, tempfile, importlib.util, numpy as np, pandas as pd
from pathlib import Path

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_root = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--data-root=")),
             os.environ.get("GWS_ROOT"))
if _root:
    os.environ["GWS_ROOT"] = _root          # must be set BEFORE the model module imports
SRC = (_args[0] if _args else
       str(Path(__file__).resolve().parent.parent / "members" / "sub_xgb_v32_era5.py"))
CUT = int(_args[1]) if len(_args) > 1 else 150                # month index cutoff
SCRATCH = Path(tempfile.mkdtemp(prefix="leak_test_"))   # portable; removed on exit
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# The member scripts are shipped EXACTLY as they were run, which means they resolve their
# own ROOT by walking up for a data/raw directory — and that happens at import time, before
# anything here can intervene. In this package no such ancestor exists, so the module is
# loaded from a scaffold instead: a temp directory holding a `data` symlink to the real data
# root and a copy of the module. ROOT then resolves to the scaffold, and the member file is
# never modified.
HOME = Path(tempfile.mkdtemp(prefix="leak_home_"))
def _scaffold(src: Path) -> Path:
    if not _root:
        return src                                   # a repo layout is already in place
    data = Path(_root) / "data"
    if not (data / "raw").is_dir():
        data = Path(_root)                           # --data-root pointed straight at data/
    (HOME / "members").mkdir(parents=True, exist_ok=True)
    link = HOME / "data"
    if not link.exists():
        link.symlink_to(Path(data).resolve())
    dst = HOME / "members" / src.name
    shutil.copy(src, dst)
    return dst

SRC_PATH = _scaffold(Path(SRC).resolve())

def load(tag):
    sp = importlib.util.spec_from_file_location(tag, SRC_PATH)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
M0 = load("probe")
REAL = M0.ROOT
# Two data layouts exist: the repo one (ROOT/data/raw/...) and the Kaggle flat one
# (ROOT/raw/... with the parquets beside it), which the notebook-extracted modules use.
FLAT = (REAL / "raw").is_dir()
if FLAT: REAL = REAL.parent
def _as_root(base):
    return Path(base) / "data" if FLAT else Path(base)
def _raw(m):
    return m.ROOT / "raw" if FLAT else m.ROOT / "data" / "raw"
mon = lambda s: (s.dt.year.to_numpy()*12 + s.dt.month.to_numpy() - 1 - M0.MONTH0)

def _corruptible(d, skip):
    """numeric columns only -- identifiers and dates are structure, not values, and
    corrupting them would change WHICH rows exist rather than what they contain."""
    return [c for c in d.columns
            if c not in skip and pd.api.types.is_numeric_dtype(d[c])]


def build_corrupt(mode):
    """copy data/ and destroy every value at month > CUT"""
    if SCRATCH.exists(): shutil.rmtree(SCRATCH)
    (SCRATCH/"data"/"raw").mkdir(parents=True)
    bad = np.nan if mode == "nan" else 999.0
    for name in ("Train.csv", "Test.csv"):
        d = pd.read_csv(REAL/"data/raw"/name)
        k = mon(pd.to_datetime(d.time)) > CUT
        for c in _corruptible(d, ("time","lat","lon","ID")):
            d[c] = d[c].astype("float64")     # some columns are bool/int; widen first
            d.loc[k, c] = bad
        d.to_csv(SCRATCH/"data/raw"/name, index=False)
    for pq in ("era5_grid.parquet","cci_grid.parquet","drought_grid.parquet"):
        f = REAL/"data"/pq
        if not f.is_file(): continue
        d = pd.read_parquet(f)
        k = mon(pd.to_datetime(d.time)) > CUT
        for c in _corruptible(d, ("time","lat","lon")):
            d[c] = d[c].astype("float64")
            d.loc[k, c] = bad
        d.to_parquet(SCRATCH/"data"/pq, index=False)
    shutil.copy(REAL/"data/raw/SampleSubmission.csv", SCRATCH/"data/raw/") if (REAL/"data/raw/SampleSubmission.csv").is_file() else None
    return SCRATCH

def features_for(root, rows=None):
    m = load("run"); m.ROOT = _as_root(root)
    tr = pd.read_csv(_raw(m)/"Train.csv",
                     usecols=["time","lat","lon","TWS_t",*m.COVARIATES,"target"])
    te = pd.read_csv(_raw(m)/"Test.csv",
                     usecols=["ID","time","lat","lon","TWS_t",*m.COVARIATES])
    y = tr["target"].to_numpy()
    p = m.P(tr, te); del tr, te
    if rows is None:
        rng = np.random.default_rng(m.SEED)
        months = np.unique(p.m_tr)
        c,t,a,_,_ = m.train_rows(p, y, 0, months[-1], m.FRAC, rng)
        keep = np.flatnonzero(t <= CUT)
        keep = np.random.default_rng(3).choice(keep, min(120_000, len(keep)), replace=False)
        rows = (c[keep], t[keep], a[keep])
    X = m.features_chunked(p, *rows)
    return X, rows, m.NAMES

# ---- guideline checks that are not about dates --------------------------------------
# test_rules.py has these, but it runs against the OLD features.py/panels.py 66-feature
# pipeline, which has not produced a submission in weeks. Carried over here so the
# evidence covers the code that actually ships.
BANNED = ("lat", "lon", "cell_id", "coord", "grid", "idx", "embed")
src_text = Path(SRC).read_text()
bad_names = [n for n in M0.NAMES if any(b in n.lower() for b in BANNED)]
assert not bad_names, f"coordinate-derived feature names: {bad_names}"
log(f"section 3 'lat/lon as a lookup key only': {len(M0.NAMES)} features, none coordinate-derived")
urls = [l.strip() for l in src_text.splitlines()
        if ("http://" in l or "https://" in l or "ftp://" in l) and not l.strip().startswith("#")]
assert not urls, f"live URL in shipping code (reproducibility): {urls[:3]}"
log("no live URL in shipping code -- inputs come from the frozen data tree")
banned_src = [w for w in ("grace","twsa","mascon","gtws","graice","total_water_storage")
              if w in src_text.lower().replace("grace/grace-fo","")]
log(f"banned-source tokens in code: {banned_src or 'none'}")

log(f"module {SRC}  cutoff month index {CUT}")
X0, rows, NAMES = features_for(REAL)
log(f"clean features {X0.shape}; all rows have t <= {CUT}")
ok = True
for mode in ("nan", "const"):
    root = build_corrupt(mode)
    log(f"rebuilt from data corrupted to {mode.upper()} at every month > {CUT}")
    X1, _, _ = features_for(root, rows)
    same = np.array([np.array_equal(np.nan_to_num(X0[:,j], nan=-9e9),
                                    np.nan_to_num(X1[:,j], nan=-9e9))
                     for j in range(X0.shape[1])])
    bad = [NAMES[j] for j in np.flatnonzero(~same)]
    log(f"  identical features: {same.sum()}/{len(same)}")
    if bad:
        ok = False
        log(f"  ❌ LEAKED ({len(bad)}): {bad[:20]}")
    else:
        log(f"  ✅ no feature moved")
shutil.rmtree(SCRATCH, ignore_errors=True); shutil.rmtree(HOME, ignore_errors=True)
log("=" * 66)
log("✅ PASS - no feature reads any month after t" if ok else "❌ FAIL - date discipline violated")
sys.exit(0 if ok else 1)
