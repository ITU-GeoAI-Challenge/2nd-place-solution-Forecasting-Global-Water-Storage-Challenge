"""XGBoost head-to-head against the frozen LightGBM v5+seeds5.

Same 73 v5 features, same delta target, same test-mix horizon weights, same
train/validation split and the same 5-seed refit as sub_v5_seeds5.py. The ONLY
variable changed is the learner. This is a like-for-like head-to-head against
LightGBM, not a blend.

Baseline to beat: sub_v5_seeds5.py -- validation 0.6175 / best_iter 217, LB 0.693870116.

Capacity is matched to the LightGBM production config deliberately:
    LightGBM  num_leaves=255, min_data_in_leaf=200, lr .05, ff/bf 0.8, l2 1.0, max_bin 127
    XGBoost   grow_policy=lossguide + max_leaves=255 (leaf-wise, like LightGBM),
              colsample/subsample 0.8, lambda 1.0, max_bin 127.
              min_child_weight=30, NOT 200: it is a sum of HESSIANS, and the rows
              carry test-mix weights averaging ~1/7, so 200 rows ~ 29 hessian.

--- v32 -------------------------------------------------------------------
v30 + 11 raw ERA5-Land features. 132 features. LB **0.677796328** (+0.001668 vs
v30) -- it LOST as a single model (NOTES 3.56).

⚠️ KEPT DESPITE LOSING, because it is a BLEND INPUT. `0.55*v30 + 0.45*v32` scores
**0.673599117**, the best submission of the competition (3.64). v32 is the only
model genuinely decorrelated from v30 (rms diff 0.1359 against 0.055-0.093 for every
other XGB variant) and carries ~0.42 weight in every optimal blend. ERA5 failed at
making the model BETTER and succeeded at making it DIFFERENT.

⚠️ RECONSTRUCTED 2026-08-30. The original was deleted after it lost, before its
value as an ensemble member was known. This rebuild is faithful to the documented
spec but has NOT been re-run end-to-end against `subs/xgb_v32_era5.csv`. The CSV is
the authoritative artifact for the blend; verify this script before relying on it to
regenerate one.

Features: e5_sd_t, e5_d_sd, e5_sd_anom_tgt, e5_smlt_win, e5_t2m_t, e5_d_t2m,
e5_tp_win, e5_e_win, e5_sro_win, e5_swvl1_t, e5_swvl_grad -- read at t/anchor only,
never at the target; date discipline asserted at runtime (3.55).
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Paths resolve against the project root, so this runs from any directory.
_here = Path(__file__).resolve()
ROOT = next(d for d in (_here.parent, *_here.parents) if (d / "data" / "raw").is_dir())

MONTH0 = 2002 * 12 + 4
N_MONTHS = (2019 * 12 + 0) - MONTH0 + 1          # 201
COVARIATES = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
SHORT = ["spei01", "spei03", "spei06", "spei12", "sm"]
TWS_LAGS = (1, 3, 6, 12)
ROLL = 12
MAX_H = 7
RESP_COV = [1, 3, 4]      # SPEI_03, SPEI_12, SOIL_MOISTURE
CURVE_MIN_N = 24           # months of own TWS needed before a quadratic is trusted
CURVE_U0, CURVE_US = 100.0, 100.0   # time is rescaled to u=(m-100)/100 in [-1,1]:
                           # a raw-month Vandermonde (m^4 up to 1.6e9) is hopeless.
ECM_COV = [3, 4]           # SPEI_12, SOIL_MOISTURE -- the two STOCK-like state
                           # variables. SPEI_03 is a flow and v19 already covers it.
NBR_COV = [2, 3, 4]        # SPEI_06, SPEI_12, SOIL_MOISTURE at rings 1/2/3
R1_EXTRA = [0, 1]          # SPEI_01, SPEI_03 at ring 1 only
E5_PATH = "data/era5_grid.parquet"      # regridded ERA5-Land, 15,715 x 204
E5_LEVEL = ["sd", "t2m", "swvl1", "swvl4"]   # read at t / anchor
E5_WIN = ["smlt", "tp", "e", "sro"]          # accumulated over (anchor, t]
VAL_MONTHS, FRAC, VAL_FRAC, SEED, ROUNDS = 24, 0.20, 0.50, 42, 3000
PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.05,
              grow_policy="lossguide", max_leaves=255, max_depth=0,
              min_child_weight=30.0, colsample_bytree=0.8, subsample=0.8,
              reg_lambda=1.0, tree_method="hist", max_bin=127, nthread=16,
              seed=SEED)
N_SEEDS = 5

NAMES = (
    ["tws_anchor", "h", "clim_anchor", "clim_target", "anom_anchor", "clim_delta",
     "cell_mean", "cell_std", "z_anchor", "cell_trend", "trend_proj",
     "tws_roll_mean_12", "anchor_minus_roll12"]
    + [f"d_tws_{k}" for k in TWS_LAGS] + [f"gap_tws_{k}" for k in TWS_LAGS]
    + [f"{c}_t" for c in SHORT] + [f"{c}_anchor" for c in SHORT]
    + [f"d_{c}" for c in SHORT] + [f"{c}_win_mean" for c in SHORT]
    + ["month_sin_target", "month_cos_target"]
    + ["nbr_tws_anchor", "nbr_minus_own", "nbr_d_tws_1", "nbr_d_tws_3"]
    + [f"nbr_{SHORT[i]}_t" for i in NBR_COV] + [f"nbr_d_{SHORT[i]}" for i in NBR_COV]
    + [f"nbr_{SHORT[i]}_t" for i in R1_EXTRA] + [f"nbr_d_{SHORT[i]}" for i in R1_EXTRA]
    + [f"nbr2_{SHORT[i]}_t" for i in NBR_COV] + [f"nbr2_d_{SHORT[i]}" for i in NBR_COV]
    + ["grad_spei06", "grad_sm"]
    + [f"nbr3_{SHORT[i]}_t" for i in NBR_COV] + [f"nbr3_d_{SHORT[i]}" for i in NBR_COV]
    + ["grad2_spei06", "grad2_sm"]
    # --- v16: volatility of the cell's OWN past TWS. Guidelines section 3 permits
    #     "per-cell statistics computed from that cell's own past TWS: climatology,
    #     variance, trend" explicitly. Motivation (NOTES 3.29): the model predicts
    #     moves of rms 0.49 in BOTH calm and volatile regimes while actual moves are
    #     0.71 and 1.03 -- it cannot tell the regimes apart, so it cannot shrink
    #     harder where uncertainty is genuinely higher.
    + ["dvol_all", "dvol_12", "dvol_ratio", "dvol_h", "clim_std_target",
       "z_clim_anchor", "d1_over_vol"]
    + [f"{k}_{n}" for n in ("spei03", "spei12", "sm") for k in ("rho", "beta", "imp")]
    + ["rho_ar", "beta_ar", "imp_ar", "rho_mr", "beta_mr", "imp_mr", "mr_decay_h"]
    + [f"ecm_{k}_{n}" for n in ("spei12", "sm")
       for k in ("k", "g", "gap", "dec")]
    + ["curve_c", "slope_now", "slope_now_proj", "curve_move", "curve_vs_lin"]
    + ["dvol_tm", "dvol_tm_ratio", "seas_amp", "seas_pos_t", "seas_rising_t",
       "anom_vs_tm", "anom_vs_tm_over_dvol"]
    + ["amr_x_n3spei06", "z_x_n3spei06", "tws_x_n3spei06", "anomtm_x_n3spei06",
       "amr_x_z"]
    # --- v36 ERA5 joint per-cell water balance ---
    + ["e5_sd_t", "e5_d_sd", "e5_sd_anom_tgt", "e5_smlt_win", "e5_t2m_t", "e5_d_t2m",
       "e5_tp_win", "e5_e_win", "e5_sro_win", "e5_swvl1_t", "e5_swvl_grad"]
)


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class P:
    """(cell x month) panels and every as-of-month statistic built on them."""

    def __init__(self, tr, te):
        gid = lambda la, lo: (np.rint(la + 55.5).astype(np.int64) * 360
                              + np.rint(lo + 179.5).astype(np.int64))
        g_tr = gid(tr["lat"].to_numpy(), tr["lon"].to_numpy())
        g_te = gid(te["lat"].to_numpy(), te["lon"].to_numpy())
        codes, uniq = pd.factorize(np.concatenate([g_tr, g_te]), sort=True)
        self.n = len(uniq); self.gids = np.asarray(uniq, dtype=np.int64)
        self.c_tr = codes[:len(g_tr)].astype(np.int32)
        self.c_te = codes[len(g_tr):].astype(np.int32)
        mid = lambda s: (s.astype("string").str.slice(0, 4).astype("int32").to_numpy() * 12
                         + s.astype("string").str.slice(5, 7).astype("int32").to_numpy()
                         - 1 - MONTH0).astype(np.int32)
        self.m_tr, self.m_te = mid(tr["time"]), mid(te["time"])

        # --- TWS from its three consistent sources -------------------------
        t = np.full((self.n, N_MONTHS), np.nan, np.float32)
        t[self.c_tr, self.m_tr] = tr["TWS_t"].to_numpy()
        nx = self.m_tr + 1; ok = nx < N_MONTHS
        t[self.c_tr[ok], nx[ok]] = tr["target"].to_numpy()[ok]
        v = te["TWS_t"].to_numpy(); s = ~np.isnan(v)
        t[self.c_te[s], self.m_te[s]] = v[s]
        self.tws = t

        self.cov = []
        for c in COVARIATES:
            m = np.full((self.n, N_MONTHS), np.nan, np.float32)
            m[self.c_tr, self.m_tr] = tr[c].to_numpy()
            m[self.c_te, self.m_te] = te[c].to_numpy()
            self.cov.append(m)

        obs = ~np.isnan(t)
        idx = np.where(obs, np.arange(N_MONTHS, dtype=np.int32)[None, :], np.int32(-1))
        self.last = np.maximum.accumulate(idx, axis=1).astype(np.int32)

        y = np.where(obs, t, 0.0).astype(np.float64)
        x = np.arange(N_MONTHS, dtype=np.float64)[None, :] * obs
        n = np.cumsum(obs, axis=1, dtype=np.float64)
        sy, syy = np.cumsum(y, 1), np.cumsum(y * y, 1)
        sx, sxx, sxy = np.cumsum(x, 1), np.cumsum(x * x, 1), np.cumsum(x * y, 1)
        self.csum, self.ccnt = sy.astype(np.float32), n.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sy / n; var = syy / n - mean * mean
            den = n * sxx - sx * sx
            slope = np.where(np.abs(den) > 1e-9, (n * sxy - sx * sy) / den, np.nan)
        bad = n < 2
        self.cmean = np.where(n < 1, np.nan, mean).astype(np.float32)
        self.cstd = np.where(bad, np.nan, np.sqrt(np.maximum(var, 0))).astype(np.float32)
        self.ctrend = np.where(bad, np.nan, slope).astype(np.float32)

        # month-to-month change volatility, from this cell's own past TWS only
        prev = np.full_like(t, np.nan); prev[:, 1:] = t[:, :-1]
        dif = t - prev                     # NaN unless both months observed
        od = ~np.isnan(dif)
        sdd = np.cumsum(np.where(od, dif * dif, 0.0), 1, dtype=np.float64)
        ndd = np.cumsum(od, 1, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.dvol = np.where(ndd >= 3, np.sqrt(sdd / ndd), np.nan).astype(np.float32)
        self.dsq, self.dcnt = sdd.astype(np.float32), ndd.astype(np.float32)

        # v19: per-cell RESPONSE coefficients. How strongly does this cell's storage
        # actually move when its soil moisture / drought index moves? Accumulated as-of,
        # over months where BOTH first differences are observed. `dif` is dTWS above.
        self.resp = {}
        for ci in RESP_COV:
            cprev = np.full_like(self.cov[ci], np.nan); cprev[:, 1:] = self.cov[ci][:, :-1]
            dc = self.cov[ci] - cprev
            ok = od & ~np.isnan(dc)
            cs_ = lambda a: np.cumsum(np.where(ok, a, 0.0), 1, dtype=np.float64).astype(np.float32)
            self.resp[ci] = (np.cumsum(ok, 1, dtype=np.float64).astype(np.float32),
                             cs_(dc), cs_(dif), cs_(dc * dc), cs_(dif * dif), cs_(dc * dif))

        # v20: per-cell DYNAMICS coefficients. `dif`/`od`/`prev` come from the v16 block.
        #   ar -- regress dTWS_t on dTWS_{t-1}: does this cell's change carry momentum
        #         or reverse?  mr -- regress dTWS_t on TWS_{t-1}: how hard does this cell
        #         pull back toward its own mean, and how fast?
        dprev = np.full_like(dif, np.nan); dprev[:, 1:] = dif[:, :-1]

        def _xmom(x, y):
            k = ~np.isnan(x) & ~np.isnan(y)
            c = lambda a: np.cumsum(np.where(k, a, 0.0), 1, dtype=np.float64).astype(np.float32)
            return (np.cumsum(k, 1, dtype=np.float64).astype(np.float32),
                    c(x), c(y), c(x * x), c(y * y), c(x * y))

        self.ar = _xmom(dprev, dif)
        self.mr = _xmom(prev, dif)

        # v22: per-cell ERROR CORRECTION. Regress dTWS_t on BOTH TWS_{t-1} and
        # Driver_{t-1}. v20's `mr` is the gamma=0 special case of this -- reversion
        # to a CONSTANT cell mean. Here the equilibrium is driver-conditional,
        # TWS* = -(gamma/k) * Driver, so a cell sitting in a wet anomaly is not
        # expected to fall back to its long-run mean at all. Differencing the LHS
        # keeps this out of spurious-regression territory: a levels-on-levels fit
        # would mostly recover co-trending.
        self.ecm = {}
        for ci in ECM_COV:
            dv_ = np.full_like(self.cov[ci], np.nan); dv_[:, 1:] = self.cov[ci][:, :-1]
            msk = ~np.isnan(prev) & ~np.isnan(dv_) & ~np.isnan(dif)
            cm_ = lambda a: np.cumsum(np.where(msk, a, 0.0), 1,
                                      dtype=np.float64).astype(np.float32)
            self.ecm[ci] = (np.cumsum(msk, 1, dtype=np.float64).astype(np.float32),
                            cm_(prev), cm_(dv_), cm_(dif),
                            cm_(prev * prev), cm_(prev * dv_), cm_(dv_ * dv_),
                            cm_(prev * dif), cm_(dv_ * dif))

        # v25: per-cell TREND CURVATURE. Every trend feature so far is LINEAR --
        # `cell_trend` is one slope over the cell's whole history and `trend_proj`
        # extrapolates it straight. But the test period is 18 months of pure forward
        # extrapolation (3.25) and real storage trends bend: aquifer depletion
        # accelerates, El Nino breaks the record. For a cell whose decline is
        # steepening, the LIFETIME-AVERAGE slope is the wrong number to extrapolate.
        # Fit TWS = a + b*u + c*u^2 as-of, over months where TWS is observed.
        u = ((np.arange(N_MONTHS, dtype=np.float64) - CURVE_U0) / CURVE_US)[None, :]
        uu = np.broadcast_to(u, t.shape)
        yq = np.where(obs, t, 0.0).astype(np.float64)
        qc = lambda a: np.cumsum(np.where(obs, a, 0.0), 1, dtype=np.float64)
        self.qd = (np.cumsum(obs, 1, dtype=np.float64),
                   qc(uu), qc(uu ** 2), qc(uu ** 3), qc(uu ** 4),
                   qc(yq), qc(uu * yq), qc(uu ** 2 * yq))

        self.covsum, self.covcnt = [], []
        for m in self.cov:
            o = ~np.isnan(m)
            self.covsum.append(np.cumsum(np.where(o, m, 0.0), 1, dtype=np.float64).astype(np.float32))
            self.covcnt.append(np.cumsum(o, 1, dtype=np.float64).astype(np.float32))

        cal = (np.arange(N_MONTHS) + MONTH0) % 12
        self.clim = np.empty((12, self.n, N_MONTHS), np.float32)
        for c in range(12):
            sel = cal == c
            a = np.cumsum(np.where(sel[None, :], y, 0.0), 1)
            k = np.cumsum(np.where(sel[None, :], obs, 0.0), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.clim[c] = np.where(k > 0, a / k, np.nan).astype(np.float32)

        # v26: per-CALENDAR-MONTH change volatility. Every per-cell parameter so far is
        # ONE number per cell, averaged over all seasons -- but a snow cell accumulates
        # in winter and dumps in melt season, and a monsoon cell fills and drains on a
        # schedule. One `dvol` averages those regimes together, the same way v19's single
        # beta averaged the wetting and drying branches.
        # NOTE `clim_std_target` (v16) is the spread of LEVELS in the target month; this
        # is the spread of CHANGES. That levels-vs-changes distinction is exactly what
        # v16 won on (3.33), applied one level deeper. Built as-of like climstd, so no
        # month at or after the target can enter -- a wet/dry mask fitted on the whole
        # record would encode months >= t+1 for training rows and is NOT used.
        self.dvolm = np.empty((12, self.n, N_MONTHS), np.float32)
        d2 = np.where(od, dif * dif, 0.0).astype(np.float64)
        for c in range(12):
            sel = (cal == c)[None, :] & od
            a = np.cumsum(np.where(sel, d2, 0.0), 1)
            k = np.cumsum(sel, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.dvolm[c] = np.where(k >= 2, np.sqrt(a / np.maximum(k, 1)),
                                         np.nan).astype(np.float32)

        self.climstd = np.empty((12, self.n, N_MONTHS), np.float32)
        for c in range(12):
            sel = cal == c
            a = np.cumsum(np.where(sel[None, :], y, 0.0), 1)
            q = np.cumsum(np.where(sel[None, :], y * y, 0.0), 1)
            k = np.cumsum(np.where(sel[None, :], obs, 0.0), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(k > 0, a / k, np.nan)
                var = np.where(k > 1, q / k - mu * mu, np.nan)
            self.climstd[c] = np.sqrt(np.maximum(var, 0)).astype(np.float32)

        # --- neighbourhoods: lat/lon builds the adjacency table, nothing else
        lat_i, lon_i = self.gids // 360, self.gids % 360
        look = np.full(140 * 360, -1, np.int32); look[self.gids] = np.arange(self.n, dtype=np.int32)

        def ring(r):
            offs = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1)
                    if max(abs(a), abs(b)) == r]
            out = np.full((self.n, len(offs)), -1, np.int32)
            for k, (dla, dlo) in enumerate(offs):
                la, lo = lat_i + dla, (lon_i + dlo) % 360
                ins = (la >= 0) & (la < 140)
                out[ins, k] = look[la[ins] * 360 + lo[ins]]
            return out

        def mean_over(panel, ix):
            vals = panel[np.where(ix >= 0, ix, 0)]; vals[ix < 0] = np.nan
            o = ~np.isnan(vals); cnt = o.sum(1); tot = np.where(o, vals, 0.0).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(cnt > 0, tot / cnt, np.nan).astype(np.float32)

        r1, r2, r3 = ring(1), ring(2), ring(3)
        self.nbr_tws = mean_over(self.tws, r1)
        self.n1 = {k: mean_over(self.cov[k], r1) for k in [0, 1, 2, 3, 4]}
        self.n2 = {k: mean_over(self.cov[k], r2) for k in NBR_COV}
        self.n3 = {k: mean_over(self.cov[k], r3) for k in NBR_COV}

        # ---- v36: ERA5 panels on the SAME (cell, month) index scheme -----------
        e5 = pd.read_parquet(ROOT / E5_PATH)
        egid = (np.rint(e5.lat.to_numpy() + 55.5).astype(np.int64) * 360
                + np.rint(e5.lon.to_numpy() + 179.5).astype(np.int64))
        elook = np.full(140 * 360, -1, np.int64); elook[self.gids] = np.arange(self.n)
        eci = elook[egid]
        ets = pd.to_datetime(e5.time)
        emi = (ets.dt.year.to_numpy() * 12 + ets.dt.month.to_numpy() - 1 - MONTH0)
        eok = (eci >= 0) & (emi >= 0) & (emi < N_MONTHS)
        eci, emi = eci[eok], emi[eok].astype(np.int64)

        def _panel(v):
            a = np.full((self.n, N_MONTHS), np.nan, np.float32)
            a[eci, emi] = e5[v].to_numpy()[eok]
            return a

        self.e5, self.e5cum = {}, {}
        for v in E5_LEVEL:
            self.e5[v] = _panel(v)
        for v in E5_WIN:
            a = _panel(v); fin = np.isfinite(a)
            self.e5cum[v] = (
                np.cumsum(np.where(fin, a, 0.0), 1, dtype=np.float64).astype(np.float32),
                np.cumsum(fin, 1, dtype=np.float64).astype(np.float32))
            del a, fin

        so = np.isfinite(self.e5["sd"]); sy = np.where(so, self.e5["sd"], 0.0).astype(np.float64)
        self.sdclim = np.empty((12, self.n, N_MONTHS), np.float32)
        for c in range(12):
            sel = (cal == c)[None, :]
            aa = np.cumsum(np.where(sel, sy, 0.0), 1)
            kk = np.cumsum(np.where(sel, so, 0.0), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.sdclim[c] = np.where(kk > 0, aa / kk, np.nan).astype(np.float32)
        del so, sy


def take64(mat, cell, month):
    """take() casts to float32; the quadratic moments must stay float64."""
    ok = month >= 0
    return np.where(ok, mat[cell, np.where(ok, month, 0)], np.nan)


def take(mat, cell, month):
    ok = month >= 0
    return np.where(ok, mat[cell, np.where(ok, month, 0)], np.nan).astype(np.float32)


def win_mean(cs, cc, cell, lo, hi):
    s = cs[cell, hi] - take(cs, cell, lo); c = cc[cell, hi] - take(cc, cell, lo)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan).astype(np.float32)


def features(p: P, cell, t, anchor):
    cell = np.asarray(cell, np.int64); t = np.asarray(t, np.int32); anchor = np.asarray(anchor, np.int32)
    h = (t + 1 - anchor).astype(np.float32)
    tgt = anchor + h.astype(np.int32)
    ca, ct = (anchor + MONTH0) % 12, (tgt + MONTH0) % 12
    tw = p.tws[cell, anchor]
    cl_a, cl_t = p.clim[ca, cell, anchor], p.clim[ct, cell, anchor]
    cm, cs, tr_ = p.cmean[cell, anchor], p.cstd[cell, anchor], p.ctrend[cell, anchor]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(cs > 1e-6, (tw - cm) / cs, np.nan)
    roll = win_mean(p.csum, p.ccnt, cell, anchor - ROLL, anchor)
    cols = [tw, h, cl_a, cl_t, tw - cl_a, cl_t - cl_a, cm, cs, z, tr_, tr_ * h,
            roll, tw - roll]
    d, g = [], []
    for k in TWS_LAGS:
        j = anchor - k; okj = j >= 0
        pr = np.where(okj, p.last[cell, np.where(okj, j, 0)], -1); ok = pr >= 0
        pv = np.where(ok, p.tws[cell, np.where(ok, pr, 0)], np.nan)
        d.append((tw - pv).astype(np.float32))
        g.append(np.where(ok, anchor - pr, np.nan).astype(np.float32))
    cols += d + g
    ct_ = [take(m, cell, t) for m in p.cov]
    ca_ = [take(m, cell, anchor) for m in p.cov]
    cw = [win_mean(p.covsum[i], p.covcnt[i], cell, anchor, t) for i in range(5)]
    cols += ct_ + ca_ + [(a - b).astype(np.float32) for a, b in zip(ct_, ca_)] + cw
    ang = 2.0 * np.pi * (ct + 1) / 12.0
    cols += [np.sin(ang), np.cos(ang)]

    na = take(p.nbr_tws, cell, anchor)
    cols += [na, (na - tw).astype(np.float32)]
    for k in (1, 3):
        j = anchor - k; okj = j >= 0
        pr = np.where(okj, p.last[cell, np.where(okj, j, 0)], -1); ok = pr >= 0
        pv = np.where(ok, take(p.nbr_tws, cell, np.where(ok, pr, 0)), np.nan)
        cols.append((na - pv).astype(np.float32))
    n1t = [take(p.n1[i], cell, t) for i in NBR_COV]
    n1a = [take(p.n1[i], cell, anchor) for i in NBR_COV]
    cols += n1t + [(a - b).astype(np.float32) for a, b in zip(n1t, n1a)]
    xt = [take(p.n1[i], cell, t) for i in R1_EXTRA]
    xa = [take(p.n1[i], cell, anchor) for i in R1_EXTRA]
    cols += xt + [(a - b).astype(np.float32) for a, b in zip(xt, xa)]
    n2t = [take(p.n2[i], cell, t) for i in NBR_COV]
    n2a = [take(p.n2[i], cell, anchor) for i in NBR_COV]
    cols += n2t + [(a - b).astype(np.float32) for a, b in zip(n2t, n2a)]
    cols += [(n1t[0] - n2t[0]).astype(np.float32), (n1t[2] - n2t[2]).astype(np.float32)]
    n3t = [take(p.n3[i], cell, t) for i in NBR_COV]
    n3a = [take(p.n3[i], cell, anchor) for i in NBR_COV]
    cols += n3t + [(a - b).astype(np.float32) for a, b in zip(n3t, n3a)]
    cols += [(n2t[0] - n3t[0]).astype(np.float32), (n2t[2] - n3t[2]).astype(np.float32)]

    # --- v16 volatility block ---------------------------------------------
    va = take(p.dvol, cell, anchor)
    s12 = p.dsq[cell, anchor] - take(p.dsq, cell, anchor - 12)
    c12 = p.dcnt[cell, anchor] - take(p.dcnt, cell, anchor - 12)
    with np.errstate(invalid="ignore", divide="ignore"):
        v12 = np.where(c12 >= 3, np.sqrt(np.maximum(s12 / c12, 0.0)), np.nan).astype(np.float32)
        vr = np.where(va > 1e-6, v12 / va, np.nan).astype(np.float32)
        csa = p.climstd[ca, cell, anchor]
        cst = p.climstd[ct, cell, anchor]
        zc = np.where(csa > 1e-6, (tw - cl_a) / csa, np.nan).astype(np.float32)
        d1v = np.where(va > 1e-6, d[0] / va, np.nan).astype(np.float32)
    cols += [va, v12, vr, (va * np.sqrt(h)).astype(np.float32), cst, zc, d1v]

    # --- v19 response coefficients ----------------------------------------
    for ci in RESP_COV:
        nn, sx, sy, sxx, syy, sxy = (take(a, cell, anchor) for a in p.resp[ci])
        with np.errstate(invalid="ignore", divide="ignore"):
            cxy = sxy - sx * sy / nn
            cxx = sxx - sx * sx / nn
            cyy = syy - sy * sy / nn
            good = (nn >= 6) & (cxx > 1e-9)
            beta = np.where(good, cxy / cxx, np.nan).astype(np.float32)
            rho = np.where(good & (cyy > 1e-9), cxy / np.sqrt(cxx * cyy), np.nan).astype(np.float32)
            # the interaction a tree cannot form for itself: this cell's own
            # sensitivity times the covariate move actually observed over the gap
            imp = (beta * (ct_[ci] - ca_[ci])).astype(np.float32)
        cols += [rho, beta, imp]

    # --- v20 per-cell dynamics --------------------------------------------
    for tag, store, drv in (("ar", p.ar, d[0]), ("mr", p.mr, (tw - cm).astype(np.float32))):
        nn, sx, sy, sxx, syy, sxy = (take(a, cell, anchor) for a in store)
        with np.errstate(invalid="ignore", divide="ignore"):
            cxy = sxy - sx * sy / nn
            cxx = sxx - sx * sx / nn
            cyy = syy - sy * sy / nn
            good = (nn >= 6) & (cxx > 1e-9)
            beta = np.where(good, cxy / cxx, np.nan).astype(np.float32)
            rho = np.where(good & (cyy > 1e-9), cxy / np.sqrt(cxx * cyy), np.nan).astype(np.float32)
            cols += [rho, beta, (beta * drv).astype(np.float32)]
            if tag == "mr":
                # total expected reversion over the ACTUAL lead, not one step:
                # sum of a geometric decay at this cell's own rate. beta<0 = reverting.
                k = np.clip(beta, -1.0, 0.5)
                cols += [((tw - cm) * (1.0 - (1.0 + k) ** h)).astype(np.float32)]

    # --- v22 per-cell error correction ------------------------------------
    # Solve the 2x2 normal equations for  dTWS = k*TWS_{t-1} + g*Drv_{t-1}.
    # The equilibrium deviation the tree can never build for itself:
    #     gap = (TWS - m_tws) + (g/k) * (Drv_t - m_drv)
    # It needs a per-cell RATIO of two per-cell regression coefficients.
    for ci in ECM_COV:
        nn, s1, s2, sy, s11, s12, s22, s1y, s2y = (take(a, cell, anchor)
                                                   for a in p.ecm[ci])
        with np.errstate(invalid="ignore", divide="ignore"):
            m1, m2, my = s1 / nn, s2 / nn, sy / nn
            S11 = s11 - s1 * m1; S12 = s12 - s1 * m2; S22 = s22 - s2 * m2
            S1y = s1y - s1 * my; S2y = s2y - s2 * my
            det = S11 * S22 - S12 * S12
            good = (nn >= 10) & (np.abs(det) > 1e-9) & (S11 > 1e-9) & (S22 > 1e-9)
            kk = np.where(good, (S22 * S1y - S12 * S2y) / det, np.nan).astype(np.float32)
            gg = np.where(good, (S11 * S2y - S12 * S1y) / det, np.nan).astype(np.float32)
            # an equilibrium only means something for a cell that actually reverts
            rev = good & (kk < -0.02)
            ratio = np.clip(np.where(rev, gg / kk, np.nan), -20.0, 20.0)
            gap = ((tw - m1) + ratio * (ct_[ci] - m2)).astype(np.float32)
            kc = np.clip(kk, -1.0, -1e-3)
            # total correction expected over the ACTUAL lead; same sign convention
            # as v20 mr_decay_h (magnitude of reversion, positive when above eq.)
            dec = np.where(rev, gap * (1.0 - (1.0 + kc) ** h), np.nan).astype(np.float32)
        cols += [kk, gg, gap, dec]

    # --- v25 per-cell trend curvature --------------------------------------
    # Normal equations for [1, u, u^2] -> (a, b, c), solved by the same 3x3
    # cofactor route verified to 2e-14 against np.linalg.solve.
    n_, su, su2, su3, su4, sy_, suy, su2y = (take64(a, cell, anchor) for a in p.qd)
    with np.errstate(invalid="ignore", divide="ignore"):
        M11, M12, M13 = n_, su, su2
        M22, M23, M33 = su2, su3, su4
        R1, R2, R3 = sy_, suy, su2y
        C11 = M22 * M33 - M23 * M23
        C12 = M13 * M23 - M12 * M33
        C13 = M12 * M23 - M22 * M13
        C22 = M11 * M33 - M13 * M13
        C23 = M12 * M13 - M11 * M23
        C33 = M11 * M22 - M12 * M12
        det = M11 * C11 + M12 * C12 + M13 * C13
        okq = (n_ >= CURVE_MIN_N) & (np.abs(det) > 1e-12 * np.abs(M11 * M22 * M33))
        qb = np.where(okq, (C12 * R1 + C22 * R2 + C23 * R3) / det, np.nan)
        qc_ = np.where(okq, (C13 * R1 + C23 * R2 + C33 * R3) / det, np.nan)
        ua = (anchor.astype(np.float64) - CURVE_U0) / CURVE_US
        ut = (tgt.astype(np.float64) - CURVE_U0) / CURVE_US
        # instantaneous slope AT the anchor, per month (not the lifetime average)
        sl_now = (qb + 2.0 * qc_ * ua) / CURVE_US
        # move implied by the quadratic over the ACTUAL lead
        cmove = (qb * (ut - ua) + qc_ * (ut * ut - ua * ua))
        clip = lambda z: np.clip(z, -5.0, 5.0).astype(np.float32)
        cols += [np.clip(qc_, -50.0, 50.0).astype(np.float32),
                 clip(sl_now * 12.0),                 # per year, comparable to cell_trend*12
                 clip(sl_now * h),
                 clip(cmove),
                 clip(cmove - tr_ * h)]               # where the quadratic disagrees

    # --- v26 seasonal (per-calendar-month) structure -----------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        dvtm = p.dvolm[ct, cell, anchor]                    # typical |dTWS| in the TARGET month
        dvr = np.where(va > 1e-6, dvtm / va, np.nan).astype(np.float32)
        allc = p.clim[:, cell, anchor]                      # (12, N) this cell's whole cycle
        cmin = np.nanmin(allc, axis=0); cmax = np.nanmax(allc, axis=0)
        amp = (cmax - cmin).astype(np.float32)
        pos = np.where(amp > 1e-6, (cl_t - cmin) / amp, np.nan).astype(np.float32)
        prevm = (ct - 1) % 12
        rising = (cl_t - p.clim[prevm, cell, anchor]).astype(np.float32)
        # distance from where this cell's storage USUALLY sits in the target month.
        # = anom_anchor - clim_delta: a DIFFERENCE of two existing features, which an
        # axis-aligned tree cannot form any more than it can form a product.
        avt = (tw - cl_t).astype(np.float32)
        avtd = np.where(dvtm > 1e-6, avt / dvtm, np.nan).astype(np.float32)
    cols += [dvtm, dvr, amp, pos, rising, avt, avtd]

    # --- v30 conjunctions: own state x REGIONAL state ----------------------
    # n3t[0] is nbr3_spei06_t, the ring-3 drought state -- the highest-SHAP
    # feature that describes the neighbourhood rather than the cell. Crossing it
    # with each measure of the cell's own displacement gives the model the sign
    # information it cannot build: how much a given anomaly is worth depends on
    # whether the region is drying out underneath it.
    amr = (tw - roll).astype(np.float32)      # anchor_minus_roll12, the top feature
    zf = z.astype(np.float32)
    reg = n3t[0]                              # nbr3_spei06_t
    cols += [(amr * reg).astype(np.float32),
             (zf * reg).astype(np.float32),
             (tw * reg).astype(np.float32),
             (avt * reg).astype(np.float32),  # v26 seasonal anomaly x region
             (amr * zf).astype(np.float32)]   # falling AND already above normal

    # --- v36 ERA5 joint per-cell water balance --------------------------------
    # 3.49 obligation (a): ERA5 is read at `t` and `anchor`, both <= t; never at the
    # target month. Asserted, not merely intended.
    assert (anchor <= t).all() and (tgt == t + 1).all(), \
        "ERA5 date discipline violated: read at t/anchor only, never at tgt"

    e5take = lambda m, mo: np.where(mo >= 0, m[cell, np.maximum(mo, 0)],
                                    np.nan).astype(np.float32)

    def e5win(v):
        """Total of v over (anchor, t] -- strictly backward. NaN at h=1 where the
        window is empty; that is 33% of test rows and is correct, not missing."""
        cs, cc = p.e5cum[v]
        tot = cs[cell, t] - e5take(cs, anchor)
        n_ = cc[cell, t] - e5take(cc, anchor)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_ > 0, tot, np.nan).astype(np.float32)

    sd_t = e5take(p.e5["sd"], t)
    t2_t = e5take(p.e5["t2m"], t)
    w1_t = e5take(p.e5["swvl1"], t)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_sd = (sd_t - e5take(p.e5["sd"], anchor)).astype(np.float32)
        sd_anom = (sd_t - p.sdclim[ct, cell, anchor]).astype(np.float32)
        d_t2m = (t2_t - e5take(p.e5["t2m"], anchor)).astype(np.float32)
        wgrad = (w1_t - e5take(p.e5["swvl4"], t)).astype(np.float32)
    cols += [sd_t, d_sd, sd_anom, e5win("smlt"), t2_t, d_t2m,
             e5win("tp"), e5win("e"), e5win("sro"), w1_t, wgrad]

    X = np.empty((len(cell), len(cols)), np.float32)
    for i, c in enumerate(cols):
        X[:, i] = c
    assert X.shape[1] == len(NAMES), (X.shape[1], len(NAMES))
    return X


def features_chunked(p, cell, t, anchor, chunk=400_000):
    """Row-blocked features(), bit-identical to calling it once.

    Every operation in features() indexes [cell, anchor] and does elementwise
    arithmetic along the row axis -- there is no reduction ACROSS rows anywhere in
    it (the one axis-0 reduction, nanmin/nanmax over p.clim, reduces the 12 calendar
    months, not the rows). So splitting the rows and writing each block into a
    preallocated X gives exactly the same matrix.

    Why it matters: features() holds a list of ~120 separate column arrays and THEN
    allocates X and copies, so the transient is 2x the result. At 2.36M training rows
    that is 2.3 GB of float32 on top of the panels -- and it was OOM-killing the run
    before a DMatrix even existed. Chunking makes the transient proportional to the
    block, not the matrix.
    """
    cell = np.asarray(cell, np.int64); t = np.asarray(t, np.int32)
    anchor = np.asarray(anchor, np.int32)
    n = len(cell)
    X = np.empty((n, len(NAMES)), np.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        X[i:j] = features(p, cell[i:j], t[i:j], anchor[i:j])
    return X


def rss(tag):
    """Peak and current RSS, so a run reports its own memory profile."""
    import resource
    with open("/proc/self/statm") as fh:
        cur = int(fh.read().split()[1]) * 4096 / 2**30
    pk = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    log(f"  [mem] {tag}: rss {cur:.2f} GB, peak {pk:.2f} GB")


def train_rows(p, y_all, lo, hi, frac, rng):
    """Replicate each training row at h = 1..7; a natural row only has h=1."""
    base = np.flatnonzero((p.m_tr >= lo) & (p.m_tr <= hi))
    C, T, A, Y, H = [], [], [], [], []
    for h in range(1, MAX_H + 1):
        a = p.m_tr[base] + 1 - h
        keep = base[(a >= 0) & ~np.isnan(p.tws[p.c_tr[base].astype(np.int64),
                                               np.maximum(a, 0)])]
        if frac < 1.0 and len(keep):
            keep = rng.choice(keep, max(1, int(round(len(keep) * frac))), replace=False)
        C.append(p.c_tr[keep].astype(np.int64)); T.append(p.m_tr[keep])
        A.append(p.m_tr[keep] + 1 - h); Y.append(y_all[keep])
        H.append(np.full(len(keep), h, np.int8))
    return (np.concatenate(C), np.concatenate(T).astype(np.int32),
            np.concatenate(A).astype(np.int32), np.concatenate(Y), np.concatenate(H))


def main():
    rng = np.random.default_rng(SEED)
    log("loading")
    tr = pd.read_csv(ROOT / "data/raw/Train.csv",
                     usecols=["time", "lat", "lon", "TWS_t", *COVARIATES, "target"])
    te = pd.read_csv(ROOT / "data/raw/Test.csv",
                     usecols=["ID", "time", "lat", "lon", "TWS_t", *COVARIATES])
    y_all = tr["target"].to_numpy(); ids = te["ID"].to_numpy()
    p = P(tr, te); del tr, te
    log(f"{p.n:,} cells, {int((~np.isnan(p.tws)).any(0).sum())} months with observed TWS")

    te_cell = p.c_te.astype(np.int64); te_t = p.m_te
    te_a = p.last[te_cell, te_t]
    te_h = te_t + 1 - te_a
    cnt = np.bincount(te_h, minlength=MAX_H + 1)[1:MAX_H + 1]
    share = cnt / cnt.sum()
    log("test horizon mix: " + " ".join(f"h{i+1}={c}" for i, c in enumerate(cnt)))

    months = np.unique(p.m_tr); val_lo = months[-VAL_MONTHS]
    # BOTH row samples are drawn here, in the original order. train_rows() consumes
    # `rng`, so swapping these two calls would resample every training row and
    # silently confound the run against v26 -- only the MATRIX build order moves.
    c_v, t_v, a_v, y_v, hv = train_rows(p, y_all, val_lo, months[-1], VAL_FRAC, rng)
    c_t, t_t, a_t, y_t, h = train_rows(p, y_all, 0, val_lo - 1, FRAC, rng)

    yv, wv = y_v, share[hv - 1].astype(np.float32)
    av = p.tws[c_v, a_v]
    log(f"validation {len(yv):,} rows | persistence "
        f"{np.sqrt(np.average((yv-av)**2, weights=wv)):.4f}")

    # Build the LARGER matrix first and hand it straight to XGBoost. DMatrix copies
    # its input, so Xt is dead the moment dtr exists; building Xv first instead would
    # hold 0.5 GB resident across dtr's 2.3 GB allocation, and that overlap -- not the
    # result -- is what OOM-killed the first two attempts.
    wt = share[h - 1].astype(np.float32)
    Xt = features_chunked(p, c_t, t_t, a_t)
    log(f"training {len(y_t):,} rows x {Xt.shape[1]}"); rss("after Xt")
    dtr = xgb.DMatrix(Xt, label=(y_t - p.tws[c_t, a_t]).astype(np.float32),
                      weight=wt, feature_names=NAMES)
    del Xt; gc.collect()
    Xv = features_chunked(p, c_v, t_v, a_v)
    dva = xgb.DMatrix(Xv, label=(yv - av).astype(np.float32), weight=wv,
                      feature_names=NAMES)
    del Xv; gc.collect()

    rss("dmatrices built")
    m = xgb.train(PARAMS, dtr, ROUNDS, evals=[(dva, "valid")],
                  early_stopping_rounds=100, verbose_eval=200)
    best = m.best_iteration + 1
    pv = av + m.predict(dva, iteration_range=(0, best)).astype(np.float32)
    del dtr, dva
    log(f"VALIDATION RMSE {np.sqrt(np.average((yv-pv)**2, weights=wv)):.4f}  best_iter={best}")
    del wt, wv
    gc.collect()

    rounds_full = int(round(best * 1.10))
    log(f"refitting on all months for {rounds_full} rounds x {N_SEEDS} seeds")
    c, t, a, y, h = train_rows(p, y_all, 0, months[-1], FRAC, rng)
    Xf, df, wf = features_chunked(p, c, t, a), (y - p.tws[c, a]).astype(np.float32), \
        share[h - 1].astype(np.float32)

    # Five fits differing only in the RNG streams that drive row bagging and
    # feature sampling. Averaging the DELTAS (not the levels) is equivalent here
    # because every seed shares the same anchor, and it keeps the arithmetic in
    # the same space the model was trained on.
    dfull = xgb.DMatrix(Xf, label=df, weight=wf, feature_names=NAMES)
    del Xf, df, wf                    # the largest allocation in the run
    gc.collect()
    # test matrix built only AFTER dfull exists, so it never overlaps that 2.7 GB
    dte = xgb.DMatrix(features_chunked(p, te_cell, te_t, te_a), feature_names=NAMES)
    gc.collect(); rss("refit dmatrix")
    preds = []
    for i in range(N_SEEDS):
        m_i = xgb.train(dict(PARAMS, seed=SEED + 1000 * i), dfull, rounds_full)
        if i == 0:
            (ROOT / "reports").mkdir(exist_ok=True)
            m_i.save_model(str(ROOT / "reports" / f"{_here.stem}_seed0.json"))
        preds.append(m_i.predict(dte).astype(np.float32))
        log(f"  seed {i + 1}/{N_SEEDS} done")
    del dfull, dte
    gc.collect()

    pred = p.tws[te_cell, te_a] + np.mean(preds, axis=0)
    out = ROOT / "subs" / "xgb_v32_era5.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"ID": ids, "Target": pred.astype(np.float64)}).to_csv(out, index=False)
    log(f"wrote {out} ({len(pred)} rows)")


if __name__ == "__main__":
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT)); _sys.path.insert(0, str(_here.parent))
        from carbon import track
    except Exception:
        from contextlib import nullcontext
        def track(_n): return nullcontext()
    with track(_here.stem):
        main()
