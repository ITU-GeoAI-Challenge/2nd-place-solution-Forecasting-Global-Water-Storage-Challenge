"""Regrid ERA5-Land onto the competition's 1 degree / 15,715-cell land grid.

Our grid: cell centres at lat X.5 for X in -56..83, lon Y.5 for Y in -180..179.
A cell centred at (X.5, Y.5) covers [X, X+1] x [Y, Y+1].

ERA5-Land native is 0.1 degree and LAND-ONLY -- ocean is NaN -- so each 1 degree box is
a 10x10 block of native points of which only the land ones are valid. We take the
nanmean over the block, which is the land-area-weighted mean and the physically correct
aggregation for a storage quantity. Coastal boxes are partly ocean and therefore
averaged over fewer points; boxes with no land at all come out NaN and are dropped,
since our 15,715 cells are land by construction.

Units, and the trap in the monthly product:
  * snow depth water equivalent, soil water layers -- instantaneous, use as-is
  * total precipitation, runoff, evaporation, snowmelt -- stored as DAILY MEAN RATES
    (m/day) in the monthly means, NOT monthly totals. Multiply by days-in-month for an
    accumulation. Evaporation is NEGATIVE for upward flux.
"""
from __future__ import annotations
import argparse, calendar
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parent
# Stored as DAILY MEAN RATES (m/day) in the monthly product, so they must be
# multiplied by days-in-month. Everything else (sd, swvl*, t2m, stl*, snowc) is
# instantaneous and must NOT be. `sf` was missing here and would have come out
# ~30x too small.
RATE_VARS = {"tp", "ro", "sro", "ssro", "e", "smlt", "es", "evavt", "evabs",
             "pev", "sf"}


def target_cells():
    """The 15,715 land cells, from the competition files."""
    tr = pd.read_csv(ROOT / "data/raw/Train.csv", usecols=["lat", "lon"])
    te = pd.read_csv(ROOT / "data/raw/Test.csv", usecols=["lat", "lon"])
    g = pd.concat([tr, te]).drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    return g


def regrid_file(path: Path, cells: pd.DataFrame, as_rate=False) -> pd.DataFrame:
    """Block-average a native ERA5-Land file onto the 15,715-cell 1 degree grid.

    Done with explicit index arithmetic rather than xarray groupby: chained groupbys
    drop the coordinate index, and this is both correct and much faster. Each native
    point is assigned to the 1 degree box containing it (floor(coord) + 0.5 is that
    box's centre), then summed with np.add.at and divided by the count of NON-NaN
    contributions -- i.e. a nanmean, which for a land-only product is the
    land-area-weighted average.
    """
    ds = xr.open_dataset(path)
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    times = pd.to_datetime(ds[tname].values)
    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)

    # target 1 degree grid
    tlat = np.arange(-55.5, 84.5, 1.0)          # 140 centres, -55.5 .. 83.5
    tlon = np.arange(-179.5, 180.5, 1.0)        # 360 centres
    nlat, nlon = len(tlat), len(tlon)

    # which target box does each native point fall in
    li = np.rint(np.floor(lat) + 0.5 + 55.5).astype(np.int64)
    lj = np.rint(np.floor(lon) + 0.5 + 179.5).astype(np.int64)
    keep_i = (li >= 0) & (li < nlat)
    keep_j = (lj >= 0) & (lj < nlon)
    li_k, lj_k = li[keep_i], lj[keep_j]
    flat = (li_k[:, None] * nlon + lj_k[None, :]).ravel()

    # map our 15,715 cells onto that grid
    ci = np.rint(cells.lat.to_numpy() + 55.5).astype(np.int64)
    cj = np.rint(cells.lon.to_numpy() + 179.5).astype(np.int64)
    cflat = ci * nlon + cj

    frames = []
    for var in ds.data_vars:
        da = ds[var]
        if set(da.dims) - {tname, "latitude", "longitude"}:
            da = da.squeeze(drop=True)
        arr = da.values                                    # (time, lat, lon)
        for m, ts in enumerate(times):
            a = arr[m][np.ix_(keep_i, keep_j)].astype(np.float64).ravel()
            good = np.isfinite(a)
            tot = np.zeros(nlat * nlon); cnt = np.zeros(nlat * nlon)
            np.add.at(tot, flat[good], a[good])
            np.add.at(cnt, flat[good], 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            v = mean[cflat].astype(np.float32)
            if as_rate and var in RATE_VARS:
                v = v * calendar.monthrange(ts.year, ts.month)[1]
            frames.append(pd.DataFrame({"time": ts.strftime("%Y-%m-%d"),
                                        "lat": cells.lat.to_numpy(),
                                        "lon": cells.lon.to_numpy(), var: v}))
    # one frame per (var, month) -> pivot into columns
    out = None
    for var in ds.data_vars:
        sub = pd.concat([f for f in frames if var in f.columns], ignore_index=True)
        out = sub if out is None else out.merge(sub, on=["time", "lat", "lon"], how="outer")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/era5")
    ap.add_argument("--out", default="data/era5_grid.parquet")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out parquet on (time, lat, lon) "
                         "instead of overwriting; use for the second source dir")
    ap.add_argument("--rates", action="store_true",
                    help="convert daily-mean rate variables to monthly totals")
    a = ap.parse_args()
    cells = target_cells()
    print(f"target grid: {len(cells):,} cells, lat {cells.lat.min()}..{cells.lat.max()}")
    files = sorted(Path(ROOT / a.src).glob("era5land_*.nc"))
    if not files:
        raise SystemExit(f"no era5land_*.nc under {a.src}")
    parts = [regrid_file(f, cells, as_rate=a.rates) for f in files]
    df = pd.concat(parts, ignore_index=True)
    if a.merge:
        # The model reads ONE panel, but main (0.25 deg) and snow (native 0.1 deg) are
        # pulled separately because only snow needs the finer grid. Merge on the cell
        # key so the whole chain from netCDF to model input is scripted:
        #   python era5_grid.py --src data/era5/nc_0p25_core --out data/era5_grid.parquet --rates
        #   python era5_grid.py --src data/era5/nc_0p10_snow --out data/era5_grid.parquet --merge
        prev = ROOT / a.out
        if prev.exists():
            df = pd.read_parquet(prev).merge(df, on=["time", "lat", "lon"], how="outer")
            print(f"merged into existing {a.out}: {df.shape[1]} cols")
    df.to_parquet(ROOT / a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows x {df.shape[1]} cols")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
