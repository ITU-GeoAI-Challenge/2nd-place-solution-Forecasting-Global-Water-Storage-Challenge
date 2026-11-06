"""Regrid ESA CCI satellite soil moisture onto the competition's 1 degree / 15,715-cell grid.

Same block-average as era5_grid.py, but CCI needs three things ERA5 did not:
  * the yearly download is a ZIP of 12 monthly netCDFs, not one file
  * the payload variable name DIFFERS by product, so `--var` is explicit:
      surface   -> `sm`
      root-zone -> `rzsm_1` (0-10cm), `rzsm_2` (10-40), `rzsm_3` (40-100), `rzsm_1m` (0-100)
    and the output column is named separately with `--col` so they never collide
  * coordinates are `lat`/`lon`, and the 0.25 deg centres sit at X.88/X.63/X.38/X.13
    against ERA5's X.00/X.75/X.50/X.25 -- a half-cell offset. HARMLESS here: each
    0.25 cell still falls unambiguously inside one 1 degree box, so every box still
    gets a clean 4x4 block. The offset would only matter for point sampling.

⚠️ COVERAGE. This is a RETRIEVAL, not a simulation. Measured on our own cells: 85.2%
mean, 75.1% in January, 91.6% in July. The gaps are snow, frozen ground and dense
vegetation -- i.e. systematically the cold cells, which are exactly the h5-h7
December->June population. Missingness here is informative, not random.
"""
from __future__ import annotations
import argparse, zipfile, tempfile
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parent


def target_cells():
    tr = pd.read_csv(ROOT / "data/raw/Train.csv", usecols=["lat", "lon"])
    te = pd.read_csv(ROOT / "data/raw/Test.csv", usecols=["lat", "lon"])
    return (pd.concat([tr, te]).drop_duplicates()
            .sort_values(["lat", "lon"]).reset_index(drop=True))


def regrid_zip(path: Path, cells: pd.DataFrame, colname: str, var: str) -> pd.DataFrame:
    nlat, nlon = 140, 360
    ci = np.rint(cells.lat.to_numpy() + 55.5).astype(np.int64)
    cj = np.rint(cells.lon.to_numpy() + 179.5).astype(np.int64)
    cflat = ci * nlon + cj
    out = []
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as d:
        for name in sorted(z.namelist()):
            if not name.endswith(".nc"):
                continue
            ds = xr.open_dataset(z.extract(name, d))
            lat = ds["lat"].values.astype(np.float64)
            lon = ds["lon"].values.astype(np.float64)
            li = np.rint(np.floor(lat) + 0.5 + 55.5).astype(np.int64)
            lj = np.rint(np.floor(lon) + 0.5 + 179.5).astype(np.int64)
            ki = (li >= 0) & (li < nlat); kj = (lj >= 0) & (lj < nlon)
            flat = (li[ki][:, None] * nlon + lj[kj][None, :]).ravel()
            ts = pd.to_datetime(ds["time"].values)[0]
            a = ds[var].isel(time=0).values[np.ix_(ki, kj)].astype(np.float64).ravel()
            good = np.isfinite(a)
            tot = np.bincount(flat[good], weights=a[good], minlength=nlat * nlon)
            cnt = np.bincount(flat[good], minlength=nlat * nlon).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            out.append(pd.DataFrame({
                "time": ts.strftime("%Y-%m-01"),
                "lat": cells.lat.to_numpy(), "lon": cells.lon.to_numpy(),
                colname: mean[cflat].astype(np.float32),
                # how many 0.25 cells actually carried a retrieval: a direct measure of
                # how much to trust the value, and a feature in its own right
                colname + "_n": cnt[cflat].astype(np.float32)}))
            ds.close()
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of yearly zips")
    ap.add_argument("--col", required=True, help="output column name")
    ap.add_argument("--var", default="sm", help="netCDF variable to read (see module docstring)")
    ap.add_argument("--out", default="data/cci_grid.parquet")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    cells = target_cells()
    print(f"target grid: {len(cells):,} cells")
    zips = sorted((ROOT / a.src).glob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips under {a.src}")
    df = pd.concat([regrid_zip(z, cells, a.col, a.var) for z in zips], ignore_index=True)
    p = ROOT / a.out
    if a.merge and p.exists():
        df = pd.read_parquet(p).merge(df, on=["time", "lat", "lon"], how="outer")
        print(f"merged into {a.out}: {df.shape[1]} cols")
    df.to_parquet(p, index=False)
    print(f"wrote {a.out}: {len(df):,} rows x {df.shape[1]} cols")
    v = df[a.col].to_numpy()
    print(f"  {a.col}: {np.isfinite(v).mean()*100:.1f}% finite, "
          f"range {np.nanmin(v):.4f}..{np.nanmax(v):.4f}")


if __name__ == "__main__":
    main()
