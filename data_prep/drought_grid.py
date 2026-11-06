"""Regrid CDS SPI/SPEI drought indices onto the competition's 1 degree / 15,715-cell grid.

Differs from cci_grid.py in one way that matters: a single yearly zip holds FOUR
accumulation windows x 12 months = 48 files, and each file names its variable after
its own window (`SPEI12`, `SPEI24`, ...). So the output column is derived from the
FILENAME, not passed in -- one pass produces all four columns at once.

The 0.25 deg grid here runs lat -56..84 and lon -180..180 INCLUSIVE, i.e. 1441
longitudes with a duplicated endpoint. The box-index arithmetic drops it naturally:
floor(180)+0.5 -> index 360, which is outside [0,360) and filtered out.

⚠️ SPEI12 is requested DELIBERATELY even though the host already supplies `SPEI_12_t`.
It is the overlap that lets us verify the two are the same product before mixing short
and long windows. If they disagree, the block carries a discontinuity at the 12/24
boundary and the long windows cannot simply be appended to the host's four.
"""
from __future__ import annotations
import argparse, zipfile, re, tempfile
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parent


def target_cells():
    tr = pd.read_csv(ROOT / "data/raw/Train.csv", usecols=["lat", "lon"])
    te = pd.read_csv(ROOT / "data/raw/Test.csv", usecols=["lat", "lon"])
    return (pd.concat([tr, te]).drop_duplicates()
            .sort_values(["lat", "lon"]).reset_index(drop=True))


def regrid_zip(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    nlat, nlon = 140, 360
    ci = np.rint(cells.lat.to_numpy() + 55.5).astype(np.int64)
    cj = np.rint(cells.lon.to_numpy() + 179.5).astype(np.int64)
    cflat = ci * nlon + cj
    frames = {}
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as d:
        for name in sorted(z.namelist()):
            if not name.endswith(".nc"):
                continue
            base = name.split("/")[-1]
            m = re.match(r"(SPEI|SPI)(\d+)_", base)
            if not m:
                continue
            col = f"{m.group(1).lower()}{int(m.group(2)):02d}"      # spei12, spei24, ...
            ds = xr.open_dataset(z.extract(name, d))
            var = m.group(1) + m.group(2)
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
            key = ts.strftime("%Y-%m-01")
            frames.setdefault(key, {})[col] = mean[cflat].astype(np.float32)
            ds.close()
    out = []
    for key, cols in sorted(frames.items()):
        df = pd.DataFrame({"time": key, "lat": cells.lat.to_numpy(),
                           "lon": cells.lon.to_numpy()})
        for c, v in cols.items():
            df[c] = v
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/drought")
    ap.add_argument("--out", default="data/drought_grid.parquet")
    a = ap.parse_args()
    cells = target_cells()
    print(f"target grid: {len(cells):,} cells")
    zips = sorted((ROOT / a.src).glob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips under {a.src}")
    df = pd.concat([regrid_zip(z, cells) for z in zips], ignore_index=True)
    df.to_parquet(ROOT / a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows x {df.shape[1]} cols")
    for c in [c for c in df.columns if c not in ("time", "lat", "lon")]:
        v = df[c].to_numpy()
        print(f"  {c:<8} {np.isfinite(v).mean()*100:5.1f}% finite  "
              f"range {np.nanmin(v):.3f}..{np.nanmax(v):.3f}")


if __name__ == "__main__":
    main()
