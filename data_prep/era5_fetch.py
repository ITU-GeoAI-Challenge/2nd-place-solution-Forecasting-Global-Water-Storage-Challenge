"""Download ERA5-Land monthly means for the competition grid.

=============================================================================
LEGALITY AUDIT against reports/important_guidelines.txt section 3
(re-checked 2026-08-27 before any bulk download)
=============================================================================

PERMITTED, and this is what we rely on:
  * "lat/lon as a lookup key - ... to sample external gridded covariates at that
    location."   -> we use lat/lon ONLY to sample ERA5, never as a feature.
  * "External non-TWS covariates, provided every source date is at or before t."
  * "ERA5T operationally, with final ERA5 as the retrospective proxy for historical
    validation, documented in the report."   -> we use final ERA5-Land, the
    retrospective proxy. THIS MUST BE STATED IN THE REPORT.

NOT PERMITTED, and why none of it applies:
  * "External GRACE or GRACE-derived total water storage, at any date ... including
    ... GRACE-trained reconstructions such as GTWS-MLrec and GRAiCE."
      -> ERA5-Land is NOT GRACE-derived. It is a single simulation of the CHTESSEL
         land-surface model forced by ERA5 atmospheric fields. It assimilates no land
         observations whatsoever, and no terrestrial water storage. GRACE assimilation
         occurs in other products (GLDAS family), not in ERA5-Land.
         Ref: Munoz-Sabater et al. 2021, ESSD 13, 4349-4383.
  * "Raw coordinates, cell IDs, coordinate-derived encodings and spatial embeddings as
    predictive features."   -> lat/lon are a lookup key here and never a feature.
  * "Any feature encoding month t+1 or later, from any source, remains grounds for
    disqualification."
      -> ERA5 monthly means for month M aggregate days WITHIN M, so month t is legal
         (identical standing to the provided SPEI_*_t / SOIL_MOISTURE_t columns, which
         are given at each row's own month t). Month tgt = t+1 is FATAL. Feature code
         must index ERA5 panels at `t` or `anchor`, NEVER at `tgt`.

TWS COMPONENTS vs TWS:
  Snow and soil water are COMPONENTS of total water storage, not TWS. The competition
  itself supplies SOIL_MOISTURE -- a TWS component -- as a legal covariate, which
  settles that components are permitted.
  ⚠️ SELF-IMPOSED RULE: do NOT sum soil water + snow + lakes into a TWS proxy feature.
  It is not GRACE-derived and is arguably legal, but it manufactures the target
  quantity from parts, and this project does not touch grey areas. Keep components
  separate and let the model combine them.

NOT DOWNLOADED, deliberately: nothing GRACE-derived, no TWSA product, no mascons.
=============================================================================

Setup once:
  1. Register at https://cds.climate.copernicus.eu
  2. Accept the licence under "Terms of use" (or: python accept_cds_licence.py --accept)
  3. CDS_API_KEY=<token> in .env (gitignored), or ~/.cdsapirc

Usage:
    python era5_fetch.py --probe                     # snow only, 2002-2003, validates the pipeline
    python era5_fetch.py --grid 0.25                 # full pull at 0.25 deg (~748 MB)
    python era5_fetch.py --vars 2m_temperature       # single variable

GRID NOTE: CDS 1.0 deg lands on WHOLE degrees (84, 83, ...) while our cell centres are
at HALF degrees (83.5, 82.5, ...) -- a half-cell offset that silently samples the wrong
cell. 0.25 deg contains our centres exactly and each 1 deg box is a clean 4x4 block.
Never request --grid 1.0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(d for d in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
            if (d / "data" / "raw").is_dir())
OUT = ROOT / "data" / "era5"
DATASET = "reanalysis-era5-land-monthly-means"
CDS_URL = "https://cds.climate.copernicus.eu/api"


def _key_from_env():
    """Read CDS_API_KEY from the project .env. Tolerates spaces around '=' and quotes.
    Kept out of git via .gitignore -- never print or log the value."""
    f = ROOT / ".env"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "CDS_API_KEY":
            return v.strip().strip('"').strip("'") or None
    return None

# Ranked by expected value for total water storage. Snow is first because it is the one
# large TWS component with NO proxy in the provided columns, and the grid reaches 83.5N.
VARIABLES = [
    "snow_depth_water_equivalent",     # THE gap: seasonal snow mass, absent from our data
    "volumetric_soil_water_layer_1",   # we get one soil-moisture column; ERA5-Land has 4
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",   # 100-289 cm, the slow store nearest to groundwater
    "total_precipitation",             # the forcing itself; SPEI is a normalised transform
    "sub_surface_runoff",              # loss terms
    "surface_runoff",
    "total_evaporation",
    "snowmelt",
    "2m_temperature",                  # drives melt and ET
]

# ---------------------------------------------------------------------------
# INVARIANT physiography.  These are the ONLY ERA5-Land fields that are not CHTESSEL
# output -- they are the model's boundary conditions, taken from FAO soil maps, GLCC
# land-cover inventories and a glacier mask.  Static: no time dimension, so nothing
# here can encode month t+1 (section 3's fatal case cannot arise).
#
# ⚠️ `geopotential` is surface elevation.  It is NOT a coordinate -- many cells share an
# elevation and it does not identify a location -- but it is the most coordinate-like of
# the ten, so it is downloaded and SCREENED SEPARATELY and is opt-in for any model.
INVARIANTS = [
    "soil_type",                       # FAO texture class -> field capacity, drainage
    "glacier_mask",                    # melt-driven vs ET-driven loss
    "lake_cover",                      # surface-water fraction
    "lake_total_depth",
    "high_vegetation_cover",           # transpiration capacity, rooting depth
    "low_vegetation_cover",
    "type_of_high_vegetation",         # categorical
    "type_of_low_vegetation",          # categorical
    "land_sea_mask",                   # coastal cells are fractional
    "geopotential",                    # elevation; see warning above
]

# our grid: lat -55.5..83.5, all longitudes.  CDS wants [North, West, South, East]
AREA = [84, -180, -56, 180]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _record(target, req):
    """Append a provenance row so the report can state exactly what was retrieved.

    Criterion: "their source and processing are fully documented for reproducibility".
    The scripts capture the METHOD; this captures the ACT -- which request, which file,
    which day, which checksum.
    """
    import hashlib, json, datetime
    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    row = {
        "downloaded_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "doi": "10.24381/cds.68d2bb30",
        "product": "ERA5-Land monthly averaged reanalysis (final ERA5-Land, "
                   "retrospective proxy per guidelines section 3)",
        "file": target.name,
        "bytes": target.stat().st_size,
        "sha256": h.hexdigest(),
        "request": req,
    }
    log = ROOT / "reports" / "era5_provenance.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")


def fetch(years, variables, outdir: Path, grid=None, months=None, stem="era5land"):
    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi not installed:  .venv/bin/pip install cdsapi")

    outdir.mkdir(parents=True, exist_ok=True)
    key = _key_from_env()
    if key:
        client = cdsapi.Client(url=CDS_URL, key=key)
    elif (Path.home() / ".cdsapirc").exists():
        client = cdsapi.Client()
    else:
        sys.exit("No CDS credentials. Put CDS_API_KEY=<token> in .env (project root), "
                 "or create ~/.cdsapirc. Token comes from your CDS profile page.")
    months = months or MONTHS
    for year in years:
        target = outdir / (f"{stem}.nc" if stem != "era5land" else f"era5land_{year}.nc")
        if target.exists() and target.stat().st_size > 0:
            print(f"  {year}  already present, skipping")
            continue
        req = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": variables,
            "year": [str(year)],
            "month": months,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": AREA,
        }
        if grid:
            # Server-side regridding. The native 0.1 deg grid is 1401x3600 and we average
            # 99% of it away, so asking CDS for 1 deg cuts the download ~100x (3.7 GB -> ~37 MB).
            # CDS interpolates rather than doing a land-masked area mean, so this must be
            # validated against a self-regridded native file before trusting it -- see
            # era5_gridcheck.py.
            req["grid"] = [grid, grid]
        print(f"  {year}  requesting {len(variables)} variables ...", flush=True)
        try:
            client.retrieve(DATASET, req, str(target))
        except Exception as e:
            msg = str(e)
            if "403" in msg or "licence" in msg.lower() or "license" in msg.lower():
                sys.exit(f"\n403 from CDS for {year}.\nAccept the licence first: open\n"
                         f"  https://cds.climate.copernicus.eu/datasets/{DATASET}\n"
                         f"and click 'Terms of use' -> Accept. Then re-run.\n\n{msg}")
            raise
        print(f"  {year}  -> {target}  ({target.stat().st_size/1e6:.1f} MB)")
        _record(target, req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", "--test", dest="probe", action="store_true",
                    help="snow only, 2002-2003 -- validates auth, licence and regridding "
                         "cheaply. NOTE: 'probe' means a trial DOWNLOAD; it has nothing to "
                         "do with Test.csv or the 2015-2018 test period, and in fact pulls "
                         "training-era years.")
    ap.add_argument("--vars", nargs="+", default=None)
    ap.add_argument("--invariant", action="store_true",
                    help="pull the 10 static physiography fields (one timestep, ~2 MB). "
                         "These carry no time dimension, so the month t+1 prohibition "
                         "cannot apply to them.")
    ap.add_argument("--from-year", type=int, default=2002)
    ap.add_argument("--to-year", type=int, default=2018)
    ap.add_argument("--grid", type=float, default=None,
                    help="request this resolution in degrees from CDS "
                         "(1.0 matches our grid and is ~100x smaller)")
    ap.add_argument("--out", default=None, help="override output subfolder")
    a = ap.parse_args()

    if a.invariant:
        years, variables = [2002], INVARIANTS
        outdir = OUT / f"nc_{str(a.grid or 0.25).replace('.', 'p')}_invariant"
        print(f"INVARIANT PULL: {len(variables)} static fields, one timestep.")
    elif a.probe:
        years, variables, outdir = [2002, 2003], ["snow_depth_water_equivalent"], OUT / "_probe"
        print("PROBE PULL: snow water equivalent, 2002-2003 only (~10 MB).")
        print("  (a trial download to validate the pipeline -- unrelated to Test.csv)")
    else:
        years = list(range(a.from_year, a.to_year + 1))
        variables = a.vars or VARIABLES
        outdir = OUT
        print(f"FULL PULL: {len(variables)} variables x {len(years)} years")
        print("Requests queue on the CDS side; expect this to take a while.")
    if a.grid is not None and abs(a.grid - 1.0) < 1e-9:
        raise SystemExit("--grid 1.0 is REFUSED: the CDS 1 deg grid sits on whole "
                         "degrees while our cell centres are on half degrees, so every "
                         "sample lands half a cell off. Use --grid 0.25.")
    # Guidelines section 3 bans external GRACE / GRACE-derived TWS at ANY date.
    # This list is deliberately broad. CDS hosts
    # `satellite-gravimetry-terrestrial-water-storage` -- the TARGET VARIABLE, 2002 to
    # present, covering the whole test window -- filed under the same "Land (hydrology)"
    # category as ERA5, which the rules DO permit. Nothing about its placement flags it.
    # An earlier version of this check missed it twice over: it matched only
    # "total_water_storage" (the product says TERRESTRIAL), had no "gravimetry" term,
    # and scanned only the variable list, never the dataset name.
    banned = ("grace", "twsa", "mascon", "gtws", "graice", "gravimetry",
              "total_water_storage", "terrestrial_water_storage", "water_storage",
              "tws_", "_tws", "jpl_mascon", "csr_", "gfz_")
    def _refuse(what, s_):
        hit = [b for b in banned if b in s_.lower().replace("-", "_")]
        if hit:
            raise SystemExit(
                f"REFUSED: {what} {s_!r} matches {hit}. Guidelines section 3 bans external "
                "GRACE / GRACE-derived TWS at ANY date -- this is the TARGET variable, not "
                "a covariate. If this is a false positive, widen the check deliberately and "
                "record why in NOTES; do not edit it around a single case.")
    _refuse("dataset", DATASET)          # not just the variables -- the dataset too
    for v_ in variables:
        _refuse("variable", v_)
    if a.out:
        outdir = OUT / a.out
    if a.invariant:
        fetch(years, variables, outdir, grid=a.grid, months=["01"], stem="era5land_invariant")
    else:
        fetch(years, variables, outdir, grid=a.grid)
    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    main()
