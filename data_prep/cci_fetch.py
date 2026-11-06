"""Fetch ESA CCI satellite soil moisture from the Copernicus CDS.

WHY, and why it is different from everything already downloaded:
every covariate we have added so far is a MODEL OUTPUT. ERA5-Land is CHTESSEL, so
`swvl1` is what a land-surface model THINKS the soil holds. The host's supplied
`SOIL_MOISTURE` comes from EDO/CDS and is very likely ERA5-derived too. That would
explain why 11 ERA5 variables across 4 encodings bought at most -0.0008: we may
largely have been re-describing information the model already had.

ESA CCI is a RETRIEVAL -- scatterometers and radiometers, independent evidence.

LEGALITY: soil moisture is a non-TWS covariate and the host supplies its own
SOIL_MOISTURE column, so the class is definitionally permitted. Guidelines section 3
allows "external non-TWS covariates, provided every source date is at or before t".
The GRACE guard below is shared with era5_fetch.py and refuses the CDS gravimetry
product by dataset name as well as variable name.

COVERAGE CAVEAT: this is a SURFACE retrieval (top ~5 cm) with real gaps under dense
vegetation, snow and frozen ground. Expect NaN where ERA5 has values -- that is the
honest difference between a measurement and a simulation, not a defect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(d for d in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
            if (d / "data" / "raw").is_dir())
OUT = ROOT / "data" / "cci_sm"
DATASET = "satellite-soil-moisture"
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


# ⚠️ Request ONE product per call. Asking for both in a single request returns only
# the surface product, silently -- no error, no warning, and the zip simply contains 12
# SSMV files. Verified 2026-08-31.
VARIABLES = ["surface_soil_moisture_volumetric"]     # top ~5 cm, the classic CCI product
# root_zone_soil_moisture_volumetric  -- 0-100 cm, nearest what GRACE integrates;
#   fetch with:  python cci_fetch.py --vars root_zone_soil_moisture_volumetric --out rootzone
SENSOR = "combined"        # active + passive merged: best spatial/temporal coverage
RECORD = "cdr"             # consolidated data record, not the interim ICDR
VERSION = "v202212"
AREA = [84, -180, -56, 180]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _record(target, req):
    import hashlib, json, datetime
    p = ROOT / "reports" / "cci_provenance.jsonl"
    p.parent.mkdir(exist_ok=True)
    h = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    with p.open("a") as f:
        f.write(json.dumps({"utc": datetime.datetime.now(datetime.UTC).isoformat(),
                            "dataset": DATASET, "target": target.name,
                            "sha256": h, "request": req}) + "\n")


def fetch(years, variables, outdir):
    import cdsapi
    outdir.mkdir(parents=True, exist_ok=True)
    key = _key_from_env()
    c = cdsapi.Client(url=CDS_URL, key=key) if key else cdsapi.Client()
    for y in years:
        target = outdir / f"cci_sm_{y}.zip"
        # Resume by VALIDITY, not existence. A run killed mid-transfer leaves a
        # non-empty but truncated zip, which a size check happily skips -- that is
        # exactly how drought_2006 ended up corrupt (311 MB of a 595 MB file) and
        # silently poisoned the regrid two hours later.
        if target.exists() and target.stat().st_size > 0:
            import zipfile as _zf
            try:
                with _zf.ZipFile(target) as _z:
                    _z.testzip()
                print(f"  {y}  already present, skipping"); continue
            except Exception:
                print(f"  {y}  present but CORRUPT -- refetching")
                target.unlink()
        # The two products need DIFFERENT parameter sets, verified against the CDS
        # constraints endpoint on 2026-08-31:
        #   surface   -> version v202212, type_of_sensor required, day required
        #   root-zone -> version v202505, NO type_of_sensor (not applicable), years 1980-2024
        # Passing a sensor with root-zone returns "not a valid combination", which is what
        # made the first attempt fail on 2002.
        rz = any("root_zone" in v for v in variables)
        # `day: ["01"]` is required even for month_average -- the constraint block lists
        # it explicitly. Omitting it returns "not a valid combination" with no hint.
        req = {"variable": variables, "time_aggregation": "month_average",
               "type_of_record": RECORD, "year": [str(y)], "month": MONTHS,
               "day": ["01"], "area": AREA}
        if rz:
            req["version"] = "v202505"      # root-zone has NO type_of_sensor key at all
        else:
            req["version"] = VERSION
            req["type_of_sensor"] = SENSOR
        print(f"    {y}  requesting {len(variables)} variables ...")
        c.retrieve(DATASET, req, str(target))
        _record(target, req)
        print(f"    {y}  -> {target}  ({target.stat().st_size/1048576:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="2002 only, to validate the request")
    ap.add_argument("--from-year", type=int, default=2002)
    ap.add_argument("--to-year", type=int, default=2018)
    ap.add_argument("--vars", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="subfolder under data/cci_sm")
    a = ap.parse_args()
    years = [2002] if a.probe else list(range(a.from_year, a.to_year + 1))
    variables = a.vars or VARIABLES
    banned = ("grace", "twsa", "mascon", "gtws", "graice", "gravimetry",
              "total_water_storage", "terrestrial_water_storage", "water_storage",
              "tws_", "_tws", "jpl_mascon", "csr_", "gfz_")
    for nm in (DATASET, *variables):
        hit = [b for b in banned if b in nm.lower().replace("-", "_")]
        if hit:
            raise SystemExit(f"REFUSED: {nm!r} matches {hit}. Guidelines section 3 bans "
                             "external GRACE / GRACE-derived TWS at ANY date.")
    print(f"{'PROBE' if a.probe else 'FULL'} PULL: {len(variables)} vars x {len(years)} years")
    outdir = OUT / a.out if a.out else OUT
    fetch(years, variables, outdir)
    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    main()
