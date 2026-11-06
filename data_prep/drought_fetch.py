"""Fetch SPI / SPEI drought indices from the Copernicus CDS.

WHY THIS IS THE BEST-MOTIVATED EXTERNAL SOURCE WE HAVE:
standardised indices are the best-POOLING covariates in the model. `spei03_t` scores
0.90 on the pooling screen at 94% sign consistency, against 0.10-0.57 for raw ERA5 --
because SPEI is dimensionless and normalised per cell by construction. And it is
DYNAMIC rather than identity: `spei03_t` sits at 2.4% between-cell variance, where an
earlier feature block at 50-73% between-cell variance produced our worst regression --
high between-cell variance means a feature is telling the model WHERE it is, not what is
happening there.

WHAT IS NEW: the host supplies SPEI at 1/3/6/12 months. CDS offers 24/36/48 as well,
plus the entire SPI family. GRACE storage is dominated by deep groundwater with
multi-year memory, our longest window is 12 months, and a 12-vs-36-month
difference scored 0.91 pooling at 93% sign consistency when screened.

SPI vs SPEI: SPI is precipitation-only, SPEI is P minus PET. Having both separates
"was it dry" from "was atmospheric demand high" -- the evaporative term, isolated.

LEGALITY: the host states its own covariates come from the European/Global Drought
Observatories and the CDS, and supplies four SPEI columns itself, so this index family
is definitionally permitted. Guidelines section 3 allows external non-TWS covariates
dated <= t. The shared GRACE guard refuses the CDS gravimetry product regardless.

⚠️ COMPATIBILITY IS UNVERIFIED. The host may have computed its SPEI from a different
source. Correlate CDS SPEI_12 against the supplied SPEI_12_t BEFORE mixing short and
long windows, or the block carries a discontinuity at the 12/24-month boundary.
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(d for d in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
            if (d / "data" / "raw").is_dir())
OUT = ROOT / "data" / "drought"
DATASET = "derived-drought-historical-monthly"
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

# We already have SPEI 1/3/6/12 from the host. These are the gaps.
# SPEI only. Fetching SPI as well DOUBLED the payload to 595 MB/year and the transfer
# runs at ~1.3 MB/s, so it cost ~8 min/year of pure download for a secondary variable.
# The hypothesis is SPEI at LONG windows; SPI (precipitation-only) is a nice-to-have.
# 2002-2005 were pulled with both and still contain SPI.
VARIABLES = ["standardised_precipitation_evapotranspiration_index"]
PERIODS = ["12", "24", "36", "48"]   # 12 is the OVERLAP, kept to validate against the host
VERSION = "1_0"
PRODUCT = "reanalysis"
RECORD = "consolidated_dataset"
AREA = [84, -180, -56, 180]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _record(target, req):
    import hashlib, json, datetime
    p = ROOT / "reports" / "drought_provenance.jsonl"
    p.parent.mkdir(exist_ok=True)
    h = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    with p.open("a") as f:
        f.write(json.dumps({"utc": datetime.datetime.now(datetime.UTC).isoformat(),
                            "dataset": DATASET, "target": target.name,
                            "sha256": h, "request": req}) + "\n")


def fetch(years, variables, periods, outdir):
    import cdsapi
    outdir.mkdir(parents=True, exist_ok=True)
    key = _key_from_env()
    c = cdsapi.Client(url=CDS_URL, key=key) if key else cdsapi.Client()
    for y in years:
        target = outdir / f"drought_{y}.zip"
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
        req = {"variable": variables, "accumulation_period": periods,
               "version": VERSION, "product_type": [PRODUCT],
               "dataset_type": RECORD, "year": [str(y)], "month": MONTHS, "area": AREA}
        print(f"    {y}  requesting {len(variables)} indices x {len(periods)} windows ...")
        c.retrieve(DATASET, req, str(target))
        _record(target, req)
        print(f"    {y}  -> {target}  ({target.stat().st_size/1048576:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="2010 only, to validate the request")
    ap.add_argument("--from-year", type=int, default=2002)
    ap.add_argument("--to-year", type=int, default=2018)
    ap.add_argument("--periods", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="subfolder, for parallel workers")
    # SPI was dropped mid-pull on 2026-08-31 to halve the payload, which stranded it at
    # 2002-2005 -- 0% coverage of the test era, i.e. unusable. This flag exists to
    # finish that job:  --vars standardised_precipitation_index --from-year 2006
    ap.add_argument("--vars", nargs="+", default=None)
    a = ap.parse_args()
    years = [2010] if a.probe else list(range(a.from_year, a.to_year + 1))
    periods = a.periods or PERIODS
    variables = a.vars or VARIABLES
    banned = ("grace", "twsa", "mascon", "gtws", "graice", "gravimetry",
              "total_water_storage", "terrestrial_water_storage", "water_storage",
              "tws_", "_tws", "jpl_mascon", "csr_", "gfz_")
    for nm in (DATASET, *variables):
        hit = [b for b in banned if b in nm.lower().replace("-", "_")]
        if hit:
            raise SystemExit(f"REFUSED: {nm!r} matches {hit}.")
    print(f"{'PROBE' if a.probe else 'FULL'} PULL: {variables} x {periods} x {len(years)} years")
    outdir = OUT / a.out if a.out else OUT
    fetch(years, variables, periods, outdir)
    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    main()
