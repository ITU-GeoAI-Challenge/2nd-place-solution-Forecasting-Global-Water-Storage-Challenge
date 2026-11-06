# Host ruling — external land-surface-model data (Zindi discussion forum, 2026-09-04)

Recorded verbatim because it is **not in `reports/important_guidelines.txt`**. It is the
sole authority for §3.129's storage-state block and for any future GLDAS work, and the
report must cite it rather than the written guidelines alone.

Relayed to this project by Himanshu on 2026-09-09.

---

**Question from a team** — 4 Sep 2026, 14:38

> Quick clarification: is NASA GLDAS-2.1 Noah Early Product allowed as an external dataset
> if we only use source months ≤ t?
>
> We would not use GLDAS-2.2 or anything GRACE-assimilated / GRACE-derived.
>
> If GLDAS-2.1 is allowed, does that cover both:
>
> - the raw component variables — soil moisture layers, snow water, canopy water; and
> - a simple sum of those components used as a modelled water-storage proxy?

**Answer by host**

> Noah variables and their combinations can be used. We would allow non-GRACE variables and
> deterministic combinations using data up to month t, while continuing to prohibit
> GLDAS-2.2 GRACE-DA and and all GRACE-derived products.

---

## What it changes

| | before | after |
|---|---|---|
| §3.68 standing decision | no land-surface-model TWS, even where the letter allows | **void** — the grey area the decision rested on is gone |
| self-imposed rule (`era5_fetch.py`) | never sum soil water + snow + lakes into a storage proxy | **void** — "deterministic combinations" named and permitted |
| GLDAS-2.1 Noah | not used | permitted, source months ≤ t |

## What it does NOT change

- GLDAS-**2.2** (GRACE-DA) remains prohibited.
- All GRACE-derived products remain prohibited at any date — GDO TWSA, JPL/CSR/GFZ
  mascons, GTWS-MLrec, GRAiCE. The refusal guard in `era5_fetch.py` stays as written.
- The ≤ t date discipline is unchanged and still absolute; §3.108's `leak_test.py` remains
  the standard of proof.
- For the 2002–2018 window use the **standard GLDAS-2.1 product, not the Early Product**,
  and document it as the retrospective proxy exactly as §3.49(b) requires for ERA5-Land.

## Measured consequence

§3.129: the ruling was tested immediately on data already held. Keeping components
separate beat summing them (−0.00738 vs −0.00661), so the permission to *combine* was not
in the end the binding constraint — the encoding was. Recorded so the report does not
overstate what the ruling bought.
