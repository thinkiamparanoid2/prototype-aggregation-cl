# A3 feasibility — comparative fixed-population forgetting grid

Prepared 2026-09-15. This is a measured planning note, not authorization or a
run receipt. No capture, scoring, benchmark, or test access was launched by the
agent.

## Measured completed reference cell

The protected frozen-B/NCM/seed-0/B400 capture
`budget_longitudinal_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01`
contains 3,148,340 bytes and 20 checkpoint NPZ files. Its matching validation
directory contains 3,593,399 bytes. Combined stored output is 6,741,739 bytes
(6.43 MiB). The capture preflight log began at 2026-09-12 03:00:34 UTC and the
recovery log was last written at 03:10:20 UTC: about 585.9 seconds (9.77 min)
for the capture. The matched validation directory was last written at 03:28:12
UTC, so capture through validation completion was about 1,657.5 seconds (27.6
min), inclusive of the gap between manual stages. This is operational elapsed
time, not a throughput measurement.

At preparation, G: free space was 72,431,595,520 bytes (67.46 GiB). The
launcher will independently enforce the established 8 GiB prelaunch headroom.

## Projection and decision

There are 36 new cells: six learners × two arms × three seeds. The prior
frozen/NCM/seed-0 capture is retained as archived audited evidence, but this
slot is deliberately recaptured under current code so every comparative cell
has one uniform code-provenance set. A straight 36× projection from the single
NCM reference is 242,702,604 bytes (231.5 MiB). This projection is deliberately
superseded by the conservative note below.

The reference is unusually small because NCM has no 768×10,000 float32
projection. HDC carries that projection (about 29.3 MiB before checkpoint
metadata); the RanPAC-style head has similarly large projection state. The
36× NCM-only byte estimate is therefore not a safe upper bound. Even a
conservative 36× 64 MiB working allocation is 2.25 GiB, leaving more than the
8 GiB guard on the measured drive. The expected grid fits the presently measured
free space, but cost differs materially by learner and outputs must be rechecked
before every user launch.

The matched capture-to-validation operational projection is about 16.6 hours
(36 × 27.6 min). It extrapolates one NCM cell and manual-stage gaps; it is a
scheduling estimate only. No user runtime ceiling is in force.

**A3 feasibility verdict: FITS for the 36-cell development grid at the current
free-space reading, subject to the per-launch 8 GiB guard.** No seed or encoder
arm is reduced. If live capacity later fails, the specified reduction order is
seed 2, then seed 1; neither encoder arm may be removed.

## Required measured cell-1 addendum before cell 2

Cell 1 is frozen/HDC/seed-0. After its PASS capture and validation, measure both
directories and inspect the capture for a single `projection_blobs` dependency
instead of per-snapshot projections. Append the measured bytes, blob layout and
new projection for the remaining 35 cells to this section. If that projection
exceeds live available space minus the 8 GiB guard, stop and report; propose only
seed reductions, never removal of an encoder arm. The present 64 MiB allocation
is explicitly unvalidated until this measurement exists.

## Measured cell-1 addendum

Measured from 2026-09-15_grid_b400_frozen_hdc_s0_t010_attempt01: capture=35985555 bytes; validation=3594465 bytes; shared content-addressed projection blob present.
A simple remaining-35 projection is 1385300700 bytes. Even the stated worst-case projection growth (~6.7 GiB across the 12 projection-bearing cells) remains within the measured drive envelope after the 8 GiB per-cell guard; the guard continues to protect the tail.
