# Shared-projection storage test - 2026-09-14

PASS: lossless saved-array storage prototype. NOT production checkpoint-loader
integration, predictive evaluation, or approval to rewrite existing artifacts.

User authorized testing deduplication and reported increased disk space.
G: available at start: 76,340,199,424 bytes (71.0974 GiB); after retained scratch
copies, about 71.0556 GiB. No raw data/checkpoints/results were deleted or moved.

## Test and preservation

Source: accepted A25 benchmark attempt02. Verified all 251 receipt-listed hashes
before the test, then compared all 252 source files (including receipt) before
and after: identical file inventory and SHA-256 values.

For each of the 46 HDC/RanPAC NPZs containing projection.npy:

1. Read the original ZIP members without pickle or learner loading.
2. Hash the complete projection.npy payload, including dtype/shape header.
3. Save each unique projection once in a compressed content-addressed blob;
   write all remaining members to a fresh compressed delta NPZ and a JSON
   reference with source-file/member hashes.
4. Reload the written reference, delta and shared blob. Verify member inventory
   and hashes; rebuild an ordinary NPZ in memory.
5. Independently load original/rebuilt arrays with NumPy and compare every name,
   dtype, shape and complete array byte sequence. Metadata/RNG bytes included.

All 46 checkpoints used one byte-identical projection blob. All 1,066 array/
metadata entries compared exactly. Ninety-two negative checks rejected altered
projection payloads or missing delta inventories (two per checkpoint). These are
bounded integrity checks, not an exhaustive production fault/security test suite.
Different learning operations still distinguish HDC and RanPAC despite sharing
these projection values. No inference, update, learner loading or experiment ran.

## Measured footprint

| Quantity | Bytes |
| --- | ---: |
| Original whole pilot | 1,408,083,356 |
| Original 46 NPZs replaced in the hypothetical layout | 1,324,852,310 |
| Written compact blobs + deltas + reference JSONs | 44,559,948 |
| Other original artifacts, counted unchanged | 83,231,046 |
| Equivalent compact whole-pilot footprint | 127,790,994 |

Equivalent footprint: 0.1190 GiB (121.87 MiB), versus 1.3114 GiB originally:
**90.9245% less storage**. This is a layout comparison, not freed disk space:
originals remain intact, with about 42.50 MiB of compact scratch copies added.
The 15,876-byte diagnostic report is excluded from the layout comparison.
No filesystem allocation, speed advantage or runtime-memory reduction is claimed.

Applying the same explicitly conditional 144-group benchmark scaling gives
17.14 GiB instead of 188.84 GiB. Assumes pilot-like other state/output sizes and
one shared projection per six-head group; it does not rely on sharing across
groups. Higher budgets, different seeds/arms, validation/uptake outputs, backups,
failures and working space still need estimates. This does not prove 71 GiB is
enough for the complete thesis or close A3/runtime ceilings.

## Evidence and next boundary

Scratch evidence: .cache/PROTOTYPE_projection_storage_20260914_attempt01/report.json
(repository-relative; local/Git-ignored), SHA-256
`d9e472a04c55d62358cacf371c45fd5bb78266ba09132ecbc23766a33a770047`.
It lists the transformed files and hashes of all compact test artifacts.
Storage conversion/round-trip checks took 35.75 seconds, excluding initial source
hashing. This is a diagnostic duration, not a scientific runtime benchmark.

The throwaway program was removed after recording the answer; scratch evidence
is retained. Program SHA-256 was
`ff0bc2a557b26175b75838ae5fa6a17410cc24715cb29cdd261087bd4da594e3`.

Next: implement and test a versioned production save/load interface with
hash-checked shared blobs, legacy compatibility, missing/corrupt-blob failures,
atomic writes, relocation/backup completeness and synthetic prediction/update
round trips. Current production code still writes self-contained NPZs.
Rebuilt ZIP container bytes/file hashes can differ even when every member is
identical; never replace old artifacts or reuse their file-hash receipts.
No official run launched; no frozen code/policy/input or scientific claim changed.
