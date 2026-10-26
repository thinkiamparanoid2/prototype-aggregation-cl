# Shared-projection production integration — 2026-09-14

User approved integrating the successful lossless prototype into future saves
and loads. Existing completed artifacts were not rewritten or rerun. New budget
longitudinal and benchmark runs share projections inside the run directory;
legacy default APIs and logical learner schema1 remain supported. See
[checkpoint guide](../LEARNER_CHECKPOINTS.md) for format and backup requirements.

Verification: 434 CPU tests and six subtests passed in133.93s through
`scripts/hdc.ps1 -Action test`; `git diff --check` passed. This includes six-head
continued-update round trips, synthetic A/B capture and validation, five-block
synthetic benchmarks, corrupt/missing blobs, exact receipt dependency closure,
copied bundles, interrupted writes, overwrite races and concurrent publication.
The Windows extended-path regression is covered explicitly. A separate focused
storage/snapshot-validation run passed62 tests in41.24s.

The attempt03 launcher read-only plan passed: 2420 bootstrap labels,976 stream
events,A25,seed0,all six heads,five paired blocks. No real learner was constructed,
trained, scored or timed by the plan. No attempt03 existed at verification.
Real production-layout storage and end-to-end acceptance remain user-owned.
The earlier90.92% saving remains a prototype result, not a measured new pilot.

## Standards

The independent review reported no blocking scientific/integrity defect in its
scoped inspection. It identified stale checkpoint-format documentation; that
documentation has been corrected. It did not deliver a completed final review
because the sub-agent hit its usage limit. The main agent checked the resulting
storage/API diff and completed synthetic verification; no full independent
review sign-off is claimed.

## Spec

The independent review reported no concrete defect found so far against the
approved shared-storage, legacy-compatibility and artifact-preservation scope.
It requested explicit receipt missing/unlisted/extra-blob rejection tests and
benchmark dependency-byte assertions; these were added and passed. It also
identified the stale guide, now corrected. Its final review was interrupted by
the usage limit, so this is partial independent coverage, not a completed sign-off.

Review baseline: scientific code/test changes against
`02986082a522d0cb61832563bbcdf95452041e6b`, including the new storage module/tests.
Unrelated prior documentation changes were preserved. No Git publication or
commit was performed. A1–A4/full-grid/untouched-test approval remains open.
