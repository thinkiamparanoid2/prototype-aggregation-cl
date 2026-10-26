# B400 feasibility launcher review — 2026-09-14

Scope: one manually launched frozen-B/seed0/Experiment-B/allowance400/threshold0.10
benchmark, six learners/five paired blocks. No runtime cap, but existing storage,
preflight, artifact-preservation and full-grid/test-release boundaries remain.
This is a new timing workload, not replacement of completed NCM B400 validation.

Review baseline:02986082a522d0cb61832563bbcdf95452041e6b to user commit
52c1ee4704afdf594a6dcf15a291f430772948f1. Scoped to new benchmark-b400.ps1 and the
B400 launcher-test parametrization; prior shared-storage changes were excluded.
Reviewers ran no tests or experiments and made no edits.

## Standards

No documented standards violations. The launcher follows the local-environment,
user-owned execution, preflight and preservation rules. It retains8GiB headroom,
explicit confirmation, the approved desktop baseline, failure preservation and
fresh preflight. Tests cover guards, log binding, confirmation and condition args.

One nonblocking smell judgment: Invoke-LoggedCommand duplicates the proven A25
stderr/exit handling. This creates future maintenance risk, but retaining that
implementation is proportionate to this bounded handoff; extraction is not
required here and would broaden changes to the completed-run launcher.

## Spec

No findings. Fixed condition and inherited six-learner/five-block engine match
the prepared feasibility policy. Fresh-attempt refusal,8GiB guard, logged preflight,
RUN BENCH B400 and manual execution are preserved. The Python CLI still defaults
to a read-only plan; the PowerShell launcher requires confirmation for execution.
Tests assert seed/experiment/allowance/threshold/output identity and guards.
No missing requirements, incorrect behavior or scope expansion identified.

Summary:Standards0 violations/1 nonblocking duplication judgment; Spec0 findings.
Targeted verification:14 launcher tests passed in9.47s; real-input PlanOnly passed
without model construction/training/scoring/timing. Full CPU suite453 tests plus6
subtests passed in208.14s; diff whitespace check passed. Initial approval-service
usage limit blocked one request; unchanged retry was accepted and completed.
No official experiment launched by the agent.
