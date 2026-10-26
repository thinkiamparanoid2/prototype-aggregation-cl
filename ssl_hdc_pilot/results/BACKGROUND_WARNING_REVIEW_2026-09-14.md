# Background-warning guard amendment - 2026-09-14

User-requested implementation after B400 preflight rejected automatically
recurring GitHub Desktop, Spotify Game Bar WebView and Edge Game Assist.
The preserved preflight log is
benchmark_development/preflight_run_d2ad9f4836c24975b2793339284290fa.log
(30,986 bytes). It reported no attempt created; B400 attempt01 was independently
confirmed absent during this change. No real experiment was launched by the agent.

## Change and claim boundaries

Unknown GPU-listed entries warn rather than abort. They remain unclassified,
not certified idle or promoted to the exact-path identified desktop baseline.
Known workload names still block through either primary inventory; malformed or
unreadable primary inventories still fail. Optional Search ownership failures
produce warnings, not fabricated ownership. Preflight PASS_WITH_WARNINGS exits0.
All in-run warning boundaries are retained in the summary and integrity receipt,
including transient entries absent at the last check. Integrity PASS does not
establish timing cleanliness. All five paired blocks remain; no isolated-machine
claim, selective block removal, process termination or completed-artifact edits.

## Verification

The synthetic exact-three-app regression failed before the change with the same
competing-workloads exception as the user. After the change, 63 targeted tests
passed in4.25s, including recurring PID/path warnings, known GPU workload names,
parent Python, malformed inventories, Search ownership and warning qualification.
An earlier broader targeted invocation had63 passes and one temporary-directory
ACL setup error, not a guard assertion failure. The repository full-suite wrapper
uses a unique workspace temporary directory and disables CUDA.

Full CPU suite:475 tests and6 subtests passed in144.61s. The first full-suite
session lost its completion result on interruption; no Python test process
remained, so the same CPU-only wrapper was repeated. No success is inferred from
the interrupted session.

Live read-only preflight:PASS_WITH_WARNINGS, exit0; no known blocked processes or
GPU entries. PhoneExperienceHost.exe (PID14180, installed YourPhone package) was
retained as an unclassified warning. The original three apps were not present
in this live inventory; their exact paths are covered by synthetic regressions.
An initial sandbox-only attempt could not read tasklist; the authorized read-only
retry outside that sandbox succeeded. No benchmark attempt was created.
The B400 read-only input plan and git diff --check passed.

Preserved original preflight log SHA256:
C5B4D310E93FFD7BD9D121BE8B4DF688819E19078F178C607C95C7661F485746.

## Standards

Independent read-only review against HEAD52c1ee4704afdf594a6dcf15a291f430772948f1
and the current working-tree changes found no hard violations or material smell
findings. Manual experiment ownership, artifact preservation, provenance and
provisional claim boundaries remain intact. The legacy other_gpu_compute key is
retained for compatibility; it now contains known excluded GPU entries only.

## Spec

Independent review found no spec defects or scope creep. Warning classification,
known exclusions, primary inventory failures, persistent qualification and block
retention match the dated development amendment. One nonblocking test-coverage
suggestion remains: aggregation and synthetic report propagation are tested
separately, not via a fully mocked warning-bearing non-smoke run_benchmark.
Inspection confirmed the wiring; no real experiment was used as a software test.

Summary: Standards0 findings; Spec0 defects,1 nonblocking coverage suggestion.
