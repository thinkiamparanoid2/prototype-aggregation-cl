# Shared-desktop Settings amendment — review, 2026-09-14

User approved accepting/logging ordinary identified Windows background activity,
including Settings recurrence, while blocking competing applications and keeping
shared-desktop measurement limitations and variability explicit. Implementation
adds the exact observed Settings path, not a blanket Windows directory allowance.
See [approved policy](../BENCHMARK_DEVELOPMENT_PLAN.md).

Independent review baseline:02986082a522d0cb61832563bbcdf95452041e6b, scoped to
benchmark_environment.py, scripts/benchmark-a25.ps1, BENCHMARK_DEVELOPMENT_PLAN.md
and the new test_desktop_settings_policy.py. Other existing changes were excluded.

## Standards

No documented standards violations found. Exact Settings-path allowance, explicit
shared-desktop limitations, retained competing-workload blocks and preservation
of attempt03 align with AGENTS.md and the approved amendment. Tests cover Settings
recurrence, path normalization and continued rejection of unrelated GPU work and
blocked applications. No material code-smell findings warranting refactoring.

## Spec

No findings. Path-based membership accepts Settings regardless of PID changes;
accepted entries retain observed PID/path, and inventories explicitly record
interference_measured=false. Tests cover disappearance/reappearance, normalized
paths, impostors, unknown GPU work and blocked applications alongside Settings.
Launcher/policy consistently permit recurring accepted desktop activity. No
process termination, timing-boundary/learner changes or rewriting of attempts.
Attempt03 preservation, all five timing blocks and variability remain explicit.

Summary:Standards0 findings; Spec0 findings; no worst issue in either axis.
Both reviewers performed read-only inspection only; the main agent ran synthetic
tests. No real benchmark was launched by the agent.

Verification:446 CPU tests plus6 subtests passed in128.75s; diff whitespace check
passed. Live read-only workload preflight reported PASS with no blocked/other GPU
entries; attempt04 input-only plan also passed. These checks do not guarantee
future machine state or zero interference; the launcher checks again.
