# Benchmark development policy - bounded approval

Prepared and explicitly approved by the user on 2026-09-13 after verified NCM
B400 longitudinal validation. Approval covers implementation and synthetic
checks under the policy below, not an experiment launch or final A4 sign-off.

## Approved bounded policy

1. Current research machine, common CUDA learner placement for all six heads;
   cached embeddings resident in host memory, including per-event host-to-device
   transfer in the timed boundary. Exclude file I/O and encoder inference.
   Record hardware/software; fix CPU intra-op and inter-op threads to one in the
   dedicated benchmark process. No CPU-deployment or mixed-device comparison.
2. Full-stream throughput and synchronized per-update latency are separate
   passes. Include prediction, acquisition, registration, updates, replay/EWC
   work and declared transfers. Exclude bootstrap, validation, snapshot writes,
   rolling diagnostic metrics and logging. Synchronize the accelerator only at
   the full region boundaries for throughput. Zero-purchase update metrics are N/A.
3. One discarded full-stream warm-up on a disposable copy per cell, followed by
   five paired measurement blocks. Restore identical bootstrap state/RNG before
   every pass. Rotate [hdc,ncm,ranpac,naive,ewc,replay] left by block index 0..4.
   Apply the same repetition policy to A and B; keep all blocks, medians/ranges
   and paired ratios. Technical repeats are not independent training streams.
4. Verify event decisions, paid registrations and final predictions against the
   corresponding untimed reference. Keep failures/interrupted attempts. Abort
   on integrity failure or known workload exclusions under the current shared-desktop policy;
   any new attempt needs
   fresh user preflight. Do not selectively discard slow successful repetitions.
5. Report prediction-state payload, extra resumable state, replay payload and
   logical/physical checkpoint storage separately. Compressed files are not
   runtime memory. Measure process/device peaks separately with declared overhead,
   not silently inside throughput timing.

Approval covers implementation and synthetic checks only. Any real development
pilot remains user-launched with preflight, exact commands and a fresh output
directory. No full-grid release, test access, tuning, deployment claim or final
protocol freeze is implied. Resource ceilings, verified feasibility, appendix,
practical-benefit criteria and A1-A4 remain separate decisions.

## Implementation checkpoint - 2026-09-13

Implemented in benchmark_core.py, benchmark_environment.py and benchmark_study.py.
The first manual acceptance pilot is ONE frozen-B / seed-0 / Experiment-A / B=25
condition, covering all six learners and all five paired blocks. This is a new
timing workload, not a rerun/replacement of completed A25 predictive evidence.
No real benchmark has been launched by the agent. See [BENCHMARKS.md](BENCHMARKS.md)
for exact boundaries, output layout, review and the one-command launcher.

Implementation details: CUDA global RNG is initialized before reference capture;
all timing passes restore saved bootstrap state/RNG. Bootstrap latency starts
from saved pre-bootstrap state. Final predictions and event decisions must match
the untimed existing-runner reference (discrete exact; score tolerance 1e-6).
Minimal decision-trace bookkeeping is included consistently in stream timing;
rolling metrics, JSON encoding/output, state inventories and scoring are excluded.
One separate pass samples process/CUDA peaks; another exports full/prediction-only
payload at retention times. Freed CUDA allocator cache is cleared before the peak
pass only. Process/native overhead remains shared, not attributed solely to a head.

At this historical checkpoint, workload checks were fail-closed boundary inventories, not a guarantee of absence
of brief/unlisted activity. They reject recognized browsers/games/recording/
overlay apps, other Python workers and unapproved GPU-listed processes. The waiting
Windows venv parent redirector is excluded from the Python-name check but not
from the GPU-compute check. No process is automatically killed. Failed attempts
remain preserved. Live operator preflight is required; avoid deliberate competing
workloads, but ordinary accepted OS activity need not be suppressed.

## Approved desktop-baseline amendment - 2026-09-13

Attempt01 stopped at the initial guard before model construction or timing:
the current driver listed Windows desktop apps in its compute-process query.
The user approved recording explicitly identified Windows desktop/Terminal/
VS Code processes as an accepted shared baseline instead of rejecting them.
This does not approve all C+G processes or treat desktop activity as free.

The exact current-machine executable paths are pinned in
benchmark_environment.DESKTOP_BASELINE_PATHS: Explorer, DWM, ShellHost,
ApplicationFrameHost, the installed Windows SystemApps paths for SearchHost,
TextInputHost, CrossDeviceResume, StartMenuExperienceHost and ShellExperienceHost,
the verified VS Code installation, and the verified Terminal package path
(both observed C: and G: aliases). Each accepted PID/full path is recorded.
Relocation/version changes require review, not automatic discovery or name-only
allowance. These path checks are an operational guard, not code-signature or
malicious-process attestation.

Overwolf/OverwolfBrowser, browsers, recording/overlay apps and other Python
workers remain blocked; unknown GPU-listed processes remain blocked. WebView,
Phone Link, Sticky Notes and Settings were NOT silently added to the September13
baseline. Settings is explicitly added by the September14 amendment below.
No automatic process termination. Keep baseline apps idle and disclose shared-
desktop interference limits in timing conclusions.

The launcher now checks workloads before RUN confirmation and attempt creation,
retains complete native stderr before checking exit code, and uses fresh
attempt02 by default. Attempt01 and its log remain unchanged. No auto-retry,
grid release, test access, or changes to timing/learner semantics are approved.

Verified Windows Search refinement: live parent inspection identified the
remaining GPU WebView as SearchHost-owned. Under the approved Windows desktop
baseline, only the pinned WebView executable version with a verified parent
chain ending at the pinned SearchHost path is accepted. Every intermediate
process must use that same WebView path; creation times reject reused parent
PIDs, cycles/missing records fail closed, and the full chain is recorded.
Generic browser/WebView allowance remains prohibited. This narrowly qualifies
the earlier WebView exclusion; it does not require killing Windows Search.

## Approved shared-desktop amendment - 2026-09-14

The user reported that Windows Settings reappeared automatically during attempt03
and explicitly approved accepting/logging ordinary Windows background activity
instead of requiring a perfectly quiet desktop. Attempt03 stopped after the HDC
latency pass in paired block4 (zero-based block3); preserve its failure and partial
outputs without resuming, reclassifying it as PASS, or combining its timing blocks
with a later attempt. Its guard failure is not evidence of measured interference.

The existing identified OS desktop baseline now also accepts the observed exact
path C:\Windows\ImmersiveControlPanel\SystemSettings.exe, regardless of PID changes.
All accepted GPU-listed PID/path entries remain logged at every boundary, and
ordinary activity from these processes is allowed to recur. No blanket Windows
directory, C+G, name-only or generic WebView exemption is introduced. Games,
recording/overlays, browsers, other training jobs and unapproved GPU entries
remain blocked; these are conservative operational exclusions, not measurements
of interference. No process is killed, suspended or prevented from restarting.

Report all resulting timings as shared-desktop measurements, not isolated-machine
or zero-interference benchmarks. Inventories explicitly record interference_measured
as false. Keep all five paired blocks, their medians/ranges and paired ratios;
do not discard slow successful repetitions or explain differences as OS effects
without measurements. The same baseline applies to every learner. Existing timing
boundaries, labels, learner updates and fidelity requirements are unchanged.

This is a dated development-policy amendment, not a retrospective change to
attempt02/03 receipts, outcome expectations, A4 or test-release approval. New
attempt04 receives new policy/code hashes and remains manually user-launched.

## Runtime decision and next feasibility condition - 2026-09-14

After compact A25 attempt04 acceptance, the user answered the remaining-runtime
question with "as long as is needed": no user-imposed time cap. Do not treat this
as unlimited storage, agent-run permission, grid/test release or a new scientific
endpoint. The next prepared manual feasibility condition uses existing policy:
frozen-B/seed0/Experiment-B/allowance400/threshold0.10,all six heads,five paired
blocks. It is a new timing/resource workload, not replacement of completed NCM
B400 longitudinal training/validation. Actual labels consumed remain method-specific.
This single high-budget condition does not prove a worst-case resource bound.
The shared-desktop and8GiB prelaunch headroom guards remain. Failed/completed
attempts are preserved, with no auto-retry/resume or selective timing omissions.

## Background-warning amendment - 2026-09-14 (current policy)

The user reported another prelaunch block on GitHub Desktop, Spotify's Game Bar
WebView and Edge Game Assist, and requested a practical fix for recurring Windows
background apps. This amendment supersedes the unknown-GPU abort rules above
for new development attempts; it does not retrospectively change old receipts.

- Known browser, game, recording/overlay and Python workload names in
  benchmark_environment.BLOCKED_NAMES still block, whether found by tasklist
  or in the GPU query. The waiting venv parent exception applies only to tasklist.
- Other GPU-listed PID/path entries are retained as unclassified warnings,
  not automatically approved baseline processes or evidence that they are idle.
  Their appearance, disappearance or version/path changes do not abort a run.
- Exact-path identified desktop entries and verified Search WebView ownership
  retain their separate baseline category. Unverified/unreadable optional Search
  ownership becomes a recorded classification warning; malformed/unreadable
  primary tasklist or GPU inventories still abort. No process is killed.
- Read-only preflight returns PASS_WITH_WARNINGS and exit0 when only warnings
  occur. Every in-run boundary records warnings. summary.json and the integrity
  receipt carry environment_qualification, warning boundary counts and entries,
  including transient warnings absent at the final boundary. Integrity PASS
  does not mean clean timing. Warning-bearing measurements are explicitly
  shared_desktop_with_unclassified_activity, not isolated-machine evidence.
- Keep all five paired blocks, medians/ranges and ratios. Do not selectively
  discard warning-bearing/slow blocks or attribute differences to Windows
  without measurements. Neither warning-free nor warning-bearing snapshots
  establish zero interference, or detect all activity between boundaries.

Timing boundaries, acquisition, learner updates, storage, fidelity and input
hash checks are unchanged. This is a development-policy amendment, not A4,
full-grid/test release or agent permission to launch an experiment. Preserve
preflight_run_d2ad9f4836c24975b2793339284290fa.log; that preflight created no attempt.
