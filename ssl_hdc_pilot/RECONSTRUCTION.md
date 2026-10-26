# Historical reconstruction: manual-run handoff

Status: **all 108 user-run historical cells PASS; independently audited 2026-09-11.**
TODO 1.2-1.4 historical reconstruction is complete, not protocol freeze.
See [the audit](results/RECONSTRUCTION_AUDIT_2026-09-11.md) for coverage,
artifact locations, hash checks and limits. No completed block should be rerun.

## What the command does

`python -m ssl_hdc_pilot.scripts.reconstruct_run` defaults to a read-only plan:
validate the source summary/history, registered cell, training provenance and
input hashes. It does not construct a learner, evaluate predictions, or write
run artifacts. Only explicit `--execute --preflight-confirmed` runs one cell.
All experiments, including reconstruction, are manually launched by the user.

Execution reuses the registered runner's bootstrap, stream ordering, acquisition,
registration, and update rules. It does not train the encoder or change budgets.
Only the approved training partition supplies labels. Official non-training
paths are rejected before opening inputs. The locked specification fixes class
selection; holdback membership is rechecked using training rows only. The old
generator's validation/test rarity-rederivation helper is not called.
The existing label-free trusted-all embedding cache is reused, but only training
IDs are passed to learning and final prediction. No held-out scoring occurs.

HDC/naive/replay retain the original spec hash `30447621...`. Replacing the
current raw spec's single `ewc_decay: 0.9` value with `1.0` reproduces the exact
full historical SHA-256. This recovery is permitted only for these three
unaffected learners and only after exact hash verification. The canonical
specification and historical artifacts are never rewritten. Other mismatches
fail closed; no hash override is provided.

## Outputs and acceptance

Use a fresh, dedicated output directory; even an existing empty directory is
refused. The command creates:

- `reconstruction_manifest.json`: source/current input hashes, ordered event
  schedule hash, complete registered configuration, environment, draft protocol
  hash, Git revision/dirty tracked-patch hash, and working-code file hashes
  (including the untracked implementation files).
- `source_stream_spec.json`: exact effective historical spec bytes.
- `<run_tag>_final_state.npz`: learner-resumable state, not whole-stream resume.
- `<run_tag>_history.csv` and `<run_tag>_summary.json`: reconstructed outputs.
- `final_training_predictions.json`: ID-aligned final bootstrap-plus-stream
  predictions sufficient to rebuild the final predictive summary.
- `fidelity_report.json`: comparisons, input/code stability checks and hashes
  linking the manifest, state, effective spec and prediction artifacts.

Acceptance requires `status: PASS`, zero history/summary mismatches, and no
input/code changes. Discrete values, sample order, prediction identities, label
decisions, charges, and update counts are exact. Finite floating values use
the draft tolerance `atol=1e-6, rtol=0`. Only history update durations, summary
update-time mean/median/p95/total, and relocated path fields are excluded.
Earlier missing `learner_configuration` is normalized only when its current
value is empty. No other missing scientific field is ignored.

PASS means observed-output fidelity, not proof that current input files are
identical to the originals: old summaries lack embedding/split content hashes.
Current hashes pin this reconstruction's inputs. It does not establish
cross-device equivalence, held-out generalization, or a speed/retention claim.
The tolerance remains proposed until validated; never loosen it to obtain PASS.

A FAIL report, exception, or missing report blocks acceptance. Partial outputs
may remain after failure; preserve them for diagnosis and do not overwrite or
delete them automatically. Use a reviewed fresh attempt directory if needed.
NPZ states remain Git-ignored and require separate backup. These directories
are non-overwriting by convention/tooling, not filesystem write-protected.

## Next manual action

Final update 2026-09-11: all 108 cells passed and their artifacts were verified.
There is no further reconstruction run to launch. The next decision concerns
Experiment A's proposed budget/shared-label schedule before implementation.
All previous next-action notes below are completed historical milestones.

Latest update 2026-09-11: all 12 representative checks now PASS. The remaining
96 historical cells have a new user-launch block at the end of RUN_COMMANDS.md,
excluding all seed-0/threshold-0.10 cells. Its syntax and read-only artifact/
input/count/disk guards were checked without launching any experiment.
The 11-cell batch's transcript ended before confirmation; the late cleanup
error does not invalidate its verified report/state artifacts. The reason for
early transcript termination is unresolved. Direct preflight/per-cell logs in
the next block avoid transcript-session dependence; completed runs stay intact.
The earlier next-step notes below record completed milestones, not rerun requests.

Update 2026-09-11: the first frozen-B/NCM cell below was manually run by the user
and passed. All six report-listed artifact hashes and current input/code hashes
were independently checked; final prediction rows total 3,396. There are no
history/summary mismatches over 976 stream steps. The next block is now
**Manual reconstruction remaining representative cells - 2026-09-11** at the
end of RUN_COMMANDS.md: 11 new cells, excluding the passed NCM cell. Stop after
these for review; this is not authorization of all 108 cells or test release.
The first transcript contains browser/overlay processes before confirmation;
it is not evidence of uncontaminated timing. No benchmark interpretation follows.

The completed first-cell procedure is retained below for provenance.

Use the new **Manual reconstruction first cell - 2026-09-10** block at the end
of [RUN_COMMANDS.md](RUN_COMMANDS.md#manual-reconstruction-first-cell---2026-09-10).
It targets frozen-B / NCM / stream seed 0 / threshold 0.10 only, not a grid.
Review contamination, close competing GPU/browser/video/game workloads yourself,
and retain the transcript. A preflight flag records your confirmation; it does
not automatically prove the machine was uncontaminated.

Return `fidelity_report.json` and the preflight transcript, or the full exception.
Pause there for review before expanding to the proposed learner/arm checks and
then all 108 cells. A1-A4 remain open; no budget grid or held-out release is
authorized by a reconstruction PASS.

## Implementation review

Final CPU suite: **114 passed, six subtests passed in 31.53 seconds** via
`scripts/hdc.ps1 -Action test` on 2026-09-11. Includes all six learner paths,
training-only filesystem guards, exact discrete comparisons, tolerance handling,
non-overwrite behavior, and rebuilding final predictive metrics from the ledger.
The manual PowerShell block passed syntax validation; it was not executed.
The real first-cell read-only metadata/hash plan passed without learner execution.

### Standards

Two blocking findings were fixed: reject non-training paths before reading;
avoid a helper that rereads held-out labels. Recheck found no remaining blockers.
The duplicated summary envelope remains a minor maintenance concern; fidelity
tests detect drift. No learner algorithm was refactored.

### Spec

Two gaps were closed: final ID-aligned prediction records and the full provenance
record/linkage. Recheck found no remaining infrastructure blockers. Official
historical fidelity is still pending, not inferred from synthetic tests.

Review summary: Standards 0 remaining blockers (one maintenance observation);
Spec 0 remaining infrastructure blockers. No official run was launched by the agent.
