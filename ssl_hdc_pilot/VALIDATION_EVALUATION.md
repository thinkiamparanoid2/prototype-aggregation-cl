# Saved-state validation evaluation — development only

## Current checkpoint

The user completed frozen-B/NCM/seed-0/B=25 Experiment-A development training.
Independent read-only audit confirmed all seven receipt-listed artifact hashes,
976 history rows, 25 purchased/scheduled labels, 2,445 supplied learner labels,
and 3,396 unique final training-prediction IDs. No training was rerun.

`scripts/evaluate_validation.py` now implements the final-state portion of TODO
2A.3. It loads that saved learner and evaluates the fixed approved validation
population. It never trains, acquires labels, chooses checkpoints, fits calibration,
or opens the trusted test split. Real scoring remains manually user-launched.
The default CLI action only verifies inputs and prepares the population; it
does not construct a learner or make predictions.

## Metrics and guards

- Canonical Option-A train/validation paths and provenance; 3,396/1,132 rows,
  identical 19-class universes, disjoint IDs, complete label-free embeddings.
  The partition reader does not load the full trusted source or test partition.
- Require a complete PASS Experiment-A receipt, verify all seven source hashes,
  source input identity and code compatibility, checkpoint/manifest agreement,
  and fixed validation IDs outside checkpoint training IDs.
- Keep sorted full 19-class denominators, including unregistered classes, with
  existing common Macro-F1, macro recall, per-class recall, stage recall and
  15-bin ECE semantics. Confusion matrix uses that same fixed class order.
- ECE uses the existing learner's softmax top probability over registered
  classes, not acquisition margin; no temperature fitting or padding/recalibration.
- Load on the source's requested device, preserving caller RNG. Fingerprint
  learner weights, registries, EWC/replay state and RNG around prediction; restore
  neural training modes. Reject detected state/RNG changes.
- Pin source artifacts, split/embedding inputs, code and draft protocol before
  evaluation; recheck after preparation/scoring/writes. Do not rewrite old
  provenance or completed artifacts. Compatibility failures need review, not
  automatic reruns or edits to historical receipts.
- Fresh output only: predictions.json, metrics.json, manifest.json and
  integrity_report.json. Missing report/exception is failure, not accepted evidence.

## Scope boundary and next action

The user completed the first saved-state validation run: 1,132 examples,
19 classes, frozen-B/NCM/seed-0/B=25. Independent audit verified all three artifact
hashes and rebuilt common classification metrics/ECE from the 1,132 unique saved
prediction IDs: Macro-F1 0.6600447531, macro recall 0.7225543250, accuracy
0.8356890459, ECE 0.7468861113. The receipt reports unchanged learner state/RNG.
No learner was rerun by this audit. The A25 validation block is complete; do not
rerun it. Scores are results, not permission to tune this cell until it wins.

TODO 2A.3 remains open for coverage and acceptance across budget conditions;
one development cell is verified. This is not the full grid, AULC analysis, longitudinal
snapshots, fixed-population forgetting/uptake, benchmark, or test release.
Validation was previously used in encoder selection, so this is development
evidence conditional on selected representations, not untouched generalization.

## Review

Verification: 22 initial focused tests passed; after the provenance refinement
and additional regression, the full wrapper suite passed 178 tests and six
subtests in 42.10 seconds. The first sandboxed test attempt hit Windows temporary-
directory permission errors; the approved out-of-sandbox synthetic runs passed.

### Standards

No blocking findings. Added the suggested preparation-time input-change
regression. A seven-element internal preparation tuple is a maintenance
observation, not a correctness blocker. Code pins now use the same digest for
compatibility checking and end-of-run stability checking.

### Spec

No blocking findings in the bounded saved-final-state validation implementation.
Budget-wide acceptance and longitudinal outcomes remain explicitly incomplete.

No official run was launched by the agent. Changes remain uncommitted to preserve
the preexisting mixed worktree; no unrelated changes were staged or committed.
