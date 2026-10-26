# SSL-HDC results

Status refreshed: 2026-09-07.

- [Scientific report](SSL_HDC_REPORT.md): refreshed six-learner analysis;
  PROVISIONAL pending equal-budget evidence and held-out generalization.
- [Verification audit](VERIFICATION_AUDIT.md): resolved and open findings.
- `continual/frozen_vitb16/` and `continual/adapted_vitb16_ep3/`: 54
  summaries plus 54 histories each; 108 comparison runs in total.
- `arm_f_*` / `arm_d_*`: encoder embeddings, provenance, checkpoints, and
  probe gates. Smoke artifacts are not scientific results.
- `unlabeled_manifest*`: sanctioned 51,906-image pool and leakage audit.

The historical `ssl_hdc_summary.png` predates the prototype controls and
report corrections. It is not the current six-learner summary.

`final_metrics` score the 3,396-image trusted-training pool after streaming.
`prequential_metrics` score predictions before each stream update.
Neither should be labelled a held-out trusted-test result.
