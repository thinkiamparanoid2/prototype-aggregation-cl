# streaming_median_fidelity.py provenance

Read-only, post-hoc offline fidelity check. No capture, evaluation,
training, or GPU work was launched by this script; it never imports
evaluate_snapshots.py, stream_runner.py, budget_experiment.py, or
experiment_a.py, and it never opens ssl_hdc_pilot/trusted_splits/test.csv.

- capture root: `ssl_hdc_pilot\results\mechanism_ablation\capture`
- validation root: `ssl_hdc_pilot\results\mechanism_ablation\validation`
- cells checked: 6

## Sanity gate

For each cell, the exact-median rebuild's macro-F1 on trusted_splits/val.csv
must reproduce the saved snapshots[-1]["metrics"]["macro_f1"] to within 1e-6.

- `2026-09-15_ablation_b400_p2_b_adapted_median_ncm_s0_t010_attempt01`: saved=0.592535250 rebuilt=0.592535250 abs_diff=0.000e+00 passed=True
- `2026-09-15_ablation_b400_p2_b_adapted_median_ncm_s1_t010_attempt01`: saved=0.586579144 rebuilt=0.586579144 abs_diff=0.000e+00 passed=True
- `2026-09-15_ablation_b400_p2_b_adapted_median_ncm_s2_t010_attempt01`: saved=0.602295346 rebuilt=0.602295346 abs_diff=0.000e+00 passed=True
- `2026-09-15_ablation_b400_p2_l_adapted_median_ncm_s0_t010_attempt01`: saved=0.670903132 rebuilt=0.670903132 abs_diff=0.000e+00 passed=True
- `2026-09-15_ablation_b400_p2_l_adapted_median_ncm_s1_t010_attempt01`: saved=0.669043869 rebuilt=0.669043869 abs_diff=0.000e+00 passed=True
- `2026-09-15_ablation_b400_p2_l_adapted_median_ncm_s2_t010_attempt01`: saved=0.673967796 rebuilt=0.673967796 abs_diff=0.000e+00 passed=True
