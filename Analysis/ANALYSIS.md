# Replication results: HDC vs NCM vs RanPAC-style

Written 2026-09-19 (Claude) from the saved cells in `ssl_hdc_pilot/results/replication/cells/` and `ssl_hdc_pilot/results/replication/comparison/COMPARISON.md`. Validation split only; sealed test sets untouched. Original = the thesis dataset (3 seeds, ViT-B only); new datasets have 5 seeds.

## 1. Full comparison (validation macro-F1, mean over seeds)

Bold = best learner in the row. *HDC vs best* = HDC minus the better of NCM/RanPAC, in points. WIN/LOSS only when the paired bootstrap 95% interval excludes zero, otherwise tie; seeds won in brackets.

### Original

| Model | HDC | NCM | RanPAC | HDC vs best | Verdict |
|---|---|---|---|---|---|
| Frozen (ViT-B) | 0.724 | 0.731 | **0.737** | -1.5 | LOSS (0/3) |
| Adapted (ViT-B) | **0.599** | 0.526 | 0.531 | +6.7 | WIN (3/3) |

### PlantDoc

| Model | HDC | NCM | RanPAC | HDC vs best | Verdict |
|---|---|---|---|---|---|
| Frozen ViT-S | 0.608 | **0.617** | 0.603 | -0.9 | LOSS (1/5) |
| Frozen ViT-B | 0.630 | **0.632** | 0.621 | -0.4 | LOSS (1/5) |
| Frozen ViT-L | **0.665** | 0.664 | 0.656 | +0.1 | tie (2/5) |
| Adapted ViT-S | 0.601 | **0.602** | 0.588 | -0.1 | tie (2/5) |
| Adapted ViT-B | 0.622 | **0.626** | 0.622 | -0.5 | tie (1/5) |
| Adapted ViT-L | **0.665** | 0.663 | 0.654 | +0.2 | tie (2/5) |

### PlantWild

| Model | HDC | NCM | RanPAC | HDC vs best | Verdict |
|---|---|---|---|---|---|
| Frozen ViT-S | 0.482 | **0.496** | 0.480 | -1.3 | LOSS (0/5) |
| Frozen ViT-B | 0.531 | **0.550** | 0.523 | -1.9 | LOSS (0/5) |
| Frozen ViT-L | 0.549 | **0.564** | 0.540 | -1.5 | LOSS (0/5) |
| Adapted ViT-S | **0.340** | 0.323 | 0.329 | +1.1 | WIN (5/5) |
| Adapted ViT-B | 0.503 | **0.512** | 0.494 | -0.9 | LOSS (0/5) |
| Adapted ViT-L | 0.481 | **0.482** | 0.472 | -0.1 | tie (2/5) |

### PlantVillage

| Model | HDC | NCM | RanPAC | HDC vs best | Verdict |
|---|---|---|---|---|---|
| Frozen ViT-S | **0.793** | 0.754 | 0.755 | +3.8 | WIN (5/5) |
| Frozen ViT-B | 0.785 | **0.789** | 0.757 | -0.4 | LOSS (0/5) |
| Frozen ViT-L | **0.826** | 0.783 | 0.752 | +4.3 | WIN (5/5) |
| Adapted ViT-S | **0.731** | 0.708 | 0.701 | +2.0 | WIN (5/5) |
| Adapted ViT-B | **0.707** | 0.652 | 0.655 | +5.2 | WIN (5/5) |
| Adapted ViT-L | 0.772 | 0.781 | **0.784** | -1.1 | LOSS (0/5) |

**Tally (20 rows):** HDC 6 WIN, 5 tie, 9 LOSS. PlantVillage 4 W / 2 L; PlantWild 1 W / 1 tie / 4 L (NCM strongest); PlantDoc 4 ties / 2 small L; Original 1 W / 1 L. Forgetting is low for all learners (about 0.3-0.6% PlantVillage, 1-4% elsewhere) and does not separate them.

Explanations of these results (why HDC wins on PlantVillage, why adaptation mostly hurts) are in `WHY_HDC_WINS_AND_ADAPTATION_HURTS.md` (same folder).
