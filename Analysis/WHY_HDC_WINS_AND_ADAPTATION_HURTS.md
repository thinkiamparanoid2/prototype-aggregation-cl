# Why HDC wins on PlantVillage and why adaptation mostly hurts

Written 2026-09-19 (Claude). Interprets the results in `ANALYSIS.md` (HDC vs NCM vs RanPAC-style by dataset and encoder). Validation split only.

## 1. Why does adaptation mostly hurt?

Self-supervised adaptation (DINOv3 recipe, 5 epochs, train images only, no labels) does **not** make features worse in general. A fully supervised logistic-regression probe (the epoch gate) finds adapted features equal or better:

| Dataset | Frozen probe F1 (S/B/L) | Best adapted probe F1 (S/B/L) | Adaptation loss start -> end |
|---|---|---|---|
| PlantDoc | 0.691 / 0.734 / 0.762 | 0.708 / 0.746 / 0.771 | 8.2 -> 8.1 (barely moves) |
| PlantWild | 0.680 / 0.725 / 0.741 | 0.689 / 0.736 / 0.760 | 8.1 -> 2.2-3.4 |
| PlantVillage | 0.988 / 0.989 / 0.989 | 0.984 / 0.991 / 0.991 | 6.4-7.6 -> 0.09-0.17 |

HDC, NCM and RanPAC-style instead classify by similarity to class prototypes built from a small label budget, with no retraining, so they depend on the *geometry* of the space, not only on linear separability.

**Working hypothesis (not yet tested):** the adaptation loss collapsing towards zero on PlantWild and PlantVillage indicates partial representation collapse: features concentrate in fewer dominant directions. A supervised probe can re-weight dimensions to undo this; prototype/cosine learners cannot. Consistent with this, PlantDoc, where the loss barely moved, is the one dataset where adaptation is neutral, while the largest drops are where the loss collapsed most (for example PlantWild ViT-S, 0.48 -> 0.34).

**Protocol limitation:** the epoch gate selects by full-label linear-probe F1, which rewards separability rather than the prototype-friendly geometry these learners need, so it can pick epochs that are good for the probe and bad for the continual learners.

**Checks to confirm (saved embeddings only, no retraining):** (a) effective rank / feature spread, frozen vs adapted, expected lower on PlantWild and PlantVillage and unchanged on PlantDoc; (b) full-label NCM accuracy, frozen vs adapted, to separate geometry from label scarcity.

## 2. Why does HDC win on PlantVillage?

**Mostly label-acquisition pacing, not better classification from the same labels.** Every learner requests a label when its similarity margin (top-1 minus top-2) is below the shared threshold 0.10, until the budget is exhausted; novel-class registration also spends budget. Because each learner's similarity scale differs, the same 0.10 threshold spends budget at different rates. A novel class introduced after a learner's budget runs out can never be registered and scores zero from then on.

Mean over 5 seeds: step at which the budget runs out, and novel classes never registered (raw data: `ssl_hdc_pilot/results/replication/comparison/budget_pacing.psv`).

| Dataset | Model | Budget out: HDC / NCM / RanPAC | Novel never registered: HDC / NCM / RanPAC | Lasts longest | Macro-F1 winner |
|---|---|---|---|---|---|
| PlantDoc | Frozen ViT-S | 208 / 221 / 183 | 1.2 / 1.0 / 2.0 | NCM | NCM |
| PlantDoc | Frozen ViT-B | 231 / 242 / 202 | 1.0 / 1.0 / 1.4 | NCM | NCM |
| PlantDoc | Frozen ViT-L | 259 / 266 / 232 | 1.0 / 1.0 / 1.0 | NCM | HDC |
| PlantDoc | Adapted ViT-S | 209 / 224 / 186 | 1.2 / 1.0 / 2.0 | NCM | NCM |
| PlantDoc | Adapted ViT-B | 233 / 240 / 204 | 1.0 / 1.0 / 1.2 | NCM | NCM |
| PlantDoc | Adapted ViT-L | 261 / 268 / 239 | 1.0 / 1.0 / 1.0 | NCM | HDC |
| PlantWild | Frozen ViT-S | 1344 / 1422 / 1199 | 9.0 / 8.8 / 10.0 | NCM | NCM |
| PlantWild | Frozen ViT-B | 1454 / 1541 / 1260 | 8.4 / 8.0 / 10.0 | NCM | NCM |
| PlantWild | Frozen ViT-L | 1525 / 1575 / 1298 | 8.0 / 7.4 / 10.0 | NCM | NCM |
| PlantWild | Adapted ViT-S | 1203 / 1170 / 1117 | 10.0 / 10.8 / 11.0 | HDC | HDC |
| PlantWild | Adapted ViT-B | 1464 / 1476 / 1270 | 8.0 / 8.0 / 10.0 | NCM | NCM |
| PlantWild | Adapted ViT-L | 1416 / 1345 / 1261 | 9.0 / 9.0 / 10.0 | HDC | NCM |
| PlantVillage | Frozen ViT-S | 4492 / 4091 / 3556 | 3.0 / 4.0 / 4.0 | HDC | HDC |
| PlantVillage | Frozen ViT-B | 4797 / 4461 / 3934 | 3.0 / 3.0 / 4.0 | HDC | NCM |
| PlantVillage | Frozen ViT-L | 5324 / 4754 / 4140 | 2.0 / 3.0 / 3.8 | HDC | HDC |
| PlantVillage | Adapted ViT-S | 4902 / 4166 / 3754 | 3.0 / 3.4 / 4.0 | HDC | HDC |
| PlantVillage | Adapted ViT-B | 3732 / 3233 / 3214 | 4.0 / 5.0 / 5.0 | HDC | HDC |
| PlantVillage | Adapted ViT-L | 3565 / 3558 / 3558 | 4.0 / 4.0 / 4.0 | HDC | RanPAC |

Stream lengths: PlantDoc 401 (3 novel classes), PlantWild 2,625 (19 novel), PlantVillage 7,488 (8 novel).

- The learner whose budget lasts longest is also the macro-F1 winner in **13 of 18** rows; most of the 5 exceptions are statistical ties.
- On PlantVillage, HDC's budget lasts longest in every arm, and it wastes fewer labels on samples it already predicts correctly (frozen ViT-L seed 0: 79% of HDC's queried samples were already correct, vs 82% NCM and 84% RanPAC).
- Plausible reason: HDC's 10,000-bit bipolar codes and majority-vote prototypes give large inter-class margins on PlantVillage's tight, clean lab-image clusters, so fewer samples fall under 0.10. On noisy in-the-wild images, NCM's full-precision class means keep the edge.

**Thesis framing:** 'HDC outperforms NCM and RanPAC-style on PlantVillage' is true under this protocol, but the mechanism is mainly how a fixed margin threshold interacts with each learner's confidence scale, not superior classification given identical labels.

**Protocol-level finding:** every learner runs out of budget before many novel classes arrive: about 8-11 of 19 on PlantWild, 2-5 of 8 on PlantVillage, about 1 of 3 on PlantDoc. Novel-class and final macro-F1 results are therefore strongly shaped by budget timing. Report this alongside any win/loss claim.

**Clean test (not yet run):** the stream runner's shared-label-schedule mode gives every learner labels for exactly the same images, removing pacing. Running PlantVillage and PlantWild under it would measure how much of HDC's advantage remains.

## 3. Standing caveats

- ECE values (about 0.6, close to accuracy) indicate these similarity scores are not calibrated probabilities; do not interpret ECE as calibration quality (the same pattern is in the Original results).
- Novel-class terminal accuracy is computed differently for Original (stream uptake-repair window) and for the new datasets (validation final-checkpoint recall proxy).
- Adapted arms used 5 DataLoader workers: same recipe, not bit-identical augmentation to a single-worker run.
- The stream design uses the amended rule (25% known holdback, 176:800 novel ratio = 18.0% novel share, equal novel quotas, novel exclusion via filtered `stream_train.csv`); see `ssl_hdc_pilot/results/replication/REPLICATION_PLAN.md` and `ssl_hdc_pilot/results/replication/CONSULT_05.md`-`CONSULT_07.md`.
- Section 1's collapse explanation is a hypothesis pending the listed checks; section 2's pacing table is computed directly from the saved `history.csv` files of all 5 seeds and all 18 new-dataset arms.
