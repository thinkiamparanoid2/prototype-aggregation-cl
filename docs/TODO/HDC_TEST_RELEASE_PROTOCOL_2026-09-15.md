# Locked test-release protocol — trusted test partition

Date written: 2026-09-15.
Status: **DRAFT — not approved, not in force.** Requires explicit user approval, and supervisor
sign-off where the supervisor owns the claim decision. Until approved, `trusted_splits/test.csv`
remains sealed and this document authorises nothing.

This closes gate **A2** of `docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md:88-90`:
*"Explicitly approve one locked release covering final AND longitudinal states, with an immutable
state list and no intermediate test-guided decisions."* `ssl_hdc_pilot/HANDOFF.md` refers to that
gate as open in roughly nineteen places and never defines it. This is the definition.

## 0. Why this is written now

It is written **before the results it will govern are known**. That is the only timing at which it
is worth anything. Every figure the project currently holds is on `trusted_splits/val.csv`, which
`RESULTS_LOG.md` correctly records as "a development split [that] has now been used for
development". Nothing produced so far is held-out evidence, and no amount of further development
work changes that.

The mechanism ablation has not yet reported. Writing the release rules while the decisive result is
still unknown is what makes them credible; writing them afterwards would make them a description of
whatever happened to work.

## 1. What is sealed

- `ssl_hdc_pilot/trusted_splits/test.csv` — 1,132 rows, never opened.
- Any embedding, prediction, metric or checkpoint derived from those rows.

Sealed means: not read, not loaded, not scored against, not used for model selection, threshold
setting, hyperparameter choice, stopping decisions, or sanity checks. Guard tests asserting this
already exist across `test_budget_experiment.py`, `test_evaluate_snapshots.py`,
`test_evaluate_validation.py`, `test_experiment_a.py`, `test_reconstruct_run.py`,
`test_forgetting_grid.py` and `test_split_guards.py`. They remain in force after release.

## 2. The release is a single event

There is **one** release. It covers final and longitudinal states together, in one execution, with
no intermediate look at any test-derived number.

Specifically prohibited:
- Scoring a subset of the state list, reading the result, and then deciding what else to score.
- Running the release, disliking the outcome, changing anything, and running it again.
- Any "pilot" or "smoke" pass against real test rows.
- Any decision taken after release that cites a test-derived number as its reason.

If the release fails on execution grounds — a crash, a corrupted artifact, a preflight block, a
power cut — the partial attempt is preserved unmodified and the run escalates to the next attempt
number, exactly as the capture grid does. **An execution failure may be retried. An unfavourable
result may not.**

## 3. Preconditions — all must hold before release

1. The mechanism ablation has completed and reported, and §4's headline learner is therefore
   determined.
2. `ssl_hdc_pilot/scripts/aggregate_grid.py` exists and reproduces the recorded development
   figures from the saved artifacts. If the development numbers are not reproducible, nothing
   downstream of them is releasable.
3. The full test suite passes, run directly, with the summary line present.
4. Every model state in §5 exists on disk with a `status: PASS` integrity receipt and hash-pinned
   artifacts.
5. This document is approved and unmodified since approval, and its SHA-256 is recorded in the
   release manifest.
6. The development record — `RESULTS_LOG.md`, `CHANGELOG.md`, `HANDOFF.md`,
   `results/SSL_HDC_REPORT.md` — is current and committed. The release commit must be separable
   from everything that preceded it.

## 4. The headline comparison is fixed by the ablation, and the rule is locked here

The release scores whichever comparison the mechanism ablation determines. That determination is
made on **development data only**, before release, by the rule already locked in
`ABLATION_PLAN.md`, restated here so it cannot drift:

Per-class F1 on `3_00_0` and `2_00_0`, adapted arms, median over seeds 0-2:

| Ablation outcome | Headline learner at release | Claim under test |
|---|---|---|
| Sign-NCM stays above 0.30 on `3_00_0` | **Sign-NCM**, with HDC reported beside it | Bounded-influence aggregation is robust to representation degradation; the hyperdimensional projection is not what provides it |
| Sign-NCM falls below 0.10, like NCM | **HDC** | The hyperdimensional projection is load-bearing; the robustness is HDC-specific |
| Anything between | **Release does not proceed** | Indeterminate — resolve on development data first |

The third row is not a formality. An indeterminate mechanism result must not be resolved by
choosing whichever framing the test set happens to favour.

Controls at release are NCM and RanPAC-style in every case. Both are reported whatever they show.

## 5. Immutable state list

Fixed here, before release, and not changeable afterwards. Any change requires a dated amendment
under §10 and resets the release.

- **Learners:** the §4 headline learner, plus NCM and RanPAC-style. Naive, EWC and replay are
  scored and reported, under §7's separation rule.
- **Encoder arms:** frozen ViT-B and adapted ViT-B. Other arms are ablation-scope only and are
  **not** released.
- **Seeds:** 0, 1, 2.
- **Condition:** Experiment B, threshold 0.10, budget 400, matched at 400 stream labels across the
  three prototype heads.
- **States:** the final state and the ten longitudinal checkpoints, per state, released together.
- **Evaluation population:** the full sealed 1,132-row test partition, fixed, unmodified, scored
  once.

No state outside this list is scored. No state inside it is omitted because of what it shows.

## 6. Primary endpoint, and its mandatory companion

**Primary:** macro-F1 on the sealed test partition, per arm, median over the three seeds.

**Mandatory companion, not optional:** the per-class F1 breakdown, reported in the same table or
immediately beside it, together with the macro delta computed both with and without the two
largest contributing classes.

This companion is compulsory because the development finding is concentrated: the adapted-arm macro
delta falls from **+0.0635 to +0.0069** once `2_00_0` and `3_00_0` are removed
(`RESULTS_LOG.md:1456`). An aggregate macro-F1 quoted alone overstates the result. **A release
table carrying an aggregate without its per-class breakdown is a protocol violation, not a
presentational choice.**

**Secondary, reported but not headline:** fixed-population forgetting, novel-class uptake,
per-class recall on `3_00_0` and `2_00_0`, labels consumed per learner, calibration (ECE).

**Not endpoints, and not eligible to become endpoints:** throughput and memory. Both are measured
losses on development data — HDC at 428.58 events/s against NCM's 1206.89 and RanPAC-style's
1370.35, and 30.75 MiB against NCM's 0.056 MiB. They are reported as losses. Release does not
revisit them.

## 7. Reporting rules, binding at release

1. **Label matching.** Supervision is matched at 400 labels only across the headline learner, NCM
   and RanPAC-style. Naive, EWC and replay consumed 17–79. They **must never appear in an
   unqualified table beside the prototype heads**, and any comparison involving them must state
   the label disparity in the same visual unit.
2. **RanPAC-style is an ablation.** It implements the fixed Gaussian projection, ReLU and online
   class prototypes, and omits the published method's Gram decorrelation and ridge inversion. The
   qualifier is not optional and the bare name "RanPAC" is not to be used.
3. **Three seeds support a consistency statement, not a significance test.** Report median, min and
   max, and a paired bootstrap interval over seeds for the headline-minus-best-control difference.
   Do not report a p-value from three seeds as though it were a powered test.
4. **Development and test figures never share a table.** The existing `SSL_HDC_REPORT.md` §2 numbers
   are training-pool scores at unmatched supervision; the grid numbers are `trusted_val` at matched
   supervision; these will be sealed-test at matched supervision. Three different populations.
5. **Report what the grid showed, including what it did not show.** Reduced forgetting is not
   established on development data (2/3 seeds, direction flips by encoder). Novel-class uptake is
   null. The best development cell overall is RanPAC-style on frozen at 0.7406, above HDC's own
   best of 0.7220. None of this is removed by a favourable test result.
6. **If a result is null, say it is null.** A null release is a valid outcome and is reported as one.

## 8. Pre-committed interpretation

Written before the numbers exist, so that the reading is not chosen to fit them.

- **The claim survives** if the headline learner's per-class advantage on the adapted arm
  reproduces on sealed test — specifically, if the controls' per-class F1 collapses below 0.10 on at
  least one class where the headline learner exceeds 0.30, in the median over three seeds, with the
  direction consistent across seeds.
- **The claim is narrowed** if the aggregate advantage reproduces but the per-class collapse does
  not. It then becomes an aggregate predictive difference on a degraded representation, with no
  mechanism attached, and must be described that way.
- **The claim fails** if neither reproduces. This is a real, reportable outcome. The thesis retains
  a complete, matched-supervision, held-out comparative evaluation of six continual learners across
  two encoder arms, plus a mechanism ablation — which is a contribution whether or not the headline
  effect holds.

The direction reversal is part of the claim, not an embarrassment to be managed: the advantage is
expected to be **absent or reversed on the frozen arm**, and `ENCODER_GEOMETRY_PREREG.md` predicted
that before the grid existed. A release in which frozen shows no advantage and adapted does is a
*confirmation*, not a mixed result.

## 9. Execution

- Capture and evaluation run through the existing guarded path. `test.csv` is admitted only by a
  release-specific code path that is added for this purpose, is explicit, and defaults off — the
  canonical train-identity check at `experiment_a.py:68-70` is **not** relaxed or deleted.
- Preflight runs before every cell; the 8 GiB free-space guard applies per cell.
- Outputs go to a dedicated `results/test_release/` parent, never into a development parent.
- The release manifest records: this document's SHA-256, the ablation result that selected the
  headline learner, the state list, every input hash, every code hash, and the operator confirmation
  timestamp.
- Guard tests are extended, not weakened: after release, `test.csv` access remains prohibited
  everywhere except the release path, and the release path is asserted to be single-use.

## 10. Amendment

Any change to §4, §5, §6 or §8 after approval requires a dated amendment recording what changed,
why, and that it was decided **without reference to any test-derived number**. An amendment made
after release has begun voids the release; the correct response is to report the released result as
specified and treat any new analysis as exploratory, labelled as such.

No amendment may remove the per-class companion requirement in §6, the label-matching rule in §7.1,
or the RanPAC-style qualifier in §7.2.

---

**Standing scope statement, to accompany every released figure:** results are from one dataset of 19
classes with a stream of 976 events and four mid-stream class introductions, on one encoder family,
in one implementation. They test this implementation and this protocol, not hyperdimensional
computing in general.
