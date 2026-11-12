# Revision 1 — protocol lock (written BEFORE any corrected-protocol comparison or reserved-cohort run)

Written 2026-11-11, after the protocol audit below and before: (i) running the typed-vs-monolithic
comparison under any new protocol, and (ii) any inference on the 50 reserved patients.

## 1. Why the protocol is being amended (finding, not a choice)

Reviewer 1 and Reviewer 2 both flagged the action-head AUROC drop (0.947 separately trained vs 0.588
jointly trained) and Reviewer 1 flagged Brier 0.564 > 0.25 with non-converging temperature scaling.
A direct audit (`a1_protocol_audit.py`) found the cause is **not** joint training or calibration:

* Region state S2 contains pooled *decoder channels* of the fold-specific base network that held the
  patient out. The five base networks are independently trained; their decoder channels are not
  aligned with one another.
* The original controlled comparison and the Stage 1.6 calibration audit cross-fit the corrector on the
  frozen base-network folds, so a corrector trained on regions from four networks is tested on regions
  from a fifth network's feature space. S2 AUROC under this protocol is 0.456 (below chance);
  S1 (class probabilities, source class, log size — comparable across networks) is 0.77.
* The 0.947 figure came from a patient-grouped CV in which train and test regions share base networks.
  Within-network CV gives 0.945.

Consequence: the original S2-based comparison is confounded by feature-space mismatch. It is retained
and reported as protocol **P0**, not discarded.

## 2. Protocols (all use the frozen oracle labels, utility, normalisation and training recipe)

* **P0 (original, retained)** — S2, 5-fold cross-fit on frozen base-network folds. Original saved
  predictions are used verbatim.
* **P1 (secondary)** — S1 only (11 features, comparable across networks), same cross-fit as P0.
  Parameters: typed 3,046 vs monolithic 3,013 (1.08 % apart, inside the predeclared 5 % bound).
* **P2 (PRIMARY)** — per-network nested protocol: corrector f sees only regions whose features come
  from base network f; patient-grouped 5-fold inside the ~31 patients of network f
  (`p2_inner_assignment`). S2 (75 features), parameters 7,142 vs 7,109 as before. By construction no
  corrector-training or corrector-test patient was ever in network f's base-model training set, so the
  non-nested dependence disclosed in the original Section 3.1 does not arise.

P2 is designated primary now because it is the only protocol in which S2 is interpretable and the only
fully nested one. Designation is made before its results are seen.

## 3. Pre-specified analyses (report regardless of direction)

1. Typed vs monolithic, patient-level normalised net utility, paired patient bootstrap (2,000
   resamples), 3 seeds averaged — under P0, P1, P2.
2. Matched coverage: global top-K ranking by P(change); typed default coverage vs monolithic at the
   same K; full utility–coverage curves with oracle-ordering and random-ordering reference.
3. Objective–metric alignment: region-size-weighted NLL training (weights = region voxels) for T and M.
4. Calibration/authorisation mitigation on the corrected Q_act: Platt and isotonic (fit on training
   folds only), RC-specific authorisation threshold.
5. Ablation: act-only head vs joint Model T under P0 and P2.
6. Ceiling decomposition (heterogeneity vs minimum-size) from the region table + saved masks.
7. Dice and HD95 before/after correction, overall and per class, in nnU-Net's preprocessed 1 mm space.
8. Variability: per-seed and per-base-network spread.

## 4. Reserved cohort (50 patients, 140 cases) — ONE-SHOT

Run only after items 1–8 are final on the development cohort. Inference only; no model, threshold,
feature set, hyperparameter or protocol may be changed after this run. Specification:

* Base networks 0–4 each score all 140 reserved cases (checkpoints unchanged).
* P2: corrector f is trained on ALL network-f development regions (3 seeds, same recipe), applied to
  reserved regions extracted by network f. P1/P0: correctors trained on all 439 development cases.
* Primary endpoint: paired T − M difference in patient-level normalised net utility on the 50 patients
  (patient value = mean over the five base-network replicates), 2,000-resample patient bootstrap.
* Secondary: matched-coverage comparison, per-network replicates, RC/NETC counts, Dice/HD95.
* Results are reported whatever they are, including if the typed advantage does not reproduce.

## 5. Not attempted (declared in advance, with reason)

* Retraining base segmenters on the default long nnU-Net schedule (≈17.5 GPU-h at 100 epochs ×5;
  the default ≈10× longer): outside the revision window.
* External cohorts (LUMIERE, RHUH-GBM): not available locally; label schemes need harmonisation.

## 6. Amendment A1 (2026-11-11, written before any sliding-window result existed and before any reserved-cohort result was inspected)

Finding: the region pipeline of the original study obtained base-network probabilities and decoder
features with **non-overlapping tiled inference, no mirroring** (`infer_volume_softmax`), not nnU-Net's default
(Gaussian-weighted sliding window, 50 % overlap, mirroring). On the development cohort, tiled predictions give
fold-level mean foreground Dice 0.567 against 0.669 in the nnU-Net validation Dice of Table 1 (largest gaps: ET, RC);
the difference was undisclosed in the original submission. Correction partly repairs a degraded baseline.

Decision: the tiled pipeline remains PRIMARY (it is the pipeline of the submitted study and the reserved-cohort
inference above already uses it). Added as a **sensitivity analysis** (same code, same protocols P0/P1/P2, same
endpoints): Gaussian-weighted sliding-window inference, step 0.5 x patch, no mirroring (a middle setting between tiled
and the full nnU-Net default), regions and features re-extracted from it. Run on the development cohort and, as a
secondary endpoint, on the reserved cohort. Reported whatever the result, including if the correction benefit shrinks.

## 7. Amendment A2 (2026-11-11, still before any sliding-window or reserved-cohort result exists)

A one-case-at-a-time check of the two inference modes (3 development cases, network 0) showed the tiled mode is not
merely different but defective for this data: volumes are ~140 x 170 x 130 voxels and the patch is 128^3, so
non-overlapping tiling gives a second tile along each axis that is mostly zero padding. ET/RC Dice on those cases
were 0.38-0.54 / 0.84 (tiled) against 0.61-0.82 / 0.89-0.94 (sliding window). Correction results obtained on tiled
predictions are therefore results on a degraded base segmenter.

Amendment: the **Gaussian-weighted sliding-window pipeline becomes PRIMARY** for every endpoint (all protocols,
matched coverage, calibration, ceiling decomposition, Dice/HD95, reserved cohort). The tiled pipeline is retained
unchanged as "as-submitted pipeline" results (already computed for the development cohort; the reserved-cohort
inference on it is also run and reported) so that the effect of this correction is visible. Primary/secondary roles of
P0/P1/P2 are unchanged (P2 primary). Nothing else in sections 2-5 changes.

## 8. Amendment A3 (2026-11-12, code completeness fix; no analysis choice changed)

The reserved-cohort Dice/HD95 job initially skipped cases with no candidate region (they are unchanged by any
correction but belong to the baseline); the development-cohort job had already been corrected for this. The reserved
job was corrected the same way and both variants (sliding window, tiled) were re-run end to end with identical
settings and seeds. PatientID_0043 has no predicted component of at least eight voxels from any base network, so the
patient-level utility endpoint is defined for 49 of the 50 patients.
