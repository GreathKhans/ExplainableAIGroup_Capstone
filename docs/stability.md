# Explanation stability

Added by Gergo, 12 Nov. Script: `analysis/explanation_stability.py` (runs in about 5 seconds on CPU, from the repo root).

## Why this check

Fidelity (baseline + SHAP = log-odds) proves the explanation is exact for the saved model file. Determinism proves the same file gives the same answer twice. Neither shows whether the explanation would survive a slightly different training sample or a different SHAP setup. The course asks for both fidelity and stability, so this check retrains the model 30 times on bootstrap resamples of the training split (same encoding, split, seed and depth as `auditor/pipeline.py`) and compares the four demo applicants. SHAP values are summed per original field, as on the screen.

## Results

| Check | Result |
|---|---|
| Fidelity, max \|baseline + SHAP − log-odds\| | about 1e-6 |
| Global importance ranking vs saved model | median Spearman 0.92 (min 0.84) |
| Row 740 predicted risk across retrains | 38%–98% (saved model 83%), flagged higher risk in 90% of runs |

Share of retrains in which each saved-model top-3 factor stays in the top 3:

| Row | Factor 1 | Factor 2 | Factor 3 |
|---|---|---|---|
| 740 | checking status 100% | credit history 80% | other installment plans 57% |
| 406 | existing credits 80% | installment rate 77% | dependents 27% |
| 29 | duration 90% | checking status 83% | property 17% |
| 11 (under 25) | duration 97% | checking status 90% | age 27% |

With two interventional TreeSHAP backgrounds (100 training rows each), the top 3 matches the saved model for rows 740, 406 and 11. For row 29 the third factor changes from property to credit amount.

## What it means

- The top two factors are stable in almost every case. That supports sending only two focus factors to retrieval.
- The third factor is not stable. It can be shown on the chart, but it should not be cited as a reason.
- Age appears as the third factor for the under-25 example in the saved model, but only in 27% of retrains. A fairness-relevant attribution that unstable should not be read as a firm finding.
- The score itself moves by tens of points under retraining for borderline-to-high cases. That is a further argument for the existing rule that a person reviews every case, and against inventing a precise deferral band.

## Limits

Four applicants, 30 retrains, one test split. The 80% stability line is a project choice, not a regulatory value. SHAP explains the model, not the cause of default.
