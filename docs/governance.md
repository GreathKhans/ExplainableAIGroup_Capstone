# Governance

## Where the data comes from

The applicants are the public Statlog / UCI German Credit table, 1,000 rows, with the UCI codebook in `data/german.doc`. Nothing in the demo is a live bank application.

The policy text has two layers. P01–P08 in `policies/` are project paraphrases of EBA/GL/2020/06. The passages whose ids look like `EBA-p010-03` are chunks of the official final report in `data/EBA_GL_2020_06.pdf`. The screen names which layer a passage belongs to.

## How a case is processed

The fields are encoded the same way as training. The saved model returns a probability of bad credit. At 50% or above the flag is Higher risk. SHAP attributes that probability. Two focus factors are sent to the clause index. A person may then request a draft.

The system never records a denial. The historical class, when the fields still match a dataset row, is shown next to the model score so the two are not confused.

## Who reviews

A person reviews every case. There is no score above which the system may act alone. If a higher-risk case has fewer than two observed risk indicators, the draft has to say the evidence does not establish two supported adverse reasons. Age, personal status and sex, and foreign worker are not adverse reasons.

The 50% flag is the notebook's cutoff. It is not a calibrated deferral band. We did not add a band such as "refer only between 40% and 60%". Referring every case is the rule.

## Errors and model updates

If scoring or retrieval throws, the page shows the error and does not invent a score. Draft warnings are listed under the text. A warning means the reviewer deletes or rewrites that sentence. It does not mean the score was withdrawn.

The shared score is the file `models/credit_xgb.json`. Replacing it changes the demo for everyone, so a new file should come from the same split, seed, and depth, and the fairness numbers in the sidebar should be re-read from that run. If the file's feature names do not match the current encoding, startup trains a new file and logs that it did so.

## What we chose not to add

Explanation fidelity for this model is the additivity check: baseline plus SHAP equals the predicted log-odds, and therefore the predicted probability. The same inputs always produce the same trees, so we did not add a perturbation-stability suite.

Runtime was measured. Training is under a second, one explanation is about a hundredth of a second, and a draft is about 15–20 seconds. We did not add a shift or coverage test suite. The form already refuses values outside the dataset.

A separate model card file is unnecessary. The dataset, metrics, fairness limits, and review rule are these notes and the sidebar.
