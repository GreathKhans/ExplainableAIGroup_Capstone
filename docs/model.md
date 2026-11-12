# Model

The scoring model is an XGBoost classifier, maximum depth 4, `eval_metric=logloss`, trained on an 80/20 split with seed 42. Categorical fields are one-hot encoded with the first level dropped, which matches the notebook. The flag is the notebook's 50% line. At 50% the log-odds are 0, so the label Higher risk means the predicted chance of bad credit is at least as large as the chance of good credit. The published German Credit cost matrix treats a missed bad credit as five times as costly as a false alarm. This demo keeps the 50% line and does not retune it to that matrix. What log-odds means on the chart is in [explanations.md](explanations.md).

On the 200 held-out applicants the saved model has accuracy 78.0% and ROC-AUC 0.815. The notebook's saved Colab run reported accuracy 0.79 and ROC-AUC 0.8234. Those numbers stay in the notebook. The app reports the model in `models/`, so a teammate sees the same score as everyone else.

`models/credit_xgb.json` is the scoring model, about 163 KB. `models/credit_xgb_no_age.json` is the same recipe without age, about 160 KB, and is used only for the fairness comparison. Both were trained on CPU. A GPU does not change the stored trees. If a checkout has no model file, or the saved feature names do not match the current encoding, startup trains a replacement and logs the new size.

The logistic regression in the notebook is a baseline for that analysis. The review screen explains the XGBoost model, because that is the model whose trees SHAP can add back to the prediction.
