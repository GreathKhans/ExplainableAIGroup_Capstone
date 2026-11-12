# Fairness

The audit group is applicants under 25. Age is stored on its own, and on this split that group receives the lower favorable score.

A favorable score means the model predicts low risk (class 0). The favorable rate is the share of a group that receives that score. The false-positive rate is a different count: among people whose historical label was good credit, the share the model still scores as higher risk. On the under-25 row, 27.3% is that second count. The 32 people are the whole test group, not the denominator of the false-positive rate.

On the 200 test applicants:

- Under 25: 65.6% favorable, 95% interval about 48%–80%, N=32. False-positive rate 27.3% among those with historical good credit.
- Age 25 or older: 81.0% favorable, N=168. False-positive rate 6.7%.
- Gap: −15.3 percentage points.

A second model trained without age narrows the gap to −9.4 points. That is not a fairness fix. The explained model is still the model that includes age. Removing the column and calling the problem solved would hide the remaining gap.

Sex is a poor protected attribute here because `personal_status_sex` mixes sex and marital status. On this split, rows coded female are not the lower-scoring group: 83.9% favorable (N=56) against 76.4% for rows coded male (N=144).

Foreign workers are 77.5% favorable (N=191). The other test rows are 100% favorable, but there are only 9 of them. That comparison is too small for a stable audit.

The notebook's earlier Colab run reported a larger age gap. The direction is the same. The screen uses the saved model's own test metrics.
