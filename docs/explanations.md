# Explanations

## What log-odds means

The percent on the screen is the predicted chance of bad credit. Log-odds is that same chance, rewritten so the features can be added.

At 50%, the log-odds are 0. Above 50% they are positive. Below 50% they are negative. The average applicant starts near −1, which is about 28% predicted default risk. That is the bottom of the waterfall.

A SHAP bar of +1 raises the odds of bad credit by about 2.7 times. From that 28% start, +1 log-odds lands near 50%. A red bar means the fact pushed the predicted risk up. A teal bar means it pushed the risk down.

The waterfall shows those pushes after they have been turned back into a percent. The log-odds chart keeps the raw additions, because that is the unit the trees use. A step in percent is not proportional to the log-odds bar, because the conversion is curved. A large log-odds push near 0% or near 100% can be a small change in the chance of default.

## What a SHAP value is

A SHAP value is this person's share of the move from the average applicant to this prediction. For these trees, each value is a log-odds contribution. The baseline plus every feature's SHAP value is the log-odds of the predicted probability. For row 740 that sum matches the model's 82.6% score. The screen rounds it to 83%.

The table lists the eight largest absolute contributions. The SHAP column is log-odds, in the same unit as the right-hand chart.

## What the 50% line is

A score of 50% or higher is labeled Higher risk. That is the dashed line on the waterfall, and it is the same flag as the notebook. The EBA guideline states no universal numeric rejection cutoff, and this project does not add one. The line tells the reviewer which side of the model's flag the person is on.

## The two charts

The left chart, "From the average applicant to this prediction", walks in probability. Each red bar raises predicted default risk. Each teal bar lowers it. The dashed line is the 50% flag. The top bar is this applicant's prediction, labeled Higher risk or Lower risk, with the percent on the bar. Features with a small absolute contribution are grouped as "Other features" so the walk still ends on the real probability.

The right chart, "Log-odds contributions", is the same eight features as horizontal bars in log-odds. Log-odds are what the trees add. Probability is what a reviewer reads. A feature's probability step is not proportional to its log-odds step, because the sigmoid is curved. Both views are shown so a large log-odds push near 0% or near 100% is not mistaken for an equally large change in the chance of default.

## What may be called an adverse reason

A positive SHAP value is not itself an adverse reason. Only two codes are observed risk indicators:

- Checking account `A11`: balance below 0 DM.
- Credit history `A33`: past repayment delays.

Credit history `A31` means previous credits at this bank were repaid duly. It is marked "Not an adverse reason" even when its SHAP value is positive. Row 740 is the example: checking `A11` is about +1.01 log-odds and is an observed risk indicator; credit history `A31` is about +0.91 and is not an adverse reason.

Age, personal status and sex, and foreign worker are never used as adverse reasons. Numeric fields such as duration and amount have no universal rejection cutoff in the project policy extracts, so they stay "Needs verification".

## One-hot columns

The model sees dummy columns such as `checking_status_A14`. A waterfall on those columns can highlight a code the person does not have, because the dropped or inactive level still carries weight. The screen sums each dummy back to the original field and labels the value the person actually has. For row 740 that label is "Checking account: Below 0 DM", not the inactive `A14` column.
