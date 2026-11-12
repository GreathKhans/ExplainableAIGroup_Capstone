# User interface

Open http://127.0.0.1:8501 after `./start.sh`. The page is wide. The sidebar is the audit. The main column is the four applicants.

## Sidebar

**Fairness audit** states the under-25 result on the held-out test set: favorable rate, sample size, confidence interval, the gap against ages 25 and older, and the two false-positive rates.

**Why this group** says why age is the audit attribute, that dropping age only shrinks the gap, that sex is mixed with marital status, and that the non-foreign comparison has only nine test rows.

**About the model** gives depth, split, seed, accuracy, ROC-AUC, and the 50% flag. It also says the two uploaded policy files are the same eight clauses, and that drafts are written by Phi-3.5-mini from the SHAP factors and the retrieved clauses.

## Applicant tabs

There are four example tabs, Applicant 1 to Applicant 4, and a **Custom** tab beside them. Each example starts from a dataset row and can be edited. **Reset applicant N** restores that row. Custom is where you type your own values. **Reset custom** restores its starting values. Editing Custom does not change the examples.

| Tab | Starts from | Historical label | What it is for |
|---|---|---|---|
| 1 | Row 740 | Good credit | The notebook example. The model still flags higher risk. |
| 2 | Row 406 | Good credit | A very low score. Features push the risk down from the average. |
| 3 | Row 29 | Bad credit | A very high score, with a checking account below 0 DM and past delays. |
| 4 | Row 11 | Bad credit | An applicant aged 24, the audit group. |
| Custom | Your own values | — | A separate form. Editing it leaves the four examples alone. |

Each tab has the twenty German Credit fields. Age, personal status and sex, and foreign worker have a help note: they are not adverse reasons. Category menus show the codebook label, not only the `Axx` code. Numbers stay inside the range present in the data.

**Evaluate all** scores applicants 1–4 only. Custom is left as it is. **Evaluate applicant N** and **Evaluate custom** score that tab alone. Editing a field does not rescore until that tab's button, or Evaluate all for an example, is pressed again. If one applicant is reset, that card's score is cleared. Custom never shows the dataset-row note, even when its values still match a row.

**Draft applicants 1–4** writes the four explanations in order, applicant 1 first, and places each draft in that applicant's tab. It does not draft Custom.

## Results

Four summary figures sit above the tabs: one percent per example applicant, with Higher risk or Lower risk. The charts, the clause list, and the draft stay inside that applicant's tab.

The blue note appears only when the fields are still the dataset row. It gives the row number and the historical label, and it says the percent below is the model's estimate.

Three figures follow:

- **Predicted default risk** is the model's probability of bad credit, rounded to the nearest percent.
- **Model flag** is Higher risk at 50% or above, otherwise Lower risk.
- **Supported adverse reasons in the top factors** counts how many of the two focus factors are observed risk indicators. Zero or one means the draft must not pretend there are two supported adverse reasons.

The left chart is the probability waterfall. Read it from the bottom. "Average applicant" is the starting chance, about 28%. Each bar starts where the previous feature left the probability. Red raises default risk. Teal lowers it. The dashed line is 50%. The top bar is this prediction, named Higher risk or Lower risk. The caption under the chart says the same thing.

The right chart is log-odds, the unit the trees add. At 50% the log-odds are 0. A bar of +1.0 multiplies the odds of bad credit by about 2.7. From the 28% average applicant, that lands near 50%. The fuller explanation is in [explanations.md](explanations.md).

The sentence under the charts names the two focus factors. Those are the ones passed to policy search and to the draft. For a higher-risk applicant they are the two strongest risk-increasing features. For a lower-risk applicant they are the two largest absolute contributions.

The table lists the eight largest absolute contributions: the field, the person's value, the code, the log-odds SHAP, and the evidence label. The evidence label is the interpretation rule, not a second score.

| Evidence | Meaning |
|---|---|
| Observed risk indicator | One of the two codes the project treats as a recorded risk indicator. |
| Not an adverse reason | Do not cite this value as a ground for an adverse action. |
| Needs verification | The value may matter, but no verified adverse cutoff is defined for it. |

**Retrieved guidelines** shows the query sentence and one expander per clause, inside the same tab. P0x is a project paraphrase. EBA-p is official text, with a page number. Open them when the talk needs the wording. They stay collapsed.

**Draft explanation** generates the written review for the open tab only. **Draft applicants 1–4** does the same for the four examples, in order from 1 to 4, and leaves each text in its tab. Until a draft exists, the caption says the score and the charts are already ready. After it finishes, the text is in the card with the time it took. Warnings appear under the text when the draft cites an unknown id, claims a completed denial, skips manual review, or mentions missed payments or grounds for denial.
