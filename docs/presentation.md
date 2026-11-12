# 10-minute presentation

About 10 minutes, including a live demo. Each block is one slide. **On the slide** is what the audience reads. **Say** is the speaker note. Do not read the note aloud word for word.

If the draft is still running when the clock is short, stay on the waterfall and the evidence table. Say that the written text takes about 20 seconds and is checked by the reviewer. The score does not depend on it.

---

## Slide 1 — Title

**On the slide**

- Credit Risk Auditor
- A review aid for German Credit applicants
- Score, explanation, lending clauses, human review

**Say**

This is a review aid, not an underwriting engine. It scores an applicant, shows why the score moved, retrieves the clauses that match those reasons, and leaves the decision with a person.

---

## Slide 2 — The problem

**On the slide**

- A model flag is not a reason a reviewer can use
- A positive feature weight is not automatically an allowed adverse reason
- The historical label and the model score are different facts

**Say**

German Credit labels people good or bad credit after the fact. A model can disagree with that label. Even when the score is clear, the reviewer still needs the person's actual values, and they need to know which of those values the policy treats as a risk indicator. We built the screen around that gap.

---

## Slide 3 — What the system does

**On the slide**

1. XGBoost scores predicted default risk
2. SHAP shows the path from the average applicant to this prediction
3. The focus factors retrieve project clauses and official EBA text
4. An optional draft is written for the reviewer
5. No loan is granted or denied by the system

**Say**

The pipeline is one applicant in, one explained score out. The draft is last and optional. If it is wrong, the charts and the clauses are still there.

---

## Slide 4 — Data

**On the slide**

- UCI / Statlog German Credit: 1,000 people, 20 attributes
- Class 1 = good credit, class 2 = bad credit
- The model learns bad credit as the positive class
- Codes matter: `A11` is a checking balance below 0 DM, `A14` is no checking account

**Say**

The table is small and public. Amounts are in Deutsche Mark. Several important fields are categories. We do not let the form invent codes that never appear in the file.

---

## Slide 5 — Model

**On the slide**

- XGBoost, depth 4, 80/20 split, seed 42
- Held-out accuracy 78%, ROC-AUC 0.815
- Flag at 50%, the same cutoff as the notebook
- The model file is about 160 KB and is shared, so every machine scores the same person the same way

**Say**

We kept a tree model so a single prediction can be taken apart. The published cost matrix says a missed default is more expensive than a false alarm. We did not retune the threshold. Fifty percent remains a flag for the reviewer, not a calibrated decision rule. The notebook has an older Colab run with a slightly higher AUC. The talk uses the saved model, because that is the model on the screen.

---

## Slide 6 — Fairness

**On the slide**

- Audit group: under 25
- Favorable rate 65.6% (N=32) versus 81.0% at 25 or older (N=168)
- Gap −15.3 points. Without age, the gap is −9.4 points
- False-positive rate 27.3% versus 6.7%
- Sex is mixed with marital status. The non-foreign test group has 9 people

**Say**

Under 25 is the group we can defend. Their low-risk rate is lower, and age is its own column. Dropping age shrinks the gap and does not remove it, so the explained model still includes age. We do not lead with sex, because this dataset stores sex together with marital status, and on this split the female-coded rows are not the lower-scoring group. We do not lead with foreign worker, because almost everyone is coded as one and the comparison group is nine people.

---

## Slide 7 — How to read the explanation

**On the slide**

- Start at the average applicant, about 28% predicted default risk
- Red raises that risk, teal lowers it
- The top bar is this applicant's prediction, in percent
- The log-odds chart is the same story in the units the trees add
- Only two codes are observed risk indicators: checking below 0 DM, and past delays
- A positive SHAP value can still be "not an adverse reason"

**Say**

The waterfall is the chart to talk through. It ends on the output the model gives, not on an abstract importance score. Log-odds stay beside it because that is the additive math. Row 740 shows the distinction that matters: the checking account is an observed risk indicator, and a duly repaid credit history can still push the score up without becoming an allowed adverse reason. We also do not explain inactive dummy columns. The label is the value this person has.

---

## Slide 8 — Clauses and the draft

**On the slide**

- Eight project paraphrases of EBA/GL/2020/06, indexed once
- Official 89-page guideline added so the search has a real corpus
- The draft must cite retrieved ids and must call for manual review
- If the draft mentions a denial, missed payments, or a cutoff, the screen warns

**Say**

The two short uploads were the same eight paraphrases, so we did not merge them into duplicates. The official report is what makes retrieval more than a lookup table. The language model is Phi-3.5-mini. It is useful and it is not fully reliable. The warning under a bad sentence is part of the design. The reviewer is responsible for the text that leaves the room.

---

## Slide 9 — Live demo

**On the slide**

- Four applicants are already filled in
- Press **Evaluate all**
- Walk applicant 1, then 2, then 3
- Draft only applicant 1 if there is time

**Say — do this on the screen**

Leave this slide up as a cue, then switch to the app.

1. Point at the sidebar. Under 25, the gap, and the sample size. One sentence.
2. Press **Evaluate all**. Read the example summary numbers: about 83% higher, 0% lower, 99% higher, 98% higher. Leave Custom for a question.
3. Applicant 1, row 740. The banner says the historical label is good credit. On the waterfall, start at about 28%, follow the red checking-account bar, and stop on the top bar, 83% higher risk. In the table, checking is "Observed risk indicator" and credit history is "Not an adverse reason".
4. Applicant 2, row 406. The bars are teal and the score is at the floor. This is the contrast: the same chart can show a score being pushed down.
5. Applicant 3, row 29. A high score where the checking account is below 0 DM and the history records past delays. Say that this is the case with two recorded risk indicators, unlike applicant 1.
6. If time remains, press **Draft explanation** on applicant 1. While it runs, say it takes about 20 seconds and does not change the score. If a warning appears, read it. That is the fallback: the chart stands, the sentence does not.

Applicant 4 is the under-25 example if someone asks. Do not open it unless they do.

---

## Slide 10 — Limits and choices

**On the slide**

- Every case is referred to a human. There is no calibrated auto-decision band
- 1,000 rows. The under-25 test group is 32 people
- The draft can invent a fact. The warning is the control we shipped
- Explanations are deterministic for a fixed model file. We did not add a separate stability suite
- No numeric rejection cutoff was invented

**Say**

The honest limits are the sample, the mixed sex column, and the draft. The design choice is that the tool prepares a review and refuses to finish it. A new deferral threshold would look more precise than the evidence. We would rather keep "a person decides" as the rule.

---

## Slide 11 — Close

**On the slide**

- Score, path from the average applicant, clauses, human review
- Same saved model for every teammate
- Questions

**Say**

What we want a reviewer to leave with is a percent, the few facts that moved it, and a clear label for which of those facts are allowed reasons. The rest stays with the person.

Offer the fairness sidebar and applicant 4 if the questions go there. Offer the evidence table if a question is about whether the draft can be wrong.
