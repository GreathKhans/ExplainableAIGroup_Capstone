# Presenter notes

Notes for `Credit Risk Auditor.pdf`. About 10 minutes, including the live demo. The lines below are what to say. Do not read them word for word.

The deck pauses on a **Live demo** card, then shows how to read the chart. Teach that chart, then switch to the app. Come back for the limits slide and questions.

If the written draft is still running near the end, stay on the waterfall and the evidence table. Say that the text takes about 20 seconds and a person still checks it. The score does not depend on it.

---

## Slide 1 — Title

This is a review aid for a person looking at a German credit applicant. It scores predicted default risk, shows which facts moved that score, retrieves the lending clauses that match those facts, and leaves the decision with a reviewer.

One sentence is enough before the problem slide: the screen prepares a review. It does not grant or refuse a loan.

---

## Slide 2 — The problem

Three gaps sit between a model and a reviewer.

A flag is only a percent. The reviewer still needs the person's actual values, and a sentence that says why those values matter.

A feature can push the score up and still be the wrong thing to cite. In this project only two recorded facts count as observed risk indicators: a checking balance below 0 DM, and a credit history that includes past delays. A positive weight on anything else is a finding to verify, not an allowed adverse reason.

The dataset label and the model score are different facts. German Credit marks people good or bad credit after the fact. The model can disagree. Applicant 1 in the demo is the example: the historical label is good credit, and the model still flags higher risk.

---

## Slide 3 — What the system does

Walk the five steps in order.

XGBoost scores predicted default risk. SHAP then walks from the average applicant to this person, so the chart ends on the prediction rather than on a ranked list of important columns. The factors that matter for this person retrieve clauses: eight project paraphrases of EBA/GL/2020/06, plus passages from the official guideline. An optional draft turns that packet into a short note for the reviewer. The last step is the person. Nothing in the pipeline approves or denies the application.

If a question comes early about the language model, say it is Phi-3.5-mini, it is optional, and the charts remain if the draft is weak.

---

## Slide 4 — Data

The table is the public UCI / Statlog German Credit set: 1,000 people and 20 attributes. Amounts are in Deutsche Mark. Many of the important fields are categories, and the form only offers values that appear in the file.

Class 1 in the file is good credit. Class 2 is bad credit. The model learns bad credit as the positive class, so a higher score means higher predicted default risk.

The checking-status line is there because that field is easy to misread. A balance below 0 DM is an observed risk indicator. Having no checking account is a different status, and the screen does not treat it as a negative balance.

If someone asks for the raw codes: below 0 DM is A11, and no checking account is A14.

---

## Slide 5 — Model

The held-out test set is 200 rows: 78% accuracy, ROC-AUC 0.815. The trees are XGBoost, depth 4, an 80/20 split, seed 42. A tree model is there so one prediction can be taken apart.

The 50% figure is the model's higher-risk label, the same flag as the notebook. It is the line on the chart. It is not a lending cutoff from the EBA guideline, and it is not a calibrated rejection rule. The published UCI cost matrix says a missed default costs more than a false alarm. We did not retune the threshold to that matrix.

The saved model is about 160 KB. Every machine loads that file and scores on CPU, so the same applicant gets the same percent. The notebook still shows an older Colab run with a slightly higher AUC. The talk uses the saved model, because that is the model on the screen.

---

## Slide 6 — Fairness

The audit group is applicants under 25. On the test set their favorable rate, the share predicted lower risk, is 65.6% (32 people) against 81.0% for age 25 or older (168 people). The gap is 15.3 points. Training without the age column shrinks it to 9.4 points and does not remove it, so the model we explain still includes age.

The false-positive rates are 27.3% and 6.7%. Those rates are among people whose historical label was good credit. The counts on the bars, 32 and 168, are the full groups. The under-25 interval is wide because 32 people is a small test group. Say that plainly if asked. The point of the slide is the gap and the sample size, not a claim that the gap is precisely 15 points in the population.

Sex is a poor lead here. The column mixes sex with marital status, and on this split the female-coded rows are not the lower-scoring group. Foreign worker is a poor lead for a different reason: almost everyone is coded as a foreign worker, and the other test group is 9 people.

---

## Slide 7 — Clauses and the draft

The two short files we started with were the same eight paraphrases of EBA/GL/2020/06. They are indexed once. They were not enough for a search, so the corpus also includes the official 89-page guideline. A factor such as checking status retrieves its mapped paraphrase, and the search can add official paragraphs beside it.

The draft is a note for the reviewer. The screen checks it. It warns if a citation was not in the retrieved set, or if a high-risk draft never says manual review is required. It warns if the text says the loan was denied, talks about grounds for denial, or invents missed payments. It also warns if a sentence stops halfway. Those checks do not catch every invented fact. The reviewer is responsible for any text that leaves the room.

---

## Slide 8 — Live demo

This card is a pause. The next slide is the legend for the charts. Stay on that legend, then open the app. The demo steps are written under slide 9 so the speaking and the clicking stay together.

---

## Slide 9 — How to read the explanation

Teach the chart, then switch to the app and do the steps below.

The waterfall starts at the average applicant, about 28% predicted default risk. Each red bar pushes that risk up. Each teal bar pulls it down. The top bar is this person's prediction, in percent, labeled higher risk or lower risk. The dashed line is the 50% flag from the model slide. The log-odds chart beside it is the same features in the units the trees actually add. The percent view is what a reviewer reads. The log-odds view is why the bars are allowed to stack.

The card on the right is the policy layer. Only two observed indicators qualify: a checking balance below 0 DM, and past payment delays. A bar can be red and still be marked "not an adverse reason." That is the sentence to land before the demo.

**On the app**

1. Point at the sidebar once: under 25, the gap, the sample of 32. One sentence.
2. Press **Evaluate all**. Read the four example numbers: about 83% higher, 0% lower, 99% higher, 98% higher. Leave **Custom** unless someone asks to type their own values.
3. Applicant 1, dataset row 740. The banner says the historical label is good credit. On the waterfall, start at about 28%, follow the red checking-account bar, and stop on 83% higher risk. In the table, checking is "Observed risk indicator." Credit history can still push the score up and read "Not an adverse reason," because this person's history is credits repaid duly at this bank.
4. Applicant 2, row 406. The bars are teal and the score sits at the floor. The same chart can show a score being pushed down.
5. Applicant 3, row 29. The checking account is below 0 DM and the history records past delays. This is the case with two observed risk indicators. Applicant 1 had one.
6. If time remains, press **Draft explanation** on applicant 1 only. Say it takes about 20 seconds and does not change the score. If a warning appears, read it. The chart stands either way.

Applicant 4 is row 11, age 24, the audit group, about 98% higher risk. Open it only if someone asks.

---

## Slide 10 — Limits and choices

Every case is handed to a person. There is no second threshold that auto-approves a confident score. A calibrated deferral band would look more precise than this sample supports.

The sample is the other limit to say out loud: 1,000 rows in total, and 32 under-25 people in the test set. The fairness gap is real on this split and too small a group for a tight claim.

The draft can invent a fact the packet never contained. The warnings on the previous slide are the control we shipped. They flag known failure patterns. They do not make the paragraph safe by themselves.

The explanations match across machines because everyone loads the same model file. The same applicant and the same trees produce the same SHAP values. We did not add a separate stability test suite. The trees are already deterministic.

---

## Slide 11 — Questions

Close on the reviewer's takeaway: a percent, the few facts that moved it, and a label for which of those facts are observed risk indicators. The decision stays with the person.

If the question is about fairness, go back to the sidebar and, if useful, applicant 4. If the question is whether the draft can be wrong, go back to the evidence table and a warning. If the question is about a custom case, the **Custom** tab edits a fifth applicant and leaves the four examples unchanged.
