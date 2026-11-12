# What we built and why

A credit model can flag an applicant as higher risk and still leave a reviewer with no usable reason. A raw feature-importance chart does not say which of this person's details pushed the score, and it does not say whether that detail is allowed as an adverse reason.

This project is a review aid for the UCI German Credit data. It does four things for an applicant:

1. Score predicted default risk with a small XGBoost model.
2. Show how the score moved away from the average applicant, in the probability the model actually outputs, and also in log-odds.
3. Retrieve the lending clauses that match those factors.
4. Draft a short explanation a person can check before any adverse-action notice.

The draft is not a decision. Every case stays with a human reviewer. A score of 50% or higher is only a flag.

We used a tree model because TreeExplainer can attribute a single prediction exactly enough to reconstruct that prediction. A neural net would need a slower, less exact explanation for a table of 1,000 rows.

The four preset applicants are there so a demo can show a disagreement with the historical label, a clearly low score, a clearly high score, and an applicant under 25, without typing twenty fields first.
