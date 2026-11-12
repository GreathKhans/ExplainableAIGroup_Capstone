# Dataset

The rows are the Statlog / UCI German Credit data: 1,000 applicants, 20 attributes, and a class label. The file is `data/german.data`. The code meanings are in `data/german.doc` and in `auditor/codebook.py`.

Class 1 is good credit. Class 2 is bad credit. The model learns the opposite direction: 1 means bad credit (the original class was 2), and 0 means good credit.

Amounts are in Deutsche Mark. Duration is in months. Several fields are codes, not numbers. `A11` on the checking account means a balance below 0 DM. `A14` means there is no checking account. Those are different facts.

Three fields are sensitive or fairness-related: age, personal status and sex, and foreign worker. Personal status and sex is one column, so sex is mixed with marital status. Foreign worker is coded `A201` for almost the whole table.

The form only offers codes that appear in the file, and numeric fields stay inside the minimum and maximum in the data. That keeps a demo applicant inside the table the model was trained on.

The historical label is not the model's decision. Row 740 is historically good credit. The saved model still scores that row as higher risk. The screen says so when the inputs are still that dataset row.
