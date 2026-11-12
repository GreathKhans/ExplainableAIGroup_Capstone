## Slide — Do the explanations hold up?  (Gergo, about 1 minute)

Fidelity we already had: the baseline plus the SHAP values gives back the model's log-odds, up to rounding. But that only proves the explanation is exact for this one model file. I wanted to know if it survives a slightly different training sample, so I retrained the same model 30 times on bootstrap resamples and compared the explanations.

Globally the ranking holds, a median rank correlation of 0.92. For applicant 1, row 740, the checking account stays in the top three in every single run, and credit history in 80%. The third factor drops out in almost half the runs. So the top two are stable, which matches the two focus factors we send to retrieval, and the third factor should never be cited as a reason.

Two more points. For the under-25 example, age is in the top three for our model, but only in 27% of retrains, so that fairness-relevant signal is not robust. And the score itself for row 740 ranges from 38 to 98 percent across retrains. That is exactly why we keep a person reviewing every case instead of trusting a precise cutoff.
