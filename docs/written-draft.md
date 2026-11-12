# Written draft

After the score and the clauses are ready, a reviewer can ask for a draft. It is optional. Scoring the four applicants does not wait for it.

The draft is Phi-3.5-mini. A CUDA machine loads `microsoft/Phi-3.5-mini-instruct` in 4-bit. A machine without CUDA uses the local 4-bit build of the same family. `NOTICE_BACKEND=openai` sends the draft to another OpenAI-compatible server instead. The scoring model does not change with that switch.

The prompt receives the model's flag, the focus factors with their review status, and the retrieved paraphrases plus one official excerpt. It is told to use only that evidence, to keep manual review in the text, to cite clause ids in square brackets, and not to say the application was denied. A high-risk case with fewer than two observed risk indicators must say the evidence does not establish two supported adverse reasons.

A finished draft is checked before it is trusted:

- citations must be ids that were actually retrieved
- it must not say the loan was denied, rejected, or declined
- a high-risk draft must mention manual review
- an unfinished last sentence is trimmed back to the previous sentence
- "missed payments" and "grounds for denial" raise a warning, because those facts were not supplied

Phi-3.5-mini still sometimes writes those phrases. The warning stays on screen. The SHAP table and the retrieved clauses are the evidence. The draft is a first text for the reviewer, and a wrong sentence has to be removed by that person.

A draft takes about 15–20 seconds once the weights are loaded. The first run also downloads them and logs that download.
