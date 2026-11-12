# Policies and retrieval

Two files were uploaded at the start. `policies/credit_risk_eba_guidelines_rag.txt` and `policies/Credit_Risk_Assessment_Condensed_EBA_Guidelines.pdf` contain the same eight clauses, P01–P08. They are project paraphrases of EBA/GL/2020/06 (29 May 2020), not the official guideline text. The PDF adds a trailing semicolon on some annex references and a shorter closing line on P08. The rules do not contradict each other. Indexing both would store every clause twice, so the text file is indexed once.

Eight clauses are too small a corpus for a meaningful search. The long document is the official 89-page final report, `data/EBA_GL_2020_06.pdf`. The index keeps the eight paraphrases plus the guideline body, 273 chunks in total. The local index is `data/chroma/` and is rebuilt when the source files change. It is not part of the git checkout.

Retrieval is two steps. First, the focus features are mapped to paraphrases. Checking account maps to P01. Credit history maps to P04 when the code is `A33` (past delays) and to P03 otherwise. Amount and duration map to P06. Second, a semantic search over the official guideline chunks adds up to four passages. The embedding model is all-MiniLM-L6-v2. For row 740 the focus is a checking account below 0 DM and a duly repaid credit history, and the search returns P01, P03, and official passages that include paragraph 85 on evidence for consumer creditworthiness.

The screen labels P0x extracts as project paraphrases and EBA-p extracts as official text. A paraphrase must not be read as a quotation. The official report does not set one numeric cutoff for amount, duration, or account balance. The app does not invent one.
