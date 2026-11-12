# Credit Risk Auditor

A reviewer sets four German Credit applicants, sees the model's predicted default risk, and reads why the score moved. The written draft is for a person. The system does not decide the loan.

Start from the repository root:

```bash
./start.sh
```

The app opens at http://127.0.0.1:8501.

## Where to look

| Path | What it is |
|---|---|
| [docs/](docs/README.md) | One short note per topic, plus the 10-minute slide script |
| [app.py](app.py) | The review screen |
| [auditor/](auditor/) | Scoring, SHAP, policy search, and the draft |
| [data/](data/) | German Credit rows, the UCI codebook, and the official EBA guideline |
| [policies/](policies/) | The project's eight paraphrased clauses |
| [models/](models/) | The saved XGBoost models |
| [notebook/](notebook/) | The original analysis notebook |
| [start.sh](start.sh) | Setup, download log, and launch |

The slide script is [docs/presentation.md](docs/presentation.md). The spoken notes for the uploaded deck are [docs/presenter-notes.md](docs/presenter-notes.md).
