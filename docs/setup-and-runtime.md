# Setup and runtime

From the repository root:

```bash
./start.sh
```

The script logs each step: the Python version, the virtualenv, pip downloads, whether `nvidia-smi` sees a GPU, the saved credit model, the policy index, and the explanation-model download. It then opens the app at http://127.0.0.1:8501.

`requirements.txt` is the shared stack. An NVIDIA GPU also installs a CUDA PyTorch wheel and `requirements-cuda.txt` (transformers, bitsandbytes, accelerate). Without a GPU, the script installs `requirements-mlx.txt`. If XGBoost cannot import, the script installs `libomp` when Homebrew is available.

The language-model weights are not in git. They are about 2 GB and download into the Hugging Face cache on first run. The credit models are in git because they are about 160 KB each.

Measured on the machine that produced the saved model:

| Step | Time |
|---|---|
| Train the depth-4 model, if no file is saved | under 1 second |
| Explain one applicant with SHAP | about 0.01 seconds |
| Build the policy index the first time | on the order of 20 seconds |
| Draft, after the weights are loaded | about 15–20 seconds |

Evaluate all four does not draft. The wait during a talk is the draft button, and only for the applicant you choose.

`NOTICE_BACKEND` can be `cuda`, `mlx`, or `openai`. Leave it unset and the script picks CUDA when a GPU is present. For `openai`, also set `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`.
