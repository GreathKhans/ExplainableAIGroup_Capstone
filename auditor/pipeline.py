"""Train the German Credit model and explain one applicant."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from auditor import log
from auditor.codebook import FIELDS, RAW_COLUMNS, field_by_name
from auditor.factors import STATUS_LABELS, assess, value_label

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "german.data"
MODEL_DIR = ROOT / "models"
SCORING_PATH = MODEL_DIR / "credit_xgb.json"
NO_AGE_PATH = MODEL_DIR / "credit_xgb_no_age.json"
EXAMPLE_ROW = 740
EXAMPLE_ROWS = (740, 406, 29, 11)
RANDOM_STATE = 42


def load_frame():
    columns = RAW_COLUMNS + ["class"]
    frame = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 German Credit rows, found {len(frame)}.")
    frame["target"] = (frame["class"] == 2).astype(int)
    return frame


def _wilson(successes, total, z=1.96):
    if total == 0:
        return (float("nan"), float("nan"))
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(center - margin), float(center + margin)


def _group_stats(y_true, y_pred, mask):
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    total = int(len(y_true))
    favorable = int((y_pred == 0).sum())
    actual_good = y_true == 0
    false_positives = int(((y_pred == 1) & actual_good).sum())
    good_total = int(actual_good.sum())
    low, high = _wilson(favorable, total)
    return {
        "n": total,
        "favorable": float(favorable / total) if total else float("nan"),
        "ci_low": low,
        "ci_high": high,
        "false_positive_rate": (
            float(false_positives / good_total) if good_total else float("nan")
        ),
    }


def _applicant_from_row(row):
    applicant = {}
    for field in FIELDS:
        name = field["name"]
        if field["kind"] == "number":
            applicant[name] = int(row[name])
        else:
            applicant[name] = str(row[name])
    return applicant


def _favorable_gap(y_true, y_pred, under_25):
    under = _group_stats(y_true, y_pred, under_25)
    older = _group_stats(y_true, y_pred, ~under_25)
    return under, older, float(under["favorable"] - older["favorable"])


def _feature_names(model):
    names = model.get_booster().feature_names
    return list(names or [])


def _load_or_train(path, x_train, y_train):
    """Use the committed CPU model so every machine scores the same applicant the same way."""
    if path.is_file():
        model = xgb.XGBClassifier()
        model.load_model(path)
        if _feature_names(model) == list(x_train.columns):
            size_kb = path.stat().st_size / 1024
            log(f"loaded {path.relative_to(ROOT)} ({size_kb:.1f} KB)")
            return model
        log(
            f"{path.name} does not match this dataset encoding. "
            "Training a replacement on CPU."
        )
    else:
        log(f"{path.name} is not in the checkout. Training it on CPU.")

    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        max_depth=4,
        device="cpu",
    )
    model.fit(x_train, y_train)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    log(f"saved {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")
    return model


def build_context():
    frame = load_frame()
    features = frame[RAW_COLUMNS]
    target = frame["target"]
    encoded = pd.get_dummies(features, drop_first=True)
    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target, test_size=0.2, random_state=RANDOM_STATE
    )

    model = _load_or_train(SCORING_PATH, x_train, y_train)
    class_index = list(model.classes_).index(1)
    proba = model.predict_proba(x_test)[:, class_index]
    pred = model.predict(x_test)

    no_age_train = x_train.drop(columns=["age"])
    no_age = _load_or_train(NO_AGE_PATH, no_age_train, y_train)
    pred_no_age = no_age.predict(x_test.drop(columns=["age"]))

    under_25 = x_test["age"].to_numpy() < 25
    y_np = y_test.to_numpy()
    young, older, gap = _favorable_gap(y_np, pred, under_25)
    _, _, gap_no_age = _favorable_gap(y_np, pred_no_age, under_25)

    raw_test = features.loc[x_test.index]
    female = raw_test["personal_status_sex"].isin(["A92", "A95"]).to_numpy()
    foreign = (raw_test["foreign_worker"] == "A201").to_numpy()

    example = _applicant_from_row(frame.loc[EXAMPLE_ROW])
    historical = (
        "good credit" if int(frame.loc[EXAMPLE_ROW, "class"]) == 1 else "bad credit"
    )
    examples = []
    for row_id in EXAMPLE_ROWS:
        row = frame.loc[row_id]
        examples.append(
            {
                "row": int(row_id),
                "historical": (
                    "good credit" if int(row["class"]) == 1 else "bad credit"
                ),
                "applicant": _applicant_from_row(row),
            }
        )
    bounds = {
        field["name"]: (
            int(features[field["name"]].min()),
            int(features[field["name"]].max()),
        )
        for field in FIELDS
        if field["kind"] == "number"
    }
    observed = {}
    for field in FIELDS:
        if field["kind"] != "category":
            continue
        present = set(features[field["name"]].astype(str))
        options = [pair for pair in field["options"] if pair[0] in present]
        known = {code for code, _ in options}
        for code in sorted(present - known):
            options.append((code, code))
        observed[field["name"]] = options

    return {
        "model": model,
        "explainer": shap.TreeExplainer(model),
        "columns": list(x_train.columns),
        "dtypes": {column: x_train[column].dtype for column in x_train.columns},
        "example": example,
        "example_historical": historical,
        "examples": examples,
        "bounds": bounds,
        "options": observed,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "test_size": int(len(y_test)),
        },
        "fairness": {
            "under_25": young,
            "older": older,
            "gap": gap,
            "gap_without_age": float(gap_no_age),
            "female": _group_stats(y_np, pred, female),
            "male": _group_stats(y_np, pred, ~female),
            "foreign_worker": _group_stats(y_np, pred, foreign),
            "not_foreign_worker": _group_stats(y_np, pred, ~foreign),
        },
    }


def parent_feature(column):
    matches = [
        name
        for name in RAW_COLUMNS
        if column == name or column.startswith(name + "_")
    ]
    if not matches:
        return column
    return max(matches, key=len)


def encode_applicant(applicant, columns, dtypes):
    row = {}
    for field in FIELDS:
        name = field["name"]
        if name not in applicant:
            raise KeyError(f"Missing applicant field: {name}")
        if field["kind"] == "number":
            row[name] = int(applicant[name])
        else:
            row[name] = str(applicant[name])
    frame = pd.DataFrame([row])
    encoded = pd.get_dummies(frame)
    aligned = encoded.reindex(columns=columns, fill_value=0)
    for column, dtype in dtypes.items():
        aligned[column] = aligned[column].astype(dtype)
    return aligned


def _shap_vector(explanation):
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        return values[0, :, class_index]
    if values.ndim == 2:
        if values.shape[0] == 1:
            return values[0]
        class_index = 1 if values.shape[1] > 1 else 0
        return values[:, class_index]
    return values


def evaluate(context, applicant):
    frame = encode_applicant(applicant, context["columns"], context["dtypes"])
    model = context["model"]
    class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0, class_index])
    predicted = int(model.predict(frame)[0])
    contributions = _shap_vector(context["explainer"](frame))
    if len(contributions) != len(frame.columns):
        raise ValueError(
            f"SHAP returned {len(contributions)} values for {len(frame.columns)} features."
        )

    grouped = {name: 0.0 for name in RAW_COLUMNS}
    for column, contribution in zip(frame.columns, contributions):
        grouped[parent_feature(column)] += float(contribution)

    factors = []
    for name, contribution in grouped.items():
        status, text = assess(name, applicant[name])
        field = field_by_name(name)
        code = int(applicant[name]) if field["kind"] == "number" else str(applicant[name])
        factors.append(
            {
                "feature": name,
                "label": field["label"],
                "value_code": code,
                "value_label": value_label(name, applicant[name]),
                "shap": float(contribution),
                "status": status,
                "status_label": STATUS_LABELS[status],
                "explanation": text,
            }
        )
    factors.sort(key=lambda item: item["shap"], reverse=True)
    positive = [item for item in factors if item["shap"] > 0]
    high_risk = predicted == 1
    if high_risk:
        focus = (positive or factors)[:2]
    else:
        focus = sorted(factors, key=lambda item: abs(item["shap"]), reverse=True)[:2]

    base_logodds = float(np.asarray(context["explainer"].expected_value).reshape(-1)[0])
    return {
        "probability_bad": probability,
        "high_risk": high_risk,
        "factors": factors,
        "focus": focus,
        "base_logodds": base_logodds,
        "matches_example": applicant == context["example"],
        "example_historical": context["example_historical"],
    }


def shap_figure(factors, limit=8):
    chosen = sorted(factors, key=lambda item: abs(item["shap"]), reverse=True)[:limit]
    chosen = sorted(chosen, key=lambda item: item["shap"])
    labels = [f"{item['label']}: {item['value_label']}" for item in chosen]
    values = [item["shap"] for item in chosen]
    colors = ["#b42318" if value > 0 else "#0f6e56" for value in values]

    figure, axis = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    axis.barh(labels, values, color=colors)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("SHAP value (log-odds of bad credit)")
    axis.set_title("Log-odds contributions")
    axis.tick_params(axis="y", labelsize=9)
    return figure


def _sigmoid(value):
    clipped = float(np.clip(value, -30, 30))
    return 1 / (1 + np.exp(-clipped))


def shap_waterfall(factors, base_logodds, high_risk, limit=8):
    """Walk from the average applicant to this prediction, in probability."""
    ranked = sorted(factors, key=lambda item: abs(item["shap"]), reverse=True)
    steps = ranked[:limit]
    remainder = sum(item["shap"] for item in ranked[limit:])
    if abs(remainder) > 1e-4:
        steps.append({"label": "Other features", "value_label": "", "shap": remainder})
    steps = list(reversed(steps))

    cursor = float(base_logodds)
    rows = [("Average applicant", _sigmoid(cursor), None)]
    for step in steps:
        start = _sigmoid(cursor)
        cursor += step["shap"]
        end = _sigmoid(cursor)
        if step["value_label"]:
            name = f"{step['label']}: {step['value_label']}"
        else:
            name = step["label"]
        rows.append((name, start, end))
    final = _sigmoid(cursor)
    outcome = "Higher risk" if high_risk else "Lower risk"
    rows.append((outcome, final, None))

    figure, axis = plt.subplots(figsize=(8.6, 5.6), layout="constrained")
    for y, (name, start, end) in enumerate(rows):
        if end is None and name == "Average applicant":
            axis.plot([start, start], [y - 0.28, y + 0.28], color="#444444", linewidth=2)
            axis.annotate(f"{start:.0%}", (start, y), xytext=(6, 0), textcoords="offset points", va="center")
        elif end is None:
            color = "#b42318" if high_risk else "#0f6e56"
            axis.barh(y, start, color=color, height=0.62)
            text_x = start - 0.02 if start > 0.82 else start + 0.01
            align = "right" if start > 0.82 else "left"
            text_color = "white" if start > 0.82 else "#222222"
            axis.annotate(
                f"{start:.0%}",
                (text_x, y),
                va="center",
                ha=align,
                color=text_color,
                fontsize=9,
            )
        else:
            left = min(start, end)
            color = "#b42318" if end > start else "#0f6e56"
            axis.barh(y, abs(end - start), left=left, color=color, height=0.62)
            axis.plot([end, end], [y + 0.31, y + 0.69], color="#8a8a8a", linewidth=0.8)
    axis.axvline(0.5, color="#b9b9b9", linewidth=0.8, linestyle="--")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([row[0] for row in rows])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Predicted default risk")
    axis.set_title("From the average applicant to this prediction")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    axis.tick_params(axis="y", labelsize=8)
    return figure
