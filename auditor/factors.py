"""Map a German Credit value to a review status.

Only two coded values are treated as observed risk indicators. That matches
the checked notebook: a positive SHAP value is not itself an adverse reason,
and the project policy extracts do not set numeric rejection cutoffs.
"""

from auditor.codebook import field_by_name, option_label

# Status, explanation. Explanations are factual descriptions of the code.
SPECIAL = {
    ("checking_status", "A11"): (
        "OBSERVED_RISK_INDICATOR",
        "Checking account balance category: below 0 DM.",
    ),
    ("checking_status", "A12"): (
        "REVIEW_ONLY",
        "Checking balance category: 0 to below 200 DM. Repayment capacity needs verification.",
    ),
    ("checking_status", "A13"): (
        "NOT_ADVERSE",
        "Checking balance category: at least 200 DM.",
    ),
    ("checking_status", "A14"): (
        "REVIEW_ONLY",
        "No checking account. This does not mean a negative account balance.",
    ),
    ("credit_history", "A30"): (
        "REVIEW_ONLY",
        "No prior credit or all previous credit repaid duly.",
    ),
    ("credit_history", "A31"): (
        "NOT_ADVERSE",
        "Previous credits at this bank repaid duly.",
    ),
    ("credit_history", "A32"): (
        "NOT_ADVERSE",
        "Existing credits repaid duly so far.",
    ),
    ("credit_history", "A33"): (
        "OBSERVED_RISK_INDICATOR",
        "Recorded category indicates past repayment delays.",
    ),
    ("credit_history", "A34"): (
        "REVIEW_ONLY",
        "Critical account or other credits recorded. Exact circumstances require verification.",
    ),
    ("other_installment_plans", "A141"): (
        "REVIEW_ONLY",
        "Other installment arrangement with a bank. Repayment burden needs verification.",
    ),
    ("other_installment_plans", "A142"): (
        "REVIEW_ONLY",
        "Other installment arrangement with a store. Repayment burden needs verification.",
    ),
    ("other_installment_plans", "A143"): (
        "NOT_ADVERSE",
        "No other installment plans recorded.",
    ),
}

SENSITIVE = {"age", "personal_status_sex", "foreign_worker"}
STATUS_LABELS = {
    "OBSERVED_RISK_INDICATOR": "Observed risk indicator",
    "REVIEW_ONLY": "Needs verification",
    "NOT_ADVERSE": "Not an adverse reason",
}


def value_label(feature, value):
    field = field_by_name(feature)
    if field["kind"] == "number":
        number = int(value)
        if feature == "duration":
            return f"{number} months"
        if feature == "credit_amount":
            return f"{number} DM"
        if feature == "age":
            return f"{number} years"
        return str(number)
    return option_label(feature, value)


def assess(feature, value):
    """Return (status, explanation) for one original feature value."""
    if feature in SENSITIVE:
        return (
            "REVIEW_ONLY",
            "Sensitive or fairness-relevant feature. Do not use as an unsupported rejection reason.",
        )

    field = field_by_name(feature)
    if field["kind"] == "number":
        shown = value_label(feature, value)
        return (
            "REVIEW_ONLY",
            f"{field['label']}: {shown}. No universal numeric rejection cutoff is stated in the project policy extracts.",
        )

    code = str(value)
    if (feature, code) in SPECIAL:
        return SPECIAL[(feature, code)]

    shown = option_label(feature, code)
    return (
        "REVIEW_ONLY",
        f"{field['label']}: {shown}. No verified adverse threshold is defined for this value.",
    )
