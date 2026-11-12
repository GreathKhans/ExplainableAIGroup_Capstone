"""Human-readable fields for the UCI German Credit (Statlog) table."""

FIELDS = [
    {
        "name": "checking_status",
        "label": "Checking account",
        "kind": "category",
        "options": [
            ("A11", "Below 0 DM"),
            ("A12", "0 to below 200 DM"),
            ("A13", "200 DM or more, or salary assignment"),
            ("A14", "No checking account"),
        ],
    },
    {"name": "duration", "label": "Loan duration (months)", "kind": "number"},
    {
        "name": "credit_history",
        "label": "Credit history",
        "kind": "category",
        "options": [
            ("A30", "No credits taken, or all credits paid back duly"),
            ("A31", "All credits at this bank paid back duly"),
            ("A32", "Existing credits paid back duly so far"),
            ("A33", "Delay in paying off credit in the past"),
            ("A34", "Critical account, or other credits elsewhere"),
        ],
    },
    {
        "name": "purpose",
        "label": "Loan purpose",
        "kind": "category",
        "options": [
            ("A40", "Car (new)"),
            ("A41", "Car (used)"),
            ("A42", "Furniture or equipment"),
            ("A43", "Radio or television"),
            ("A44", "Domestic appliances"),
            ("A45", "Repairs"),
            ("A46", "Education"),
            ("A48", "Retraining"),
            ("A49", "Business"),
            ("A410", "Other"),
        ],
    },
    {"name": "credit_amount", "label": "Credit amount (DM)", "kind": "number"},
    {
        "name": "savings_status",
        "label": "Savings account or bonds",
        "kind": "category",
        "options": [
            ("A61", "Below 100 DM"),
            ("A62", "100 to below 500 DM"),
            ("A63", "500 to below 1000 DM"),
            ("A64", "1000 DM or more"),
            ("A65", "Unknown, or no savings account"),
        ],
    },
    {
        "name": "employment_since",
        "label": "Present employment since",
        "kind": "category",
        "options": [
            ("A71", "Unemployed"),
            ("A72", "Less than 1 year"),
            ("A73", "1 to less than 4 years"),
            ("A74", "4 to less than 7 years"),
            ("A75", "7 years or more"),
        ],
    },
    {
        "name": "installment_rate",
        "label": "Installment rate (1–4, share of disposable income)",
        "kind": "number",
    },
    {
        "name": "personal_status_sex",
        "label": "Personal status and sex",
        "kind": "category",
        "sensitive": True,
        "options": [
            ("A91", "Male, divorced or separated"),
            ("A92", "Female, divorced, separated, or married"),
            ("A93", "Male, single"),
            ("A94", "Male, married or widowed"),
            ("A95", "Female, single"),
        ],
    },
    {
        "name": "other_debtors",
        "label": "Other debtors or guarantors",
        "kind": "category",
        "options": [
            ("A101", "None"),
            ("A102", "Co-applicant"),
            ("A103", "Guarantor"),
        ],
    },
    {
        "name": "residence_since",
        "label": "Present residence since (1–4)",
        "kind": "number",
    },
    {
        "name": "property",
        "label": "Property",
        "kind": "category",
        "options": [
            ("A121", "Real estate"),
            ("A122", "Building-society savings or life insurance"),
            ("A123", "Car or other property"),
            ("A124", "Unknown, or no property"),
        ],
    },
    {
        "name": "age",
        "label": "Age (years)",
        "kind": "number",
        "sensitive": True,
    },
    {
        "name": "other_installment_plans",
        "label": "Other installment plans",
        "kind": "category",
        "options": [
            ("A141", "Bank"),
            ("A142", "Stores"),
            ("A143", "None"),
        ],
    },
    {
        "name": "housing",
        "label": "Housing",
        "kind": "category",
        "options": [
            ("A151", "Rent"),
            ("A152", "Own"),
            ("A153", "For free"),
        ],
    },
    {
        "name": "existing_credits",
        "label": "Existing credits at this bank",
        "kind": "number",
    },
    {
        "name": "job",
        "label": "Job",
        "kind": "category",
        "options": [
            ("A171", "Unemployed or unskilled, non-resident"),
            ("A172", "Unskilled, resident"),
            ("A173", "Skilled employee or official"),
            ("A174", "Management, self-employed, or highly qualified"),
        ],
    },
    {
        "name": "num_dependents",
        "label": "Number of dependents",
        "kind": "number",
    },
    {
        "name": "telephone",
        "label": "Telephone",
        "kind": "category",
        "options": [
            ("A191", "None"),
            ("A192", "Yes, registered in the customer's name"),
        ],
    },
    {
        "name": "foreign_worker",
        "label": "Foreign worker",
        "kind": "category",
        "sensitive": True,
        "options": [
            ("A201", "Yes"),
            ("A202", "No"),
        ],
    },
]

RAW_COLUMNS = [field["name"] for field in FIELDS]
SENSITIVE_HELP = (
    "The model can use this value. This demo does not treat it as "
    "an adverse-action reason."
)


def field_by_name(name):
    for field in FIELDS:
        if field["name"] == name:
            return field
    raise KeyError(name)


def option_label(name, code):
    field = field_by_name(name)
    labels = dict(field.get("options", []))
    return labels.get(str(code), str(code))
