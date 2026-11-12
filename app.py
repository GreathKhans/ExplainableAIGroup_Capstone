"""Interactive credit-risk review: applicant inputs, SHAP, policy clauses, and a draft notice."""

import streamlit as st

from auditor.codebook import FIELDS, SENSITIVE_HELP
from auditor.notice import draft_notice, select_backend
from auditor.pipeline import build_context, evaluate, shap_figure, shap_waterfall
from auditor.rag import SOURCE_NOTE, open_store, retrieve

st.set_page_config(page_title="Credit Risk Auditor", layout="wide")


@st.cache_resource(show_spinner="Loading the credit model...")
def cached_context():
    # Preset applicants are part of this context.
    return build_context()


@st.cache_resource(show_spinner="Indexing the lending guidelines...")
def cached_store():
    return open_store()


def _slot_key(slot, name):
    return f"slot{slot}_{name}"


def _queue_eval(slot):
    st.session_state["pending_eval"] = slot


def _reset_slot(slot, applicant):
    for name, value in applicant.items():
        st.session_state[_slot_key(slot, name)] = value
    st.session_state.pop(f"notice_{slot}", None)
    results = st.session_state.get("results")
    if results and slot < len(results):
        results[slot] = None


def _read_slot(slot):
    return {field["name"]: st.session_state[_slot_key(slot, field["name"])] for field in FIELDS}


def _custom_start(context):
    return dict(context["example"])


def _results_list(count):
    results = st.session_state.get("results")
    if not isinstance(results, list):
        results = []
    if len(results) < count:
        results = results + [None] * (count - len(results))
    elif len(results) > count:
        results = results[:count]
    st.session_state["results"] = results
    return results


def _score_slot(context, store, slot, examples, custom_slot):
    applicant = _read_slot(slot)
    evaluation = evaluate(context, applicant)
    evaluation["retrieval"] = retrieve(store, evaluation["focus"])
    if slot == custom_slot:
        evaluation["source_row"] = None
        evaluation["historical"] = None
        evaluation["unchanged"] = False
    else:
        example = examples[slot]
        evaluation["source_row"] = example["row"]
        evaluation["historical"] = example["historical"]
        evaluation["unchanged"] = applicant == example["applicant"]
    st.session_state.pop(f"notice_{slot}", None)
    return evaluation


def _show_fairness(fairness, metrics):
    young = fairness["under_25"]
    older = fairness["older"]
    st.sidebar.header("Fairness audit")
    st.sidebar.markdown(
        f"Audit group: **applicants under 25**. On the held-out test set, "
        f"**{young['favorable']:.1%}** were predicted low risk "
        f"(95% interval {young['ci_low']:.0%}–{young['ci_high']:.0%}, N={young['n']}). "
        f"Applicants 25 or older were predicted low risk **{older['favorable']:.1%}** "
        f"of the time (N={older['n']}). "
        f"The gap is **{fairness['gap']:+.1%}**."
    )
    st.sidebar.markdown(
        f"False-positive rate, good-credit applicants scored high risk: "
        f"**{young['false_positive_rate']:.1%}** under 25 and "
        f"**{older['false_positive_rate']:.1%}** for ages 25 and older."
    )
    with st.sidebar.expander("Why this group"):
        female = fairness["female"]
        foreign = fairness["foreign_worker"]
        st.markdown(
            "Under-25 applicants are the audit group because their favorable-score rate "
            "is lower on this split, and age is recorded on its own. "
            f"Removing age from a second model changes the gap from "
            f"{fairness['gap']:+.1%} to {fairness['gap_without_age']:+.1%}. "
            "That smaller gap is not a fairness fix."
        )
        st.markdown(
            "Sex is stored together with marital status, so it is a poor protected attribute. "
            f"On this split, rows coded female have a {female['favorable']:.1%} low-risk rate "
            f"(N={female['n']}), against {fairness['male']['favorable']:.1%} for rows coded male "
            f"(N={fairness['male']['n']}). The female-coded rows are not the lower-scoring group."
        )
        other = fairness["not_foreign_worker"]
        st.markdown(
            f"Foreign workers are {foreign['favorable']:.1%} low-risk (N={foreign['n']}), "
            f"against {other['favorable']:.1%} for other test rows (N={other['n']}). "
            "Almost every row is coded as a foreign worker, so that comparison group is too small for a stable audit."
        )
    with st.sidebar.expander("About the model"):
        st.markdown(
            f"XGBoost, max depth 4, 80/20 split, seed 42. "
            f"Test accuracy {metrics['accuracy']:.1%}, ROC-AUC {metrics['roc_auc']:.3f} "
            f"on {metrics['test_size']} applicants. "
            "A score of 50% or higher is flagged as higher credit risk. "
            "The published German Credit cost matrix treats a missed default as more costly "
            "than a false alarm. This demo keeps the 50% cutoff used in the notebook."
        )
        st.markdown(SOURCE_NOTE)
        if select_backend() == "openai":
            explanation = "Explanations are requested from the server in OPENAI_BASE_URL."
        else:
            explanation = (
                "Drafts are written by Phi-3.5-mini from the SHAP factors "
                "and the retrieved clauses."
            )
        st.markdown(explanation)


def _applicant_fields(slot, context):
    columns = st.columns(3)
    for index, field in enumerate(FIELDS):
        with columns[index % 3]:
            help_text = SENSITIVE_HELP if field.get("sensitive") else None
            if field["kind"] == "number":
                low, high = context["bounds"][field["name"]]
                st.number_input(
                    field["label"],
                    min_value=low,
                    max_value=high,
                    step=1,
                    key=_slot_key(slot, field["name"]),
                    help=help_text,
                )
            else:
                options = context["options"][field["name"]]
                labels = dict(options)
                st.selectbox(
                    field["label"],
                    options=[code for code, _ in options],
                    format_func=lambda code, labels=labels: labels[code],
                    key=_slot_key(slot, field["name"]),
                    help=help_text,
                )


def _show_result(slot, fast):
    if fast.get("unchanged"):
        st.info(
            f"These inputs are dataset row {fast['source_row']}. "
            f"The historical label is {fast['historical']}. "
            "The score below is the model's estimate."
        )

    focus = fast["focus"]
    observed = sum(item["status"] == "OBSERVED_RISK_INDICATOR" for item in focus)
    left, middle, right = st.columns(3)
    left.metric("Predicted default risk", f"{fast['probability_bad']:.0%}")
    middle.metric("Model flag", "Higher risk" if fast["high_risk"] else "Lower risk")
    right.metric("Supported adverse reasons in the top factors", observed)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(
            shap_waterfall(fast["factors"], fast["base_logodds"], fast["high_risk"]),
            clear_figure=True,
            width=640,
        )
        st.caption(
            "Red raises predicted default risk and teal lowers it. "
            "The dashed line is the 50% flag. The top bar is this prediction."
        )
    with chart_right:
        st.pyplot(shap_figure(fast["factors"]), clear_figure=True, width=640)
        st.caption("The same features, shown as log-odds contributions.")
    queried = ", ".join(f"{item['label']}: {item['value_label']}" for item in focus)
    st.caption(f"The policy search and the draft use these factors: {queried}.")

    table = [
        {
            "Feature": item["label"],
            "Value": item["value_label"],
            "Code": str(item["value_code"]),
            "SHAP (log-odds)": round(item["shap"], 3),
            "Evidence": item["status_label"],
        }
        for item in sorted(fast["factors"], key=lambda item: abs(item["shap"]), reverse=True)[:8]
    ]
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Retrieved guidelines")
    st.caption(fast["retrieval"]["query"])
    for index, chunk in enumerate(fast["retrieval"]["chunks"]):
        page = f", page {chunk['page']}" if chunk["page"] else ""
        with st.expander(
            f"{chunk['id']} · {chunk['source']}{page}",
            expanded=False,
            key=f"policy_{slot}_{chunk['id']}",
        ):
            st.write(chunk["text"])

    st.subheader("Draft for the reviewer" if fast["high_risk"] else "Explanation")
    notice = st.session_state.get(f"notice_{slot}")
    if st.button("Draft explanation", key=f"draft_{slot}"):
        with st.spinner("Drafting the explanation..."):
            try:
                notice = draft_notice(fast, fast["retrieval"]["chunks"])
            except Exception as error:
                notice = {"error": str(error)}
        st.session_state[f"notice_{slot}"] = notice
    if notice is None:
        st.caption("The score and the charts are ready. Draft an explanation when you want the written review.")
        return
    if notice.get("error"):
        st.error(notice["error"])
        return
    st.markdown(notice["text"])
    st.caption(f"Drafted in {notice['seconds']:.1f}s.")
    for warning in notice["warnings"]:
        st.warning(warning)


def main():
    context = cached_context()
    store = cached_store()
    _show_fairness(context["fairness"], context["metrics"])
    examples = context["examples"]

    st.title("Credit Risk Auditor")
    st.caption(
        "Compare the example applicants, or fill in Custom without changing them. "
        "The written draft is for a human reviewer."
    )

    custom_slot = len(examples)
    custom_start = _custom_start(context)
    labels = [f"Applicant {slot + 1}" for slot in range(len(examples))] + ["Custom"]

    for slot, example in enumerate(examples):
        for name, value in example["applicant"].items():
            st.session_state.setdefault(_slot_key(slot, name), value)
    for name, value in custom_start.items():
        st.session_state.setdefault(_slot_key(custom_slot, name), value)

    action_left, action_right = st.columns(2)
    with action_left:
        evaluate_all = st.button("Evaluate all", type="primary")
    with action_right:
        draft_all = st.button("Draft applicants 1–4")

    results = _results_list(len(labels))
    pending = st.session_state.pop("pending_eval", None)
    if isinstance(pending, int) and 0 <= pending < len(labels):
        try:
            with st.spinner(f"Scoring {labels[pending].lower()}..."):
                results[pending] = _score_slot(context, store, pending, examples, custom_slot)
        except Exception as error:
            st.error(f"The evaluation failed: {error}")

    if evaluate_all:
        errors = []
        with st.spinner("Scoring applicants 1–4 and retrieving policy clauses..."):
            for slot in range(custom_slot):
                try:
                    results[slot] = _score_slot(context, store, slot, examples, custom_slot)
                except Exception as error:
                    errors.append(f"Applicant {slot + 1}: {error}")
        for message in errors:
            st.error(message)

    if draft_all:
        if not any(results[slot] for slot in range(custom_slot)):
            st.info("Evaluate applicants 1–4 before drafting.")
        else:
            with st.status("Drafting applicants 1–4 in order...", expanded=True) as status:
                for slot in range(custom_slot):
                    fast = results[slot]
                    if not fast:
                        st.write(f"Applicant {slot + 1} has no score yet.")
                        continue
                    status.update(label=f"Drafting applicant {slot + 1} of 4...")
                    try:
                        notice = draft_notice(fast, fast["retrieval"]["chunks"])
                    except Exception as error:
                        notice = {"error": str(error)}
                    st.session_state[f"notice_{slot}"] = notice
                    if notice.get("error"):
                        st.write(f"Applicant {slot + 1} failed: {notice['error']}")
                    else:
                        st.write(f"Applicant {slot + 1} drafted.")
                status.update(label="Drafts for applicants 1–4 are in their cards.", state="complete")

    summary = st.columns(custom_slot)
    for slot in range(custom_slot):
        fast = results[slot]
        if not fast:
            summary[slot].metric(labels[slot], "—")
            continue
        flag = "Higher risk" if fast["high_risk"] else "Lower risk"
        summary[slot].metric(labels[slot], f"{fast['probability_bad']:.0%}", flag, delta_color="off")

    st.caption(
        "Evaluate all scores applicants 1–4. Draft applicants 1–4 writes their explanations in that order. "
        "Each tab can score one applicant. Custom stays separate."
    )

    tabs = st.tabs(labels)
    for slot, tab in enumerate(tabs):
        with tab:
            if slot == custom_slot:
                st.caption("Set your own values here. The example applicants stay as they are.")
                reset_label = "Reset custom"
                evaluate_label = "Evaluate custom"
                restore = custom_start
            else:
                example = examples[slot]
                st.caption(
                    f"Starts from dataset row {example['row']} "
                    f"(historical {example['historical']}). Edit any field."
                )
                reset_label = f"Reset applicant {slot + 1}"
                evaluate_label = f"Evaluate applicant {slot + 1}"
                restore = example["applicant"]
            control_left, control_right = st.columns(2)
            with control_left:
                st.button(
                    reset_label,
                    key=f"reset_{slot}",
                    on_click=_reset_slot,
                    args=(slot, restore),
                )
            with control_right:
                st.button(evaluate_label, key=f"eval_{slot}", on_click=_queue_eval, args=(slot,))
            _applicant_fields(slot, context)
            fast = results[slot]
            if not fast:
                st.caption(f"Choose {evaluate_label} to score this card.")
                continue
            flag = "Higher risk" if fast["high_risk"] else "Lower risk"
            st.subheader(f"{fast['probability_bad']:.0%} {flag.lower()}")
            _show_result(slot, fast)


if __name__ == "__main__":
    main()
