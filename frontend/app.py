"""Streamlit frontend for HireFair resume screening.

Locked Phase 9 implementation choices:
- Real HTTP communication: Communicates with FastAPI backend exclusively via requests.
  Does NOT import FastAPI, pipeline, or agent modules directly.
- .txt upload interface: Accepts .txt job description and multiple .txt resumes.
- Manual smoke-testing oriented: Simple, clean UI for verifying API integration
  and decision categorization before building the full reviewer dashboard.
- Clear decision differentiation: Visually differentiates CLEARED, FLAGGED_FOR_REVIEW,
  NOT_QUALIFIED, INCOMPLETE_DATA, DUPLICATE, and pipeline failures.
"""

import os
from typing import Any, Dict, List
import requests
import streamlit as st

# Default backend URL (configurable via environment variable)
DEFAULT_API_URL = os.environ.get("HIREFAIR_API_URL", "http://127.0.0.1:8000")

# Streamlit page setup
st.set_page_config(
    page_title="HireFair — Resume Screening",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ HireFair — Fairness-Aware Resume Screening")
st.markdown(
    "Automated screening with independent fairness auditing and deterministic routing. "
    "Upload a job description and candidate resumes to evaluate qualifications and detect potential biases."
)

# Sidebar configuration & health indicator
with st.sidebar:
    st.header("⚙️ Server Configuration")
    api_url = st.text_input("FastAPI Backend URL", value=DEFAULT_API_URL)

    if st.button("Check Backend Health"):
        try:
            health_resp = requests.get(f"{api_url}/health", timeout=5)
            if health_resp.status_code == 200:
                st.success(f"Backend healthy: {health_resp.json()}")
            else:
                st.error(f"Backend returned status {health_resp.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Connection failed: {exc}")

    st.markdown("---")
    st.markdown(
        "**Phase 9 Architecture**\n"
        "- Synchronous /screen endpoint\n"
        "- .txt files only (in-memory)\n"
        "- Real HTTP requests via `requests`\n"
        "- Independent Fairness Auditor critic"
    )

# File Upload Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    jd_file = st.file_uploader(
        "Upload Job Description (.txt only)",
        type=["txt"],
        key="jd_uploader",
        help="Single plain-text job description file",
    )

with col2:
    st.subheader("2. Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload Candidate Resumes (.txt only)",
        type=["txt"],
        accept_multiple_files=True,
        key="resumes_uploader",
        help="One or more plain-text resume files",
    )

st.markdown("---")

# Screening Action
if st.button("🚀 Screen Candidates", type="primary", use_container_width=True):
    if not jd_file:
        st.warning("Please upload a job description file (.txt) before submitting.")
    elif not resume_files:
        st.warning("Please upload at least one candidate resume file (.txt) before submitting.")
    else:
        # Prepare multipart/form-data upload payload in-memory
        files_payload: List[Any] = [
            ("job_description", (jd_file.name, jd_file.getvalue(), "text/plain"))
        ]
        for rf in resume_files:
            files_payload.append(("resumes", (rf.name, rf.getvalue(), "text/plain")))

        try:
            with st.spinner("Processing candidates through HireFair LangGraph Pipeline and Router..."):
                response = requests.post(
                    f"{api_url}/screen",
                    files=files_payload,
                    timeout=180,
                )

            if response.status_code == 200:
                data: Dict[str, Any] = response.json()
                results = data.get("results", [])
                failed_candidates = data.get("failed_candidates", [])

                st.success(f"Screening complete! Processed {len(results)} candidate(s).")

                # Summary metrics
                cleared_count = sum(1 for r in results if r.get("decision") == "cleared")
                flagged_count = sum(1 for r in results if r.get("decision") == "flagged_for_review")
                unqual_count = sum(1 for r in results if r.get("decision") == "not_qualified")
                inc_count = sum(1 for r in results if r.get("decision") == "incomplete_data")
                dup_count = sum(1 for r in results if r.get("decision") == "duplicate")
                fail_count = len(failed_candidates)

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Cleared", cleared_count)
                m2.metric("Flagged", flagged_count)
                m3.metric("Not Qualified", unqual_count)
                m4.metric("Incomplete", inc_count)
                m5.metric("Duplicate", dup_count)
                m6.metric("Failed", fail_count)

                st.markdown("### Candidate Screening Results")

                # Render successful candidate decisions
                for r in results:
                    cid = r.get("candidate_id", "Unknown")
                    decision = r.get("decision", "Unknown").upper()
                    reason = r.get("reason", "")
                    final_score = r.get("final_score")
                    dup_of = r.get("duplicate_of")
                    profile = r.get("candidate_profile") or {}
                    cand_name = profile.get("candidate_name") or cid
                    audit = r.get("audit_record") or {}
                    scores = r.get("scores") or []

                    score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

                    # Category styling
                    if decision == "CLEARED":
                        badge = "🟢 CLEARED"
                        border_color = "green"
                    elif decision == "FLAGGED_FOR_REVIEW":
                        badge = "🟠 FLAGGED FOR REVIEW"
                        border_color = "orange"
                    elif decision == "NOT_QUALIFIED":
                        badge = "⚪ NOT QUALIFIED"
                        border_color = "gray"
                    elif decision == "INCOMPLETE_DATA":
                        badge = "🟡 INCOMPLETE DATA"
                        border_color = "gold"
                    elif decision == "DUPLICATE":
                        badge = "🟣 DUPLICATE"
                        border_color = "purple"
                    else:
                        badge = f"🔵 {decision}"
                        border_color = "blue"

                    with st.expander(f"{badge} — {cand_name} (ID: `{cid}`) | Final Score: {score_str}", expanded=(decision == "FLAGGED_FOR_REVIEW")):
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.markdown(f"**Decision:** `{decision}`")
                            st.markdown(f"**Reason:** {reason}")
                            if dup_of:
                                st.markdown(f"**Duplicate Of:** `{dup_of}`")

                        with col_b:
                            st.metric("Final Weighted Score", score_str)

                        # Expose Auditor details and evidence for reviewer visibility
                        if audit:
                            st.markdown("#### 🔍 Fairness Auditor Critic Details")
                            aud_col1, aud_col2, aud_col3 = st.columns(3)
                            orig_sc = audit.get("original_score")
                            anon_sc = audit.get("anonymized_score")
                            delta_sc = audit.get("delta")

                            aud_col1.metric("Original Score", f"{orig_sc:.2f}" if orig_sc is not None else "N/A")
                            aud_col2.metric("Anonymized Score", f"{anon_sc:.2f}" if anon_sc is not None else "N/A")
                            aud_col3.metric("Counterfactual Delta", f"{delta_sc:.2f}" if delta_sc is not None else "N/A")

                            st.markdown(f"- **Citation Valid:** {'✅ Yes' if audit.get('citation_valid') else '❌ No'}")
                            st.markdown(f"- **Non-Traditional Evidence Detected:** {'Yes' if audit.get('non_traditional_evidence_found') else 'No'}")
                            st.markdown(f"- **Self-Repair Applied (Option A):** {'Yes' if audit.get('repair_applied') else 'No'}")
                            if audit.get("flagged_reason"):
                                st.info(f"**Auditor Flagged Rationale:** {audit.get('flagged_reason')}")

                        # Expose rubric criteria scores
                        if scores:
                            st.markdown("#### 📊 Criterion Scores & Quotes")
                            score_rows = []
                            for sc in scores:
                                score_rows.append({
                                    "Criterion ID": sc.get("criterion_id"),
                                    "Score": f"{sc.get('score', 0):.2f}",
                                    "Confidence": f"{sc.get('confidence', 0):.2f}",
                                    "Evidence Quote": sc.get("evidence_quote") or "(No evidence quote)",
                                })
                            st.table(score_rows)

                # Render pipeline failures
                if failed_candidates:
                    st.markdown("### ⚠️ Pipeline Execution Failures")
                    for fc in failed_candidates:
                        st.error(
                            f"**Candidate:** `{fc.get('candidate_id')}` — **Error:** {fc.get('reason')}"
                        )

            else:
                try:
                    err_detail = response.json().get("detail", response.text)
                except Exception:
                    err_detail = response.text
                st.error(f"API Error ({response.status_code}): {err_detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to FastAPI server at `{api_url}`. "
                "Please verify that the backend is running (`uvicorn app.api.main:app`)."
            )
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")
