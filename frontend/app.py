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

                # Summary metrics based on Router decisions
                cleared_results = [r for r in results if r.get("decision") == "cleared"]
                flagged_results = [r for r in results if r.get("decision") == "flagged_for_review"]
                unqual_results = [r for r in results if r.get("decision") == "not_qualified"]
                inc_results = [r for r in results if r.get("decision") == "incomplete_data"]
                dup_results = [r for r in results if r.get("decision") == "duplicate"]
                fail_count = len(failed_candidates)

                scheduled_count = sum(
                    1 for r in cleared_results
                    if (r.get("scheduling_result") or {}).get("status") == "scheduled"
                )
                no_avail_count = sum(
                    1 for r in cleared_results
                    if (r.get("scheduling_result") or {}).get("status") == "not_scheduled_no_availability"
                )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Cleared", len(cleared_results))
                m2.metric("Flagged", len(flagged_results))
                m3.metric("Not Qualified", len(unqual_results))
                m4.metric("Incomplete Data", len(inc_results))
                m5.metric("Duplicate", len(dup_results))

                # Nested breakdown under Cleared
                st.info(
                    f"**🟢 Cleared Scheduling Breakdown:** "
                    f"📅 **Scheduled:** `{scheduled_count}` | "
                    f"⏳ **Not Scheduled (No Availability in 48h):** `{no_avail_count}`"
                )

                # ===================================================================
                # Section 1: Reviewer Queue (Flagged, Incomplete Data, Duplicate)
                # ===================================================================
                reviewer_queue = flagged_results + inc_results + dup_results
                st.markdown("---")
                st.subheader(f"📋 Reviewer Queue ({len(reviewer_queue)})")
                st.markdown(
                    "Candidates requiring manual human review due to fairness audit flags, "
                    "incomplete profile information, or potential duplicate submissions."
                )

                if not reviewer_queue:
                    st.success("✅ No candidates currently in the Reviewer Queue.")
                else:
                    for r in reviewer_queue:
                        cid = r.get("candidate_id", "Unknown")
                        decision = r.get("decision", "Unknown")
                        reason = r.get("reason", "")
                        final_score = r.get("final_score")
                        dup_of = r.get("duplicate_of")
                        profile = r.get("candidate_profile") or {}
                        cand_name = profile.get("candidate_name") or cid
                        audit = r.get("audit_record") or {}
                        scores = r.get("scores") or []

                        score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

                        if decision == "flagged_for_review":
                            badge = "🟠 FLAGGED FOR REVIEW"
                        elif decision == "incomplete_data":
                            badge = "🟡 INCOMPLETE DATA"
                        elif decision == "duplicate":
                            badge = "🟣 DUPLICATE"
                        else:
                            badge = f"🔵 {decision.upper()}"

                        with st.expander(
                            f"{badge} — {cand_name} (ID: `{cid}`) | Final Score: {score_str}",
                            expanded=True,
                        ):
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.markdown(f"**Decision:** `{decision}`")
                                st.markdown(f"**Reason:** {reason}")
                                if decision == "duplicate" and dup_of:
                                    st.warning(f"⚠️ **Duplicate Of Candidate:** `{dup_of}`")
                            with col_b:
                                st.metric("Final Weighted Score", score_str)

                            # Candidate Profile
                            if profile:
                                st.markdown("#### 👤 Candidate Profile")
                                skills = profile.get("skills", [])
                                if skills:
                                    st.markdown(f"**Skills:** {', '.join(skills)}")
                                exp = profile.get("experience", [])
                                if exp:
                                    st.markdown("**Work Experience:**")
                                    for e in exp:
                                        st.markdown(
                                            f"- **{e.get('title')}** at {e.get('company')} "
                                            f"({e.get('duration_months', 0)} mos): {e.get('description', '')}"
                                        )
                                edu = profile.get("education", [])
                                if edu:
                                    st.markdown("**Education:**")
                                    for ed in edu:
                                        st.markdown(
                                            f"- {ed.get('degree')} in {ed.get('field_of_study')} from "
                                            f"{ed.get('institution')} ({ed.get('graduation_year', 'N/A')})"
                                        )

                            # Auditor details
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

                            # Rubric criterion scores and quotes
                            if scores:
                                st.markdown("#### 📊 Criterion Scores & Quotes")
                                score_rows = [
                                    {
                                        "Criterion ID": sc.get("criterion_id"),
                                        "Score": f"{sc.get('score', 0):.2f}",
                                        "Confidence": f"{sc.get('confidence', 0):.2f}",
                                        "Evidence Quote": sc.get("evidence_quote") or "(No evidence quote)",
                                    }
                                    for sc in scores
                                ]
                                st.table(score_rows)

                # ===================================================================
                # Section 2: Cleared Candidates & Automated Interview Scheduling
                # ===================================================================
                st.markdown("---")
                st.subheader(f"🟢 Cleared Candidates ({len(cleared_results)})")
                st.markdown(
                    "Candidates who met all qualification criteria without bias flags, "
                    "automatically scheduled for interviews via deterministic slot assignment."
                )

                if not cleared_results:
                    st.info("No cleared candidates in this screening batch.")
                else:
                    for r in cleared_results:
                        cid = r.get("candidate_id", "Unknown")
                        final_score = r.get("final_score")
                        profile = r.get("candidate_profile") or {}
                        cand_name = profile.get("candidate_name") or cid
                        sched = r.get("scheduling_result") or {}
                        sched_status = sched.get("status", "ineligible")
                        sched_at = sched.get("scheduled_at")
                        duration = sched.get("duration_minutes", 30)
                        reason = r.get("reason", "")
                        scores = r.get("scores") or []

                        score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

                        if sched_status == "scheduled":
                            exp_title = (
                                f"🟢 CLEARED & SCHEDULED — {cand_name} (ID: `{cid}`) | "
                                f"📅 {sched_at} UTC"
                            )
                        elif sched_status == "not_scheduled_no_availability":
                            exp_title = (
                                f"🟠 CLEARED (NO AVAILABILITY) — {cand_name} (ID: `{cid}`) | "
                                f"Final Score: {score_str}"
                            )
                        else:
                            exp_title = f"🟢 CLEARED — {cand_name} (ID: `{cid}`) | Final Score: {score_str}"

                        with st.expander(exp_title, expanded=False):
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.markdown(f"**Qualification Reason:** {reason}")
                                if sched_status == "scheduled":
                                    st.success(
                                        f"📅 **Interview Scheduled:** `{sched_at} UTC` "
                                        f"(Duration: {duration} minutes)"
                                    )
                                    if sched.get("reason"):
                                        st.caption(sched.get("reason"))
                                elif sched_status == "not_scheduled_no_availability":
                                    st.warning(
                                        f"⏳ **Interview Not Scheduled:** {sched.get('reason', 'No slots available in 48h.')}"
                                    )
                            with col_b:
                                st.metric("Final Weighted Score", score_str)

                            if profile:
                                skills = profile.get("skills", [])
                                if skills:
                                    st.markdown(f"**Skills:** {', '.join(skills)}")

                            if scores:
                                st.markdown("#### 📊 Criterion Scores & Quotes")
                                score_rows = [
                                    {
                                        "Criterion ID": sc.get("criterion_id"),
                                        "Score": f"{sc.get('score', 0):.2f}",
                                        "Confidence": f"{sc.get('confidence', 0):.2f}",
                                        "Evidence Quote": sc.get("evidence_quote") or "(No evidence quote)",
                                    }
                                    for sc in scores
                                ]
                                st.table(score_rows)

                # ===================================================================
                # Section 3: Not Qualified Candidates (Informational)
                # ===================================================================
                st.markdown("---")
                st.subheader(f"⚪ Not Qualified ({len(unqual_results)})")
                st.info(
                    "ℹ️ The following candidate(s) did not meet the minimum qualification bar "
                    "(final score ≥ 0.50 and all must-have requirements ≥ 0.30). "
                    "No reviewer action is required."
                )

                if unqual_results:
                    for r in unqual_results:
                        cid = r.get("candidate_id", "Unknown")
                        final_score = r.get("final_score")
                        profile = r.get("candidate_profile") or {}
                        cand_name = profile.get("candidate_name") or cid
                        reason = r.get("reason", "")
                        scores = r.get("scores") or []

                        score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

                        with st.expander(
                            f"⚪ NOT QUALIFIED — {cand_name} (ID: `{cid}`) | Final Score: {score_str}",
                            expanded=False,
                        ):
                            st.markdown(f"**Reason:** {reason}")
                            if scores:
                                st.markdown("#### 📊 Criterion Scores")
                                score_rows = [
                                    {
                                        "Criterion ID": sc.get("criterion_id"),
                                        "Score": f"{sc.get('score', 0):.2f}",
                                        "Confidence": f"{sc.get('confidence', 0):.2f}",
                                        "Evidence Quote": sc.get("evidence_quote") or "(No evidence quote)",
                                    }
                                    for sc in scores
                                ]
                                st.table(score_rows)

                # ===================================================================
                # Section 4: Pipeline Failures (if any)
                # ===================================================================
                if failed_candidates:
                    st.markdown("---")
                    st.subheader(f"⚠️ Pipeline Failures ({len(failed_candidates)})")
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
