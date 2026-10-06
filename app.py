"""Smart ATS: a local candidate workspace powered by private model inference."""

import html
import json
import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ats.client import APIClient, BackendError, api_url_from_env
from ats.extraction import extract_cv
from ats.matching import csv_export, split_skills
from ats.validation import validate_analysis, validate_rank, validate_search
from ats.workspace import (
    add_cv,
    analyze_cv,
    ask_cv,
    assess_candidates,
    clear_cvs,
    get_cv,
    list_cvs,
    open_workspace,
    prepare_criteria,
    remove_cv,
)

ROOT = Path(__file__).resolve().parent
# Environment variables take precedence, so deployments and isolated checks can override .env.
load_dotenv(ROOT / ".env", override=False)
st.set_page_config(page_title="Smart ATS", page_icon="◈", layout="wide")
st.markdown(
    "<style>" + (ROOT / "assets/style.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "client": None,
        "search_results": None,
        "rank_results": None,
        "analysis_cache": {},
        "answers": {},
        "selected_cv": None,
        "page": "CV library",
        "upload_results": [],
        "upload_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def invalidate():
    st.session_state.search_results = None
    st.session_state.rank_results = None
    st.session_state.analysis_cache = {}
    st.session_state.answers = {}


def navigate(page, cv_id=None):
    st.session_state.page = page
    if cv_id:
        st.session_state.selected_cv = cv_id


def show_error(exc):
    if isinstance(exc, (BackendError, ValueError, KeyError)):
        st.error(str(exc))
    else:
        logging.getLogger(__name__).error("Candidate operation failed: %s", type(exc).__name__)
        st.error("This operation could not finish. Please retry.")


def chips(values, missing=False):
    if not values:
        st.caption("None listed")
        return
    cls = "chip missing" if missing else "chip"
    st.markdown(
        "".join(f'<span class="{cls}">{html.escape(str(value))}</span>' for value in values),
        unsafe_allow_html=True,
    )


def empty(title, description, symbol=""):
    st.markdown(
        f'<div class="empty"><strong>{html.escape(title)}</strong>'
        f"<p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def bullet_details(values):
    for value in values:
        st.text("• " + value)
    if not values:
        st.caption("Not stated in the CV.")


def page_header(title, description):
    st.markdown(
        f'<div class="page-heading"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def configured_client():
    url, key = api_url_from_env(), os.getenv("ATS_API_KEY", "").strip()
    signature = (url, key)
    if st.session_state.get("endpoint_signature") != signature:
        st.session_state.client = None
        st.session_state.endpoint_signature = signature
        invalidate()
    if not url or not key:
        return None
    if st.session_state.client is None:
        client = APIClient(url, key)
        health = client.health()
        if health.get("status") != "ok" or health.get("version") != "4.1.0":
            raise BackendError(
                "The AI service needs the updated Kaggle notebook (API 4.1). "
                "Restart its server with the new notebook, then retry."
            )
        st.session_state.client = client
    return st.session_state.client


init_state()
try:
    workspace = open_workspace(os.getenv("ATS_DATA_DIR") or ROOT / ".data", configured_client)
    cvs = list_cvs(workspace)
except Exception as exc:
    logging.getLogger(__name__).error("Local storage unavailable: %s", type(exc).__name__)
    st.error("Your library could not be opened. Check that its folder is writable.")
    st.stop()

library_ids = tuple(sorted(cv["id"] for cv in cvs))
if st.session_state.get("library_ids") != library_ids:
    invalidate()
    st.session_state.library_ids = library_ids

st.markdown('<div class="brand">Smart <span>ATS</span></div>', unsafe_allow_html=True)
with st.container(key="navigation"):
    columns = st.columns(4, gap="small")
    for column, route, label in zip(
        columns,
        ["CV library", "Search candidates", "Rank candidates", "CV analysis"],
        ["Candidates", "Search", "Shortlist", "Profile"],
    ):
        column.button(
            label,
            key="nav_" + route.replace(" ", "_").lower(),
            type="primary" if st.session_state.page == route else "secondary",
            use_container_width=True,
            on_click=navigate,
            args=(route,),
        )
page = st.session_state.page


def library_page():
    page_header(
        "Your candidate library", "Add CVs, find the right experience, and explore each profile."
    )
    files = st.file_uploader(
        "Upload CVs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_version}",
        help="Upload one CV or a batch. Scanned PDFs need readable text.",
    )
    st.caption("PDF, DOCX or TXT · Documents are stored in your local workspace.")
    if st.button("Add to library", type="primary", disabled=not files, key="import_cvs"):
        outcomes = []
        progress = st.progress(0, text="Reading documents…")
        for index, file in enumerate(files, 1):
            try:
                result = add_cv(workspace, extract_cv(file.name, file.getvalue()).model_dump())
                outcomes.append(
                    {
                        "Document": file.name,
                        "Status": "Already added" if result["duplicate"] else "Added",
                        "Details": "Ready to search",
                    }
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "Document": file.name,
                        "Status": "Needs attention",
                        "Details": str(exc)
                        if isinstance(exc, ValueError)
                        else "Could not read this document.",
                    }
                )
            progress.progress(index / len(files), text=f"Read {index} of {len(files)} documents")
        st.session_state.upload_results = outcomes
        st.session_state.upload_version += 1
        invalidate()
        st.rerun()
    if st.session_state.upload_results:
        outcomes = st.session_state.upload_results
        st.caption(f"Last import · {sum(row['Status'] == 'Added' for row in outcomes)} added")
        with st.expander(
            "Import details", expanded=any(row["Status"] == "Needs attention" for row in outcomes)
        ):
            st.dataframe(outcomes, hide_index=True, use_container_width=True)
    st.divider()
    heading, action = st.columns([4, 1.5])
    heading.subheader(f"Candidates · {len(cvs)}")
    action.button(
        "Search library",
        on_click=navigate,
        args=("Search candidates",),
        use_container_width=True,
        disabled=not cvs,
    )
    if not cvs:
        empty(
            "Start with a CV",
            "Upload one or more documents above. Your library stays here between sessions.",
        )
        return
    filter_text = st.text_input("Find a candidate", placeholder="Search name or document…")
    visible = [
        cv
        for cv in cvs
        if filter_text.casefold() in ((cv.get("name") or "") + " " + cv["filename"]).casefold()
    ]
    for cv in visible:
        with st.container(border=True):
            info, action = st.columns([4, 1.5])
            info.markdown(f"**{html.escape(cv.get('name') or Path(cv['filename']).stem)}**")
            info.caption(cv.get("headline") or cv["filename"])
            info.caption("Profile ready" if cv["analyzed"] else "Ready to prepare profile")
            action.button(
                "Open profile",
                key="view_" + cv["id"],
                use_container_width=True,
                on_click=navigate,
                args=("CV analysis", cv["id"]),
            )
    if not visible:
        empty("No candidate found", "Try another name or filename.")
    with st.expander("Manage documents"):
        labels = {cv["id"]: cv["filename"] for cv in cvs}
        delete_id = st.selectbox("Document to remove", list(labels), format_func=labels.get)
        if st.button("Remove document"):
            remove_cv(workspace, delete_id)
            invalidate()
            st.rerun()
        clear_check = st.checkbox("Remove every document from this workspace")
        if st.button("Clear library", disabled=not clear_check):
            clear_cvs(workspace)
            invalidate()
            st.rerun()


def run_matching(query, required, preferred=None):
    progress = st.progress(0, text="Understanding your request…")
    client = None
    stage = "Understanding your request"
    completion = 0
    try:
        client = configured_client()

        def model_update(status, elapsed):
            label = "Waiting for the model" if status == "queued" else stage
            progress.progress(completion, text=f"{label} · {elapsed}s")

        if client:
            client.on_progress = model_update
        criteria = prepare_criteria(workspace, query, required, preferred)

        def update(done, total):
            nonlocal stage, completion
            completion = done / max(total, 1)
            stage = f"Reviewing candidate {min(done + 1, total)} of {total}"
            progress.progress(completion, text=stage + "…")

        rows = assess_candidates(workspace, criteria, update)
        return {
            "rows": rows,
            "criteria": criteria,
            "query": query,
            "reviewed": len(cvs),
            "incomplete": workspace.get("assessment_incomplete", 0),
        }
    finally:
        if client:
            client.on_progress = None
        progress.empty()


def render_results(result, shortlist=False):
    if result is None:
        empty("Describe what matters", "Combine skills and practical experience in one request.")
        return
    rows, criteria = result["rows"], result["criteria"]
    st.divider()
    heading, download = st.columns([4, 1.5])
    heading.subheader(("Your shortlist" if shortlist else "Results") + f" · {len(rows)}")
    st.caption(f"Reviewed {result['reviewed']} candidates · Showing supported matches only")
    if result.get("incomplete"):
        st.warning(
            f"{result['incomplete']} candidate review(s) had unverified requirements. "
            "Those requirements were not counted as matches. Retry the search or review the original CVs."
        )
    with st.expander("What the AI looked for"):
        st.text(criteria["intent"])
        for item in criteria["requirements"]:
            st.text(("Required: " if item["required"] else "Preferred: ") + item["label"])
    if rows:
        export = [
            {
                "Rank": index,
                "Candidate": row["name"] or row["filename"],
                "Criteria coverage": row["score"],
                "Matches": ", ".join(row["matched"]),
                "Preferred gaps": ", ".join(row["missing"]),
                "Why it fits": row["explanation"],
            }
            for index, row in enumerate(rows, 1)
        ]
        download.download_button(
            "Export results",
            csv_export(export),
            "candidate_shortlist.csv" if shortlist else "candidate_search.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        empty(
            "No supported matches",
            "No CV met all required criteria. Review the interpreted "
            "requirements, adjust your request, or add more candidates.",
        )
    if shortlist:
        st.caption(
            "Ordered by evidence-backed criteria coverage. Required criteria carry twice "
            "the weight of preferred criteria; concrete experience breaks ties. "
            "This score measures coverage, not ability."
        )
    for index, row in enumerate(rows, 1):
        with st.container(border=True):
            info, action = st.columns([4, 1.5])
            name = row["name"] or Path(row["filename"]).stem
            info.markdown(f"**{str(index) + '. ' if shortlist else ''}{html.escape(name)}**")
            info.caption(row["headline"] or row["filename"])
            if shortlist:
                info.caption(f"{row['score']:.0f}% criteria coverage")
            action.button(
                "Open profile",
                key=("rank_" if shortlist else "search_") + row["id"],
                on_click=navigate,
                args=("CV analysis", row["id"]),
                use_container_width=True,
            )
            st.text(row["explanation"])
            with st.expander("Why this candidate matters"):
                for reason in row["reasons"]:
                    st.markdown("**" + html.escape(reason["requirement"]) + "**")
                    st.text(reason["reason"])
                if row["missing"]:
                    st.caption(
                        "Preferred criteria not fully supported: " + ", ".join(row["missing"])
                    )
            with st.expander("CV evidence"):
                for item in row["evidence"]:
                    st.caption(item["skill"])
                    st.text(item["quote"])


def search_page():
    page_header("Find the right experience", "Search skills, keywords and experience together.")
    with st.form("search_form", border=False):
        query = st.text_area(
            "What are you looking for?",
            placeholder="Python and ROS 2 with hands-on autonomous robot navigation experience",
            height=110,
            max_chars=5000,
        )
        required = st.text_input(
            "Required skills (optional)",
            placeholder="Python, ROS 2",
            help="Separate skills with commas. These must be supported in the CV.",
        )
        submitted = st.form_submit_button(
            "Find candidates", type="primary", use_container_width=True, disabled=not cvs
        )
    if submitted:
        st.session_state.search_results = None
        try:
            request = validate_search({"query": query, "required": split_skills(required)})
            st.session_state.search_results = run_matching(request["query"], request["required"])
        except Exception as exc:
            show_error(exc)
    if not cvs:
        empty("Add your first candidates", "Upload CVs in Candidates to start searching.")
    else:
        render_results(st.session_state.search_results)


def rank_page():
    page_header(
        "Build your shortlist", "Compare practical experience against the work your role needs."
    )
    with st.form("rank_form", border=False):
        left, right = st.columns(2)
        required = left.text_input("Required skills", placeholder="Python, ROS 2")
        preferred = right.text_input("Preferred skills", placeholder="C++, SLAM, Linux")
        description = st.text_area(
            "Role and experience",
            height=140,
            max_chars=5000,
            placeholder="Describe the work, relevant projects and experience you need.",
        )
        submitted = st.form_submit_button(
            "Create shortlist", type="primary", use_container_width=True, disabled=not cvs
        )
    if submitted:
        st.session_state.rank_results = None
        try:
            criteria = validate_rank(
                {
                    "required": split_skills(required),
                    "preferred": split_skills(preferred),
                    "job_description": description,
                }
            )
            st.session_state.rank_results = run_matching(
                criteria["job_description"], criteria["required"], criteria["preferred"]
            )
        except Exception as exc:
            show_error(exc)
    if not cvs:
        empty("Start with your candidates", "Upload CVs in Candidates, then define your role.")
    else:
        render_results(st.session_state.rank_results, shortlist=True)


def analysis_page():
    page_header(
        "Candidate profile", "See the details, understand the experience, and explore the source."
    )
    if not cvs:
        empty(
            "Every candidate has a story",
            "Upload a CV in Candidates to explore a detailed profile.",
        )
        return
    ids = [cv["id"] for cv in cvs]
    labels = {cv["id"]: cv.get("name") or cv["filename"] for cv in cvs}
    selected = st.session_state.selected_cv
    cv_id = st.selectbox(
        "Choose a candidate",
        ids,
        index=ids.index(selected) if selected in ids else 0,
        format_func=labels.get,
    )
    st.session_state.selected_cv = cv_id
    try:
        record = get_cv(workspace, cv_id)
    except Exception as exc:
        show_error(exc)
        return
    analyzed = next(cv["analyzed"] for cv in cvs if cv["id"] == cv_id)
    data = st.session_state.analysis_cache.get(cv_id)
    if analyzed and data is None:
        try:
            data = validate_analysis(analyze_cv(workspace, cv_id))
            st.session_state.analysis_cache[cv_id] = data
        except Exception as exc:
            show_error(exc)
    if data is None:
        with st.container(border=True):
            st.subheader("Ready for a closer look")
            st.write(
                "Prepare a structured profile with experience, education, skills, projects, and supporting evidence."
            )
            if st.button("Prepare profile", type="primary", icon=":material/auto_awesome:"):
                client = None
                status_text = st.empty()
                try:
                    client = configured_client()
                    if client:
                        client.on_progress = lambda status, elapsed: status_text.caption(
                            (
                                "Waiting for the model"
                                if status == "queued"
                                else "Preparing your profile"
                            )
                            + f" · {elapsed}s"
                        )
                    with st.spinner("Reading the CV and preparing the profile…"):
                        st.session_state.analysis_cache[cv_id] = validate_analysis(
                            analyze_cv(workspace, cv_id)
                        )
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
                finally:
                    if client:
                        client.on_progress = None
                    status_text.empty()
        with st.expander("Preview document", expanded=True):
            st.text_area(
                "CV text", record["text"], height=340, disabled=True, key="source_" + cv_id
            )
        return

    with st.container(border=True):
        st.subheader(data["name"] or Path(record["filename"]).stem)
        if data["review_notes"]:
            st.warning(
                "Some profile details need review. Check Review notes and the original document below."
            )
        st.text(data["headline"] or "Professional headline not stated")
        left, middle, right = st.columns(3)
        for column, label, key in [
            (left, "EMAIL", "email"),
            (middle, "PHONE", "phone"),
            (right, "LOCATION", "location"),
        ]:
            column.caption(label)
            column.text(data[key] or "Not stated")
        st.text(data["summary"] or "Summary not provided.")
        chips(data["skills"])

    overview, experience, education, projects, questions, source = st.tabs(
        ["Overview", "Experience", "Education", "Projects", "Ask a question", "Document & export"]
    )
    with overview:
        st.subheader("Skills & evidence")
        if not data["evidence"]:
            st.caption(
                "No supporting skill excerpts were extracted. Review the document for context."
            )
        for item in data["evidence"]:
            with st.expander(item["skill"]):
                st.text(item["quote"])
        c1, c2 = st.columns(2)
        for column, title, key in [
            (c1, "Certifications", "certifications"),
            (c2, "Languages", "languages"),
            (c1, "Achievements", "achievements"),
            (c2, "Professional links", "links"),
        ]:
            with column:
                st.subheader(title)
                bullet_details(data[key])
        if data["other_details"]:
            st.subheader("Additional details")
            bullet_details(data["other_details"])
        if data["stated_years_experience"]:
            st.caption("Experience explicitly stated: " + data["stated_years_experience"])
        if data["review_notes"]:
            with st.expander("Details to review"):
                bullet_details(data["review_notes"])
    with experience:
        for item in data["experience"]:
            with st.container(border=True):
                st.text(item["role"] or "Role not stated")
                st.text(item["organization"] or "Organization not stated")
                st.caption(item["dates"] or "Dates not stated")
                bullet_details(item["details"])
        if not data["experience"]:
            empty("Experience not listed", "No work experience was found in this CV.")
    with education:
        for item in data["education"]:
            with st.container(border=True):
                st.text(item["qualification"] or "Qualification not stated")
                st.text(item["institution"] or "Institution not stated")
                st.caption(item["dates"] or "Dates not stated")
        if not data["education"]:
            empty("Education not listed", "No education details were found in this CV.")
    with projects:
        for item in data["projects"]:
            with st.container(border=True):
                st.text(item["name"] or "Project name not stated")
                st.text(item["details"])
                chips(item["technologies"])
        if not data["projects"]:
            empty("Projects not listed", "No project details were found in this CV.")
    with questions:
        st.subheader("Explore this candidate’s experience")
        with st.form("ask_" + cv_id):
            question = st.text_input(
                "Your question", placeholder="Which projects show NLP experience?", max_chars=1000
            )
            ask = st.form_submit_button(
                "Ask question", type="primary", icon=":material/chat_bubble_outline:"
            )
        if ask:
            try:
                with st.spinner("Finding the relevant details…"):
                    answer = ask_cv(workspace, cv_id, question)
                st.session_state.answers[cv_id] = {"question": question, **answer}
            except Exception as exc:
                show_error(exc)
        if cv_id in st.session_state.answers:
            answer = st.session_state.answers[cv_id]
            with st.container(border=True):
                st.text(answer["question"])
                st.text(answer["answer"])
            with st.expander("Supporting passages"):
                for i, passage in enumerate(answer["passages"], 1):
                    st.text(f"[Passage {i}] " + passage)
    with source:
        st.caption("Review extracted details against the original wording.")
        st.text_area(
            "CV text", record["text"], height=400, disabled=True, key="analyzed_source_" + cv_id
        )
        st.download_button(
            "Export profile",
            json.dumps(data, ensure_ascii=False, indent=2),
            cv_id + "_profile.json",
            "application/json",
            icon=":material/download:",
        )


{
    "CV library": library_page,
    "Search candidates": search_page,
    "Rank candidates": rank_page,
    "CV analysis": analysis_page,
}[page]()
