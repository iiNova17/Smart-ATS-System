"""Smart ATS: local candidate workflows with privately configured model inference."""

import html
import json
import logging
import os
from pathlib import Path

import pandas as pd
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
    clear_cvs,
    get_cv,
    list_cvs,
    open_workspace,
    rank_cvs,
    remove_cv,
    search_cvs,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
st.set_page_config(page_title="Smart ATS · Talent workspace", page_icon="◈", layout="wide")
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
        st.error("We couldn’t complete this request. Please try again.")


def chips(values, missing=False):
    if not values:
        st.caption("None listed")
        return
    cls = "chip missing" if missing else "chip"
    st.markdown(
        "".join(f'<span class="{cls}">{html.escape(str(value))}</span>' for value in values),
        unsafe_allow_html=True,
    )


def empty(title, description, symbol="◈"):
    st.markdown(
        f'<div class="empty"><div class="empty-symbol">{html.escape(symbol)}</div>'
        f"<strong>{html.escape(title)}</strong><p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def bullet_details(values):
    for value in values:
        st.text("• " + value)
    if not values:
        st.caption("Not stated in the CV.")


def page_header(title, description):
    st.title(title)
    st.caption(description)


def configured_client():
    url = api_url_from_env()
    key = os.getenv("ATS_API_KEY", "").strip()
    signature = (url, key)
    if st.session_state.get("endpoint_signature") != signature:
        st.session_state.client = None
        st.session_state.endpoint_signature = signature
    if not url or not key:
        return None
    if st.session_state.client is None:
        client = APIClient(url, key)
        health = client.health()
        if health.get("status") != "ok" or health.get("version") != "3.0.0":
            raise BackendError("AI needs an update. Please contact your administrator.")
        st.session_state.client = client
    return st.session_state.client


init_state()
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">Smart ATS</div>'
        '<div class="brand-sub">Talent workspace</div></div></div>'
        '<div class="sidebar-section">WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    for route, label, icon in [
        ("CV library", "Candidates", ":material/folder_open:"),
        ("Search candidates", "Search", ":material/manage_search:"),
        ("Rank candidates", "Shortlist", ":material/format_list_numbered:"),
        ("CV analysis", "Candidate profile", ":material/badge:"),
    ]:
        st.button(
            label,
            icon=icon,
            key="nav_" + route.replace(" ", "_").lower(),
            use_container_width=True,
            type="primary" if st.session_state.page == route else "secondary",
            on_click=navigate,
            args=(route,),
        )
    st.markdown(
        '<div class="sidebar-note"><div class="sidebar-section">THOUGHTFUL HIRING</div>'
        "<p>A clear view of every candidate.<br>Evidence behind every match.</p></div>",
        unsafe_allow_html=True,
    )

try:
    workspace = open_workspace(os.getenv("ATS_DATA_DIR") or ROOT / ".data", configured_client)
    cvs = list_cvs(workspace)
except Exception as exc:
    logging.getLogger(__name__).error("Local storage unavailable: %s", type(exc).__name__)
    st.error("Your local library could not be opened. Check that its folder is writable.")
    st.stop()

library_ids = tuple(sorted(cv["id"] for cv in cvs))
if st.session_state.get("library_ids") != library_ids:
    invalidate()
    st.session_state.library_ids = library_ids

page = st.session_state.page
breadcrumbs = {
    "CV library": "CANDIDATES",
    "Search candidates": "SEARCH",
    "Rank candidates": "SHORTLIST",
    "CV analysis": "CANDIDATE PROFILE",
}
st.markdown(
    '<div class="workspace-top"><span class="eyebrow">WORKSPACE / '
    + breadcrumbs[page]
    + '</span><span class="workspace-tag">CV intelligence</span></div>',
    unsafe_allow_html=True,
)


def library_page():
    page_header("Candidate library", "Bring your candidates together. Find the skills that matter.")
    metrics = st.columns(3)
    metrics[0].metric("Candidates", len(cvs))
    analyzed = sum(cv["analyzed"] for cv in cvs)
    metrics[1].metric("Profiles ready", analyzed)
    metrics[2].metric("Awaiting analysis", len(cvs) - analyzed)

    import_column, library_column = st.columns([1, 1.75], gap="large")
    with import_column:
        with st.container(border=True):
            st.markdown(
                '<div class="card-eyebrow">GROW YOUR TALENT POOL</div>', unsafe_allow_html=True
            )
            st.subheader("Add candidates")
            st.caption("Upload one CV or add a batch in a single step.")
            files = st.file_uploader(
                "Upload CVs",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.upload_version}",
                help="PDF, Word, or text documents. Scanned documents need readable text first.",
            )
            st.caption("PDF, DOCX, TXT · Original documents stay on your device.")
            if st.button(
                "Add to library",
                icon=":material/add:",
                type="primary",
                use_container_width=True,
                disabled=not files,
            ):
                outcomes = []
                progress = st.progress(0, text="Preparing documents…")
                for i, file in enumerate(files):
                    try:
                        cv = extract_cv(file.name, file.getvalue())
                        result = add_cv(workspace, cv.model_dump())
                        status = "Already added" if result["duplicate"] else "Added"
                        outcomes.append(
                            {"Document": file.name, "Status": status, "Details": "Ready to review"}
                        )
                    except Exception as exc:
                        detail = (
                            str(exc)
                            if isinstance(exc, (ValueError, BackendError))
                            else "Could not read this document."
                        )
                        outcomes.append(
                            {"Document": file.name, "Status": "Needs attention", "Details": detail}
                        )
                    progress.progress(
                        (i + 1) / len(files), text=f"Processed {i + 1} of {len(files)} documents"
                    )
                st.session_state.upload_results = outcomes
                invalidate()
                st.rerun()
            st.markdown(
                '<div class="import-note"><strong>From upload to understanding</strong>'
                "<p>Search skills across your library, build a shortlist, then open a profile to explore the details.</p></div>",
                unsafe_allow_html=True,
            )
        if st.session_state.upload_results:
            outcomes = st.session_state.upload_results
            added = sum(row["Status"] == "Added" for row in outcomes)
            errors = sum(row["Status"] == "Needs attention" for row in outcomes)
            st.caption(f"Last import · {added} added · {errors} need attention")
            with st.expander("Import details", expanded=bool(errors)):
                st.dataframe(outcomes, hide_index=True, use_container_width=True)

    with library_column:
        title, refresh = st.columns([4, 1])
        title.subheader("Your candidates")
        if refresh.button("Refresh", icon=":material/refresh:"):
            invalidate()
            st.rerun()
        if not cvs:
            empty(
                "Your talent pool starts here",
                "Add CVs to search, compare, and understand your candidates.",
            )
            return
        filter_text = st.text_input(
            "Find a candidate", placeholder="Search name or document…", label_visibility="collapsed"
        )
        visible = [
            cv
            for cv in cvs
            if filter_text.casefold() in ((cv.get("name") or "") + " " + cv["filename"]).casefold()
        ]
        st.caption(f"{len(visible)} " + ("candidate" if len(visible) == 1 else "candidates"))
        for cv in visible:
            with st.container(border=True):
                info, action = st.columns([4, 1.3])
                name = (
                    cv.get("name")
                    or Path(cv["filename"]).stem.replace("_", " ").replace("-", " ").title()
                )
                label = "Profile ready" if cv["analyzed"] else "Ready to analyze"
                info.markdown(
                    '<div class="candidate-row"><div class="candidate-avatar">'
                    + html.escape("".join(word[0] for word in name.split()[:2]))
                    + '</div><div><div class="candidate-name">'
                    + html.escape(name)
                    + '</div><div class="candidate-meta">'
                    + html.escape(cv.get("headline") or cv["filename"])
                    + "</div></div></div>",
                    unsafe_allow_html=True,
                )
                info.caption(label)
                action.button(
                    "View profile",
                    key="view_" + cv["id"],
                    use_container_width=True,
                    on_click=navigate,
                    args=("CV analysis", cv["id"]),
                )
        if not visible:
            empty("No candidates found", "Try another name or document.")
        with st.expander("Manage documents"):
            labels = {cv["id"]: cv["filename"] for cv in cvs}
            delete_id = st.selectbox("Document to remove", list(labels), format_func=labels.get)
            if st.button("Remove document"):
                try:
                    remove_cv(workspace, delete_id)
                    invalidate()
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
            clear_check = st.checkbox("Remove every document from this workspace")
            if st.button("Clear library", disabled=not clear_check):
                try:
                    clear_cvs(workspace)
                    invalidate()
                    st.session_state.upload_results = []
                    st.session_state.upload_version += 1
                    st.rerun()
                except Exception as exc:
                    show_error(exc)


def search_page():
    page_header(
        "Find your next match", "Search across your candidates by skills, keywords, or experience."
    )
    with st.form("search_form", border=True):
        query = st.text_input(
            "What are you looking for?", placeholder="Python, SQL, LangChain", max_chars=1000
        )
        left, right = st.columns(2)
        method = left.selectbox("Search by", ["Skills & keywords", "Experience & meaning"])
        rule = right.selectbox("Match", ["All keywords", "Any keyword"])
        st.caption("Separate skills with commas, or describe the experience you need.")
        submitted = st.form_submit_button(
            "Find candidates",
            icon=":material/search:",
            type="primary",
            disabled=not cvs,
        )
    if submitted:
        st.session_state.search_results = None
        try:
            request = validate_search(
                {
                    "query": query,
                    "mode": "keyword" if method == "Skills & keywords" else "semantic",
                    "match": "all" if rule == "All keywords" else "any",
                }
            )
            with st.spinner("Finding relevant candidates…"):
                rows = search_cvs(workspace, request)
            st.session_state.search_results = {
                "rows": rows,
                "query": query,
                "mode": request["mode"],
            }
        except Exception as exc:
            show_error(exc)
    result = st.session_state.search_results
    if not cvs:
        empty("Add your first candidates", "Upload CVs in Candidates to start searching.")
    elif result is None:
        empty(
            "The right experience, within reach",
            "Search your library to see relevant candidates and the passages behind each match.",
        )
    else:
        rows = result["rows"]
        heading, download = st.columns([4, 1.6])
        heading.subheader(f"{len(rows)} " + ("match" if len(rows) == 1 else "matches"))
        st.caption("Results for: " + result["query"])
        if rows:
            export = [
                {
                    "Candidate": row["filename"],
                    "Matches": ", ".join(row["matched"]),
                    "Score": row["score"],
                }
                for row in rows
            ]
            download.download_button(
                "Export results",
                csv_export(export),
                "candidate_search.csv",
                "text/csv",
                icon=":material/download:",
            )
        if not rows:
            empty("No matches yet", "Try fewer keywords, another phrase, or matching any keyword.")
        for row in rows:
            with st.container(border=True):
                info, action = st.columns([4, 1.3])
                info.text(row["filename"])
                if result["mode"] == "keyword":
                    with info:
                        chips(row["matched"])
                else:
                    info.caption("Matched by relevant experience")
                action.button(
                    "View profile",
                    key="search_" + row["id"],
                    on_click=navigate,
                    args=("CV analysis", row["id"]),
                    use_container_width=True,
                )
                with st.expander("Why this candidate matches"):
                    for item in row["evidence"]:
                        st.text(item["skill"])
                        st.text(item["quote"])


def rank_page():
    page_header(
        "Build a stronger shortlist", "Compare candidates against the skills your role needs."
    )
    with st.form("rank_form", border=True):
        left, right = st.columns(2)
        required = left.text_area(
            "Required skills",
            placeholder="Python, SQL",
            height=105,
            help="Separate skills with commas.",
        )
        preferred = right.text_area(
            "Nice-to-have skills",
            placeholder="LangChain, RAG",
            height=105,
            help="Separate skills with commas.",
        )
        job_description = st.text_area(
            "Role description",
            placeholder="Add context about the role (optional).",
            help="Uses AI to order candidates with equal skill coverage. Leave empty for skill-only ranking.",
            height=115,
            max_chars=6000,
        )
        submitted = st.form_submit_button(
            "Create shortlist",
            icon=":material/format_list_numbered:",
            type="primary",
            disabled=not cvs,
        )
    if submitted:
        st.session_state.rank_results = None
        try:
            criteria = validate_rank(
                {
                    "required": split_skills(required),
                    "preferred": split_skills(preferred),
                    "job_description": job_description,
                }
            )
            with st.spinner("Comparing your candidates…"):
                rows = rank_cvs(workspace, criteria)
            st.session_state.rank_results = {"rows": rows, "criteria": criteria}
        except Exception as exc:
            show_error(exc)
    result = st.session_state.rank_results
    if not cvs:
        empty(
            "Your shortlist starts with candidates",
            "Upload CVs in Candidates, then define the skills for your role.",
        )
    elif result is None:
        empty(
            "A clear comparison, backed by evidence",
            "Add the skills you need to see who matches and where information is missing.",
        )
    else:
        rows, criteria = result["rows"], result["criteria"]
        heading, download = st.columns([4, 1.6])
        heading.subheader("Your shortlist")
        table = [
            {
                "Rank": row["rank"],
                "Candidate": row["filename"],
                "Skill coverage": row["score"],
                "Required matches": ", ".join(row["matched_required"]),
                "Not found": ", ".join(row["missing_required"]),
                "Nice-to-have matches": ", ".join(row["matched_preferred"]),
            }
            for row in rows
        ]
        download.download_button(
            "Export shortlist",
            csv_export(table),
            "candidate_shortlist.csv",
            "text/csv",
            icon=":material/download:",
        )
        st.caption("Based on: " + ", ".join(criteria["required"] + criteria["preferred"]))
        st.dataframe(
            pd.DataFrame(table),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Skill coverage": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                )
            },
        )
        with st.expander("How the comparison works"):
            st.write(
                "Required skills contribute 80% and nice-to-have skills 20%. If you use only one group, it contributes the full score."
            )
            st.write(
                "Coverage reflects terms found in a CV, not proficiency. Check the evidence and context. A role description breaks ties by relevant experience."
            )
        for row in rows:
            with st.expander(
                f"{row['rank']:02d} · {row['filename']} · {row['score']:.0f}% coverage"
            ):
                match, missing = st.columns(2)
                with match:
                    st.caption("MATCHED SKILLS")
                    chips(row["matched_required"] + row["matched_preferred"])
                with missing:
                    st.caption("NOT FOUND IN THE CV")
                    chips(row["missing_required"] + row["missing_preferred"], missing=True)
                for item in row["evidence"]:
                    st.text(item["skill"] + ": " + item["quote"])
                st.button(
                    "View candidate",
                    key="rank_" + row["id"],
                    on_click=navigate,
                    args=("CV analysis", row["id"]),
                )


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
                try:
                    with st.spinner("Reading the CV and preparing the profile…"):
                        st.session_state.analysis_cache[cv_id] = validate_analysis(
                            analyze_cv(workspace, cv_id)
                        )
                    st.rerun()
                except Exception as exc:
                    show_error(exc)
        with st.expander("Preview document", expanded=True):
            st.text_area(
                "CV text", record["text"], height=340, disabled=True, key="source_" + cv_id
            )
        return

    with st.container(border=True):
        st.subheader(data["name"] or Path(record["filename"]).stem)
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
