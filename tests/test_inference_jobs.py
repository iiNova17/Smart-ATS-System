"""Inference transport and compact-output regressions, without GPU or candidate files."""

import threading
from time import monotonic
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from ats.client import APIClient, BackendError
from backend.api import create_app
from backend.runtime import expand_assessment, expand_profile, numbered_source, source_quotes

KEY = "private-test-key-at-least-24-characters"
CV = "Alex\nSkills: Python, ROS 2.\nBuilt Python navigation nodes using ROS 2."
HEADERS = {"Authorization": "Bearer " + KEY}
CRITERIA = {
    "intent": "Python robotics",
    "requirements": [{"label": "Python", "kind": "skill", "required": True}],
}


def test_compact_profile_preserves_source_details_without_generating_them():
    lines, source = numbered_source(CV)
    assert "[L3]: Built Python" in source
    result = expand_profile(
        {
            "name": "Alex",
            "skills": ["Python", "ROS 2"],
            "experience": [{"role": "Engineer", "details": ["L3"]}],
            "projects": [{"name": "Navigation", "details": ["L3"]}],
            "achievements": ["L3"],
            "evidence": [{"skill": "Python", "quote": ["L2"]}],
        },
        lines,
    )
    assert result["experience"][0]["details"] == [lines[2]]
    assert result["projects"][0]["details"] == lines[2]
    assert result["achievements"] == [lines[2]]
    assert result["evidence"][0]["quote"] == lines[1]
    partial = expand_profile({"name": "Alex", "experience": [{"details": ["L3", 999]}]}, lines)
    assert partial["name"] == "Alex" and partial["experience"][0]["details"] == [lines[2]]
    assert partial["review_notes"]


def test_compact_assessment_restores_quotes_and_marks_missing_requirements_unverified():
    lines, _ = numbered_source(CV)
    wire = {
        "relevant": True,
        "explanation": "The navigation work uses Python.",
        "checks": [["met", "applied", "Python is used in navigation.", ["L3"]]],
    }
    result = expand_assessment(wire, CRITERIA, lines)
    assert result["assessments"][0]["index"] == 0
    assert result["assessments"][0]["quotes"] == [lines[2]]
    assert result["explanation"] != lines[2]
    missing = expand_assessment({**wire, "checks": []}, CRITERIA, lines)
    assert not missing["relevant"]
    assert missing["assessments"][0]["reason"].startswith("Assessment incomplete:")
    unsupported = expand_assessment(
        {**wire, "checks": [["met", "applied", "Reason", [-1]]]}, CRITERIA, lines
    )
    assert unsupported["assessments"][0]["status"] == "not_found"
    assert unsupported["assessments"][0]["quotes"] == []


def test_keyed_checks_do_not_shift_missing_requirements_and_normalize_depth():
    criteria = {"intent": "Python and SQL", "requirements": CRITERIA["requirements"] * 2}
    result = expand_assessment(
        {
            "checks": {
                "R2": {
                    "status": "MET",
                    "depth": "hands-on",
                    "reason": "Documented project.",
                    "refs": ["L1"],
                },
            }
        },
        criteria,
        ["Built SQL reporting."],
    )
    assert result["assessments"][0]["status"] == "not_found"
    assert result["assessments"][1]["status"] == "met"
    assert result["assessments"][1]["depth"] == "listed"
    assert result["assessments"][1]["quotes"] == ["Built SQL reporting."]
    ambiguous = expand_assessment(
        {"checks": [["met", "Project evidence.", ["L1"]]]}, criteria, ["Built SQL reporting."]
    )
    assert all(item["status"] == "not_found" for item in ambiguous["assessments"])


@pytest.mark.parametrize(
    "row",
    [
        ["met", "Documented Python project.", ["L1"]],
        {"status": "met", "reason": "Documented Python project.", "source_labels": "L1"},
    ],
)
def test_short_rows_and_object_checks_need_no_regeneration(row):
    result = expand_assessment({"checks": {"R1": row}}, CRITERIA, ["Built Python navigation."])
    assert result["assessments"][0]["status"] == "met"
    assert result["assessments"][0]["depth"] == "listed"


def test_duplicate_ids_and_invalid_status_cannot_create_matches():
    result = expand_assessment(
        {
            "checks": [
                {"id": "R1", "status": "met", "reason": "Project.", "refs": ["L1"]},
                {"id": "R1", "status": "not_found", "reason": "Absent."},
            ]
        },
        CRITERIA,
        ["Python project."],
    )
    assert not result["relevant"]
    result = expand_assessment(
        {"checks": {"R1": ["maybe", "expert", "Unclear.", ["L1"]]}}, CRITERIA, ["Python project."]
    )
    assert not result["relevant"]


def test_profile_aliases_and_extra_fields_preserve_recognized_facts():
    result = expand_profile(
        {
            "profile": {
                "full_name": "Alex",
                "technical_skills": "Python",
                "unexpected": "ignore",
                "work_experience": [
                    {"title": "Engineer", "company": "Org", "source_refs": "L1", "extra": "ignore"}
                ],
                "education": [{"degree": "BSc", "university": "University", "grade": "A"}],
                "projects": [{"title": "Navigation", "description": "L1", "technologies": None}],
            }
        },
        ["Built Python navigation."],
    )
    assert result["name"] == "Alex" and result["skills"] == ["Python"]
    assert result["experience"][0]["role"] == "Engineer"
    assert result["experience"][0]["details"] == ["Built Python navigation."]
    assert result["education"][0]["qualification"] == "BSc"
    assert result["review_notes"]
    with pytest.raises(ValueError, match="recognizable"):
        expand_profile({"unrelated": "bad output"}, ["Python"])


@pytest.mark.parametrize("reference", ["L1", "[L1]", "l1", "1", 1])
def test_one_line_cv_citation_forms_resolve_to_the_same_printed_line(reference):
    text = "Connection check. Skills: Python. Project: Built Python document retrieval tools."
    lines, source = numbered_source(text)
    assert source == "[L1]: " + text
    result = expand_assessment(
        {
            "relevant": True,
            "explanation": "Python document retrieval is supported.",
            "checks": [["met", "applied", "Python is used in a project.", [reference]]],
        },
        CRITERIA,
        lines,
    )
    assert result["assessments"][0]["quotes"] == [text]


@pytest.mark.parametrize("reference", ["L0", "L2", 0, 2, "2", True, 1.0, "Invented SQL", "L1–L2"])
def test_one_line_cv_rejects_missing_labels_and_does_not_guess_offsets(reference):
    with pytest.raises(ValueError, match="Only L1 through L1"):
        source_quotes([reference], ["Skills: Python."])


def test_labels_in_a_multiline_cv_do_not_shift_to_a_different_fact():
    lines, _ = numbered_source("Alex\nSkills: SQL.\nProject: Built Python navigation tools.")
    assert source_quotes(["L2", 2, "2"], lines) == ["Skills: SQL."]
    assert source_quotes(["L3"], lines) == ["Project: Built Python navigation tools."]


@pytest.mark.parametrize("reference", ["L2-L3", "L2–L3", "[L2—L3]", "2:3"])
def test_source_ranges_preserve_all_lines_without_a_second_model_call(reference):
    lines, _ = numbered_source(CV)
    assert source_quotes(reference, lines) == ["\n".join(lines[1:3])]
    result = expand_profile(
        {
            "experience": [["Engineer", "Org", "2024", reference]],
            "projects": [["Navigation", reference, ["Python"]]],
            "education": [["BSc", "University", "2023"]],
        },
        lines,
    )
    assert result["experience"][0]["details"] == ["\n".join(lines[1:3])]
    assert result["projects"][0]["details"] == "\n".join(lines[1:3])
    assert result["education"][0]["qualification"] == "BSc"
    assert not result["review_notes"]


def test_copied_facts_are_accepted_only_when_present_in_source():
    lines, _ = numbered_source(CV)
    result = expand_profile(
        {
            "projects": [["Navigation", "Built Python navigation nodes using ROS 2.", []]],
            "achievements": ["Won an invented award."],
        },
        lines,
    )
    assert result["projects"][0]["details"] == lines[2]
    assert result["achievements"] == []
    assert result["review_notes"]


def test_dense_103_line_profile_keeps_every_detail_from_a_compact_range():
    lines = [
        f"Professional responsibility {index}: built and maintained software."
        for index in range(103)
    ]
    result = expand_profile(
        {
            "experience": [["Engineer", "Org", "2024", "L1–L103"]],
            "skills": ["Python"],
            "certifications": ["L999"],
        },
        lines,
    )
    assert result["experience"][0]["details"][0].splitlines() == lines
    assert result["skills"] == ["Python"]
    assert result["certifications"] == []
    assert result["review_notes"]


def test_model_job_returns_promptly_while_inference_is_still_running():
    release = threading.Event()

    def analyze(text):
        assert release.wait(10), "Test did not release the model."
        return {"name": "Alex"}

    with TestClient(create_app(KEY, {"analyze_text": analyze})) as server:
        assert (
            server.post("/jobs", json={"task": "analyze", "input": {"text": CV}}).status_code == 401
        )
        assert (
            server.post("/jobs", headers=HEADERS, json={"task": "search", "input": {}}).status_code
            == 422
        )
        started = monotonic()
        response = server.post(
            "/jobs", headers=HEADERS, json={"task": "analyze", "input": {"text": CV}}
        )
        try:
            assert response.status_code == 202 and monotonic() - started < 2
            path = "/jobs/" + response.json()["job_id"]
            assert server.get(path).status_code == 401
            assert server.get(path, headers=HEADERS).json()["status"] in {"queued", "running"}
        finally:
            release.set()
        for _ in range(100):
            result = server.get(path, headers=HEADERS).json()
            if result["status"] == "completed":
                break
            threading.Event().wait(0.001)
        assert result["result"]["analysis"]["name"] == "Alex"
        server.delete(path, headers=HEADERS)
        assert server.get(path, headers=HEADERS).status_code == 404


def test_client_polls_same_job_after_transient_network_failure_and_releases_result(monkeypatch):
    job_id = "a" * 32
    calls, progress = [], []
    replies = iter(
        [
            requests.ConnectionError("temporary"),
            {"status": "running"},
            {"status": "completed", "result": {"analysis": {"name": "Alex"}}},
        ]
    )

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            data = {"job_id": job_id}
        elif method == "DELETE":
            data = {"status": "released"}
        else:
            data = next(replies)
            if isinstance(data, Exception):
                raise data
        return SimpleNamespace(ok=True, json=lambda: data)

    monkeypatch.setattr("ats.client.requests.request", request)
    monkeypatch.setattr("ats.client.time.sleep", lambda _: None)
    client = APIClient("http://localhost:8000", KEY)
    client.on_progress = lambda status, elapsed: progress.append(status)
    assert client.analyze(CV)["name"] == "Alex"
    assert [method for method, _, _ in calls] == ["POST", "GET", "GET", "GET", "DELETE"]
    assert all(job_id in url for _, url, _ in calls[1:])
    assert progress == ["running"]
    assert all(kwargs["timeout"][1] <= 30 for _, _, kwargs in calls)


def test_job_validation_failures_are_forwarded_and_not_regenerated(monkeypatch):
    job_id = "b" * 32
    calls = []

    def request(method, path, payload=None, timeout=30):
        calls.append(method)
        if method == "POST":
            return {"job_id": job_id}
        if method == "DELETE":
            return {"status": "released"}
        return {"status": "failed", "error": {"status_code": 422, "detail": "Invalid source line."}}

    client = APIClient("http://localhost:8000", KEY)
    monkeypatch.setattr(client, "request", request)
    with pytest.raises(BackendError, match="Invalid source line"):
        client.analyze(CV)
    assert calls == ["POST", "GET", "DELETE"]
