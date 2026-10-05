"""Regression checks for relevance decisions, source grounding and model failures."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException

from ats.matching import skill_evidence
from ats.workspace import add_cv, assess_candidates, open_workspace, prepare_criteria
from backend.api import create_app
from backend.runtime import build_analysis_parser, make_runtime, merge_assessments, merge_profiles

KEY = "private-test-key-at-least-24-characters"
CRITERIA = {
    "intent": "ROS 2 autonomous navigation",
    "requirements": [
        {"label": "ROS 2", "kind": "skill", "required": True},
        {"label": "autonomous navigation", "kind": "experience", "required": True},
    ],
}
CV = "Skills: Python, ROS2. Built autonomous navigation for a mobile robot using Nav2."


def assessment(text, statuses=None, relevant=True):
    return {
        "relevant": relevant,
        "explanation": "The navigation project connects ROS 2 to practical robot work.",
        "assessments": [
            {
                "index": index,
                "status": status,
                "depth": "applied",
                "reason": "The robot project demonstrates this requirement.",
                "quotes": [text] if status != "not_found" else [],
            }
            for index, status in enumerate(statuses or ["met", "met"])
        ],
    }


def test_spacing_case_and_version_aliases():
    for query in ["ros 2", "ROS2", "ROS-2", "ROS_2"]:
        assert skill_evidence(CV, query)
    assert skill_evidence(CV, "ROS")
    assert skill_evidence("Used Robot Operating System 2 for navigation", "ROS 2")
    assert not skill_evidence("Used ROS 1 for navigation", "ROS 2")
    assert not skill_evidence("Used Microsoft software", "ROS")


def test_combined_search_excludes_related_but_insufficient_and_irrelevant(tmp_path):
    calls = []
    texts = [
        CV,
        "Skills: ROS 2. No practical robotics experience is documented in this CV.",
        "Python web developer. Built websites and database APIs using Flask and SQL.",
    ]
    decisions = [
        assessment(texts[0]),
        assessment(texts[1], ["met", "not_found"]),
        assessment(texts[2], ["met", "met"], relevant=False),
    ]

    def evaluate(text, criteria):
        calls.append(text)
        return decisions[texts.index(text)]

    client = SimpleNamespace(evaluate=evaluate)
    workspace = open_workspace(tmp_path, lambda: client)
    for index, text in enumerate(texts):
        add_cv(workspace, {"filename": f"cv{index}.txt", "text": text})
    rows = assess_candidates(workspace, CRITERIA)
    assert len(rows) == 1 and rows[0]["filename"] == "cv0.txt"
    assert rows[0]["explanation"] != rows[0]["evidence"][0]["quote"]
    assert len(rows[0]["reasons"]) == 2
    assert len(calls) == 3
    assert assess_candidates(workspace, CRITERIA) == rows
    assert len(calls) == 3  # No repeated generation for identical criteria and CV content.


@pytest.mark.parametrize(
    "text,quote",
    [
        (
            "Used ROS 1 for autonomous navigation on an indoor mobile robot.",
            "Used ROS 1 for autonomous navigation on an indoor mobile robot.",
        ),
        (CV, "Invented ROS 2 evidence that does not appear in this CV."),
    ],
)
def test_unsupported_quotes_or_wrong_ros_version_cannot_pass(tmp_path, text, quote):
    client = SimpleNamespace(evaluate=lambda *args: assessment(quote))
    workspace = open_workspace(tmp_path, lambda: client)
    add_cv(workspace, {"filename": "cv.txt", "text": text})
    assert assess_candidates(workspace, CRITERIA) == []


@pytest.mark.parametrize("alias", ["ros2", "ROS-2", "Robot Operating System 2"])
def test_explicit_required_skill_cannot_be_dropped_by_interpretation(tmp_path, alias):
    client = SimpleNamespace(
        interpret=lambda query: {
            "intent": "Find robotics work",
            "requirements": [{"label": "ROS 2", "kind": "skill", "required": False}],
        }
    )
    workspace = open_workspace(tmp_path, lambda: client)
    criteria = prepare_criteria(workspace, "Robotics work", [alias, "Python"])
    assert len(criteria["requirements"]) == 2
    assert criteria["requirements"][0]["required"]
    assert any(item["label"] == "Python" and item["required"] for item in criteria["requirements"])


def test_fenced_json_and_unknown_details_are_supported_but_truncation_is_rejected():
    parser = build_analysis_parser()
    result = parser.invoke(
        '```json\n{"experience":[{"role":"Engineer","dates":null}],'
        '"stated_years_experience":3}\n```'
    )
    assert result["experience"][0]["dates"] == ""
    assert result["stated_years_experience"] == "3"
    with pytest.raises(OutputParserException):
        parser.invoke('```json\n{"name":"Incomplete"')


def test_section_merging_keeps_later_cv_details_and_partial_is_not_promoted():
    parser = build_analysis_parser()
    profiles = [
        parser.invoke('{"skills":["Python"],"summary":"Software work."}'),
        parser.invoke(
            '{"education":[{"qualification":"BSc"}],"skills":["ROS 2"],"summary":"Robotics work."}'
        ),
    ]
    result = merge_profiles(profiles)
    assert result["skills"] == ["Python", "ROS 2"]
    assert result["education"][0]["qualification"] == "BSc"
    combined = merge_assessments(
        [assessment(CV, ["met", "not_found"]), assessment(CV, ["partial", "partial"])], CRITERIA
    )
    assert [item["status"] for item in combined["assessments"]] == ["met", "partial"]


def test_failure_reference_is_actionable_without_exposing_cv_or_credentials(capsys):
    def broken(text):
        raise RuntimeError("PRIVATE candidate text and credentials")

    server = TestClient(create_app(KEY, {"analyze_text": broken}))
    response = server.post(
        "/analyze", headers={"Authorization": "Bearer " + KEY}, json={"text": CV}
    )
    detail = response.json()["detail"]
    assert response.status_code == 503 and detail["reference"] and detail["code"] == "model_error"
    assert "PRIVATE" not in response.text and KEY not in response.text


def test_generation_chains_accept_all_prompt_variables_and_complete_json(monkeypatch):
    """Exercise the real chain wiring without downloading a model or importing local Torch."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "langchain_text_splitters",
        SimpleNamespace(
            RecursiveCharacterTextSplitter=SimpleNamespace(
                from_huggingface_tokenizer=lambda *args, **kwargs: SimpleNamespace(
                    split_text=lambda text: [text]
                )
            )
        ),
    )
    tensor = MagicMock()
    tensor.shape = (1, 3)
    tensor.to.return_value = tensor
    tokenizer = MagicMock(return_value={"input_ids": tensor})
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    tokenizer.decode.side_effect = [
        json.dumps(CRITERIA),
        json.dumps(assessment(CV)),
        '{"name":"Alex","skills":["Python"]}',
        "The project uses ROS 2. [Passage 1]",
    ]
    model = SimpleNamespace(device="cpu", generate=lambda **kwargs: [[0, 0, 0, 9]])
    runtime = make_runtime(tokenizer, model, None, None)
    assert runtime["interpret_query"]("ros2 robotics")["requirements"][0]["label"] == "ROS 2"
    assert runtime["evaluate_text"](CV, CRITERIA)["assessments"][0]["status"] == "met"
    assert runtime["analyze_text"](CV)["skills"] == ["Python"]
    assert "[Passage 1]" in runtime["answer_question"](CV, "Which framework?")
    assert all(
        call.kwargs["tokenize"] is False for call in tokenizer.apply_chat_template.call_args_list
    )


def test_concrete_project_evidence_breaks_equal_coverage_ties(tmp_path):
    criteria = {
        "intent": "Python",
        "requirements": [{"label": "Python", "kind": "skill", "required": True}],
    }
    texts = [
        "Skills: Python. No professional projects are listed in this CV.",
        "Built a Python navigation project and a Python mapping project for robots.",
    ]

    def evaluate(text, criteria):
        return {
            "relevant": True,
            "explanation": "The CV documents Python experience.",
            "assessments": [
                {
                    "index": 0,
                    "status": "met",
                    "depth": "extensive" if text == texts[1] else "listed",
                    "reason": "Supported Python evidence.",
                    "quotes": [text],
                }
            ],
        }

    workspace = open_workspace(tmp_path, lambda: SimpleNamespace(evaluate=evaluate))
    for filename, text in zip(["a_list_only.txt", "z_projects.txt"], texts):
        add_cv(workspace, {"filename": filename, "text": text})
    rows = assess_candidates(workspace, criteria)
    assert rows[0]["filename"] == "z_projects.txt"
    assert rows[0]["score"] == rows[1]["score"] == 100
