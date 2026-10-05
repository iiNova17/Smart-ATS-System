"""Plain dictionary validation shared by the local app and model requests."""

from copy import deepcopy

PROFILE_TEMPLATE = {
    "name": None,
    "headline": None,
    "email": None,
    "phone": None,
    "location": None,
    "links": [],
    "summary": "",
    "skills": [],
    "experience": [],
    "education": [],
    "projects": [],
    "certifications": [],
    "languages": [],
    "achievements": [],
    "other_details": [],
    "stated_years_experience": None,
    "evidence": [],
    "review_notes": [],
}
DETAIL_TEMPLATES = {
    "experience": {"role": "", "organization": "", "dates": "", "details": []},
    "education": {"qualification": "", "institution": "", "dates": ""},
    "projects": {"name": "", "details": "", "technologies": []},
    "evidence": {"skill": "", "quote": ""},
}


def validate_text(value, label, minimum=0, maximum=30000):
    if (
        not isinstance(value, str)
        or not minimum <= len(value.strip()) <= maximum
        or "\x00" in value
    ):
        raise ValueError(f"{label} must contain {minimum}–{maximum:,} readable characters.")
    return value.strip()


def validate_cv(payload):
    return {
        "filename": validate_text(payload.get("filename"), "Filename", 1, 200),
        "text": validate_text(payload.get("text"), "CV text", 40, 30000),
    }


def clean_skills(values):
    if not isinstance(values, list) or len(values) > 30:
        raise ValueError("Enter a list of at most 30 skills per group.")
    output, seen = [], set()
    for value in values:
        skill = validate_text(value, "Skill", 0, 100)
        if skill and skill.casefold() not in seen:
            output.append(skill)
            seen.add(skill.casefold())
    return output


def validate_rank(payload):
    required = clean_skills(payload.get("required", []))
    preferred = clean_skills(payload.get("preferred", []))
    preferred = [
        skill for skill in preferred if skill.casefold() not in {x.casefold() for x in required}
    ]
    if not required and not preferred and not payload.get("job_description", "").strip():
        raise ValueError("Add skills or describe the experience your role needs.")
    return {
        "required": required,
        "preferred": preferred,
        "job_description": validate_text(
            payload.get("job_description", ""), "Job description", 0, 6000
        ),
    }


def validate_search(payload):
    required = clean_skills(payload.get("required", []))
    query = payload.get("query") or ", ".join(required)
    if not query:
        raise ValueError("Describe the skills or experience you need, or add a required skill.")
    query = validate_text(query, "Search", 1, 6000)
    mode, match, limit = (
        payload.get("mode", "smart"),
        payload.get("match", "all"),
        payload.get("limit"),
    )
    if mode not in {"smart", "keyword", "semantic"} or match not in {"all", "any"}:
        raise ValueError("Choose a valid search method and match rule.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Result count must be a positive integer.")
    return {
        "query": query,
        "mode": mode,
        "match": match,
        "limit": limit,
        "required": required,
    }


def validate_criteria(value):
    """A small, readable schema for the model's interpretation of a request."""
    if not isinstance(value, dict):
        raise ValueError("AI returned an invalid search interpretation.")
    intent = validate_text(value.get("intent"), "Search intent", 1, 6000)
    requirements = value.get("requirements")
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 60:
        raise ValueError("AI must return the requirements in your request.")
    clean = []
    for item in requirements:
        if not isinstance(item, dict):
            raise ValueError("Invalid search requirement.")
        label = validate_text(item.get("label"), "Requirement", 1, 300)
        kind, required = item.get("kind"), item.get("required")
        if kind not in {"skill", "experience"} or type(required) is not bool:
            raise ValueError("Invalid requirement type.")
        clean.append({"label": label, "kind": kind, "required": required})
    return {"intent": intent, "requirements": clean}


def validate_assessment(value, criteria):
    if not isinstance(value, dict) or type(value.get("relevant")) is not bool:
        raise ValueError("AI returned an invalid candidate assessment.")
    explanation = validate_text(value.get("explanation"), "Match explanation", 1, 2400)
    assessments = value.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(criteria["requirements"]):
        raise ValueError("AI must assess every requirement, including missing evidence.")
    clean, seen = [], set()
    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError("Invalid requirement assessment.")
        index, status = item.get("index"), item.get("status")
        if type(index) is not int or not 0 <= index < len(assessments) or index in seen:
            raise ValueError("AI returned incorrect requirement indices.")
        if status not in {"met", "partial", "not_found"}:
            raise ValueError("Invalid evidence status.")
        depth = item.get("depth", "listed")
        if depth not in {"listed", "applied", "extensive"}:
            raise ValueError("Invalid experience evidence depth.")
        quotes = item.get("quotes", [])
        if not isinstance(quotes, list) or len(quotes) > 4:
            raise ValueError("Invalid supporting evidence.")
        quotes = [validate_text(quote, "Evidence quote", 1, 1600) for quote in quotes]
        clean.append(
            {
                "index": index,
                "status": status,
                "depth": depth,
                "reason": validate_text(item.get("reason"), "Reason", 1, 1000),
                "quotes": quotes,
            }
        )
        seen.add(index)
    return {
        "relevant": value["relevant"],
        "name": validate_text(value["name"], "Candidate name", 1, 200)
        if value.get("name")
        else None,
        "headline": validate_text(value["headline"], "Headline", 1, 200)
        if value.get("headline")
        else None,
        "explanation": explanation,
        "assessments": sorted(clean, key=lambda item: item["index"]),
    }


def fill_fields(value, template):
    if not isinstance(value, dict) or set(value) - set(template):
        raise ValueError("Analysis does not match the expected profile fields.")
    result = deepcopy(template)
    for key, item in value.items():
        default = template[key]
        # JSON null is a common, valid representation of an unknown text detail.
        if item is None and isinstance(default, str):
            item = default
        if default is None:
            valid = item is None or isinstance(item, str)
        elif isinstance(default, list):
            valid = isinstance(item, list)
        else:
            valid = isinstance(item, str)
        if not valid:
            raise ValueError(f"Invalid analysis field: {key}.")
        result[key] = item
    return result


def validate_analysis(value):
    if isinstance(value, dict) and type(value.get("stated_years_experience")) in {int, float}:
        value = {**value, "stated_years_experience": str(value["stated_years_experience"])}
    result = fill_fields(value, PROFILE_TEMPLATE)
    for key, items in result.items():
        if not isinstance(items, list):
            continue
        if key in DETAIL_TEMPLATES:
            result[key] = [fill_fields(item, DETAIL_TEMPLATES[key]) for item in items]
            for entry in result[key]:
                for field in entry.values():
                    if isinstance(field, list) and any(not isinstance(x, str) for x in field):
                        raise ValueError("Analysis detail lists must contain text.")
        elif any(not isinstance(item, str) for item in items):
            raise ValueError(f"Analysis field {key} must be a list of text.")
    return result
