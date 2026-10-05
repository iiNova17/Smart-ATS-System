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
    if not required and not preferred:
        raise ValueError("Add at least one required or preferred skill.")
    return {
        "required": required,
        "preferred": preferred,
        "job_description": validate_text(
            payload.get("job_description", ""), "Job description", 0, 6000
        ),
    }


def validate_search(payload):
    query = validate_text(payload.get("query"), "Search", 1, 6000)
    mode, match, limit = (
        payload.get("mode", "keyword"),
        payload.get("match", "all"),
        payload.get("limit"),
    )
    if mode not in {"keyword", "semantic"} or match not in {"all", "any"}:
        raise ValueError("Choose a valid search method and match rule.")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Result count must be a positive integer.")
    return {"query": query, "mode": mode, "match": match, "limit": limit}


def fill_fields(value, template):
    if not isinstance(value, dict) or set(value) - set(template):
        raise ValueError("Analysis does not match the expected profile fields.")
    result = deepcopy(template)
    for key, item in value.items():
        default = template[key]
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
