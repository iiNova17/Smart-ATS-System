"""Local skill evidence, coverage scores, and safe CSV exports."""

import csv
import io
import re
import unicodedata


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold()


def split_skills(text):
    return list(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))


def skill_pattern(skill):
    # Lookarounds preserve punctuation-heavy skills such as C++, C#, and .NET.
    parts = re.split(r"\s+", normalize(skill.strip()))
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(x) for x in parts) + r"(?!\w)")


def skill_evidence(text, skill):
    # Normalize per line so the returned evidence is the original CV wording.
    for line in text.splitlines():
        if skill_pattern(skill).search(normalize(line)):
            return line.strip()
    # A multi-word phrase may span line breaks in a PDF.
    lines = text.splitlines()
    for first, second in zip(lines, lines[1:]):
        joined = first + " " + second
        if skill_pattern(skill).search(normalize(joined)):
            return joined.strip()
    return None


def skill_matches(text, skills):
    evidence, missing = [], []
    for skill in skills:
        quote = skill_evidence(text, skill)
        if quote is None:
            missing.append(skill)
        else:
            evidence.append({"skill": skill, "quote": quote})
    return evidence, missing


def coverage_score(required_count, preferred_count, required_matches, preferred_matches):
    # Missing groups redistribute their weight, avoiding artificial penalties.
    weights = (80 if required_count else 0, 20 if preferred_count else 0)
    total = sum(weights)
    if not total:
        return 0.0
    return round(
        100
        * (
            weights[0] * required_matches / max(required_count, 1)
            + weights[1] * preferred_matches / max(preferred_count, 1)
        )
        / total,
        1,
    )


def csv_export(rows):
    """Neutralize spreadsheet formulas in untrusted filenames/quotes."""
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        clean = {}
        for key, value in row.items():
            value = str(value)
            clean[key] = "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
        writer.writerow(clean)
    return output.getvalue()
