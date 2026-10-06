"""Local CV library, persistent caches, similarity search, ranking, and RAG retrieval.

SQLite is built into Python. Kaggle is contacted only to compute model outputs.
"""

import hashlib
import json
import math
import sqlite3
from pathlib import Path

from ats.client import BackendError
from ats.matching import canonical_skill, normalize, skill_evidence, skill_matches, split_skills
from ats.validation import (
    validate_analysis,
    validate_assessment,
    validate_criteria,
    validate_cv,
    validate_rank,
    validate_search,
    validate_text,
)


def query(workspace, statement, parameters=()):
    connection = sqlite3.connect(str(workspace["database"]), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            cursor = connection.execute(statement, parameters)
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def open_workspace(directory, client_provider=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    workspace = {"database": directory / "candidates.sqlite3", "client_provider": client_provider}
    query(
        workspace,
        """CREATE TABLE IF NOT EXISTS cvs (
        id TEXT PRIMARY KEY, filename TEXT NOT NULL, text TEXT NOT NULL,
        analysis TEXT, embedding TEXT)""",
    )
    query(
        workspace,
        """CREATE TABLE IF NOT EXISTS model_cache (
          cache_key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    )
    return workspace


def model_client(workspace):
    provider = workspace.get("client_provider")
    client = provider() if provider else None
    if client is None:
        raise BackendError("AI is temporarily unavailable. Your local library is still available.")
    return client


def list_cvs(workspace):
    rows = query(
        workspace,
        "SELECT id, filename, length(text) AS characters, analysis FROM cvs ORDER BY rowid",
    )
    result = []
    for row in rows:
        profile = json.loads(row.pop("analysis") or "null")
        result.append(
            {
                **row,
                "analyzed": profile is not None,
                "name": (profile or {}).get("name"),
                "headline": (profile or {}).get("headline"),
            }
        )
    return result


def add_cv(workspace, payload):
    cv = validate_cv(payload)
    cv_id = hashlib.sha256(cv["text"].encode("utf-8")).hexdigest()[:24]
    # No embedding/model request on upload; a stopped notebook cannot block imports.
    added = query(
        workspace,
        "INSERT OR IGNORE INTO cvs (id, filename, text) VALUES (?, ?, ?) RETURNING id",
        (cv_id, cv["filename"], cv["text"]),
    )
    return {"cv": get_cv(workspace, cv_id), "duplicate": not bool(added)}


def get_cv(workspace, cv_id):
    records = query(workspace, "SELECT id, filename, text FROM cvs WHERE id = ?", (cv_id,))
    if not records:
        raise KeyError("Candidate no longer exists. Refresh your library.")
    return records[0]


def remove_cv(workspace, cv_id):
    # Cached analysis and vectors live in the same row and are removed with it.
    query(workspace, "DELETE FROM cvs WHERE id = ?", (cv_id,))
    query(workspace, "DELETE FROM model_cache WHERE cache_key LIKE ?", ("assess:" + cv_id + ":%",))


def clear_cvs(workspace):
    query(workspace, "DELETE FROM cvs")
    query(workspace, "DELETE FROM model_cache")


def unit_vector(values):
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= 4096
        or any(type(value) not in {float, int} or not math.isfinite(value) for value in values)
    ):
        raise BackendError("AI returned an invalid embedding. Please try again.")
    length = math.sqrt(math.fsum(value * value for value in values))
    if not length or not math.isfinite(length):
        raise BackendError("AI returned an empty embedding. Please try again.")
    return [value / length for value in values]


def embedding_response(response, texts):
    version, items = response.get("embedding_version"), response.get("items")
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(items, list)
        or len(items) != len(texts)
    ):
        raise BackendError("AI returned incomplete embeddings. Please try again.")
    result, dimensions = [], set()
    for item, source in zip(items, texts):
        vector = unit_vector(item.get("vector"))
        dimensions.add(len(vector))
        chunks = item.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise BackendError("AI returned no passages. Please try again.")
        clean = []
        normalized_source = " ".join(normalize(source).split())
        for chunk in chunks:
            text = chunk.get("text")
            if (
                not isinstance(text, str)
                or not text.strip()
                or " ".join(normalize(text).split()) not in normalized_source
            ):
                raise BackendError("AI returned an unsupported passage. Please try again.")
            chunk_vector = unit_vector(chunk.get("vector"))
            dimensions.add(len(chunk_vector))
            clean.append({"text": text, "vector": chunk_vector})
        result.append({"embedding_version": version, "vector": vector, "chunks": clean})
    if len(dimensions) != 1:
        raise BackendError("AI embedding dimensions changed. Please try again.")
    return result


def retrieve(workspace, text, cv_id=None):
    """Compute similarities locally. Restrict to one CV before question retrieval."""
    records = query(
        workspace,
        "SELECT id, filename, text, embedding FROM cvs" + (" WHERE id = ?" if cv_id else ""),
        (cv_id,) if cv_id else (),
    )
    if not records:
        return []
    client = model_client(workspace)
    query_embedding = embedding_response(client.embed([text]), [text])[0]
    version, vector = query_embedding["embedding_version"], query_embedding["vector"]
    pending = []
    for record in records:
        cached = json.loads(record["embedding"] or "null")
        if (
            cached
            and cached.get("embedding_version") == version
            and len(cached.get("vector", [])) == len(vector)
        ):
            record["embedding"] = cached
        else:
            pending.append(record)
    # Batch inference without any limit on the total number of CVs in the library.
    for start in range(0, len(pending), 4):
        batch = pending[start : start + 4]
        texts = [record["text"] for record in batch]
        computed = embedding_response(client.embed(texts), texts)
        for record, embedding in zip(batch, computed):
            if embedding["embedding_version"] != version or len(embedding["vector"]) != len(vector):
                raise BackendError("The AI model changed during this request. Please try again.")
            query(
                workspace,
                "UPDATE cvs SET embedding = ? WHERE id = ?",
                (json.dumps(embedding, ensure_ascii=False), record["id"]),
            )
            record["embedding"] = embedding
    hits = []
    for record in records:
        for chunk in record["embedding"]["chunks"]:
            score = math.fsum(a * b for a, b in zip(vector, chunk["vector"]))
            hits.append(
                {
                    "id": record["id"],
                    "filename": record["filename"],
                    "score": max(-1.0, min(1.0, score)),
                    "text": chunk["text"],
                }
            )
    return sorted(hits, key=lambda hit: (-hit["score"], hit["filename"]))


def search_cvs(workspace, payload, progress=None):
    request = validate_search(payload)
    if request["mode"] == "keyword":
        keywords = split_skills(request["query"])
        if not keywords:
            raise ValueError("Enter one or more keywords.")
        results = []
        for cv in query(workspace, "SELECT id, filename, text FROM cvs"):
            evidence, missing = skill_matches(cv["text"], keywords)
            if evidence and (request["match"] == "any" or not missing):
                results.append(
                    {
                        "id": cv["id"],
                        "filename": cv["filename"],
                        "matched": [item["skill"] for item in evidence],
                        "missing": missing,
                        "evidence": evidence,
                        "score": len(evidence) / len(keywords),
                    }
                )
        results.sort(key=lambda row: (-row["score"], row["filename"]))
    else:
        criteria = prepare_criteria(workspace, request["query"], request["required"])
        results = assess_candidates(workspace, criteria, progress)
    return results[: request["limit"]]


def cached_model_output(workspace, key, compute):
    rows = query(workspace, "SELECT value FROM model_cache WHERE cache_key = ?", (key,))
    if rows:
        return json.loads(rows[0]["value"])
    result = compute()
    query(
        workspace,
        "INSERT OR REPLACE INTO model_cache (cache_key, value) VALUES (?, ?)",
        (key, json.dumps(result, ensure_ascii=False)),
    )
    return result


def prepare_criteria(workspace, text, required=None, preferred=None):
    client = model_client(workspace)
    required, preferred = required or [], preferred or []
    request = text or "Find candidates with the following skills."
    if required:
        request += "\nExplicit required skills: " + ", ".join(required)
    if preferred:
        request += "\nExplicit preferred skills (not mandatory): " + ", ".join(preferred)
    digest = hashlib.sha256(("criteria-v2:" + request).encode()).hexdigest()
    criteria = validate_criteria(
        cached_model_output(
            workspace, "intent:" + digest, lambda: validate_criteria(client.interpret(request))
        )
    )
    # Explicit form constraints cannot be dropped or made optional by the model.
    for labels, mandatory in [(preferred, False), (required, True)]:
        for label in labels:
            found = next(
                (
                    item
                    for item in criteria["requirements"]
                    if canonical_skill(item["label"]) == canonical_skill(label)
                ),
                None,
            )
            if found:
                found["required"] = mandatory
            else:
                criteria["requirements"].append(
                    {"label": label, "kind": "skill", "required": mandatory}
                )
    return validate_criteria(criteria)


def verified_assessment(value, criteria, text):
    result = validate_assessment(value, criteria)
    source = " ".join(normalize(text).split())
    for field in ("name", "headline"):
        if result[field] and " ".join(normalize(result[field]).split()) not in source:
            result[field] = None
    for item, requirement in zip(result["assessments"], criteria["requirements"]):
        item["quotes"] = [
            quote for quote in item["quotes"] if " ".join(normalize(quote).split()) in source
        ]
        if item["status"] != "not_found" and not item["quotes"]:
            item.update(status="not_found", reason="The model's evidence could not be verified.")
        # Prevent a model from treating ROS 1 or an unversioned ROS mention as ROS 2.
        key = canonical_skill(requirement["label"])
        if key in {"ros1", "ros2"} and not any(
            skill_evidence(quote, requirement["label"]) for quote in item["quotes"]
        ):
            item.update(
                status="not_found", reason="The requested ROS version is not stated.", quotes=[]
            )
    return result


def assess_candidates(workspace, criteria, progress=None):
    """Use model reasoning as evidence; filtering and ordering happen entirely locally."""
    criteria = validate_criteria(criteria)
    client = model_client(workspace)
    digest = hashlib.sha256(
        ("assessment-v5:" + json.dumps(criteria, sort_keys=True)).encode()
    ).hexdigest()
    records = query(workspace, "SELECT id, filename, text, analysis FROM cvs")
    workspace["assessment_incomplete"] = 0
    rows = []
    for position, cv in enumerate(records, 1):
        if progress:
            progress(position - 1, len(records))
        assessment = verified_assessment(
            cached_model_output(
                workspace,
                "assess:" + cv["id"] + ":" + digest,
                lambda: verified_assessment(
                    client.evaluate(cv["text"], criteria), criteria, cv["text"]
                ),
            ),
            criteria,
            cv["text"],
        )
        items = assessment["assessments"]
        if any(item["reason"].startswith("Assessment incomplete:") for item in items):
            workspace["assessment_incomplete"] += 1
            # Retry unresolved checks on the next search rather than permanently caching omissions.
            query(
                workspace,
                "DELETE FROM model_cache WHERE cache_key = ?",
                ("assess:" + cv["id"] + ":" + digest,),
            )
        required = [item for item, req in zip(items, criteria["requirements"]) if req["required"]]
        # No related-but-insufficient candidate is presented as a required match.
        if not assessment["relevant"] or any(item["status"] != "met" for item in required):
            continue
        if not any(item["status"] == "met" for item in items):
            continue
        total_weight = sum(2 if req["required"] else 1 for req in criteria["requirements"])
        earned = sum(
            (2 if req["required"] else 1)
            * {"met": 1, "partial": 0.5, "not_found": 0}[item["status"]]
            for item, req in zip(items, criteria["requirements"])
        )
        profile = json.loads(cv["analysis"] or "null") or {}
        depth = sum(
            {"listed": 1, "applied": 2, "extensive": 3}[item["depth"]]
            for item in items
            if item["status"] == "met"
        ) / len(items)
        matched, gaps, evidence, reasons = [], [], [], []
        for item, req in zip(items, criteria["requirements"]):
            (matched if item["status"] == "met" else gaps).append(req["label"])
            if item["status"] == "met":
                reasons.append({"requirement": req["label"], "reason": item["reason"]})
            for quote in item["quotes"]:
                evidence.append({"skill": req["label"], "quote": quote})
        rows.append(
            {
                "id": cv["id"],
                "filename": cv["filename"],
                "name": profile.get("name") or assessment["name"],
                "headline": profile.get("headline") or assessment["headline"],
                "score": round(100 * earned / total_weight, 1),
                "matched": matched,
                "missing": gaps,
                "evidence": evidence,
                "reasons": reasons,
                "explanation": assessment["explanation"],
                "criteria": criteria,
                "evidence_depth": depth,
            }
        )
    if progress:
        progress(len(records), len(records))
    return sorted(rows, key=lambda row: (-row["score"], -row["evidence_depth"], row["filename"]))


def rank_cvs(workspace, payload, progress=None):
    request = validate_rank(payload)
    criteria = prepare_criteria(
        workspace, request["job_description"], request["required"], request["preferred"]
    )
    rows = assess_candidates(workspace, criteria, progress)
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    return rows


def analyze_cv(workspace, cv_id):
    cv = get_cv(workspace, cv_id)
    cached = query(workspace, "SELECT analysis FROM cvs WHERE id = ?", (cv_id,))[0]["analysis"]
    if cached:
        return validate_analysis(json.loads(cached))
    result = validate_analysis(model_client(workspace).analyze(cv["text"]))
    source = " ".join(normalize(cv["text"]).split())
    valid = []
    for item in result["evidence"]:
        quote = " ".join(normalize(item["quote"]).split())
        if quote and quote in source:
            valid.append(item)
        else:
            result["review_notes"].append("An unsupported evidence quote was removed.")
    result["evidence"] = valid
    # Supporting excerpts can be located locally; the model need not regenerate this text.
    evidenced = {item["skill"].casefold() for item in valid}
    for skill in result["skills"]:
        if skill.casefold() not in evidenced:
            quote = skill_evidence(cv["text"], skill)
            if quote:
                result["evidence"].append({"skill": skill, "quote": quote})
                evidenced.add(skill.casefold())
    query(
        workspace,
        "UPDATE cvs SET analysis = ? WHERE id = ?",
        (json.dumps(result, ensure_ascii=False), cv_id),
    )
    return result


def ask_cv(workspace, cv_id, question):
    get_cv(workspace, cv_id)
    question = validate_text(question, "Question", 1, 1000)
    passages = list(dict.fromkeys(hit["text"] for hit in retrieve(workspace, question, cv_id)))[:4]
    answer = model_client(workspace).answer(question, passages)
    return {"answer": answer, "passages": passages}
