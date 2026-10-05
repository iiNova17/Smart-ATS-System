# Kaggle model API · version 3.0.0

Set `NGROK_DOMAIN` and `ATS_API_KEY` in local `.env`. The app builds the HTTPS base URL
from your account's domain. An optional `ATS_API_URL` overrides it for development.
The app calls this API only for
model inference; candidate storage, extraction, search, ranking, and RAG retrieval are local.
There is no session header or remote CV collection. All requests authenticate with
`Authorization: Bearer <ATS_API_KEY>` and send `ngrok-skip-browser-warning: true`.
Use the same domain as Kaggle's `NGROK_DOMAIN` secret. Notebook Block 8 opens its Agent
Endpoint with the official `ngrok` Python SDK. Local HTTP is supported for tests.

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | No body | `{status: "ok", version: "3.0.0"}` |
| POST | `/embed` | `{texts: [string, ...]}` | `{embedding_version, items: [{vector, chunks: [{text, vector}]}]}` |
| POST | `/analyze` | `{text: string}` | `{analysis: profile_dictionary}` |
| POST | `/answer` | `{question: string, passages: [string, ...]}` | `{answer: string}` |

The previous `/cvs`, `/search`, `/rank`, and per-CV analysis/question routes are removed.
Use the updated notebook and app together; the app expects version `3.0.0`.

## Embeddings

```json
{"texts": ["Skills: Python, SQL. Built document search using LangChain."]}
```

LangChain splits each input into 100-token passages with 20-token overlap and embeds
them using multilingual MiniLM. `chunks` contains those passage texts and vectors;
`vector` is the normalized average of their vectors, used for queries/descriptions.
No index is created or retained on Kaggle. The local app caches the returned passages
and vectors, checks model version/dimensions, and calculates similarity locally.

A single request accepts one to eight texts, each 1–30,000 characters. The app batches
CV requests in groups of four. This request-size safeguard is independent of library size:
there is no fixed total CV cap. The embedding version includes the model and chunk settings;
change it if you change the embedding model, preprocessing, or splitting configuration.

## Profile extraction

```json
{"text": "Alex Example. Skills: Python, SQL. Built document search at Example Labs."}
```

The analysis chain reads the supplied CV, generates complete JSON, and validates the
dictionary's field types. Missing fields receive null/empty defaults. The response includes
contact details, summary, skills, experience, education, projects, certifications,
languages, achievements, links, other details, evidence, and review notes.
The local app checks evidence quotes against the source and persists the profile locally.
No candidate ID or filename is needed. Input: 40–30,000 characters; the model additionally
rejects prompts above 7,000 tokens. Output: at most 3,000 generated tokens, with one retry
for malformed JSON. Structure validation does not guarantee factual accuracy.

## Answers from retrieved passages

```json
{
  "question": "Which skills are stated?",
  "passages": ["Skills: Python, SQL, LangChain.", "Built document search using RAG."]
}
```

The local app selects passages from the chosen CV before making this call. Kaggle formats
them as `[Passage 1]`, `[Passage 2]`, etc., and invokes the question chain. The prompt
requests grounded answers with passage citations and an explicit missing-information response.
The local UI displays the supplied passages for review. No retrieval runs on Kaggle.

Questions accept 1–1,000 characters. Supply one to four passages, each 1–6,000 characters;
the combined prompt still must fit the model token limit. The API trusts the caller to
select the correct candidate's passages. The shipped local app scopes retrieval by CV ID.

## Local operations

`ats/workspace.py` contains plain functions for SQLite persistence, CRUD, keyword search,
coverage ranking, local cosine similarity, scoped retrieval, cached profile analysis, and
question orchestration. SQLite data defaults to `.data/candidates.sqlite3`; optionally set
`ATS_DATA_DIR` privately. No model weights or LangChain dependencies load in the local app.

Upload, library operations, keyword search, and skill ranking with no role description
make **zero API calls**. New profiles, semantic searches, questions, and optional description
tie-breakers need inference. Cached profiles remain available offline.

Errors: 401 for invalid key, 422 for invalid input/output structure, 503 for model failures.
Remote failures do not clear the local library. HTTP errors omit internal exception details.
This service uses a shared project API key and is intended for a private local workspace.
It does not implement user accounts or multiuser data isolation.
