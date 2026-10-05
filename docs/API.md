# Kaggle model API · version 4.0.0

The API performs model inference and LangChain parsing only. Candidate storage, uploads,
source verification, filtering, scoring, ranking, RAG retrieval and exports run locally.
No candidate library, vector store or search/rank endpoints exist on Kaggle.

All requests use `Authorization: Bearer <ATS_API_KEY>` and
`ngrok-skip-browser-warning: true`. The app derives the HTTPS base URL from your private
`NGROK_DOMAIN`; `ATS_API_URL` optionally overrides it for development. The notebook and
app must both use API **4.0.0**. No session header or candidate ID is sent.

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | No body | `{status: "ok", version: "4.0.0"}` |
| POST | `/interpret` | `{query: string}` | `{criteria: {intent, requirements}}` |
| POST | `/evaluate` | `{text: string, criteria: object}` | `{assessment: {name, headline, relevant, explanation, assessments}}` |
| POST | `/analyze` | `{text: string}` | `{analysis: profile_dictionary}` |
| POST | `/embed` | `{texts: [string, ...]}` | `{embedding_version, items: [{vector, chunks: [{text, vector}]}]}` |
| POST | `/answer` | `{question: string, passages: [string, ...]}` | `{answer: string}` |

## Search interpretation

`/interpret` accepts 1–6,000 characters. Qwen combines keywords, technical skills and
experience requirements; aliases and spelling variations are interpreted in context.
An explicit ROS 2 request requires that version; generic ROS covers the broader ecosystem.
The prompt must not invent extra requirements. OR alternatives remain one criterion.

```json
{
  "criteria": {
    "intent": "Python and ROS 2 with practical autonomous navigation experience",
    "requirements": [
      {"label": "Python", "kind": "skill", "required": true},
      {"label": "ROS 2", "kind": "skill", "required": true},
      {"label": "autonomous navigation experience", "kind": "experience", "required": true}
    ]
  }
}
```

Requirements contain a label, `kind` (`skill` or `experience`) and a Boolean `required`.
The local app preserves explicit required/preferred form entries even if interpretation
omits them. The user can inspect the interpreted criteria in the results.

## Candidate assessment

`/evaluate` accepts a CV (40–30,000 characters) plus validated criteria. It returns model
reasoning about one supplied text; it never fetches candidates, filters a library or assigns
rank positions. Every criterion must have an assessment using its zero-based index.

```json
{
  "assessment": {
    "relevant": true,
    "explanation": "The navigation project connects ROS 2 to practical robot work.",
    "assessments": [
      {"index": 0, "status": "met", "depth": "applied",
       "reason": "The mapping project uses Python for practical robotics work.",
       "quotes": ["Implemented indoor mapping in Python with SLAM."]}
    ]
  }
}
```

The example shows one criterion; real output must cover every supplied criterion.
`status` is `met`, `partial` or `not_found`. `depth` is `listed`, `applied` or `extensive`
according to concrete CV evidence. `reason` connects source facts to the request; the
separate `quotes` list holds verbatim evidence. A skill mention alone cannot prove hands-on
experience, seniority or years. Numeric experience constraints require clearly stated
relevant durations; the model must not infer years from overlapping jobs.

The local app checks quotes against the CV and enforces explicit ROS-version evidence.
An unsupported quote cannot establish a match. Irrelevant CVs and unmet/partial required
criteria are excluded. Remaining candidates are ordered locally by weighted criteria
coverage (required 2, preferred 1; met 1, partial 0.5, not_found 0), then evidence depth.
The score measures criteria coverage, not proficiency or hiring probability.
Interpretations and assessments are cached in local SQLite by request/CV content and
prompt version. A first search reviews every CV; there is no silent top-k candidate cutoff.
Structural and quote validation cannot independently prove the model's reasoning correct.

## Profiles and JSON parsing

`/analyze` extracts contact details, summary, skills, experience, education, projects,
certifications, languages, achievements, links, other professional details, evidence and
review notes. Unknown fields use null/empty defaults. Fenced JSON is accepted; incomplete
JSON is rejected. Unknown nested text values may be null; explicitly stated numeric years
are stored as text. The local app checks quotes and persists profiles.

Analysis and assessment split long CVs into overlapping 2,200-token sections with 100-token
overlap, processing every section. Profile facts are merged without inferring new facts;
review notes flag potential overlapping entries. Assessments retain the strongest supported
status for each criterion. Partial evidence is never promoted to met during merging.
Each generation has a 2,600-token output budget, with one retry for invalid JSON.
The local client allows up to 15 minutes for section-based profile/assessment requests.

Chains use familiar LangChain composition:

```text
Interpret / assess / profile: prompt | model | complete JSON check | JsonOutputParser | validation
Question:                    prompt | model | StrOutputParser
```

## Embeddings and local RAG

`/embed` accepts one to eight texts, each 1–30,000 characters. Multilingual MiniLM runs on
Kaggle CPU. LangChain splits inputs into 100-token passages with 20-token overlap. The API
returns passages, vectors and a normalized average; it retains no index. The embedding
version identifies model and chunk settings. The local app caches vectors, checks version
and dimensions, and computes cosine similarity locally.

`/answer` accepts a question (1–1,000 characters) and one to four locally retrieved passages,
each 1–6,000 characters. The model answers using the supplied facts and cites `[Passage N]`.
The app scopes retrieval to the selected CV before sending passages. Embedding similarity
is used for question retrieval, while search relevance comes from explicit AI assessment.

## Failures and privacy

401 means an invalid API key; 422 means invalid input or model output; 503 means a model
runtime failure. A 503 contains a safe message, code and error reference:

```json
{"detail": {"message": "The model could not finish this request. Check the Kaggle error reference.",
            "code": "model_error", "reference": "12ab34cd"}}
```

Match the reference to the Kaggle cell output. Logs contain operation name, exception
class and stack frames, not CV text, keys, model output or arbitrary exception messages.
GPU out-of-memory errors have a specific restart message. Failures preserve local CVs.
Application code stores no candidate data on Kaggle. Provider/ngrok retention and traffic
inspection are separate settings. This is one private workspace using a shared project
key; separate user accounts and multiuser isolation are not implemented.
