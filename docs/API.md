# Kaggle model API · version 4.1.0

The API performs model inference and LangChain parsing only. Candidate storage, uploads,
source verification, filtering, scoring, ranking, RAG retrieval and exports run locally.
No candidate library, vector store or search/rank endpoints exist on Kaggle.

All requests use `Authorization: Bearer <ATS_API_KEY>` and
`ngrok-skip-browser-warning: true`. The app derives the HTTPS base URL from your private
`NGROK_DOMAIN`; `ATS_API_URL` optionally overrides it for development. The notebook and
app must both use API **4.1.0**. No session header or candidate ID is sent.

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | No body | `{status: "ok", version: "4.1.0"}` |
| POST | `/interpret` | `{query: string}` | `{criteria: {intent, requirements}}` |
| POST | `/evaluate` | `{text: string, criteria: object}` | `{assessment: {name, headline, relevant, explanation, assessments}}` |
| POST | `/analyze` | `{text: string}` | `{analysis: profile_dictionary}` |
| POST | `/embed` | `{texts: [string, ...]}` | `{embedding_version, items: [{vector, chunks: [{text, vector}]}]}` |
| POST | `/answer` | `{question: string, passages: [string, ...]}` | `{answer: string}` |
| POST | `/jobs` | `{task: "analyze"/"evaluate"/"interpret"/"embed"/"answer", input: object}` | HTTP 202, `{job_id}` |
| GET | `/jobs/{job_id}` | No body | `{status: "queued"/"running"/"completed"/"failed", result?, error?}` |
| DELETE | `/jobs/{job_id}` | No body | `{status: "released"}` |

The local app uses `/jobs` for every inference operation. Submission and polling are short
HTTP requests; a slow generation does not hold one tunnel response open. One worker executes
requests sequentially. Completed results are dictionaries with the same shape as the direct
endpoints above. A failed job returns `error: {status_code, detail}`, using the same safe
error contract. Polling a failed job does not rerun generation.

The client polls once per second, tolerates two consecutive transport failures, and releases
results after reading them. Inference has a ten-minute overall deadline. Running jobs cannot
be interrupted by DELETE; queued jobs can be canceled. Finished requests older than
15 minutes are pruned on submission/polling; shutdown clears remaining results.
Jobs and results exist only in memory; no CV library or index
is created on Kaggle. The direct endpoints remain available for development checks.

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

Profiles use 1,800-token sections; assessments use 1,200-token sections, both with 100-token
overlap, processing every section. Profile facts are merged without inferring new facts;
review notes flag potential overlapping entries. Assessments retain the strongest supported
status for each criterion. Partial evidence is never promoted to met during merging.
Profile output is capped at 768 tokens. Assessment uses groups of up to four requirements,
with 360–720 output tokens per group. Interpretation normally uses 768 tokens and scales
to 4,096 for long descriptions/lists; answers use 384.
One retry includes the actual format error. Incomplete JSON is rejected, never silently repaired.

Internally, profile descriptions and evidence use explicit source labels `[L1]`, `[L2]`, etc.
Prompts state the valid label range for each CV section. Assessment
checks use fixed-order rows: `[status, depth, reason, source_labels]`, e.g. `[..., ["L1"]]`.
The parser also accepts equivalent one-based integer/string references (1 or "1" for L1),
inclusive ranges (`L2-L6`, `L2–L6`), and verbatim text verified against the CV, with no offset
guessing. Zero, absent labels and non-integer numeric references are rejected.
Invalid optional citations are isolated: valid profile fields remain, with a review note
and a visible notice in the UI. In assessments, a claim without any verified evidence becomes
`not_found`. Missing requirement rows and malformed JSON still fail validation. A bad citation
does not cause the entire model response to be regenerated.

Profile experience, education and projects use short positional arrays inside the chain;
the output parser restores the public dictionaries. Descriptions use ranges to preserve
all source lines without repeating their text. The local client locates skill excerpts
after extraction, reducing the text the model must generate.
Greedy decoding resets Qwen's sampling defaults to avoid ignored-generation-flags warnings.

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
class and stack frames, token counts, generation timing and controlled schema-validation
messages. They exclude CV text, keys, model output and arbitrary runtime exception messages.
GPU out-of-memory errors have a specific restart message. Failures preserve local CVs.
Application code saves no candidate data to disk on Kaggle; transient inference results
exist until released or expired. Provider/ngrok retention and traffic
inspection are separate settings. This is one private workspace using a shared project
key; separate user accounts and multiuser isolation are not implemented.
