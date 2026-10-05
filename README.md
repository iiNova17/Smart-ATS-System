# Smart ATS

A local candidate workspace for importing CVs, finding skills, building shortlists, and
exploring individual profiles. Streamlit runs on your computer; Kaggle runs the models.
The official `ngrok` Python SDK connects them through **your account's fixed domain**.

**[Installation & setup guide](docs/KAGGLE_GUIDE.md)** ·
**[Kaggle notebook](notebooks/smart_ats_kaggle.ipynb)** ·
**[Model API reference](docs/API.md)**

## Candidate workflows

- **Candidates:** import one CV or a batch of PDF, DOCX, or TXT files, inspect per-file results,
  filter the library, preview extracted text, and remove documents.
- **Search:** combine skills, keywords and experience in one natural-language request.
  The AI understands equivalent wording, assesses practical relevance, and explains why
  each candidate matters. Only candidates with supported required criteria appear.
- **Shortlist:** compare required skills, preferred skills and role experience together.
  Order matches by verified criteria coverage, then depth of relevant experience; export CSV.
- **Candidate profile:** prepare structured contact, skills, experience, education, projects,
  and other professional details. Ask questions grounded in that candidate's CV and export JSON.

Connection settings stay in private configuration. The recruiter interface has no backend
selector, API-key form, or capacity controls. No example CVs are bundled; use your own documents.

## Architecture

```text
Browser → Streamlit on your computer
             ├─ CV extraction, SQLite library, caches, and exports
             ├─ Evidence verification, relevance filtering, scoring, ranking and RAG retrieval
             └─ HTTPS model requests → your fixed ngrok domain → Kaggle model API
                                                                ├─ Qwen + LangChain
                                                                └─ MiniLM embeddings
```

| Your computer | Kaggle |
|---|---|
| Streamlit UI and document extraction | Language model and embedding model |
| Persistent CV library and profile/vector caches | LangChain prompts, chains, and output parsers |
| Scoring, filtering, similarity, and retrieval | Query interpretation and candidate assessment |
| Ranking, evidence validation, and exports | Stateless model API and ngrok Agent Endpoint |

Kaggle stores no candidate library or search index. The local app loads no model weights
and needs no GPU. CV text, cached profiles, and vectors persist in `.data/candidates.sqlite3`.

## Quick start

### 1. Set up your ngrok domain

Sign in to [ngrok Domains](https://dashboard.ngrok.com/domains) and copy a domain owned by
your account. Free accounts use their automatically assigned dev domain, which stays fixed.
The SDK does not reserve a name merely because you put it in `domain=`.
See [ngrok's domain rules](https://ngrok.com/docs/gateway/domains).

Copy your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken), then
generate a **separate API key** locally:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep this key for both Kaggle and the local client.

### 2. Start Kaggle

Import [`notebooks/smart_ats_kaggle.ipynb`](notebooks/smart_ats_kaggle.ipynb) into a private
Kaggle Python notebook. Enable **GPU** and **Internet**. Add these notebook secrets and
enable access to all three:

| Secret | Value |
|---|---|
| `NGROK_AUTHTOKEN` | Authtoken from your ngrok account |
| `NGROK_DOMAIN` | Your assigned/reserved hostname, without `https://` or a path |
| `ATS_API_KEY` | The random API key you generated |

Run **Blocks 1–9**. Block 8 opens your fixed domain; Block 9 checks public connectivity.
Leave shutdown **Block 10** until you finish. The notebook contains functions and
dictionaries with zero custom classes. See the [block-by-block guide](docs/KAGGLE_GUIDE.md#4-run-the-kaggle-notebook).

### 3. Start the local client

From the project folder in PowerShell (Python 3.10 or 3.11 recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your own values:

```dotenv
NGROK_DOMAIN=your-assigned-domain.ngrok-free.dev
ATS_API_KEY=the-same-private-api-key-as-kaggle
```

The app builds `https://<NGROK_DOMAIN>` automatically. Do not put the ngrok authtoken in
the local file. If `.env` already exists, edit it instead of overwriting it with the example.

Check the model connection, then launch the app:

```powershell
.\.venv\Scripts\python.exe -m tools.check_connection
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) and upload your PDFs through **Candidates**.
For macOS/Linux commands, restart instructions, and troubleshooting, use the
[complete setup guide](docs/KAGGLE_GUIDE.md).

The URL stays the same when reconnecting with the same account/domain. **Kaggle and the
SDK connection must still be running for AI requests.** A fixed URL does not provide
continuous hosting when the notebook stops.

## Models and scoring

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) extracts profiles
and answers questions, interprets searches, and assesses candidate evidence in 4-bit on Kaggle GPU 0.
[Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
embeds passages on Kaggle CPU. The profile chain uses **JsonOutputParser** with complete-JSON
and field validation; the question chain uses **StrOutputParser**. RAG retrieval stays local.

The model turns your request into required/preferred skill and experience criteria. It
assesses each CV against every criterion and produces an explanation with verbatim evidence.
The local app verifies quotes, excludes irrelevant candidates or missing required evidence,
and computes criteria coverage (required weight 2; preferred weight 1). Concrete project/work
evidence breaks equal-coverage ties. Coverage is a review aid, not a proficiency or hiring score.

`ros 2`, `ROS2`, and `ROS-2` are equivalent. Generic `ROS` is broad enough to include ROS 1
or ROS 2; asking explicitly for ROS 2 requires evidence of that version. The model can also
interpret related terminology; review **What the AI looked for** to check its interpretation.
Model judgments can be wrong even when a quoted excerpt exists; review the explanation and source.

## Local storage and offline use

Imports, browsing/removal and viewing/exporting cached profiles work without Kaggle.
AI search, shortlisting, new profiles and questions require the model service. Search
interpretations and assessments are cached locally for repeated criteria and unchanged CVs.
A first search reviews every CV, which can take time; the UI shows progress. No candidates
are silently excluded by a fixed retrieval or library-size cap.

Local data survives app, browser, and Kaggle restarts. Back up `.data/` to preserve the library
between machines. Set `ATS_DATA_DIR` privately to choose another storage folder. This is one
local workspace shared by browser sessions; separate recruiter accounts are not implemented.

PDF/DOCX/TXT imports support 10 MB and 30,000 extracted characters; PDFs support up to 30
pages. Scanned/partly scanned or encrypted PDFs need preprocessing. Complex Word layouts
can lose text. Long CVs are processed in overlapping 2,200-token sections and merged;
each generation has a 2,600-token output budget and one retry for invalid JSON. There is
no fixed CV-count cap, but serial AI review suits a modest personal library.

Original files stay local. Extracted text crosses ngrok only when inference needs it.
Application code does not persist CVs on Kaggle; ngrok traffic inspection/provider retention
are separate settings. Keep credentials and candidate data private. `.env` and `.data/`
are ignored by Git.

## Repository layout

```text
app.py                         Streamlit entry point
ats/                           Local extraction, storage, matching, retrieval, and model client
backend/                       Kaggle model/chain functions and stateless API
assets/                        Product styles
.streamlit/                    Streamlit theme and upload settings
notebooks/smart_ats_kaggle.ipynb Standalone Kaggle notebook
docs/KAGGLE_GUIDE.md            Installation, ngrok, startup, and troubleshooting
docs/API.md                    Model request contract
tools/                         Notebook generator and connection check
tests/                         Automated checks with inline synthetic test text
requirements.txt               Local runtime dependencies
requirements-kaggle.txt        Kaggle dependencies; embedded in notebook Block 1
requirements-dev.txt           Local verification dependencies
```

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/generate_notebook.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py ats backend tests tools
```

Regenerate the notebook after changing the model/API source or Kaggle dependency list.
Tests download no models and contact no ngrok account; SDK lifecycle tests use a substitute
listener. They do not prove domain ownership or GPU inference. Verify connectivity with
Block 9 and `tools.check_connection`, then exercise the model workflows with your own PDFs.

## Updating an existing installation

This release requires **model API 4.0.0**. Keep your existing `.env`, ngrok domain, secrets
and `.data/`. Replace the Kaggle notebook with the updated file. For the simplest update,
restart the Kaggle session, run Block 1, restart the kernel if packages changed, then run
Blocks 2–9. Restart Streamlit to load the dark theme and new client.
See [the update instructions](docs/KAGGLE_GUIDE.md#updating-from-api-30).
