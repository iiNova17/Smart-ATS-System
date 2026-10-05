# Installation and setup guide

Set up your own ngrok account/domain, start the models on Kaggle, and connect the local
Streamlit client. No sample CVs are bundled. You can test the complete workflow with
your own PDFs after the connection checks succeed.

## 1. Before you start

You need:

- Python on your computer; 3.10 or 3.11 is recommended for a fresh environment.
- A Kaggle account with a GPU accelerator available and Internet enabled in the notebook.
- An ngrok account with an assigned/reserved domain and its authtoken.
- This project folder and your readable PDF, DOCX, or TXT CVs.

Only the models and LangChain chains/parsers run on Kaggle. Your computer handles document
extraction, persistent storage, search, ranking, similarity, RAG retrieval, and exports.
Neither model runs on your computer. No database server or local ngrok installation is needed.

## 2. Set up your ngrok domain and endpoint

### Find your fixed domain

1. Sign in to [ngrok Domains](https://dashboard.ngrok.com/domains).
2. Find your account's assigned development domain or an existing reserved domain.
3. Copy the **hostname only**, such as `your-assigned-name.ngrok-free.dev`.
4. Copy the authtoken from [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).

Free accounts receive one stable, automatically assigned dev domain. You must use the
domain assigned to your account; you cannot choose an arbitrary hostname by typing it
into Python. Choosing/reserving another name requires a compatible paid plan, and the
available domain suffixes depend on ngrok's offerings. Use the exact name shown in your
dashboard. See [ngrok domain ownership and plan rules](https://ngrok.com/docs/gateway/domains).

This repository has **no hardcoded personal domain**. Everyone can use their own account
by setting `NGROK_DOMAIN` and `NGROK_AUTHTOKEN`.

### How the endpoint is created

The notebook's SDK call creates an **Agent Endpoint** using your domain and forwards it
to `localhost:8000` inside Kaggle. You do not need to create a separate Cloud Endpoint or
point it at your laptop. After running Block 8, check the ngrok dashboard's endpoint list
to confirm that your hostname is online. Existing domain ownership and the SDK listener
are different things: the hostname is fixed; the listener makes it reachable.

```text
Local Streamlit → HTTPS at your ngrok domain → SDK running in Kaggle
                                                    ↓
                                         Kaggle API localhost:8000
                                                    ↓
                                          Models and LangChain
```

The link stays the same across reconnections with the same domain. The model API is
reachable only while Kaggle, FastAPI, and the ngrok listener are running. Free account
usage quotas still apply; check [current ngrok free-plan limits](https://ngrok.com/docs/pricing-limits/free-plan-limits).

## 3. Prepare the Kaggle notebook

### Generate a separate application key

Run locally:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the output privately as your `ATS_API_KEY`. It must contain at least 24 characters.
It protects model API requests and is separate from your ngrok authtoken.

### Import and configure the notebook

1. Create a private Kaggle Python notebook.
2. Import [`notebooks/smart_ats_kaggle.ipynb`](../notebooks/smart_ats_kaggle.ipynb).
3. Enable a GPU accelerator and Internet in notebook settings.
4. Open the notebook secrets manager (usually **Add-ons → Secrets**).
5. Add the following values and enable this notebook's access to all three:

| Kaggle secret | Value | Purpose |
|---|---|---|
| `NGROK_AUTHTOKEN` | Token from your ngrok account | Authorize the SDK connection |
| `NGROK_DOMAIN` | Your account's domain, hostname only | Reuse the same public address |
| `ATS_API_KEY` | The random key you generated | Authenticate local model requests |

`NGROK_DOMAIN` is configuration rather than a sensitive credential; Kaggle's secrets
manager is used as a convenient way to supply it without editing notebook code.
Block 2 reads the values and sets `NGROK_AUTHTOKEN` and `NGROK_DOMAIN` in the notebook's
environment. Tokens are never printed.

If you are replacing the older pyngrok notebook, start with a fresh Kaggle kernel/session
so the old tunnel and loaded models do not interfere with this setup.

## 4. Run the Kaggle notebook

Run the blocks in order. The first model download can take several minutes.

| Block | Action | Expected result |
|---|---|---|
| 1 | Install dependencies, including the official `ngrok` SDK | Installation completes |
| 2 | Read settings from Kaggle secrets and export SDK environment values | `Private settings loaded.` |
| 3 | Define plain profile dictionaries and validation functions | Functions defined |
| 4 | Load Qwen in 4-bit on GPU and MiniLM on Kaggle CPU | `Models ready.` |
| 5 | Create embedding, profile, question, interpretation and assessment chains | `Chains ready.` |
| 6 | Define the model API | API function defined |
| 7 | Start FastAPI on Kaggle port 8000 | `Model API ready on port 8000.` |
| 8 | Connect your fixed-domain ngrok endpoint | `Available at: https://<your-domain>` |
| 9 | Check the public endpoint with an authenticated health request | `Fixed endpoint is ready...` |
| 10 | Close the listener and API | Run only when finished |

**Run Blocks 1–9. Leave Block 10 until you finish using AI features.**
Keep Kaggle's existing PyTorch/CUDA installation. Block 1's dependencies come from
`requirements-kaggle.txt` and are embedded in the notebook, so the notebook needs no
repository files uploaded. If already-imported packages conflict after installation,
restart the kernel once and resume from Block 2.

Block 1 uses `!{sys.executable} -m pip install ...`. The `!` runs a shell command;
`sys.executable` selects the Python used by the notebook kernel. Run the cell once,
then restart the kernel if already-imported packages changed. Do not replace Kaggle's
PyTorch/CUDA installation.

Block 9 checks the URL, authentication, and API availability; it does not generate a
profile or prove model quality. Test the model workflows with your own CVs in Section 6.

### The ngrok code

Block 8 uses the official SDK and the domain supplied in Block 2:

```python
import ngrok

async def connect_ngrok():
    forwarder = await ngrok.forward(
        "localhost:8000",
        authtoken_from_env=True,
        domain=NGROK_DOMAIN,
    )
    print(f"Available at: {forwarder.url()}")
    return forwarder

if "ats_forwarder" in globals():
    await ats_forwarder.close()
ats_forwarder = await connect_ngrok()
```

Kaggle/Jupyter already runs an event loop. The SDK returns an awaitable there, so the
notebook uses `await` to obtain the listener before calling `.url()`. This small adjustment
avoids the “Future has no attribute url” error that can occur with the synchronous example.
The listener is retained in `ats_forwarder`; rerunning the cell closes the old listener
first. See the [official SDK connection and cleanup documentation](https://ngrok.github.io/ngrok-python/).

In a regular Python script without a running event loop, your original synchronous pattern
is appropriate:

```python
import ngrok
import os

def connect_ngrok():
    forwarder = ngrok.forward(
        "localhost:8000",
        authtoken_from_env=True,
        domain=os.environ["NGROK_DOMAIN"],
    )
    print(f"Available at: {forwarder.url()}")
    return forwarder

forwarder = connect_ngrok()
```

That script must set both environment variables and remain running with its API. Use the
notebook version on Kaggle. `localhost:8000` refers to Kaggle in these cells, not your laptop.

## 5. Install and configure the local client

Open a terminal in this project's root folder, containing `app.py`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

If your environment is already installed, skip creation. If `.env` already exists, edit it
instead of overwriting it with the example.

### Set your private configuration

Open `.env` and replace the placeholders:

```dotenv
NGROK_DOMAIN=your-assigned-domain.ngrok-free.dev
ATS_API_KEY=the-exact-same-private-api-key-as-kaggle
```

Use the **same hostname** as Kaggle's `NGROK_DOMAIN`, without `https://`, a port, trailing
slash, or endpoint path. The app builds the HTTPS API base URL automatically.
The local file does not need `NGROK_AUTHTOKEN`; only Kaggle opens the SDK connection.

For development, an optional `ATS_API_URL` full base URL overrides `NGROK_DOMAIN`.
Leave it unset for normal ngrok use. If you migrate an older `.env`, remove its old
`ATS_API_URL` value or update it so it does not override your new domain.
An optional `ATS_DATA_DIR` chooses another local storage directory.

### Check connectivity and launch

Windows:

```powershell
.\.venv\Scripts\python.exe -m tools.check_connection
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```bash
.venv/bin/python -m tools.check_connection
.venv/bin/python -m streamlit run app.py
```

The check should report `Connected to the Smart ATS model API (version 4.0.0).`
It sends no CV and prints no credential. Open [localhost:8501](http://localhost:8501).

Settings load privately; the recruiter sees only the candidate workspace. Restart Streamlit
after editing `.env`. There is no need to change the hostname after each Kaggle restart.

## 6. Test with your own PDFs

1. Open **Candidates** and upload one or several CVs. Check each import result.
2. Open a candidate and review **Preview document**. Confirm extracted text is readable
   before requesting AI analysis.
3. In **Search**, describe skills and experience together, for example `Python and ros 2
   with hands-on autonomous navigation experience`. Add explicit required skills if needed.
   Check **What the AI looked for**, **Why this candidate matters**, and **CV evidence**.
   A CV listing ROS 2 without navigation work should not pass this request. A web developer
   should not appear merely because Python is present.
4. In **Shortlist**, enter required/preferred skills and a role/experience description.
   Compare coverage and the explanations. Required gaps exclude a CV; preferred gaps are shown.
5. Open **Candidate profile** and click **Prepare profile**. This invokes Qwen; generation
   may take a minute or more on a small GPU. Verify the professional details and JSON export.
6. Ask a question about that candidate's experience. Check the numbered source passages.
7. Repeat a search with `ros2`, `ROS 2`, and generic `ROS`. The first two mean the same
   version; generic ROS covers the broader ecosystem. Confirm an explicit ROS 2 search
   does not admit a CV stating only ROS 1. Repeat the same query to check local cache reuse.

No test CV files are shipped or inserted into your library. Automated checks use synthetic
text defined inside the test module; it never becomes product data.

### How the models and scoring work

The language model is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
loaded in 4-bit on Kaggle GPU 0. The embedding model is
[multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
running on Kaggle CPU. LangChain splits embedding inputs into 100-token passages with
20-token overlap, avoiding silent truncation of long CVs by the small embedding model.

Profile chain: `prompt | model | complete JSON check | JsonOutputParser | validation`.
Question chain: `prompt | model | StrOutputParser`.
Search chains: `prompt | model | complete JSON check | JsonOutputParser | validation`.
They use ordinary functions and dictionaries, with one retry for invalid JSON.
Structural validation does not guarantee factual accuracy; review the original wording.

Questions use local RAG: request embeddings, cache CV vectors locally, calculate cosine
similarity against the selected CV only, retrieve up to four passages, and send those
passages plus the question to the model API. Kaggle has no vector store or retrieval logic.

Search has two model steps. First Qwen interprets the request into skill and experience
criteria. Next it assesses each CV, returning `met`, `partial` or `not_found`, a prose reason,
and supporting quotes for each criterion. It distinguishes a listed skill from practical
work and must respect negation and numeric constraints; unstated experience remains uncertain.
The local app checks the quotes, excludes irrelevant CVs and any unmet required criterion,
then scores coverage with weight 2 for required criteria and 1 for preferred criteria.
Concrete experience evidence breaks ties. There is no embedding-score threshold presented
as proof of relevance. AI interpretation and reasoning can still be wrong; inspect the source.

Long CVs are processed in 2,200-token sections with 100-token overlap. Profiles merge all
section outputs; review notes flag possible overlapping entries. Assessment combines the
strongest supported status per criterion without promoting partial evidence to a full match.

## 7. Stop and restart

- Stop the local app with **Ctrl+C** in its terminal.
- Run notebook **Block 10** when finished using AI. It closes the SDK listener and API.
- In a new Kaggle session, run **Blocks 1–9** again with the same secrets/domain.
- Start the local app again with the same command. The API address stays the same.
- In an existing session with loaded models, Blocks **7–9** restart/check the API and endpoint;
  do not reload models unnecessarily. Stop other sessions using the same domain first.

The fixed hostname remains associated with your ngrok account, while its online endpoint
depends on the running session. Notebook stopping, GPU/session quotas, and network failures
can interrupt AI requests even though the address stays fixed.

## 8. Storage, offline use, and document limits

The local library, profiles, and cached vectors persist in `.data/candidates.sqlite3`.
SQLite is part of Python, so there is no database server to install. Back up `.data/` to
preserve the workspace between machines. Browser sessions on the same app share one
library; separate recruiter accounts are not implemented.

| Works when Kaggle is stopped | Requires Kaggle online |
|---|---|
| Import, preview, browse, remove, or clear CVs | Prepare new AI profiles |
| View/export cached profiles and existing local results | AI search and shortlisting |
| Cached extracted candidate details | New question embeddings and answers |

Cached CV embeddings are reused, but a new question still requires a query embedding.
Search interpretations and candidate assessments are also cached locally; an unchanged
request reuses them. The service must be online to start the AI workflow.
Changing embedding versions rebuilds affected local caches. Kaggle/app/browser restarts do
not erase the local library. Removal clears that CV's text, profile, and vectors locally.

Documents may be up to 10 MB and 30,000 extracted characters; PDFs may have up to 30 pages.
Scanned/partly scanned and encrypted PDFs need preprocessing. Complex Word layouts may
lose text. Each generation has a 2,600-token output budget and one retry for malformed
JSON; incomplete output is rejected rather than silently repaired. Long CVs use multiple
generations. There is no fixed CV-count cap; the serial AI review is intended for a modest
personal collection. First searches and long-CV profiles can take several minutes.

Original files stay local; text crosses ngrok when a model operation needs it. Application
code does not persist CVs on Kaggle. Review ngrok traffic inspection and provider retention
settings before using real candidate data. `.env`, `.data/`, and local SQLite files are
ignored by Git; keep credentials and documents private.

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| Domain is rejected or not authorized | Use a domain shown in your account's Domains page and an authtoken from that same account |
| Endpoint is already online | Close another agent/notebook using the domain; rerun Block 8 |
| Future has no `.url()` | Use the notebook's `await` version unchanged |
| API unreachable or ngrok endpoint offline | Keep Kaggle alive; confirm Blocks 7–9 completed |
| Connection still uses an old URL | Remove/update `ATS_API_URL` in `.env`; it overrides `NGROK_DOMAIN` |
| AI access cannot be verified | Local `ATS_API_KEY` must exactly match the Kaggle secret |
| AI needs an update | Use this notebook; the local app expects API version `4.0.0` |
| Secret lookup fails | Check exact names and enable notebook access to all three secrets |
| GPU/download error | Enable GPU/Internet and check Kaggle availability/quota |
| GPU memory error | Restart the kernel to unload old models; avoid loading the model twice |
| Profile/server request fails | Match the error reference in the app to the Kaggle output. Error class and stack frames identify the failing operation without logging candidate text or credentials |
| Invalid/incomplete model JSON | One retry is automatic; fenced JSON and unknown text details are supported. Retry or inspect the extracted document if failure persists |
| PDF needs OCR | Convert scanned pages to searchable text before importing |
| Local library cannot open | Ensure `.data/` or the configured folder is writable |
| ngrok quota error | Check the account's current usage and plan limits in the dashboard |

## 10. Maintainer checks

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/generate_notebook.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py ats backend tests tools
```

Regenerate the notebook after editing model/API source or `requirements-kaggle.txt`.
Tests use deterministic model substitutes and a fake SDK listener, so they need no GPU,
ngrok credential, or candidate files. They verify local workflows and notebook behavior;
your account's domain and real model inference are verified with the steps above.
See [the model API contract](API.md) for request details.

## Updating from API 3.0

Keep your domain and all three secrets. Keep local `.env` and `.data/`; the app adds a
cache table automatically and preserves every saved CV/profile.

1. Stop the old notebook API/tunnel with its Block 10.
2. Replace the notebook with this repository's updated `smart_ats_kaggle.ipynb`.
3. Restart the Kaggle session for a clean installation. Run Block 1, restart the kernel
   if imports changed, then run Blocks 2–9. Block 9 must report API `4.0.0`.
4. Restart Streamlit with the usual command to load the new dark theme and client.
5. Test profile preparation, a combined robotics search, an irrelevant CV, and shortlisting.

If the models are already loaded and dependencies have not changed, you can stop the old
server/tunnel, replace Blocks 3, 5 and 6, run those updated blocks, then run Blocks 7–9.
Do not reload Block 4 into an already populated GPU unnecessarily. The new endpoints are
`/interpret` and `/evaluate`; both perform model inference only.
