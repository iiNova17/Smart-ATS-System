"""Generate a standalone, function-based notebook containing model inference only."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


def selected_source(path, names):
    text = (ROOT / path).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    selected = []
    for node in ast.parse(text).body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            selected.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n\n".join(selected)


def api_source():
    text = (ROOT / "backend/api.py").read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith("from ats."))


def install_command():
    packages = [
        line.strip()
        for line in (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return (
        "import sys\n!{sys.executable} -m pip install -q "
        + " ".join('"' + package + '"' for package in packages)
        + "\n"
    )


markdown("""# Smart ATS · Kaggle model service
Only model inference and LangChain chains/parsers run here. **No CV collection,
file uploads, search, ranking, database, or retrieval runs in this notebook.**
All code uses functions and dictionaries, with zero custom classes.

Your local Streamlit app stores CV text, profiles, and vectors in SQLite; it performs
evidence verification, relevance filtering, scoring, passage retrieval, and exports locally.
Both the language model and embedding model run on Kaggle.

Enable GPU and Internet in a private notebook. Add `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`, and `ATS_API_KEY`
as Kaggle secrets and enable notebook access to both. Use separate values.
Set `NGROK_DOMAIN` to a domain that belongs to your own ngrok account. Free accounts must use
their assigned dev domain; specifying a name does not reserve it.
Run Blocks 1–9; leave Block 10 until you finish using AI features.
""")
markdown("""## Block 1 — Install libraries
Keep Kaggle's existing PyTorch/CUDA installation. If already-imported packages conflict,
restart the kernel once after installing, then run from Block 2.
""")
code(install_command())
markdown("""## Block 2 — Read secrets
The ngrok token authenticates the tunnel agent. The API key protects model requests.
Never print either token. Generate an API key locally with `secrets.token_urlsafe(32)`.
""")
code("""import os
from kaggle_secrets import UserSecretsClient

settings = UserSecretsClient()
NGROK_AUTHTOKEN = settings.get_secret("NGROK_AUTHTOKEN").strip()
ATS_API_KEY = settings.get_secret("ATS_API_KEY").strip()
NGROK_DOMAIN = settings.get_secret("NGROK_DOMAIN").strip()
os.environ["NGROK_AUTHTOKEN"] = NGROK_AUTHTOKEN
os.environ["NGROK_DOMAIN"] = NGROK_DOMAIN
assert NGROK_AUTHTOKEN, "Add your ngrok authtoken to Kaggle secrets."
assert NGROK_DOMAIN and ":" not in NGROK_DOMAIN and "/" not in NGROK_DOMAIN, \
    "NGROK_DOMAIN must be a hostname only, without https://, port, or path."
assert len(ATS_API_KEY) >= 24, "Use a random API key of at least 24 characters."
print("Private settings loaded.")
""")
markdown("""## Block 3 — Describe the model's JSON output
Profiles are plain dictionaries. These templates and functions check the parser output;
they do not store CVs or rank candidates.
""")
code(
    "from copy import deepcopy\n\n"
    + selected_source(
        "ats/validation.py",
        {
            "PROFILE_TEMPLATE",
            "DETAIL_TEMPLATES",
            "validate_text",
            "fill_fields",
            "validate_analysis",
            "validate_criteria",
            "validate_assessment",
        },
    )
)
markdown("""## Block 4 — Load the two models
Qwen3-4B-Instruct-2507 runs in 4-bit on GPU 0. Multilingual MiniLM runs on Kaggle CPU.
LangChain splits embedding inputs into 100-token passages with 20-token overlap,
so the small embedding model can process long CVs without silently truncating them.
""")
code(
    selected_source("backend/runtime.py", {"load_models"})
    + "\n\ntokenizer, model, embeddings, splitter = load_models()\nprint('Models ready.')\n"
)
markdown("""## Block 5 — Build inference functions and chains
Profile: `prompt | model | complete JSON check | JsonOutputParser | validation`.
Search interpretation and candidate assessment use the same JSON chain pattern.
Long CVs are processed in 2,200-token sections; no CV section is silently discarded.
Question: `prompt | model | StrOutputParser`.

`embed_texts` returns vectors and passage text; it keeps no index. The local app caches
these vectors and retrieves relevant passages before calling the question chain.
Long query embeddings average their passage vectors. No local model is required.
""")
code(
    "import json\nfrom threading import Lock\n\n"
    + selected_source(
        "backend/runtime.py",
        {
            "EMBEDDING_VERSION",
            "build_analysis_parser",
            "build_json_parser",
            "merge_profiles",
            "merge_assessments",
            "make_runtime",
        },
    )
    + "\n\nruntime = make_runtime(tokenizer, model, embeddings, splitter)\nprint('Chains ready.')\n"
)
markdown("""## Block 6 — Expose only model requests
FastAPI is a small HTTP wrapper around the functions:
- `GET /health`: check the connection and API version.
- `POST /embed`: text → passages and vectors.
- `POST /analyze`: CV text → parsed profile.
- `POST /answer`: question + locally retrieved passages → answer.
- `POST /interpret`: search request → normalized requirements and experience criteria.
- `POST /evaluate`: CV text + criteria → evidence-backed assessment and explanation.

There are no `/cvs`, `/search`, or `/rank` endpoints and no collection/session state.
Text exists transiently during inference; no application code saves CVs here.
""")
code(api_source())
markdown("""## Block 7 — Start the API on Kaggle
The server listens on Kaggle's `127.0.0.1:8000`. A background thread lets the next cell
open ngrok while this server continues running. Access logs are disabled.
""")
code("""import threading
import time
import requests
import uvicorn

if "api_server" in globals():
    api_server.should_exit = True
    api_thread.join(timeout=15)
    if api_thread.is_alive():
        raise RuntimeError("Previous server is busy. Wait, then rerun.")

app = create_app(ATS_API_KEY, runtime)
api_server = uvicorn.Server(uvicorn.Config(
    app, host="127.0.0.1", port=8000, log_level="warning", access_log=False))
api_thread = threading.Thread(target=api_server.run, daemon=True)
api_thread.start()
for attempt in range(40):
    if not api_thread.is_alive():
        raise RuntimeError("Server stopped. Check the preceding output.")
    if api_server.started:
        response = requests.get("http://127.0.0.1:8000/health",
                                headers={"Authorization": "Bearer " + ATS_API_KEY}, timeout=2)
        response.raise_for_status()
        assert response.json()["version"] == "4.0.0"
        print("Model API ready on port 8000.")
        break
    time.sleep(0.5)
else:
    raise RuntimeError("Server did not become ready.")
""")
markdown("""## Block 8 — Start the fixed-domain ngrok endpoint
The official `ngrok` Python SDK forwards your account's fixed HTTPS domain to port 8000
**inside Kaggle**. No ngrok installation is needed on your laptop.

Your domain must already be assigned/reserved in the same account as your authtoken.
This creates an **Agent Endpoint**, so no separate Cloud Endpoint is needed in the dashboard.
Kaggle/Jupyter has a running event loop; `await` resolves the SDK's listener before `.url()`.
Retain the returned listener in `ats_forwarder` and close it before rerunning this cell.

Set local `.env` to the same `NGROK_DOMAIN` and `ATS_API_KEY` as this notebook.
This URL remains the same when reconnecting with the same domain. Keep Kaggle running;
a fixed address does not keep a stopped notebook online.
""")
code("""import ngrok

async def connect_ngrok():
    forwarder = await ngrok.forward(
        "localhost:8000", authtoken_from_env=True, domain=NGROK_DOMAIN
    )
    print(f"Available at: {forwarder.url()}")
    return forwarder

if "ats_forwarder" in globals():
    await ats_forwarder.close()
ats_forwarder = await connect_ngrok()
""")
markdown("""## Block 9 — Check the fixed endpoint
This authenticated health request checks the public endpoint without sending a CV.
Use your own PDFs in the local app to test extraction, embeddings, profiles, and answers.
""")
code("""headers = {"Authorization": "Bearer " + ATS_API_KEY,
           "ngrok-skip-browser-warning": "true"}
url = ats_forwarder.url()
assert url.rstrip("/") == "https://" + NGROK_DOMAIN, "Unexpected endpoint URL."

response = requests.get(url + "/health", headers=headers, timeout=30)
response.raise_for_status()
assert response.json()["version"] == "4.0.0"
print("Fixed endpoint is ready. Start the local app and upload your PDFs.")
""")
markdown("""## Block 10 — Stop when finished
This closes the tunnel and API. Local CVs, profiles, and cached vectors remain on your
laptop. Next time run Blocks 1–9 using the same domain; your local API URL stays the same.
""")
code("""if "ats_forwarder" in globals():
    await ats_forwarder.close()
    del ats_forwarder
if "api_server" in globals():
    api_server.should_exit = True
    api_thread.join(timeout=15)
print("Shutdown requested. Your local candidate library is unaffected.")
""")
markdown("""## Troubleshooting
- Download/GPU: enable Internet and GPU; check your Kaggle quota.
- Memory: restart to unload old models; begin with a shorter CV.
- Invalid JSON: one retry is included; fenced JSON is accepted, incomplete JSON is rejected.
- Profile/server failure: match the app's error reference to the Kaggle cell output. The error
  class and stack frames are logged without credentials or CV text. Restart the kernel if the
  error is GPU memory related; run the model-loading block once, then rebuild chains/server.
- Connection: check Block 7, Block 8, `.env` URL, and matching API key.
- Endpoint offline: keep Kaggle alive; rerun server/endpoint cells using the same domain.
- Domain rejected: verify ownership and the authtoken's account in the ngrok dashboard.
- Endpoint already online: stop another agent/session using this domain, then rerun Block 8.
- A Future has no `.url()`: use Block 8 unchanged with its `await` calls.
- Offline: imports, library browsing, cached profiles and exports still work locally.
  AI search, shortlisting and new profiles/answers require this service.

References: [Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
[LangChain embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers),
[ngrok Python SDK](https://ngrok.github.io/ngrok-python/),
[ngrok domains](https://ngrok.com/docs/gateway/domains).
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}
for i, cell in enumerate(cells):
    cell["id"] = f"ats-cell-{i:02d}"
output = ROOT / "notebooks/smart_ats_kaggle.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Generated {output.name}: {len(cells)} cells")
