"""Model inference with transient requests; no CV library, search, ranking, or vector store."""

import logging
import secrets
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic

from fastapi import Body, Depends, FastAPI, Header, HTTPException

from ats.validation import validate_analysis, validate_assessment, validate_criteria, validate_text


def create_app(api_key, runtime):
    if len(api_key) < 24:
        raise ValueError("ATS_API_KEY must contain at least 24 characters.")
    jobs = {}
    jobs_lock = Lock()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")

    @asynccontextmanager
    async def lifespan(app):
        yield
        worker.shutdown(wait=True, cancel_futures=True)
        jobs.clear()

    app = FastAPI(
        title="Smart ATS · Model service",
        version="4.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authenticate(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, "Bearer " + api_key):
            raise HTTPException(401, "AI access could not be verified.")

    def run(operation, *args):
        try:
            return operation(*args)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception as exc:
            reference = secrets.token_hex(4)
            # Print stack frames and error class, never CV text, credentials or model output.
            logging.getLogger("smart_ats").error(
                "Model request %s failed in %s: %s",
                reference,
                operation.__name__,
                type(exc).__name__,
            )
            traceback.print_tb(exc.__traceback__)
            memory_error = "outofmemory" in type(exc).__name__.lower()
            message = (
                "The model ran out of GPU memory. Restart the Kaggle session and load the "
                "model once."
                if memory_error
                else "The model could not finish this request. Check the Kaggle error reference."
            )
            raise HTTPException(
                503,
                {
                    "message": message,
                    "reference": reference,
                    "code": "gpu_memory" if memory_error else "model_error",
                },
            ) from None

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        return {"status": "ok", "version": "4.1.0"}

    def interpretation_request(payload):
        text = validate_text(payload.get("query"), "Search request", 1, 6000)
        return validate_criteria(runtime["interpret_query"](text))

    @app.post("/interpret", dependencies=[Depends(authenticate)])
    def interpret(payload: dict = Body()):
        return {"criteria": run(interpretation_request, payload)}

    def evaluation_request(payload):
        text = validate_text(payload.get("text"), "CV text", 40, 30000)
        criteria = validate_criteria(payload.get("criteria"))
        return validate_assessment(runtime["evaluate_text"](text, criteria), criteria)

    @app.post("/evaluate", dependencies=[Depends(authenticate)])
    def evaluate(payload: dict = Body()):
        return {"assessment": run(evaluation_request, payload)}

    def embedding_request(payload):
        texts = payload.get("texts")
        # A request-size safeguard, independent of the size of the local CV library.
        if not isinstance(texts, list) or not 1 <= len(texts) <= 8:
            raise ValueError("Send between one and eight texts per embedding request.")
        texts = [validate_text(text, "Text", 1, 30000) for text in texts]
        return runtime["embed_texts"](texts)

    @app.post("/embed", dependencies=[Depends(authenticate)])
    def embed(payload: dict = Body()):
        return run(embedding_request, payload)

    def analysis_request(payload):
        text = validate_text(payload.get("text"), "CV text", 40, 30000)
        return validate_analysis(runtime["analyze_text"](text))

    @app.post("/analyze", dependencies=[Depends(authenticate)])
    def analyze(payload: dict = Body()):
        return {"analysis": run(analysis_request, payload)}

    def answer_request(payload):
        question = validate_text(payload.get("question"), "Question", 1, 1000)
        passages = payload.get("passages")
        if not isinstance(passages, list) or not 1 <= len(passages) <= 4:
            raise ValueError("Supply one to four retrieved passages.")
        passages = [validate_text(text, "Passage", 1, 6000) for text in passages]
        context = "\n\n".join(f"[Passage {i}] {text}" for i, text in enumerate(passages, 1))
        return runtime["answer_question"](context, question)

    @app.post("/answer", dependencies=[Depends(authenticate)])
    def answer(payload: dict = Body()):
        return {"answer": run(answer_request, payload)}

    # These are transient model requests, not a candidate store or search service.
    operations = {
        "interpret": (interpretation_request, "criteria"),
        "evaluate": (evaluation_request, "assessment"),
        "analyze": (analysis_request, "analysis"),
        "embed": (embedding_request, None),
        "answer": (answer_request, "answer"),
    }

    def infer(task, payload):
        operation, field = operations[task]
        result = run(operation, payload)
        return {field: result} if field else result

    def clean_jobs():
        for job_id, record in list(jobs.items()):
            future = record["future"]
            if future.done() and monotonic() - record["created"] > 900:
                del jobs[job_id]

    @app.post("/jobs", status_code=202, dependencies=[Depends(authenticate)])
    def submit(payload: dict = Body()):
        task, inputs = payload.get("task"), payload.get("input")
        if not isinstance(task, str) or task not in operations or not isinstance(inputs, dict):
            raise HTTPException(422, "Supply a supported model task and input object.")
        with jobs_lock:
            clean_jobs()
            if sum(not record["future"].done() for record in jobs.values()) >= 8:
                raise HTTPException(
                    429, "The model is busy. Wait for the current request to finish."
                )
            job_id = secrets.token_hex(16)
            jobs[job_id] = {"future": worker.submit(infer, task, inputs), "created": monotonic()}
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def poll(job_id: str):
        with jobs_lock:
            clean_jobs()
            record = jobs.get(job_id)
            if record is None:
                raise HTTPException(404, "The model request expired. Please retry.")
            future = record["future"]
        if not future.done():
            return {"status": "running" if future.running() else "queued"}
        try:
            result = future.result()
        except HTTPException as exc:
            return {
                "status": "failed",
                "error": {"status_code": exc.status_code, "detail": exc.detail},
            }
        return {"status": "completed", "result": result}

    @app.delete("/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def discard(job_id: str):
        with jobs_lock:
            record = jobs.get(job_id)
            if record is not None and (record["future"].done() or record["future"].cancel()):
                del jobs[job_id]
        return {"status": "released"}

    return app
