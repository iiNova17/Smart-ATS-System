"""Stateless model API: no CV library, search, ranking, or vector store."""

import logging
import secrets
import traceback

from fastapi import Body, Depends, FastAPI, Header, HTTPException

from ats.validation import validate_analysis, validate_assessment, validate_criteria, validate_text


def create_app(api_key, runtime):
    if len(api_key) < 24:
        raise ValueError("ATS_API_KEY must contain at least 24 characters.")
    app = FastAPI(
        title="Smart ATS · Model service",
        version="4.0.0",
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
        return {"status": "ok", "version": "4.0.0"}

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

    return app
