"""Stateless model API: no CV library, search, ranking, or vector store."""

import secrets

from fastapi import Body, Depends, FastAPI, Header, HTTPException

from ats.validation import validate_analysis, validate_text


def create_app(api_key, runtime):
    if len(api_key) < 24:
        raise ValueError("ATS_API_KEY must contain at least 24 characters.")
    app = FastAPI(
        title="Smart ATS · Model service",
        version="3.0.0",
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
        except Exception:
            raise HTTPException(503, "AI is temporarily unavailable. Please try again.") from None

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        return {"status": "ok", "version": "3.0.0"}

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
