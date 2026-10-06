"""Authenticated client for stateless Kaggle model requests only."""

import os
import time
from urllib.parse import urlparse

import requests


def api_url_from_env():
    """Private developer configuration; no endpoint selection in the product UI."""
    override = os.getenv("ATS_API_URL", "").strip()
    if override:
        return override
    domain = os.getenv("NGROK_DOMAIN", "").strip()
    if not domain:
        return ""
    if "/" in domain or ":" in domain or any(char.isspace() for char in domain):
        raise BackendError(
            "AI connection settings need attention. Please contact your administrator."
        )
    return "https://" + domain


class BackendError(Exception):
    pass


class APIClient:
    def __init__(self, url, api_key):
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise BackendError(
                "The workspace connection needs attention. Please contact your administrator."
            )
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise BackendError(
                "The workspace connection needs attention. Please contact your administrator."
            )
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/")
        ):
            raise BackendError(
                "The workspace connection needs attention. Please contact your administrator."
            )
        if len(api_key) < 24:
            raise BackendError(
                "Workspace access needs attention. Please contact your administrator."
            )
        self.url = url.strip().rstrip("/")
        self.headers = {
            "Authorization": "Bearer " + api_key,
            "ngrok-skip-browser-warning": "true",
        }
        self.on_progress = None

    def request(self, method, path, payload=None, timeout=30):
        try:
            response = requests.request(
                method,
                self.url + path,
                json=payload,
                headers=self.headers,
                timeout=(10, timeout),
                allow_redirects=False,
            )
        except requests.Timeout:
            raise BackendError(
                "AI is taking longer than expected. Please try again shortly."
            ) from None
        except requests.ConnectionError:
            raise BackendError(
                "AI is temporarily unavailable. Your local library is still available."
            ) from None
        except requests.RequestException:
            raise BackendError(
                "We couldn’t complete this request. Please try again shortly."
            ) from None
        try:
            data = response.json()
        except ValueError:
            raise BackendError(
                "AI is temporarily unavailable. Your local library is still available."
            ) from None
        if not isinstance(data, dict):
            raise BackendError(
                "AI is temporarily unavailable. Your local library is still available."
            )
        if not response.ok:
            detail = data.get("detail", "Request failed.")
            self.raise_error(response.status_code, detail)
        return data

    def raise_error(self, status_code, detail):
        if status_code == 503 and isinstance(detail, dict):
            message = detail.get("message", "The model could not finish this request.")
            reference = detail.get("reference", "unknown")
            raise BackendError(f"{message} Reference: {reference}.")
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", "Invalid request") for item in detail)
        if status_code in {400, 404, 422, 429}:
            raise BackendError(str(detail))
        if status_code == 401:
            raise BackendError(
                "Workspace access needs attention. Please contact your administrator."
            )
        raise BackendError("We couldn’t complete this request. Please try again shortly.")

    def infer(self, task, payload, timeout=600):
        """Poll short HTTP requests; a slow generation never holds the tunnel response open."""
        submitted = self.request("POST", "/jobs", {"task": task, "input": payload})
        job_id = submitted.get("job_id")
        if (
            not isinstance(job_id, str)
            or len(job_id) != 32
            or any(char not in "0123456789abcdef" for char in job_id)
        ):
            raise BackendError("The AI service returned an invalid request identifier.")
        started = time.monotonic()
        path = "/jobs/" + job_id
        failures = 0
        try:
            while time.monotonic() - started < timeout:
                try:
                    state = self.request("GET", path)
                except BackendError:
                    failures += 1
                    if failures >= 3:
                        raise
                    time.sleep(1)
                    continue
                failures = 0
                if state.get("status") == "completed" and isinstance(state.get("result"), dict):
                    return state["result"]
                if state.get("status") == "failed":
                    error = state.get("error", {})
                    self.raise_error(
                        error.get("status_code", 503), error.get("detail", "Model request failed.")
                    )
                if state.get("status") not in {"running", "queued"}:
                    raise BackendError("The AI service returned an invalid request status.")
                if self.on_progress:
                    self.on_progress(state["status"], int(time.monotonic() - started))
                time.sleep(1)
            raise BackendError(
                "This model request exceeded ten minutes. Check generation timings in Kaggle."
            )
        finally:
            try:
                self.request("DELETE", path, timeout=5)
            except BackendError:
                pass

    def health(self):
        return self.request("GET", "/health", timeout=15)

    def embed(self, texts):
        return self.infer("embed", {"texts": texts})

    def analyze(self, text):
        return self.infer("analyze", {"text": text})["analysis"]

    def interpret(self, query):
        return self.infer("interpret", {"query": query})["criteria"]

    def evaluate(self, text, criteria):
        return self.infer("evaluate", {"text": text, "criteria": criteria})["assessment"]

    def answer(self, question, passages):
        return self.infer("answer", {"question": question, "passages": passages})["answer"]
