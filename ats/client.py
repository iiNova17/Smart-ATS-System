"""Authenticated client for stateless Kaggle model requests only."""

import os
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

    def request(self, method, path, payload=None, timeout=180):
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
            if isinstance(detail, list):
                detail = "; ".join(item.get("msg", "Invalid request") for item in detail)
            if response.status_code in {400, 404, 422}:
                raise BackendError(str(detail))
            if response.status_code == 401:
                raise BackendError(
                    "Workspace access needs attention. Please contact your administrator."
                )
            raise BackendError("We couldn’t complete this request. Please try again shortly.")
        return data

    def health(self):
        return self.request("GET", "/health", timeout=15)

    def embed(self, texts):
        return self.request("POST", "/embed", {"texts": texts}, timeout=300)

    def analyze(self, text):
        return self.request("POST", "/analyze", {"text": text}, timeout=300)["analysis"]

    def answer(self, question, passages):
        return self.request(
            "POST", "/answer", {"question": question, "passages": passages}, timeout=300
        )["answer"]
