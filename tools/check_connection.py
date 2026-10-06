"""Check the configured model endpoint without sending a CV or printing credentials.

Run from the project root: python -m tools.check_connection
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ats.client import APIClient, BackendError, api_url_from_env


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    key = os.getenv("ATS_API_KEY", "").strip()
    if not key:
        print("Set ATS_API_KEY in your local .env to the same private key as Kaggle.")
        return 1
    try:
        url = api_url_from_env()
        if not url:
            print(
                "Set NGROK_DOMAIN in your local .env to your account's assigned/reserved hostname."
            )
            return 1
        client = APIClient(url, key)
        health = client.health()
        if health.get("status") != "ok" or health.get("version") != "4.1.0":
            print("Unexpected API version. Run the updated Kaggle notebook.")
            return 1
    except BackendError as exc:
        print(str(exc))
        print("Check Kaggle Blocks 7–9, domain ownership, and your local .env.")
        return 1
    print("Connected to the Smart ATS model API (version 4.1.0).")
    print("Start Streamlit and upload your CVs to test the model workflows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
