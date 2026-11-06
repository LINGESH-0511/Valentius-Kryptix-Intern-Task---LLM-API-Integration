"""Live smoke test for the Gemini API key stored in .env."""

import os
import sys

from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    print("FAIL: GEMINI_API_KEY is missing. Add your real key to .env.")
    sys.exit(1)

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents="Reply with exactly: Gemini API connection successful.",
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("The API returned an empty response.")
    print("PASS: Gemini API responded successfully.")
    print(f"Model: {model_name}")
    print(f"Response: {answer}")
except Exception as exc:
    print(f"FAIL: Gemini API test failed: {exc}")
    print("Check the key, model name, internet connection, quota, and billing settings.")
    sys.exit(1)
