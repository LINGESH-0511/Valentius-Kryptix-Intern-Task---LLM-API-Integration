# Gemini Chatbot

A local Flask chatbot using Google's `google-genai` SDK. The interface includes a ChatGPT-style conversation sidebar, persistent chat history, and a compact message composer.

## Features

- Real Gemini API calls using `GEMINI_API_KEY` from `.env`.
- Independent conversation IDs.
- SQLite transcript storage, so previous chats remain visible after a page refresh or Flask restart.
- Context-window management: only the latest `MAX_HISTORY_MESSAGES` previous messages are sent to Gemini on each request (10 by default).
- A defined assistant system prompt.
- User-friendly API, timeout, quota, and server error messages.
- New chat, open previous chat, and delete chat actions.
- Automated backend tests that mock Gemini; no API key is needed for those tests.

## 1. Create the environment

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add your real Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.8-flash
MAX_HISTORY_MESSAGES=10
FLASK_DEBUG=false
```

Do not commit `.env`. It is excluded by `.gitignore`. The `.env.example` file contains only a placeholder.

## 2. Test the real Gemini API key

With `.env` configured, run:

```powershell
python check_gemini_api.py
```

The script prints `PASS` only after Gemini returns a non-empty response. This is the live API-key check; it cannot pass with the placeholder key.

## 3. Run automated backend tests

```powershell
python -m pytest -q
```

These tests use a mocked Gemini response and verify the Flask routes, multi-turn context, conversation isolation, context truncation, new-chat creation, error handling, and Git ignore configuration.

## 4. Start the application

```powershell
python app.py
```

Open <http://127.0.0.1:5000>.

## Context management

The complete transcript is saved in `instance/chat_history.sqlite3` for the sidebar and transcript view. Before each Gemini request, the backend sends only the latest `MAX_HISTORY_MESSAGES` previous messages plus the new user message. The default is 10 previous messages (five complete exchanges). Use an even number such as 6, 10, or 20.

## Project structure

```text
 gemini_chatbot_upgraded/
 ├── app.py
 ├── test_app.py
 ├── check_gemini_api.py
 ├── requirements.txt
 ├── .env.example
 ├── .gitignore
 ├── README.md
 ├── templates/
 │   └── index.html
 ├── static/
 │   ├── css/style.css
 │   └── js/app.js
 └── instance/
     └── chat_history.sqlite3  (created automatically; ignored by Git)
```

## Notes

- Conversation history belongs to this local app and is not shared between users or devices.
- This starter project does not include user authentication. Keep it bound to localhost; do not expose it directly to the public internet.
- The automated tests do not prove that a real API key works. Run `python check_gemini_api.py` for that live check.


## Verification checklist

1. Install dependencies: `pip install -r requirements.txt`
2. Add your real API key to `.env` (do not share or commit it).
3. Run mocked backend tests: `python -m pytest -q`
4. Verify the real Gemini connection: `python check_gemini_api.py`
5. Start the app: `python app.py`, then open `http://127.0.0.1:5000`.

The mocked tests verify the Flask routes and conversation behavior without making live API calls. The live API check is separate and passes only when Gemini returns a non-empty response using your actual key.
