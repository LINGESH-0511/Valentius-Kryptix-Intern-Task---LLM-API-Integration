"""Automated backend tests; Gemini calls are mocked, so no API key is needed."""

import json

import pytest

import app as chatbot


@pytest.fixture()
def test_client(tmp_path, monkeypatch):
    database = tmp_path / "test_chat.sqlite3"
    monkeypatch.setitem(chatbot.app.config, "DATABASE", str(database))
    monkeypatch.setitem(chatbot.app.config, "TESTING", True)
    monkeypatch.setitem(chatbot.app.config, "MAX_HISTORY_MESSAGES", 6)
    chatbot.initialize_database()
    monkeypatch.setattr(chatbot, "client", object())
    with chatbot.app.test_client() as client:
        yield client


def new_chat(client):
    response = client.post("/api/conversations", json={})
    assert response.status_code == 201
    return response.get_json()["conversation_id"]


def test_home_route_returns_200(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"New chat" in response.data
    assert b"Powered by Google Gemini" not in response.data
    assert b"Enter to send" not in response.data


def test_chat_endpoint_returns_model_response(test_client, monkeypatch):
    monkeypatch.setattr(chatbot, "generate_gemini_response", lambda messages: "Mock Gemini response")
    conversation_id = new_chat(test_client)
    response = test_client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "Hello"},
    )
    assert response.status_code == 200
    assert response.get_json()["reply"] == "Mock Gemini response"


def test_same_conversation_keeps_multi_turn_context(test_client, monkeypatch):
    received = []

    def fake_generate(messages):
        received.append(messages)
        return "I have kept the context."

    monkeypatch.setattr(chatbot, "generate_gemini_response", fake_generate)
    conversation_id = new_chat(test_client)
    prompts = [
        "Remember this code name: ORBIT-27.",
        "The project inspects bottle caps.",
        "It uses a camera.",
        "What is the code name?",
    ]
    for prompt in prompts:
        result = test_client.post(
            "/api/chat",
            json={"conversation_id": conversation_id, "message": prompt},
        )
        assert result.status_code == 200

    final_context = received[-1]
    assert any("ORBIT-27" in item["content"] for item in final_context)
    assert len(final_context) == 7  # 6 previous messages + current message


def test_different_conversations_are_isolated(test_client, monkeypatch):
    received = []

    def fake_generate(messages):
        received.append(messages)
        return "Reply"

    monkeypatch.setattr(chatbot, "generate_gemini_response", fake_generate)
    conversation_a = new_chat(test_client)
    conversation_b = new_chat(test_client)

    test_client.post("/api/chat", json={"conversation_id": conversation_a, "message": "Alpha secret"})
    test_client.post("/api/chat", json={"conversation_id": conversation_b, "message": "Beta topic"})
    test_client.post("/api/chat", json={"conversation_id": conversation_b, "message": "Follow up"})

    context_for_b = received[-1]
    assert any("Beta topic" in item["content"] for item in context_for_b)
    assert not any("Alpha secret" in item["content"] for item in context_for_b)


def test_context_is_truncated_to_configured_limit(test_client, monkeypatch):
    monkeypatch.setitem(chatbot.app.config, "MAX_HISTORY_MESSAGES", 4)
    sent_lengths = []

    def fake_generate(messages):
        sent_lengths.append(len(messages))
        return "Reply"

    monkeypatch.setattr(chatbot, "generate_gemini_response", fake_generate)
    conversation_id = new_chat(test_client)
    for index in range(5):
        result = test_client.post(
            "/api/chat",
            json={"conversation_id": conversation_id, "message": f"Message {index + 1}"},
        )
        assert result.status_code == 200

    assert sent_lengths[-1] == 5  # 4 retained messages plus the current user message
    assert result.get_json()["history_messages_sent"] == 5

    saved = test_client.get(f"/api/conversations/{conversation_id}").get_json()
    assert len(saved["messages"]) == 10  # Full visible transcript remains available


def test_new_chat_creates_independent_conversation(test_client):
    first_id = new_chat(test_client)
    second_id = new_chat(test_client)
    assert first_id != second_id
    listing = test_client.get("/api/conversations").get_json()["conversations"]
    assert {item["conversation_id"] for item in listing} == {first_id, second_id}


def test_api_failure_returns_clear_error(test_client, monkeypatch):
    def fail(_messages):
        raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(chatbot, "generate_gemini_response", fail)
    conversation_id = new_chat(test_client)
    response = test_client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "Hello"},
    )
    assert response.status_code == 429
    assert "request limit" in response.get_json()["error"].lower()


def test_actual_env_is_ignored_by_gitignore():
    from pathlib import Path

    gitignore = Path(__file__).with_name(".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()


def test_health_endpoint_reports_context_configuration(test_client):
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["max_history_messages"] == 6
