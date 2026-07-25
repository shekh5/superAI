import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from reasoning_chain.context import ContextBundle
from reasoning_chain.documents import DocumentMetadata
from reasoning_chain.schemas import ChainTrace, Plan, PlanStep, VerifyResult

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "tools" in r.json()


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"version": "1.1.0"}


def test_chat_ui():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SuperAI" in r.text


def test_chat_ui_has_message_history_fallback():
    r = client.get("/chat")
    assert "agentic_chat_messages:" in r.text
    assert "loadMessagesFromLocalStorage" in r.text


def test_chat_ui_has_temperature_selector_and_persistence():
    r = client.get("/chat")
    assert 'id="temperature-select"' in r.text
    assert "Precise · 0.0" in r.text
    assert "Reliable · 0.1" in r.text
    assert "agentic_react_temperature" in r.text
    assert "temperature" in r.text


def test_chat_ui_renders_grounding_links_safely():
    r = client.get("/chat")
    assert "function renderMessageText" in r.text
    assert 'rel="noopener noreferrer"' in r.text


def test_chat_ui_has_document_upload_and_session_document_loading():
    r = client.get("/chat")
    assert 'id="document-input"' in r.text
    assert "uploadDocuments" in r.text
    assert "multiple hidden" in r.text
    assert ".docx,.txt,.md,.csv,.tsv,.xlsx,.pptx" in r.text
    assert "/documents`" in r.text


def test_chat_ui_has_evidence_workspace_controls():
    r = client.get("/chat")
    assert 'id="evidence-panel"' in r.text
    assert "openEvidence" in r.text
    assert "verify.evidence" in r.text
    assert 'id="login-overlay"' in r.text
    assert "pollDocument" in r.text


def test_chat_ui_has_modern_responsive_workspace_shell():
    r = client.get("/chat")
    assert 'class="chat-topbar"' in r.text
    assert 'class="welcome-shell"' in r.text
    assert "What can I help" in r.text
    assert "suggestion-grid" in r.text
    assert "toggleMobileSidebar" in r.text
    assert "toggleSidebar" in r.text
    assert "superai_sidebar_collapsed" in r.text
    assert "submitComposer" in r.text
    assert "message-avatar" in r.text
    assert 'class="composer-disclaimer"' in r.text


def test_dashboard_displays_web_search_sources():
    r = client.get("/dashboard")
    assert "call.response_payload.sources" in r.text
    assert "source.url" in r.text


def test_dashboard_displays_prompt_version_telemetry():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "call.prompt_version" in r.text
    assert "context.utilization_percent" in r.text
    assert "context.compression_level" in r.text
    assert "selectedTrace.corrections" in r.text
    assert "selectedTrace.total_corrections" in r.text
    assert "context.document_chunks_included" in r.text
    assert "selectedTrace.verify.claims" in r.text


def test_compose_uses_durable_redis_storage():
    project_root = Path(__file__).resolve().parents[1]
    for compose_name in ("docker-compose.yml", "docker-compose.prod.yml"):
        compose = (project_root / compose_name).read_text(encoding="utf-8")
        assert "redis-server --appendonly yes" in compose
        assert "redis_data:/data" in compose


def test_agent_calculator():
    r = client.post("/agent", json={"tool": "calculator", "argument": "2 + 2"})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] == "4"
    assert body["tool"] == "calculator"


def test_agent_calculator_bad_expression():
    r = client.post("/agent", json={"tool": "calculator", "argument": "import os"})
    assert r.status_code == 400


def test_agent_calculator_rejects_python_syntax():
    r = client.post(
        "/agent",
        json={"tool": "calculator", "argument": "().__class__.__base__"},
    )
    assert r.status_code == 400


def test_agent_get_time():
    r = client.post("/agent", json={"tool": "get_time"})
    assert r.status_code == 200
    assert "T" in r.json()["result"]  # ISO format check


def test_agent_get_weather_known_city():
    r = client.post("/agent", json={"tool": "get_weather", "argument": "Delhi"})
    assert r.status_code == 200
    assert "hazy" in r.json()["result"]


def test_agent_unknown_tool():
    r = client.post("/agent", json={"tool": "not_a_tool"})
    assert r.status_code == 400


def test_chain_plan_route_is_mounted():
    fake_plan = Plan(
        goal="what time is it",
        steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="user asked")],
    )

    with patch("reasoning_chain.router.decompose_goal", return_value=fake_plan):
        r = client.post("/chain/plan", params={"goal": "what time is it"})

    assert r.status_code == 200
    assert r.json()["goal"] == "what time is it"
    assert r.json()["steps"][0]["tool"] == "get_time"


def test_chain_run_route_is_mounted():
    fake_trace = ChainTrace(
        request_id="req-1",
        goal="what time is it",
        plan=Plan(
            goal="what time is it",
            steps=[PlanStep(step_id=1, tool="get_time", tool_input={}, reason="user asked")],
        ),
        results=[],
        verify=VerifyResult(satisfied=True, final_summary="done"),
        repair_rounds=0,
    )

    with patch("reasoning_chain.router.run_chain", return_value=fake_trace):
        r = client.post("/chain/run", params={"goal": "what time is it"})

    assert r.status_code == 200
    assert r.json()["request_id"] == "req-1"


def test_chain_run_forwards_user_temperature():
    fake_trace = ChainTrace(
        request_id="req-temperature",
        goal="test temperature",
        plan=Plan(goal="test temperature", steps=[]),
        results=[],
        verify=VerifyResult(satisfied=True, final_summary="done"),
        repair_rounds=0,
        temperature=0.7,
    )

    with patch("reasoning_chain.router.run_chain", return_value=fake_trace) as mock_run:
        r = client.post(
            "/chain/run",
            params={"goal": "test temperature", "temperature": 0.7},
        )

    assert r.status_code == 200
    assert r.json()["temperature"] == 0.7
    assert mock_run.call_args.kwargs["temperature"] == 0.7


def test_chain_run_rejects_temperature_outside_user_range():
    with patch("reasoning_chain.router.run_chain") as mock_run:
        r = client.post(
            "/chain/run",
            params={"goal": "test temperature", "temperature": 1.5},
        )

    assert r.status_code == 422
    mock_run.assert_not_called()


def test_chain_run_returns_502_for_model_transport_failure():
    with patch("reasoning_chain.router.run_chain", side_effect=RuntimeError("model unavailable")):
        r = client.post("/chain/run", params={"goal": "test failure"})

    assert r.status_code == 502
    assert "model unavailable" in r.json()["detail"]


def test_chain_run_persists_session_messages():
    fake_trace = ChainTrace(
        request_id="req-session",
        goal="what time is it",
        plan=Plan(goal="what time is it", steps=[]),
        results=[],
        verify=VerifyResult(satisfied=True, final_summary="done"),
        repair_rounds=0,
    )
    mock_redis = MagicMock()
    mock_store = MagicMock()
    mock_store.load.return_value = ContextBundle()

    with (
        patch("reasoning_chain.router._redis", mock_redis),
        patch("reasoning_chain.router.RedisContextStore", return_value=mock_store),
        patch("reasoning_chain.router.run_chain", return_value=fake_trace),
    ):
        r = client.post(
            "/chain/run",
            params={"goal": "what time is it", "session_id": "session-1"},
        )

    assert r.status_code == 200
    display_records = mock_store.append_display.call_args.args[1:]
    assert display_records[0]["sender"] == "user"
    assert display_records[0]["timestamp"].endswith("+00:00")
    clean_messages = [call.args[1] for call in mock_store.append.call_args_list]
    assert [message.role for message in clean_messages] == ["user", "model"]
    assert all(not hasattr(message, "trace") for message in clean_messages)


def test_pdf_upload_endpoint_attaches_processed_document_to_session():
    metadata = DocumentMetadata(
        document_id="doc-1",
        session_id="session-1",
        filename="report.pdf",
        page_count=2,
        chunk_count=3,
        extracted_chars=100,
        created_at="2026-07-20T00:00:00+00:00",
    )
    mock_store = MagicMock()
    mock_store.ingest.return_value = metadata

    with patch("reasoning_chain.router._document_store", return_value=mock_store):
        r = client.post(
            "/chain/session/session-1/documents",
            files={"file": ("report.pdf", b"%PDF-test", "application/pdf")},
        )

    assert r.status_code == 201
    assert r.json()["document_id"] == "doc-1"
    assert mock_store.ingest.call_args.args[:2] == ("session-1", "report.pdf")


def test_pdf_upload_rejects_file_over_configured_limit():
    with (
        patch("reasoning_chain.router._document_store", return_value=MagicMock()),
        patch("reasoning_chain.router.document_settings") as settings,
    ):
        settings.max_file_bytes = 4
        r = client.post(
            "/chain/session/session-1/documents",
            files={"file": ("report.pdf", b"%PDF-test", "application/pdf")},
        )

    assert r.status_code == 400
    assert "exceeds" in r.json()["detail"]


def test_chain_traces_route_is_mounted():
    with patch("reasoning_chain.router._redis") as mock_redis:
        mock_redis.lrange.return_value = ["req-1"]
        mock_redis.get.return_value = json.dumps({
            "request_id": "req-1",
            "goal": "test goal",
            "start_time": "2026-07-15T00:00:00Z",
            "total_latency_ms": 150.0,
            "verify": {"satisfied": True, "final_summary": "done"},
            "repair_rounds": 0,
            "results": []
        })

        r = client.get("/chain/traces")
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert r.json()[0]["request_id"] == "req-1"
        assert r.json()[0]["satisfied"] is True
