"""Integration and hygiene tests for AiTutor API and pipeline.

Verifies:
- HTTP endpoints: GET /, GET /api/health, GET /api/topics, GET /api/export_csv
- Static web assets: /style.css, /app.js, /avatar.js, /affect_engine.js
- WebSocket protocol: session_config, telemetry -> affect_ack, student_text -> tutor_turn,
  and proactive_help_trigger -> tutor_turn with proactive=True
- Repository hygiene: zero hardcoded user home paths, zero TODO placeholders,
  zero emojis across backend and frontend, and zero border-left in style.css.
"""

from pathlib import Path
import re
from typing import Any
from fastapi.testclient import TestClient
import pytest

from backend.curriculum import generate_offline_socratic_response
from backend.main import app
from backend.tts_engine import (
    LocalTTSEngine,
    list_voices,
    normalize_text_for_speech,
    split_into_speech_chunks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHttpAndStaticEndpoints:
    """Verifies all REST and static file delivery endpoints."""

    def test_root_index(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "AiTutor" in response.text

    def test_api_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert data.get("offline_edge_mode") is True
        assert "platform" in data
        assert "stt_engine" in data
        assert "tts_engine" in data
        assert "ollama" in data
        assert "voices" in data
        assert data.get("default_voice") == "warm_mentor"
        assert len(data.get("voices", [])) >= 4

    def test_api_topics(self, client: TestClient) -> None:
        response = client.get("/api/topics")
        assert response.status_code == 200
        topics = response.json()
        assert isinstance(topics, list)
        assert len(topics) >= 3
        topic_ids = [t.get("id") for t in topics]
        assert "linear_equations" in topic_ids
        assert "python_loops" in topic_ids
        assert any(t in topic_ids for t in ["fraction_word_problems", "fractions_word_problems"])

    def test_api_export_csv(self, client: TestClient) -> None:
        response = client.get("/api/export_csv")
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")
        lines = response.text.strip().split("\n")
        assert len(lines) >= 1
        header = lines[0].split(",")
        assert "timestamp" in header
        assert "attentiveness" in header
        assert "frustration" in header
        assert "pedagogical_state" in header

    def test_static_assets_delivered(self, client: TestClient) -> None:
        for route in ["/style.css", "/app.js", "/avatar.js", "/affect_engine.js"]:
            response = client.get(route)
            assert response.status_code == 200, f"Route {route} returned {response.status_code}"
            assert len(response.content) > 0, f"Route {route} was empty"


class TestWebSocketSessionProtocol:
    """Verifies the bidirectional WebSocket tutoring protocol."""

    def test_session_lifecycle_and_telemetry(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/session") as ws:
            # 1. session_config handshake without automatic greeting
            ws.send_json({
                "type": "session_config",
                "topic_id": "linear_equations",
                "step_idx": 0,
                "reset": True,
                "greet": False,
            })
            ready_msg = ws.receive_json()
            assert ready_msg.get("type") == "session_ready"
            assert ready_msg.get("topic_id") == "linear_equations"
            assert ready_msg.get("step_idx") == 0

            # 2. Ingest telemetry -> affect_ack
            ws.send_json({
                "type": "telemetry",
                "telemetry": {
                    "attentiveness": 0.95,
                    "frustration": 0.05,
                    "yaw": 2.0,
                    "pitch": 1.0,
                    "roll": 0.0,
                    "hesitation_seconds": 0.0,
                    "is_speaking": False,
                },
            })
            ack = ws.receive_json()
            assert ack.get("type") == "affect_ack"
            assert ack.get("pedagogical_state") == "CONFIDENT_ADVANCE"
            assert "tts_speed" in ack
            assert "avatar_mode" in ack

            # 3. student_text -> tutor_turn
            ws.send_json({
                "type": "student_text",
                "text": "subtract 7 from both sides",
            })
            turn = ws.receive_json()
            assert turn.get("type") == "tutor_turn"
            assert turn.get("proactive") is False
            assert "spoken_response" in turn
            assert "audio_wav_base64" in turn
            assert len(turn["audio_wav_base64"]) > 0
            assert "whiteboard" in turn
            assert isinstance(turn["whiteboard"], dict)

    def test_proactive_help_websocket_trigger(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "type": "session_config",
                "topic_id": "linear_equations",
                "step_idx": 0,
                "reset": True,
                "greet": False,
            })
            _ = ws.receive_json()

            ws.send_json({"type": "proactive_help_trigger"})
            turn = ws.receive_json()
            assert turn.get("type") == "tutor_turn"
            assert turn.get("proactive") is True
            assert len(turn.get("audio_wav_base64", "")) > 0


class TestSpeechProcessingAndTTS:
    """Verifies text normalization for speech, sentence chunking, and voice listing."""

    def test_normalize_text_for_speech_fractions(self) -> None:
        assert "three fourths" in normalize_text_for_speech("Take 3/4 of the pizza.")
        assert "two thirds" in normalize_text_for_speech("Add 2/3 cup.")
        assert "one half" in normalize_text_for_speech("Half is 1/2.")

    def test_normalize_text_for_speech_equations_and_code(self) -> None:
        normalized = normalize_text_for_speech("Solve 3x + 7 = 19 where i <= 5 and range(1, 5).")
        assert "3 x" in normalized
        assert "less than or equal to" in normalized
        assert "range 1 to 5" in normalized

        norm_code = normalize_text_for_speech("Use len(items) and array[-1].")
        assert "length of items" in norm_code
        assert "array at negative 1" in norm_code

    def test_normalize_text_markdown_stripping(self) -> None:
        text = "Here is **bold** text and `inline_code` with *italic*."
        normalized = normalize_text_for_speech(text)
        assert "**" not in normalized
        assert "`" not in normalized
        assert "*" not in normalized

    def test_split_into_speech_chunks(self) -> None:
        text = "Hello there! Let us solve this problem together. What is your first step? Take your time."
        chunks = split_into_speech_chunks(text)
        assert len(chunks) >= 2
        assert len(chunks) <= 3
        # Ensure chunks don't have dangling punctuation-only fragments
        for c in chunks:
            assert len(c.strip()) >= 5

    def test_list_voices_presets(self) -> None:
        voices = list_voices()
        assert len(voices) >= 4
        voice_ids = [v["id"] for v in voices]
        assert "warm_mentor" in voice_ids
        assert "bright_encourager" in voice_ids
        assert "calm_british" in voice_ids
        assert "friendly_coach" in voice_ids


class TestCurriculumSocraticRotation:
    """Verifies rotating non-repeating Socratic responses across attempts."""

    def test_non_repeating_rotating_responses_per_state(self) -> None:
        states = ["DISTRACTED_REENGAGE", "FRUSTRATED_SCAFFOLD", "CONFUSED_HINT"]
        for st in states:
            resp_0 = generate_offline_socratic_response(
                "linear_equations", 0, "", st, attempt_num=0
            )["spoken_response"]
            resp_1 = generate_offline_socratic_response(
                "linear_equations", 0, "", st, attempt_num=1
            )["spoken_response"]
            resp_2 = generate_offline_socratic_response(
                "linear_equations", 0, "", st, attempt_num=2
            )["spoken_response"]
            assert resp_0 != resp_1
            assert resp_1 != resp_2
            assert resp_0 != resp_2

    def test_high_attempt_scaffolding_in_linear_equations(self) -> None:
        # attempt_num >= 2 should give direct scaffolding step
        res = generate_offline_socratic_response(
            "linear_equations", 0, "", "FRUSTRATED_SCAFFOLD", attempt_num=2
        )
        assert "subtract 7" in res["spoken_response"].lower()


class TestStreamingAudioWebSocket:
    """Verifies WebSocket sentence-chunked streaming audio."""

    def test_websocket_stream_audio_delivery(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/session") as ws:
            ws.send_json({
                "type": "session_config",
                "topic_id": "linear_equations",
                "step_idx": 0,
                "reset": True,
                "greet": False,
                "stream_audio": True,
                "voice": "warm_mentor",
            })
            ready = ws.receive_json()
            assert ready.get("type") == "session_ready"

            # Send student text
            ws.send_json({
                "type": "student_text",
                "text": "what should I do?",
                "stream_audio": True,
            })
            first_msg = ws.receive_json()
            assert first_msg.get("type") == "tutor_turn"
            assert "audio_wav_base64" in first_msg
            total_chunks = first_msg.get("total_chunks", 1)
            assert total_chunks >= 1
            assert first_msg.get("chunk_index") == 0

            # If there are multiple chunks, receive subsequent tutor_audio_chunk messages
            for idx in range(1, total_chunks):
                chunk_msg = ws.receive_json()
                assert chunk_msg.get("type") == "tutor_audio_chunk"
                assert chunk_msg.get("chunk_index") == idx
                assert chunk_msg.get("total_chunks") == total_chunks
                assert len(chunk_msg.get("audio_wav_base64", "")) > 0


class TestRepositoryHygiene:
    """Verifies repository cleanliness, hygiene, and strict formatting invariants."""

    def test_zero_hardcoded_user_home_paths(self) -> None:
        # Dynamically construct forbidden regex to avoid embedding literal home paths
        prefix_part = r"/(?:usr/local/" + r"google/home|Users|home)/"
        forbidden_regex = re.compile(prefix_part + r"[A-Za-z0-9_.-]+")

        violations: list[str] = []
        for subdir in ["backend", "frontend", "scripts", "tests"]:
            target_dir = REPO_ROOT / subdir
            if not target_dir.is_dir():
                continue
            for file_path in target_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                # Skip third-party vendor assets (e.g. emscripten vendor builds)
                if "vendor" in file_path.parts:
                    continue
                if file_path.suffix in [".py", ".js", ".html", ".css", ".md", ".sh"]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    matches = forbidden_regex.findall(content)
                    if matches:
                        violations.append(f"{file_path.relative_to(REPO_ROOT)}: {matches}")

        assert not violations, f"Hardcoded home paths found: {violations}"

    def test_zero_todo_placeholders(self) -> None:
        todo_regex = re.compile(r"\bTODO\b|\bFIXME\b|\bXXX\b")
        violations: list[str] = []

        for subdir in ["backend", "frontend"]:
            target_dir = REPO_ROOT / subdir
            if not target_dir.is_dir():
                continue
            for file_path in target_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix in [".py", ".js", ".html", ".css"]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if todo_regex.search(content):
                        violations.append(str(file_path.relative_to(REPO_ROOT)))

        assert not violations, f"TODO/FIXME placeholders found in: {violations}"

    def test_zero_emojis(self) -> None:
        # Match common emoji ranges in unicode
        emoji_regex = re.compile(
            "[\U00010000-\U0010ffff]|"
            "[\u2600-\u26ff]|"
            "[\u2700-\u27bf]"
        )
        violations: list[str] = []

        for subdir in ["backend", "frontend"]:
            target_dir = REPO_ROOT / subdir
            if not target_dir.is_dir():
                continue
            for file_path in target_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix in [".py", ".js", ".html", ".css"]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    matches = emoji_regex.findall(content)
                    if matches:
                        violations.append(f"{file_path.relative_to(REPO_ROOT)}: {len(matches)} emojis")

        assert not violations, f"Emojis found in source code: {violations}"

    def test_zero_border_left_in_style_css(self) -> None:
        style_file = REPO_ROOT / "frontend" / "style.css"
        assert style_file.is_file(), "frontend/style.css must exist"
        content = style_file.read_text(encoding="utf-8")
        assert "border-left" not in content, "Found forbidden border-left accent stripe in frontend/style.css"
