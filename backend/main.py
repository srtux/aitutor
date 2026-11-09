"""AiTutor Local Multimodal Server.

FastAPI application delivering real-time affective tutoring, WebSocket session
orchestration, local Whisper STT, Ollama LLM Socratic streaming, and Kokoro/macOS TTS.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
import csv
import io
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .affect_router import AffectRouter
from . import curriculum
from .llm_client import LocalOllamaClient
from .stt_engine import LocalSTTEngine
from .tts_engine import LocalTTSEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Engine singletons
stt_engine = LocalSTTEngine()
tts_engine = LocalTTSEngine()
ollama_client = LocalOllamaClient()

# Global telemetry ring buffer for CSV export
GLOBAL_TELEMETRY_LOG: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to probe models non-blockingly at startup."""
    asyncio.create_task(ollama_client.probe_models())
    yield


app = FastAPI(title="AiTutor Local Multimodal Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Return health status, device platforms, and local engine readiness."""
    face_landmarker_file = FRONTEND_DIR / "assets" / "face_landmarker.task"
    return {
        "status": "ok",
        "offline_edge_mode": True,
        "platform": {
            "system": sys.platform,
            "machine": platform.machine(),
            "apple_silicon": bool(sys.platform == "darwin" and platform.machine() == "arm64"),
        },
        "ollama": {
            "connected": ollama_client.connected,
            "base_url": ollama_client.base_url,
            "default_model": ollama_client.default_model,
            "available_models": ollama_client.available_models,
            "supported_models": ollama_client.supported_models,
        },
        "stt_engine": stt_engine.get_engine_name(),
        "tts_engine": tts_engine.get_engine_name(),
        "voices": tts_engine.list_voices(),
        "default_voice": "warm_mentor",
        "face_landmarker_present": face_landmarker_file.is_file(),
    }


@app.get("/api/topics")
async def get_topics() -> list[dict[str, Any]]:
    """Return list of available curriculum topics."""
    return curriculum.list_topics()


@app.get("/api/export_csv")
async def export_telemetry_csv(session_id: str | None = None) -> Response:
    """Export recorded telemetry and turn latency rows as downloadable CSV."""
    rows = GLOBAL_TELEMETRY_LOG
    if session_id:
        rows = [r for r in rows if r.get("session_id") == session_id]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "timestamp",
            "session_id",
            "topic_id",
            "step_idx",
            "attentiveness",
            "frustration",
            "unmasked_frustration",
            "yaw",
            "pitch",
            "roll",
            "hesitation_seconds",
            "speech_mask_active",
            "pedagogical_state",
            "tts_speed",
            "stt_ms",
            "llm_ttft_ms",
            "llm_total_ms",
            "tts_ms",
            "total_turn_ms",
            "llm_engine",
            "tts_engine",
        ]
    )

    for r in rows:
        writer.writerow(
            [
                r.get("timestamp", ""),
                r.get("session_id", ""),
                r.get("topic_id", ""),
                r.get("step_idx", ""),
                r.get("attentiveness", ""),
                r.get("frustration", ""),
                r.get("unmasked_frustration", ""),
                r.get("yaw", ""),
                r.get("pitch", ""),
                r.get("roll", ""),
                r.get("hesitation_seconds", ""),
                r.get("speech_mask_active", ""),
                r.get("pedagogical_state", ""),
                r.get("tts_speed", ""),
                r.get("stt_ms", ""),
                r.get("llm_ttft_ms", ""),
                r.get("llm_total_ms", ""),
                r.get("tts_ms", ""),
                r.get("total_turn_ms", ""),
                r.get("llm_engine", ""),
                r.get("tts_engine", ""),
            ]
        )

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aitutor_telemetry.csv"},
    )


class SessionState:
    """State manager for an active WebSocket tutoring connection."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.router = AffectRouter()
        self.topic_id = "linear_equations"
        self.step_idx = 0
        self.model = ollama_client.default_model
        self.history: list[dict[str, str]] = []
        self.awaiting_answer = True
        self.telemetry_log: list[dict[str, Any]] = []
        self.voice = "warm_mentor"
        self.consecutive_misses = 0
        self.stream_audio = False


@app.websocket("/ws/session")
async def websocket_session_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint handling live multimodal tutoring interaction."""
    await websocket.accept()
    session = SessionState(session_id=uuid.uuid4().hex[:8])

    async def execute_turn(
        student_text: str,
        stt_ms: float = 0.0,
        proactive: bool = False,
    ) -> None:
        turn_start = time.perf_counter()
        router_start = time.perf_counter()

        pedagogical_state = session.router.fuse_turn_signals(
            student_text=student_text,
            consecutive_misses=session.consecutive_misses,
            proactive=proactive,
        )
        directive = session.router.get_pedagogical_directive(pedagogical_state)
        tts_speed = 0.95 if proactive else directive["tts_speed"]
        avatar_mode = "empathetic_nod" if proactive else directive["avatar_mode"]
        router_ms = (time.perf_counter() - router_start) * 1000.0

        telemetry_snap = {
            "attentiveness": session.router.attentiveness,
            "frustration": session.router.frustration,
            "unmasked_frustration": session.router.unmasked_frustration,
            "yaw": session.router.yaw,
            "pitch": session.router.pitch,
            "roll": session.router.roll,
            "hesitation_seconds": session.router.hesitation_seconds,
            "speech_mask_active": session.router.speech_mask_active,
        }

        # Query LLM (or fallback Socratic generator)
        llm_turn = await ollama_client.generate_tutor_turn(
            topic_id=session.topic_id,
            step_idx=session.step_idx,
            history=session.history,
            student_text=student_text,
            pedagogical_state=pedagogical_state,
            directive=directive,
            telemetry=telemetry_snap,
            model=session.model,
            proactive=proactive,
            attempt_num=session.consecutive_misses,
        )

        spoken_response = llm_turn.get("spoken_response", "")
        whiteboard = llm_turn.get("whiteboard", {})
        advance_step = bool(llm_turn.get("advance_step", False))

        if advance_step:
            session.step_idx += 1
            session.consecutive_misses = 0
            session.router.hesitation_seconds = 0.0
        elif student_text and not proactive:
            session.consecutive_misses += 1

        if student_text:
            session.history.append({"role": "user", "content": student_text})
        if spoken_response:
            session.history.append({"role": "assistant", "content": spoken_response})

        turn_id = uuid.uuid4().hex[:8]

        if session.stream_audio:
            chunks = tts_engine.split_into_speech_chunks(spoken_response)
            if not chunks:
                chunks = [spoken_response or "Let us continue."]

            # Synthesize chunk 0 first for rapid audio start
            tts_res0 = await asyncio.to_thread(
                tts_engine.synthesize_speech, chunks[0], tts_speed, session.voice
            )
            audio_b64_0 = base64.b64encode(tts_res0["wav_bytes"]).decode("ascii")

            total_turn_ms = (time.perf_counter() - turn_start) * 1000.0 + stt_ms
            latency_dict = {
                "stt_ms": round(stt_ms, 2),
                "router_ms": round(router_ms, 2),
                "llm_ttft_ms": round(llm_turn.get("ttft_ms", 0.0), 2),
                "llm_total_ms": round(llm_turn.get("total_ms", 0.0), 2),
                "tts_ms": round(tts_res0.get("latency_ms", 0.0), 2),
                "total_ms": round(total_turn_ms, 2),
            }

            log_row = {
                "timestamp": time.time(),
                "session_id": session.session_id,
                "topic_id": session.topic_id,
                "step_idx": session.step_idx,
                "attentiveness": round(session.router.attentiveness, 4),
                "frustration": round(session.router.frustration, 4),
                "unmasked_frustration": round(session.router.unmasked_frustration, 4),
                "yaw": round(session.router.yaw, 2),
                "pitch": round(session.router.pitch, 2),
                "roll": round(session.router.roll, 2),
                "hesitation_seconds": round(session.router.hesitation_seconds, 2),
                "speech_mask_active": session.router.speech_mask_active,
                "pedagogical_state": pedagogical_state,
                "tts_speed": tts_speed,
                "stt_ms": latency_dict["stt_ms"],
                "llm_ttft_ms": latency_dict["llm_ttft_ms"],
                "llm_total_ms": latency_dict["llm_total_ms"],
                "tts_ms": latency_dict["tts_ms"],
                "total_turn_ms": latency_dict["total_ms"],
                "llm_engine": llm_turn.get("engine", session.model),
                "tts_engine": tts_res0.get("engine", "tts"),
                "voice": session.voice,
            }
            session.telemetry_log.append(log_row)
            GLOBAL_TELEMETRY_LOG.append(log_row)

            await websocket.send_json(
                {
                    "type": "tutor_turn",
                    "turn_id": turn_id,
                    "text": spoken_response,
                    "spoken_response": spoken_response,
                    "pedagogical_state": pedagogical_state,
                    "tts_speed": tts_speed,
                    "avatar_mode": avatar_mode,
                    "whiteboard": whiteboard,
                    "audio_wav_base64": audio_b64_0,
                    "sample_rate": tts_res0.get("sample_rate", 24000),
                    "chunked_audio": len(chunks) > 1,
                    "total_chunks": len(chunks),
                    "chunk_index": 0,
                    "latency": latency_dict,
                    "llm_engine": llm_turn.get("engine", session.model),
                    "tts_engine": tts_res0.get("engine", "tts"),
                    "voice": session.voice,
                    "proactive": proactive,
                }
            )

            for c_idx in range(1, len(chunks)):
                tts_res_c = await asyncio.to_thread(
                    tts_engine.synthesize_speech, chunks[c_idx], tts_speed, session.voice
                )
                audio_b64_c = base64.b64encode(tts_res_c["wav_bytes"]).decode("ascii")
                await websocket.send_json(
                    {
                        "type": "tutor_audio_chunk",
                        "turn_id": turn_id,
                        "chunk_index": c_idx,
                        "total_chunks": len(chunks),
                        "is_final": c_idx == len(chunks) - 1,
                        "audio_wav_base64": audio_b64_c,
                        "sample_rate": tts_res_c.get("sample_rate", 24000),
                        "tts_ms": tts_res_c.get("latency_ms", 0.0),
                    }
                )
        else:
            tts_res = await asyncio.to_thread(
                tts_engine.synthesize_speech, spoken_response, tts_speed, session.voice
            )
            audio_b64 = base64.b64encode(tts_res["wav_bytes"]).decode("ascii")

            total_turn_ms = (time.perf_counter() - turn_start) * 1000.0 + stt_ms
            latency_dict = {
                "stt_ms": round(stt_ms, 2),
                "router_ms": round(router_ms, 2),
                "llm_ttft_ms": round(llm_turn.get("ttft_ms", 0.0), 2),
                "llm_total_ms": round(llm_turn.get("total_ms", 0.0), 2),
                "tts_ms": round(tts_res.get("latency_ms", 0.0), 2),
                "total_ms": round(total_turn_ms, 2),
            }

            log_row = {
                "timestamp": time.time(),
                "session_id": session.session_id,
                "topic_id": session.topic_id,
                "step_idx": session.step_idx,
                "attentiveness": round(session.router.attentiveness, 4),
                "frustration": round(session.router.frustration, 4),
                "unmasked_frustration": round(session.router.unmasked_frustration, 4),
                "yaw": round(session.router.yaw, 2),
                "pitch": round(session.router.pitch, 2),
                "roll": round(session.router.roll, 2),
                "hesitation_seconds": round(session.router.hesitation_seconds, 2),
                "speech_mask_active": session.router.speech_mask_active,
                "pedagogical_state": pedagogical_state,
                "tts_speed": tts_speed,
                "stt_ms": latency_dict["stt_ms"],
                "llm_ttft_ms": latency_dict["llm_ttft_ms"],
                "llm_total_ms": latency_dict["llm_total_ms"],
                "tts_ms": latency_dict["tts_ms"],
                "total_turn_ms": latency_dict["total_ms"],
                "llm_engine": llm_turn.get("engine", session.model),
                "tts_engine": tts_res.get("engine", "tts"),
                "voice": session.voice,
            }
            session.telemetry_log.append(log_row)
            GLOBAL_TELEMETRY_LOG.append(log_row)

            await websocket.send_json(
                {
                    "type": "tutor_turn",
                    "turn_id": turn_id,
                    "text": spoken_response,
                    "spoken_response": spoken_response,
                    "pedagogical_state": pedagogical_state,
                    "tts_speed": tts_speed,
                    "avatar_mode": avatar_mode,
                    "whiteboard": whiteboard,
                    "audio_wav_base64": audio_b64,
                    "sample_rate": tts_res.get("sample_rate", 24000),
                    "chunked_audio": False,
                    "total_chunks": 1,
                    "chunk_index": 0,
                    "latency": latency_dict,
                    "llm_engine": llm_turn.get("engine", session.model),
                    "tts_engine": tts_res.get("engine", "tts"),
                    "voice": session.voice,
                    "proactive": proactive,
                }
            )

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "session_config":
                if msg.get("reset", True):
                    session.router.reset()
                    session.history.clear()
                    session.telemetry_log.clear()
                    session.consecutive_misses = 0

                session.topic_id = msg.get("topic_id", session.topic_id)
                session.model = msg.get("model", session.model)
                session.step_idx = int(msg.get("step_idx", 0))
                session.voice = msg.get("voice", session.voice)
                session.stream_audio = bool(msg.get("stream_audio", session.stream_audio))

                await websocket.send_json(
                    {
                        "type": "session_ready",
                        "session_id": session.session_id,
                        "topic_id": session.topic_id,
                        "step_idx": session.step_idx,
                        "model": session.model,
                        "voice": session.voice,
                        "stream_audio": session.stream_audio,
                    }
                )

                # Send initial greeting and whiteboard state
                if msg.get("greet", True):
                    greeting = curriculum.get_opening_greeting(session.topic_id)
                    whiteboard = curriculum.get_initial_whiteboard(
                        session.topic_id, session.step_idx
                    )
                    tts_res = await asyncio.to_thread(
                        tts_engine.synthesize_speech, greeting, 1.0, session.voice
                    )
                    audio_b64 = base64.b64encode(tts_res["wav_bytes"]).decode("ascii")

                    await websocket.send_json(
                        {
                            "type": "tutor_turn",
                            "turn_id": uuid.uuid4().hex[:8],
                            "text": greeting,
                            "spoken_response": greeting,
                            "pedagogical_state": "CONFIDENT_ADVANCE",
                            "tts_speed": 1.02,
                            "avatar_mode": "happy_attentive",
                            "whiteboard": whiteboard,
                            "audio_wav_base64": audio_b64,
                            "sample_rate": tts_res.get("sample_rate", 24000),
                            "chunked_audio": False,
                            "total_chunks": 1,
                            "chunk_index": 0,
                            "latency": {
                                "stt_ms": 0.0,
                                "router_ms": 0.0,
                                "llm_ttft_ms": 0.0,
                                "llm_total_ms": 0.0,
                                "tts_ms": tts_res.get("latency_ms", 0.0),
                                "total_ms": tts_res.get("latency_ms", 0.0),
                            },
                            "llm_engine": "greeting",
                            "tts_engine": tts_res.get("engine", "tts"),
                            "voice": session.voice,
                            "proactive": False,
                        }
                    )

            elif msg_type == "telemetry":
                telemetry_payload = msg.get("telemetry", msg)
                ack = session.router.ingest_telemetry(telemetry_payload)
                await websocket.send_json(
                    {
                        "type": "affect_ack",
                        **ack,
                    }
                )

            elif msg_type == "student_audio":
                if "stream_audio" in msg:
                    session.stream_audio = bool(msg["stream_audio"])
                if "voice" in msg:
                    session.voice = str(msg["voice"])
                if isinstance(msg.get("telemetry"), dict):
                    session.router.ingest_telemetry(msg["telemetry"])

                audio_b64 = msg.get("wav_base64") or msg.get("audio_wav_base64", "")
                wav_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

                stt_res = await asyncio.to_thread(
                    stt_engine.transcribe_wav_bytes, wav_bytes
                )
                student_text = stt_res.get("text", "")
                stt_ms = stt_res.get("latency_ms", 0.0)

                await websocket.send_json(
                    {
                        "type": "transcript_student",
                        "text": student_text,
                        "stt_ms": stt_ms,
                        "rms": stt_res.get("rms", 0.0),
                        "engine": stt_res.get("engine", "stt"),
                        "pedagogical_state": session.router.evaluate_pedagogical_state(),
                        "attentiveness": round(session.router.attentiveness, 4),
                        "frustration": round(session.router.frustration, 4),
                        "unmasked_frustration": round(session.router.unmasked_frustration, 4),
                    }
                )

                if student_text:
                    await execute_turn(
                        student_text=student_text,
                        stt_ms=stt_ms,
                        proactive=False,
                    )

            elif msg_type == "student_text":
                if "stream_audio" in msg:
                    session.stream_audio = bool(msg["stream_audio"])
                if "voice" in msg:
                    session.voice = str(msg["voice"])
                if isinstance(msg.get("telemetry"), dict):
                    session.router.ingest_telemetry(msg["telemetry"])

                student_text = str(msg.get("text", "")).strip()
                if student_text:
                    await execute_turn(
                        student_text=student_text,
                        stt_ms=0.0,
                        proactive=False,
                    )

            elif msg_type == "proactive_help_trigger":
                if "stream_audio" in msg:
                    session.stream_audio = bool(msg["stream_audio"])
                if "voice" in msg:
                    session.voice = str(msg["voice"])
                if isinstance(msg.get("telemetry"), dict):
                    session.router.ingest_telemetry(msg["telemetry"])

                now_s = time.time()
                if session.router.last_proactive_ts == 0.0 or (
                    now_s - session.router.last_proactive_ts >= 20.0
                ):
                    session.router.last_proactive_ts = now_s
                    await execute_turn(student_text="", proactive=True)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exc()
        pass


# Static file serving: index.html at root and assets from FRONTEND_DIR
@app.get("/", response_class=FileResponse)
async def serve_index() -> Response:
    """Serve the single-page application frontend index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return Response(
        content="<html><body><h3>AiTutor Server Active</h3><p>Frontend assets loading...</p></body></html>",
        media_type="text/html",
    )


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
