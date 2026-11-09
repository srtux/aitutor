"""Local Ollama client with streaming token metrics and Socratic fallback.

Connects to Ollama over HTTP, measures Time-To-First-Token (TTFT), extracts
structured JSON pedagogic responses, and falls back to deterministic curriculum
generation when Ollama is offline.
"""

import json
import os
import time
from typing import Any

from . import curriculum


class LocalOllamaClient:
    """Async Ollama client providing low-latency Socratic turns and model probing."""

    DEFAULT_FALLBACK_MODELS = ["qwen2.5:7b-instruct", "llama3.2:3b"]

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.connected = False
        self.default_model = "qwen2.5:7b-instruct"
        self.available_models: list[str] = []
        self.supported_models: list[str] = list(self.DEFAULT_FALLBACK_MODELS)

    async def probe_models(self) -> dict[str, Any]:
        """Probe local Ollama daemon for installed models and connectivity."""
        try:
            import httpx  # type: ignore

            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    raw_models = [
                        m.get("name")
                        for m in data.get("models", [])
                        if isinstance(m, dict) and m.get("name")
                    ]
                    self.available_models = [m for m in raw_models if m]
                    self.connected = True

                    # Prioritize models: qwen2.5:7b-instruct, then llama3.2:3b, then patterns
                    selected_model = None
                    for candidate in ["qwen2.5:7b-instruct", "llama3.2:3b"]:
                        if any(m.startswith(candidate) for m in self.available_models):
                            selected_model = next(
                                m for m in self.available_models if m.startswith(candidate)
                            )
                            break

                    if not selected_model:
                        for m in self.available_models:
                            m_lower = m.lower()
                            if "qwen2.5" in m_lower or "llama3" in m_lower:
                                selected_model = m
                                break

                    if not selected_model and self.available_models:
                        selected_model = self.available_models[0]

                    if selected_model:
                        self.default_model = selected_model

                    # Deduplicated union of supported models
                    combined = list(self.DEFAULT_FALLBACK_MODELS)
                    for m in self.available_models:
                        if m not in combined:
                            combined.append(m)
                    self.supported_models = combined
        except Exception:
            self.connected = False
            self.available_models = []
            self.supported_models = list(self.DEFAULT_FALLBACK_MODELS)

        return {
            "connected": self.connected,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "available_models": self.available_models,
            "supported_models": self.supported_models,
        }

    def _build_system_prompt(
        self,
        topic_id: str,
        step_idx: int,
        pedagogical_state: str,
        directive: dict[str, Any],
        telemetry: dict[str, Any],
        proactive: bool,
    ) -> str:
        topic = curriculum.get_topic(topic_id)
        steps = topic.get("steps", [])
        clamped_step = max(0, min(step_idx, len(steps) - 1)) if steps else 0
        step_data = steps[clamped_step] if steps else {}

        directive_text = directive.get("directive", "")
        att = telemetry.get("attentiveness", 1.0)
        frust = telemetry.get("frustration", 0.0)
        unmasked = telemetry.get("unmasked_frustration", frust)
        yaw = telemetry.get("yaw", 0.0)
        hesitation = telemetry.get("hesitation_seconds", 0.0)

        schema_spec = (
            '{\n'
            '  "spoken_response": "1-2 spoken sentences directly addressing the student.",\n'
            '  "whiteboard": {\n'
            '    "title": "Problem title",\n'
            '    "problem": "Current equation or code snippet",\n'
            '    "steps": [{"label": "Step description", "expr": "expression", "status": "completed|current|pending"}],\n'
            '    "socratic_hint": "Concrete targeted hint",\n'
            '    "visual_diagram": "balance_scale|loop_trace|fraction_bars",\n'
            '    "binary_choices": ["Choice A", "Choice B"]\n'
            '  },\n'
            '  "pedagogical_action": "advance_step|socratic_hint|reengage_focus|scaffold_binary_choice|elaborate_reasoning|complete_problem",\n'
            '  "advance_step": false\n'
            '}'
        )

        return (
            "You are AiTutor, a warm, encouraging, human-like Socratic tutor.\n"
            "Speak naturally with friendly conversational contractions (I'll, let's, that's, you're).\n"
            "If the student is frustrated, validate their effort first and keep anxiety low.\n"
            "Keep spoken_response concise (under 30 words, 1-2 sentences) so voice synthesis begins immediately.\n"
            "Write out mathematical symbols in words inside spoken_response (e.g. 'three fourths' instead of '3/4', 'equals' instead of '=').\n"
            "Keep symbolic math clean inside whiteboard.\n"
            "Never give away final answers directly; guide the student to discover the move.\n"
            f"Active Topic: {topic.get('label', topic_id)}\n"
            f"Active Problem: {step_data.get('problem', '')}\n"
            f"Step Index: {clamped_step}\n"
            f"Current Visual Diagram: {step_data.get('visual_diagram', 'balance_scale')}\n\n"
            "STUDENT AFFECTIVE TELEMETRY:\n"
            f"- Pedagogical State: {pedagogical_state}\n"
            f"- Attentiveness: {att:.2f}\n"
            f"- Frustration: {frust:.2f} (Unmasked: {unmasked:.2f})\n"
            f"- Head Pose Yaw: {yaw:.1f} degrees\n"
            f"- Hesitation Duration: {hesitation:.1f} seconds\n"
            f"- Proactive Silence Help Triggered: {proactive}\n\n"
            "PEDAGOGICAL DIRECTIVE:\n"
            f"{directive_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Deliver a natural, friendly Socratic response in spoken_response (1-2 sentences, max 30 words).\n"
            "2. If the student answers correctly, set advance_step to true and celebrate their insight.\n"
            "3. If pedagogical_state is FRUSTRATED_SCAFFOLD, offer a binary A/B choice in binary_choices.\n"
            "4. Return strict JSON ONLY matching this schema:\n"
            f"{schema_spec}\n"
        )

    async def generate_tutor_turn(
        self,
        topic_id: str,
        step_idx: int,
        history: list[dict[str, str]],
        student_text: str,
        pedagogical_state: str,
        directive: dict[str, Any],
        telemetry: dict[str, Any],
        model: str | None = None,
        proactive: bool = False,
        attempt_num: int = 0,
    ) -> dict[str, Any]:
        """Generate multimodal tutor turn with streaming TTFT and fallback safety."""
        start_time = time.perf_counter()
        active_model = model or self.default_model

        system_prompt = self._build_system_prompt(
            topic_id=topic_id,
            step_idx=step_idx,
            pedagogical_state=pedagogical_state,
            directive=directive,
            telemetry=telemetry,
            proactive=proactive,
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:  # Keep recent history concise
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": turn["role"], "content": turn["content"]})

        if student_text:
            messages.append({"role": "user", "content": student_text})
        elif proactive:
            messages.append(
                {
                    "role": "user",
                    "content": "[Student has been silent and appears hesitant on this step.]",
                }
            )

        ttft_ms = 0.0
        first_token_seen = False
        accumulated_chunks: list[str] = []

        try:
            import httpx  # type: ignore

            timeout = httpx.Timeout(connect=2.0, read=15.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": active_model,
                        "messages": messages,
                        "stream": True,
                        "format": "json",
                        "options": {
                            "temperature": 0.35,
                            "num_predict": 320,
                        },
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Ollama returned HTTP status {response.status_code}")

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})
                            content = msg.get("content", "")
                            if content:
                                if not first_token_seen:
                                    ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                    first_token_seen = True
                                accumulated_chunks.append(content)
                        except json.JSONDecodeError:
                            continue

            full_content = "".join(accumulated_chunks).strip()
            total_ms = (time.perf_counter() - start_time) * 1000.0
            if not ttft_ms:
                ttft_ms = total_ms

            # Parse JSON content from model
            parsed_data = self._parse_and_repair_json(
                raw_json=full_content,
                topic_id=topic_id,
                step_idx=step_idx,
            )

            return {
                "spoken_response": parsed_data["spoken_response"],
                "whiteboard": parsed_data["whiteboard"],
                "pedagogical_action": parsed_data["pedagogical_action"],
                "advance_step": parsed_data["advance_step"],
                "engine": active_model,
                "ttft_ms": round(ttft_ms, 2),
                "total_ms": round(total_ms, 2),
            }

        except Exception:
            # Clean fallback to deterministic Socratic generator
            total_ms = (time.perf_counter() - start_time) * 1000.0
            fallback = curriculum.generate_offline_socratic_response(
                topic_id=topic_id,
                step_idx=step_idx,
                student_text=student_text,
                pedagogical_state=pedagogical_state,
                proactive=proactive,
                attempt_num=attempt_num,
            )
            fallback["engine"] = "offline-socratic-fallback"
            fallback["ttft_ms"] = round(total_ms, 2)
            fallback["total_ms"] = round(total_ms, 2)
            return fallback

    def _parse_and_repair_json(
        self,
        raw_json: str,
        topic_id: str,
        step_idx: int,
    ) -> dict[str, Any]:
        """Safely parse model JSON output and fill missing keys from curriculum."""
        default_whiteboard = curriculum.get_initial_whiteboard(topic_id, step_idx)

        # Attempt to isolate JSON block if wrapped in markdown fences
        clean_text = raw_json.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(clean_text)
            if not isinstance(data, dict):
                raise ValueError("Model output did not parse into a dictionary.")
        except Exception:
            return {
                "spoken_response": (
                    "Let's reflect on this step together. What do you think our next move should be?"
                ),
                "whiteboard": default_whiteboard,
                "pedagogical_action": "socratic_hint",
                "advance_step": False,
            }

        spoken_response = str(
            data.get(
                "spoken_response",
                "Let's look at this step carefully. How can we simplify the equation next?",
            )
        ).strip()

        wb = data.get("whiteboard")
        if not isinstance(wb, dict):
            wb = default_whiteboard
        else:
            if "title" not in wb:
                wb["title"] = default_whiteboard.get("title", "")
            if "problem" not in wb:
                wb["problem"] = default_whiteboard.get("problem", "")
            if "steps" not in wb or not isinstance(wb["steps"], list):
                wb["steps"] = default_whiteboard.get("steps", [])
            if "visual_diagram" not in wb:
                wb["visual_diagram"] = default_whiteboard.get("visual_diagram", "balance_scale")
            if "binary_choices" not in wb:
                wb["binary_choices"] = default_whiteboard.get("binary_choices", [])
            if "socratic_hint" not in wb:
                wb["socratic_hint"] = default_whiteboard.get("socratic_hint", "")

        pedagogical_action = str(data.get("pedagogical_action", "socratic_hint"))
        advance_step = bool(data.get("advance_step", False))

        return {
            "spoken_response": spoken_response,
            "whiteboard": wb,
            "pedagogical_action": pedagogical_action,
            "advance_step": advance_step,
        }
