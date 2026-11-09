"""Affect Router and Pedagogical State Engine for multimodal tutoring.

Computes 3D head pose Euler angles, attentiveness, VAD-masked frustration,
sustain durations, a 4-second rolling affect latch, multimodal lexical and miss fusion,
and maps the student's affective state to four pedagogical directives.
"""

from collections import deque
import math
import time
from typing import Any


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a floating point value between lower and upper bounds."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def extract_euler_angles(matrix_4x4: list[float]) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll in degrees from a 16-element 4x4 transformation matrix.

    Auto-detects OpenGL/MediaPipe column-major vs row-major representation.
    Returns (yaw, pitch, roll) in degrees.
    """
    if not isinstance(matrix_4x4, (list, tuple)) or len(matrix_4x4) != 16:
        return (0.0, 0.0, 0.0)

    try:
        m = [float(x) for x in matrix_4x4]
    except (ValueError, TypeError):
        return (0.0, 0.0, 0.0)

    # OpenGL column-major check: translation sits at indices 12, 13, 14.
    # In row-major, translation sits at indices 3, 7, 11.
    col_major_trans = abs(m[12]) + abs(m[13]) + abs(m[14])
    row_major_trans = abs(m[3]) + abs(m[7]) + abs(m[11])

    if col_major_trans > row_major_trans:
        # Column-major rotation components
        r00 = m[0]
        r10 = m[1]
        r20 = m[2]
        r01 = m[4]
        r11 = m[5]
        r21 = m[6]
        r02 = m[8]
        r12 = m[9]
        r22 = m[10]
    else:
        # Row-major rotation components
        r00 = m[0]
        r01 = m[1]
        r02 = m[2]
        r10 = m[4]
        r11 = m[5]
        r12 = m[6]
        r20 = m[8]
        r21 = m[9]
        r22 = m[10]

    pitch = math.degrees(math.asin(clamp(-r12, -1.0, 1.0)))
    yaw = math.degrees(math.atan2(r02, r22))
    roll = math.degrees(math.atan2(r10, r11))

    return (yaw, pitch, roll)


class AffectRouter:
    """Stateful affective router tracking engagement, frustration, and pedagogical state."""

    PEDAGOGICAL_DIRECTIVES = {
        "DISTRACTED_REENGAGE": {
            "state": "DISTRACTED_REENGAGE",
            "tts_speed": 1.03,
            "avatar_mode": "wave_track_yaw",
            "directive": (
                "The student appears disengaged or looking away. Re-engage them warmly with a "
                "concise, gamified 1-sentence check-in. Ask an immediate, low-friction question "
                "about the current step to regain focus."
            ),
            "description": "Student has shown sustained low attentiveness (>2.5s). Bring their attention back warmly without scolding.",
        },
        "FRUSTRATED_SCAFFOLD": {
            "state": "FRUSTRATED_SCAFFOLD",
            "tts_speed": 0.95,
            "avatar_mode": "empathetic_nod",
            "directive": (
                "The student is experiencing high frustration. Validate that this step is tricky, "
                "speak calmly, provide a grounding real-world analogy or simplified visual hint, "
                "and offer a binary A/B choice. NEVER give the final answer away."
            ),
            "description": "Student exhibits sustained high facial strain / frustration (>2.0s). De-escalate anxiety and scaffold the problem.",
        },
        "CONFUSED_HINT": {
            "state": "CONFUSED_HINT",
            "tts_speed": 0.98,
            "avatar_mode": "curious_tilt",
            "directive": (
                "The student seems puzzled or stuck on this step. Offer a targeted Socratic hint "
                "pointing to the specific variable, operation, or line. Ask a guided question that "
                "leads them to spot the next move themselves."
            ),
            "description": "Student shows moderate frustration or extended hesitation (>5s). Provide a focused Socratic nudge.",
        },
        "CONFIDENT_ADVANCE": {
            "state": "CONFIDENT_ADVANCE",
            "tts_speed": 1.02,
            "avatar_mode": "happy_attentive",
            "directive": (
                "The student is focused and confident. Acknowledge their progress enthusiastically, "
                "ask them to explain their reasoning or propose the next step, and advance the problem "
                "when correct."
            ),
            "description": "Student is attentive and low frustration. Maintain momentum and encourage deeper reasoning.",
        },
    }

    def __init__(self, alpha: float = 0.15, history_window_s: float = 3.0) -> None:
        self.alpha = alpha
        self.history_window_s = history_window_s

        self.attentiveness = 1.0
        self.frustration = 0.0
        self.unmasked_frustration = 0.0

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.is_speaking = False
        self.speech_mask_active = False
        self.hesitation_seconds = 0.0

        self.low_attentiveness_duration = 0.0
        self.high_frustration_duration = 0.0

        self.last_ts: float | None = None
        self.last_proactive_ts: float = 0.0
        self._has_received_sample: bool = False

        # Rolling sample history (ts, attentiveness, frustration)
        self.history: deque[tuple[float, float, float]] = deque()

        # 4.0-second rolling turn window: (now_s, attentiveness, frustration, high_frustration_duration, dt)
        self.turn_window: deque[tuple[float, float, float, float, float]] = deque()
        self.client_latched_frustration: float = 0.0
        self.client_latched_state: str = ""

    def reset(self) -> None:
        """Reset internal filter states and sustain timers."""
        self.attentiveness = 1.0
        self.frustration = 0.0
        self.unmasked_frustration = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.is_speaking = False
        self.speech_mask_active = False
        self.hesitation_seconds = 0.0
        self.low_attentiveness_duration = 0.0
        self.high_frustration_duration = 0.0
        self.last_ts = None
        self.last_proactive_ts = 0.0
        self._has_received_sample = False
        self.history.clear()
        self.turn_window.clear()
        self.client_latched_frustration = 0.0
        self.client_latched_state = ""

    def ingest_blendshapes(
        self,
        blendshapes: dict[str, float],
        matrix_4x4: list[float] | None = None,
        is_speaking: bool = False,
        time_since_last_speech_ms: float = 999999.0,
        hesitation_seconds: float = 0.0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Compute affective state from raw MediaPipe facial blendshapes and head pose."""
        now_s = time.time() if timestamp is None else float(timestamp)

        # 3D Head Pose
        if matrix_4x4:
            self.yaw, self.pitch, self.roll = extract_euler_angles(matrix_4x4)

        # Attentiveness computation
        gaze_away = max(
            blendshapes.get("eyeLookOutLeft", 0.0),
            blendshapes.get("eyeLookOutRight", 0.0),
            blendshapes.get("eyeLookInLeft", 0.0),
            blendshapes.get("eyeLookInRight", 0.0),
            blendshapes.get("eyeLookUpLeft", 0.0),
            blendshapes.get("eyeLookDownLeft", 0.0),
        )

        head_penalty = (
            clamp((abs(self.yaw) - 12.0) / 25.0, 0.0, 1.0) * 0.6
            + clamp((abs(self.pitch) - 10.0) / 20.0, 0.0, 1.0) * 0.4
        )
        gaze_penalty = clamp((gaze_away - 0.35) / 0.45, 0.0, 1.0)
        raw_attentiveness = clamp(1.0 - max(head_penalty, gaze_penalty * 0.8), 0.0, 1.0)

        # VAD-Masked Frustration computation
        brow_furrow = (
            blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)
        ) / 2.0
        eye_squint = (
            blendshapes.get("eyeSquintLeft", 0.0) + blendshapes.get("eyeSquintRight", 0.0)
        ) / 2.0
        brow_inner_up = blendshapes.get("browInnerUp", 0.0)
        nose_sneer = (
            blendshapes.get("noseSneerLeft", 0.0) + blendshapes.get("noseSneerRight", 0.0)
        ) / 2.0

        speech_mask_active = bool(is_speaking or time_since_last_speech_ms <= 300.0)
        self.is_speaking = is_speaking
        self.speech_mask_active = speech_mask_active
        self.hesitation_seconds = max(0.0, float(hesitation_seconds))

        inner_weight = 0.15 if speech_mask_active else 0.55
        upper_face_strain = clamp(
            brow_furrow * 1.4 + eye_squint * 0.6 + nose_sneer * 0.8 + brow_inner_up * inner_weight,
            0.0,
            1.0,
        )

        mouth_frown = (
            blendshapes.get("mouthFrownLeft", 0.0) + blendshapes.get("mouthFrownRight", 0.0)
        ) / 2.0
        mouth_press = (
            blendshapes.get("mouthPressLeft", 0.0) + blendshapes.get("mouthPressRight", 0.0)
        ) / 2.0
        unmasked_lower_face_tension = clamp(mouth_frown * 1.2 + mouth_press * 1.1, 0.0, 1.0)

        lower_face_tension = 0.0 if speech_mask_active else unmasked_lower_face_tension
        hesitation_factor = clamp((self.hesitation_seconds - 3.0) / 8.0, 0.0, 1.0)

        if speech_mask_active:
            raw_frustration = clamp(
                0.67 * upper_face_strain + 0.33 * hesitation_factor, 0.0, 1.0
            )
        else:
            raw_frustration = clamp(
                0.50 * upper_face_strain + 0.25 * lower_face_tension + 0.25 * hesitation_factor,
                0.0,
                1.0,
            )

        raw_unmasked_frustration = clamp(
            0.50 * upper_face_strain
            + 0.25 * unmasked_lower_face_tension
            + 0.25 * hesitation_factor,
            0.0,
            1.0,
        )

        # EMA smoothing
        if not self._has_received_sample:
            self.attentiveness = raw_attentiveness
            self.frustration = raw_frustration
            self.unmasked_frustration = raw_unmasked_frustration
            self._has_received_sample = True
        else:
            self.attentiveness = (
                (1.0 - self.alpha) * self.attentiveness + self.alpha * raw_attentiveness
            )
            self.frustration = (
                (1.0 - self.alpha) * self.frustration + self.alpha * raw_frustration
            )
            self.unmasked_frustration = (
                (1.0 - self.alpha) * self.unmasked_frustration
                + self.alpha * raw_unmasked_frustration
            )

        # Update durations and rolling history windows
        self._update_durations_and_history(now_s)

        pedagogical_state = self.evaluate_pedagogical_state()
        directive = self.get_pedagogical_directive(pedagogical_state)

        return self._format_status_payload(pedagogical_state, directive)

    def ingest_telemetry(
        self,
        telemetry: dict[str, Any],
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Ingest pre-computed or live telemetry from frontend WebSocket."""
        now_s = time.time() if timestamp is None else float(timestamp)

        # If raw blendshapes are passed, route directly to ingest_blendshapes
        if "blendshapes" in telemetry:
            is_speaking_val = bool(
                telemetry.get("is_speaking", telemetry.get("isSpeaking", False))
            )
            time_since_speech = float(
                telemetry.get(
                    "time_since_last_speech_ms",
                    telemetry.get("timeSinceLastSpeechMs", 999999.0),
                )
            )
            hesitation_val = float(
                telemetry.get(
                    "hesitation_seconds", telemetry.get("hesitationSeconds", 0.0)
                )
            )
            return self.ingest_blendshapes(
                blendshapes=telemetry["blendshapes"],
                matrix_4x4=telemetry.get("matrix"),
                is_speaking=is_speaking_val,
                time_since_last_speech_ms=time_since_speech,
                hesitation_seconds=hesitation_val,
                timestamp=now_s,
            )

        # Ingest pre-computed metrics (accepting snake_case or camelCase)
        raw_att = telemetry.get("raw_attentiveness", telemetry.get("rawAttentiveness"))
        if raw_att is not None:
            raw_att_f = float(raw_att)
            if not self._has_received_sample:
                self.attentiveness = raw_att_f
            else:
                self.attentiveness = (
                    (1.0 - self.alpha) * self.attentiveness + self.alpha * raw_att_f
                )
        elif "attentiveness" in telemetry:
            self.attentiveness = float(telemetry["attentiveness"])

        raw_frust = telemetry.get("raw_frustration", telemetry.get("rawFrustration"))
        if raw_frust is not None:
            raw_frust_f = float(raw_frust)
            if not self._has_received_sample:
                self.frustration = raw_frust_f
            else:
                self.frustration = (
                    (1.0 - self.alpha) * self.frustration + self.alpha * raw_frust_f
                )
        elif "frustration" in telemetry:
            self.frustration = float(telemetry["frustration"])

        raw_unmasked = telemetry.get(
            "raw_unmasked_frustration", telemetry.get("rawUnmaskedFrustration")
        )
        if raw_unmasked is not None:
            raw_unmasked_f = float(raw_unmasked)
            if not self._has_received_sample:
                self.unmasked_frustration = raw_unmasked_f
            else:
                self.unmasked_frustration = (
                    (1.0 - self.alpha) * self.unmasked_frustration
                    + self.alpha * raw_unmasked_f
                )
        elif "unmasked_frustration" in telemetry or "unmaskedFrustration" in telemetry:
            self.unmasked_frustration = float(
                telemetry.get("unmasked_frustration", telemetry.get("unmaskedFrustration"))
            )
        else:
            self.unmasked_frustration = self.frustration

        latched_f = telemetry.get("latched_frustration", telemetry.get("latchedFrustration"))
        if latched_f is not None:
            self.client_latched_frustration = float(latched_f)
        latched_s = telemetry.get("latched_state", telemetry.get("latchedState"))
        if latched_s is not None:
            self.client_latched_state = str(latched_s)

        self._has_received_sample = True

        self.yaw = float(telemetry.get("yaw", self.yaw))
        self.pitch = float(telemetry.get("pitch", self.pitch))
        self.roll = float(telemetry.get("roll", self.roll))
        hesitation_val = telemetry.get(
            "hesitation_seconds", telemetry.get("hesitationSeconds", self.hesitation_seconds)
        )
        self.hesitation_seconds = max(0.0, float(hesitation_val))
        self.is_speaking = bool(
            telemetry.get("is_speaking", telemetry.get("isSpeaking", self.is_speaking))
        )
        self.speech_mask_active = bool(
            telemetry.get(
                "speech_mask_active",
                telemetry.get("speechMaskActive", self.speech_mask_active),
            )
        )

        client_low_att = telemetry.get(
            "low_attentiveness_duration", telemetry.get("lowAttentivenessDuration")
        )
        client_high_frust = telemetry.get(
            "high_frustration_duration", telemetry.get("highFrustrationDuration")
        )

        if client_low_att is not None and client_high_frust is not None:
            self.low_attentiveness_duration = float(client_low_att)
            self.high_frustration_duration = float(client_high_frust)
            self._update_durations_and_history(now_s, skip_dt_update=True)
        else:
            self._update_durations_and_history(now_s, skip_dt_update=False)

        pedagogical_state = self.evaluate_pedagogical_state()
        directive = self.get_pedagogical_directive(pedagogical_state)

        return self._format_status_payload(pedagogical_state, directive)

    def _update_durations_and_history(
        self, now_s: float, skip_dt_update: bool = False
    ) -> None:
        """Update continuous condition durations and rolling sample history."""
        dt = 0.0
        if not skip_dt_update:
            dt = max(0.0, now_s - self.last_ts) if self.last_ts is not None else 0.0
            if dt > 10.0:
                # Discard massive gaps from suspended tabs
                dt = 0.0

            if self.attentiveness < 0.45:
                self.low_attentiveness_duration += dt
            else:
                self.low_attentiveness_duration = 0.0

            if self.frustration >= 0.62:
                self.high_frustration_duration += dt
            else:
                self.high_frustration_duration = 0.0

        self.last_ts = now_s

        # Rolling history maintainer (3.0s window)
        self.history.append((now_s, self.attentiveness, self.frustration))
        cutoff = now_s - self.history_window_s
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # Rolling 4.0s turn window: (now_s, attentiveness, frustration, high_frustration_duration, dt)
        self.turn_window.append(
            (now_s, self.attentiveness, self.frustration, self.high_frustration_duration, dt)
        )
        cutoff_turn = now_s - 4.0
        while self.turn_window and self.turn_window[0][0] < cutoff_turn:
            self.turn_window.popleft()

    def latch_turn_affect(self) -> str:
        """Latch peak pre-speech frustration and sustained strain over the last 4 seconds."""
        frust_samples = [s[2] for s in self.turn_window]
        frust_samples.append(self.frustration)
        if self.client_latched_frustration > 0.0:
            frust_samples.append(self.client_latched_frustration)

        if len(frust_samples) == 1:
            peak_p90_frust = frust_samples[0]
        else:
            sorted_frusts = sorted(frust_samples)
            p90_idx = int(math.ceil(0.90 * (len(sorted_frusts) - 1)))
            peak_p90_frust = sorted_frusts[p90_idx]

        dur_samples = [s[3] for s in self.turn_window] + [self.high_frustration_duration]
        max_high_dur = max(dur_samples) if dur_samples else self.high_frustration_duration

        frustrated_time = sum(s[4] for s in self.turn_window if s[2] >= 0.56)

        self.frustration = max(self.frustration, peak_p90_frust)
        self.high_frustration_duration = max(self.high_frustration_duration, max_high_dur)

        # Speaking should not wipe out pre-utterance frustration
        if (frustrated_time >= 1.4 and peak_p90_frust >= 0.58) or self.client_latched_state == "FRUSTRATED_SCAFFOLD":
            self.high_frustration_duration = max(self.high_frustration_duration, 2.1)

        return self.evaluate_pedagogical_state()

    def fuse_turn_signals(
        self,
        student_text: str = "",
        consecutive_misses: int = 0,
        proactive: bool = False,
    ) -> str:
        """Multimodal fusion of affective latch, verbal cues, and consecutive mistakes."""
        # 1. Latch rolling affective window first
        self.latch_turn_affect()

        # 2. Proactive silence with high facial strain routes to scaffolding
        if proactive and self.frustration >= 0.58:
            self.high_frustration_duration = max(self.high_frustration_duration, 2.1)

        text_l = student_text.lower().strip()

        # 3. Strong verbal frustration phrases
        strong_phrases = [
            "don't get",
            "dont get",
            "no sense",
            "makes no sense",
            "so confused",
            "i'm lost",
            "im lost",
            "give up",
            "too hard",
            "frustrated",
            "hate this",
            "ugh",
            "no idea",
            "impossible",
            "stuck",
        ]
        if any(p in text_l for p in strong_phrases):
            self.frustration = max(self.frustration, 0.72)
            self.high_frustration_duration = max(self.high_frustration_duration, 2.5)

        # 4. Mild confusion or uncertainty phrases
        mild_phrases = [
            "i don't know",
            "i dont know",
            "idk",
            "not sure",
            "confused",
            "help",
            "hint",
            "wait what",
            "huh",
        ]
        if any(p in text_l for p in mild_phrases):
            if consecutive_misses >= 1 or self.frustration >= 0.35:
                self.frustration = max(self.frustration, 0.66)
                self.high_frustration_duration = max(self.high_frustration_duration, 2.2)
            else:
                self.frustration = max(self.frustration, 0.52)

        # 5. Consecutive wrong attempts on the same step
        if consecutive_misses == 1:
            self.frustration = max(self.frustration, 0.50)
        elif consecutive_misses >= 2:
            self.frustration = max(self.frustration, 0.68)
            self.high_frustration_duration = max(self.high_frustration_duration, 2.4)

        return self.evaluate_pedagogical_state()

    def evaluate_pedagogical_state(self) -> str:
        """Evaluate pedagogical routing state according to exact precedence rules."""
        # 1. Distracted Re-engage: attentiveness < 0.45 sustained > 2.5s
        if self.low_attentiveness_duration > 2.5:
            return "DISTRACTED_REENGAGE"

        # 2. Frustrated Scaffold: frustration >= 0.62 sustained > 2.0s
        if self.high_frustration_duration > 2.0:
            return "FRUSTRATED_SCAFFOLD"

        # 3. Confused Hint: moderate frustration, short high-frustration spike, or long hesitation
        if (
            (0.45 <= self.frustration < 0.62)
            or (self.frustration >= 0.62 and self.high_frustration_duration <= 2.0)
            or (self.hesitation_seconds > 5.0)
        ):
            return "CONFUSED_HINT"

        # 4. Confident Advance: default attentive state
        return "CONFIDENT_ADVANCE"

    def should_trigger_proactive_help(
        self, now_s: float | None = None, awaiting_answer: bool = True
    ) -> bool:
        """Determine if a proactive tutor intervention should be triggered."""
        current_time = time.time() if now_s is None else float(now_s)

        if not awaiting_answer:
            return False

        if (
            self.hesitation_seconds > 6.5
            and self.frustration > 0.55
            and (current_time - self.last_proactive_ts >= 20.0)
        ):
            self.last_proactive_ts = current_time
            return True

        return False

    @classmethod
    def get_pedagogical_directive(cls, state: str) -> dict[str, Any]:
        """Return the pedagogical directive details for a given state name."""
        return cls.PEDAGOGICAL_DIRECTIVES.get(
            state, cls.PEDAGOGICAL_DIRECTIVES["CONFIDENT_ADVANCE"]
        )

    def _format_status_payload(
        self, pedagogical_state: str, directive: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "pedagogical_state": pedagogical_state,
            "tts_speed": directive["tts_speed"],
            "avatar_mode": directive["avatar_mode"],
            "attentiveness": round(self.attentiveness, 4),
            "frustration": round(self.frustration, 4),
            "unmasked_frustration": round(self.unmasked_frustration, 4),
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "hesitation_seconds": round(self.hesitation_seconds, 2),
            "is_speaking": self.is_speaking,
            "speech_mask_active": self.speech_mask_active,
            "low_attentiveness_duration": round(self.low_attentiveness_duration, 2),
            "high_frustration_duration": round(self.high_frustration_duration, 2),
        }
