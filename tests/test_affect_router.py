"""Unit tests for AffectRouter, Euler angle extraction, and pedagogical routing.

Verifies:
- 3D head pose Euler angle extraction across column-major and row-major matrices.
- Attentiveness penalty math and bounds.
- VAD Speech Mask behavior during speech and the <=300ms post-speech tail.
- Exact 4-state pedagogical router transitions and sustain duration requirements.
- Proactive silence intervention conditions, awaiting_answer gating, and 20s cooldown.
"""

import math
from typing import Any
import pytest

from backend.affect_router import AffectRouter, clamp, extract_euler_angles


class TestEulerAngles:
    """Verifies Euler angle extraction for head pose."""

    def test_identity_matrix(self) -> None:
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        yaw, pitch, roll = extract_euler_angles(identity)
        assert pytest.approx(yaw, abs=1e-3) == 0.0
        assert pytest.approx(pitch, abs=1e-3) == 0.0
        assert pytest.approx(roll, abs=1e-3) == 0.0

    def test_pure_yaw_positive_and_negative(self) -> None:
        for target_yaw in [25.0, -25.0]:
            rad = math.radians(target_yaw)
            c, s = math.cos(rad), math.sin(rad)
            # Row-major Y-rotation
            m_row = [
                c, 0.0, s, 0.0,
                0.0, 1.0, 0.0, 0.0,
                -s, 0.0, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            yaw, pitch, roll = extract_euler_angles(m_row)
            assert pytest.approx(yaw, abs=1e-2) == target_yaw
            assert pytest.approx(pitch, abs=1e-2) == 0.0
            assert pytest.approx(roll, abs=1e-2) == 0.0

            # Column-major Y-rotation with non-zero translation
            m_col = [
                c, 0.0, -s, 0.0,
                0.0, 1.0, 0.0, 0.0,
                s, 0.0, c, 0.0,
                12.0, 34.0, 56.0, 1.0,
            ]
            yaw_col, pitch_col, roll_col = extract_euler_angles(m_col)
            assert pytest.approx(yaw_col, abs=1e-2) == target_yaw
            assert pytest.approx(pitch_col, abs=1e-2) == 0.0
            assert pytest.approx(roll_col, abs=1e-2) == 0.0

    def test_pure_pitch_positive_and_negative(self) -> None:
        for target_pitch in [15.0, -15.0]:
            rad = math.radians(target_pitch)
            c, s = math.cos(rad), math.sin(rad)
            # Row-major X-rotation
            # Pitch formula: pitch = asin(-r12). To get +15 deg, r12 must be -sin(rad).
            m_row = [
                1.0, 0.0, 0.0, 0.0,
                0.0, c, -s, 0.0,
                0.0, s, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            yaw, pitch, roll = extract_euler_angles(m_row)
            assert pytest.approx(yaw, abs=1e-2) == 0.0
            assert pytest.approx(pitch, abs=1e-2) == target_pitch
            assert pytest.approx(roll, abs=1e-2) == 0.0

            # Column-major X-rotation with non-zero translation
            m_col = [
                1.0, 0.0, 0.0, 0.0,
                0.0, c, s, 0.0,
                0.0, -s, c, 0.0,
                5.0, 10.0, 15.0, 1.0,
            ]
            yaw_col, pitch_col, roll_col = extract_euler_angles(m_col)
            assert pytest.approx(yaw_col, abs=1e-2) == 0.0
            assert pytest.approx(pitch_col, abs=1e-2) == target_pitch
            assert pytest.approx(roll_col, abs=1e-2) == 0.0

    def test_invalid_matrix_fallback(self) -> None:
        assert extract_euler_angles([]) == (0.0, 0.0, 0.0)
        assert extract_euler_angles([1.0, 2.0]) == (0.0, 0.0, 0.0)
        assert extract_euler_angles(["invalid"] * 16) == (0.0, 0.0, 0.0)  # type: ignore[list-item]


class TestAttentivenessMath:
    """Verifies attentiveness calculation and penalties."""

    def test_ideal_attentiveness_no_penalties(self) -> None:
        router = AffectRouter()
        # |yaw| <= 12, |pitch| <= 10, gazeAway <= 0.35
        payload = router.ingest_blendshapes(
            blendshapes={"eyeLookOutLeft": 0.20, "eyeLookInRight": 0.25},
            is_speaking=False,
            time_since_last_speech_ms=999999.0,
            timestamp=1.0,
        )
        assert router.attentiveness == 1.0
        assert payload["attentiveness"] == 1.0

    def test_head_penalty_exact_calculation(self) -> None:
        router = AffectRouter()
        # yaw = 37 deg: head_penalty = ((37 - 12) / 25) * 0.6 = (25 / 25) * 0.6 = 0.6
        # pitch = 0, gaze = 0 -> rawAttentiveness = 1.0 - 0.6 = 0.4
        router.yaw = 37.0
        router.pitch = 0.0
        router.roll = 0.0
        payload = router.ingest_blendshapes(
            blendshapes={},
            is_speaking=False,
            time_since_last_speech_ms=999999.0,
            timestamp=1.0,
        )
        assert pytest.approx(router.attentiveness, abs=1e-4) == 0.40
        assert payload["attentiveness"] == 0.40

    def test_gaze_penalty_clamping(self) -> None:
        router = AffectRouter()
        # gazeAway = 0.80: gaze_penalty = (0.80 - 0.35) / 0.45 = 1.0
        # rawAttentiveness = 1.0 - (1.0 * 0.8) = 0.20
        router.ingest_blendshapes(
            blendshapes={"eyeLookOutLeft": 0.80},
            is_speaking=False,
            timestamp=1.0,
        )
        assert pytest.approx(router.attentiveness, abs=1e-4) == 0.20


class TestVADSpeechMask:
    """Verifies VAD Speech Mask suppresses lower face tension during speech and <=300ms tail."""

    def test_speech_mask_active_during_speaking(self) -> None:
        router = AffectRouter()
        # High lower-face tension: mouthFrown and mouthPress
        bs: dict[str, float] = {
            "mouthFrownLeft": 0.70,
            "mouthFrownRight": 0.70,
            "mouthPressLeft": 0.60,
            "mouthPressRight": 0.60,
        }

        # Case 1: is_speaking=True
        router.ingest_blendshapes(
            blendshapes=bs,
            is_speaking=True,
            time_since_last_speech_ms=0.0,
            timestamp=1.0,
        )
        assert router.speech_mask_active is True
        # Frustration without lower face is 0.0 because upper face strain and hesitation are 0
        assert router.frustration == 0.0
        assert router.unmasked_frustration > 0.0

    def test_speech_mask_active_during_post_speech_tail(self) -> None:
        router = AffectRouter()
        bs: dict[str, float] = {
            "mouthFrownLeft": 0.70,
            "mouthFrownRight": 0.70,
            "mouthPressLeft": 0.60,
            "mouthPressRight": 0.60,
        }

        # Case 2: is_speaking=False, but time_since_last_speech_ms = 250ms (<= 300ms tail)
        router.ingest_blendshapes(
            blendshapes=bs,
            is_speaking=False,
            time_since_last_speech_ms=250.0,
            timestamp=2.0,
        )
        assert router.speech_mask_active is True
        assert router.frustration == 0.0
        assert router.unmasked_frustration > 0.0

    def test_speech_mask_inactive_after_tail_expires(self) -> None:
        router = AffectRouter()
        bs: dict[str, float] = {
            "mouthFrownLeft": 0.70,
            "mouthFrownRight": 0.70,
            "mouthPressLeft": 0.60,
            "mouthPressRight": 0.60,
        }

        # Case 3: is_speaking=False, time_since_last_speech_ms = 350ms (> 300ms tail)
        router.ingest_blendshapes(
            blendshapes=bs,
            is_speaking=False,
            time_since_last_speech_ms=350.0,
            timestamp=3.0,
        )
        assert router.speech_mask_active is False
        assert router.frustration > 0.0
        assert pytest.approx(router.frustration, abs=1e-4) == router.unmasked_frustration


class TestPedagogicalRouterStates:
    """Verifies 4-state pedagogical state transitions and duration requirements."""

    def test_distracted_reengage_requires_sustained_duration(self) -> None:
        router = AffectRouter(alpha=1.0)  # instantaneous for clean state verification
        t = 10.0

        # Feed low attentiveness (0.30 < 0.45) for 2.0s: duration is 2.0s (<= 2.5s)
        router.ingest_telemetry({"attentiveness": 0.30}, timestamp=t)
        t += 2.0
        router.ingest_telemetry({"attentiveness": 0.30}, timestamp=t)
        assert pytest.approx(router.low_attentiveness_duration, abs=1e-4) == 2.0
        assert router.evaluate_pedagogical_state() != "DISTRACTED_REENGAGE"

        # Advance to 2.6s sustained: triggers DISTRACTED_REENGAGE
        t += 0.6
        payload = router.ingest_telemetry({"attentiveness": 0.30}, timestamp=t)
        assert pytest.approx(router.low_attentiveness_duration, abs=1e-4) == 2.6
        assert router.evaluate_pedagogical_state() == "DISTRACTED_REENGAGE"
        assert payload["pedagogical_state"] == "DISTRACTED_REENGAGE"
        assert payload["tts_speed"] == 1.03
        assert payload["avatar_mode"] == "wave_track_yaw"

    def test_frustrated_scaffold_requires_sustained_duration(self) -> None:
        router = AffectRouter(alpha=1.0)
        t = 20.0

        # Feed high frustration (0.75 >= 0.62) for 1.8s (<= 2.0s) -> should be CONFUSED_HINT
        router.ingest_telemetry({"attentiveness": 1.0, "frustration": 0.75}, timestamp=t)
        t += 1.8
        router.ingest_telemetry({"attentiveness": 1.0, "frustration": 0.75}, timestamp=t)
        assert pytest.approx(router.high_frustration_duration, abs=1e-4) == 1.8
        assert router.evaluate_pedagogical_state() == "CONFUSED_HINT"

        # Advance past 2.0s (total 2.3s) -> triggers FRUSTRATED_SCAFFOLD
        t += 0.5
        payload = router.ingest_telemetry({"attentiveness": 1.0, "frustration": 0.75}, timestamp=t)
        assert pytest.approx(router.high_frustration_duration, abs=1e-4) == 2.3
        assert router.evaluate_pedagogical_state() == "FRUSTRATED_SCAFFOLD"
        assert payload["pedagogical_state"] == "FRUSTRATED_SCAFFOLD"
        assert payload["tts_speed"] == 0.95
        assert payload["avatar_mode"] == "empathetic_nod"

    def test_confused_hint_triggers(self) -> None:
        router = AffectRouter(alpha=1.0)

        # Sub-case A: moderate frustration 0.45 <= frustration < 0.62
        router.reset()
        router.ingest_telemetry({"attentiveness": 1.0, "frustration": 0.52}, timestamp=1.0)
        assert router.evaluate_pedagogical_state() == "CONFUSED_HINT"
        directive = router.get_pedagogical_directive("CONFUSED_HINT")
        assert directive["tts_speed"] == 0.98
        assert directive["avatar_mode"] == "curious_tilt"

        # Sub-case B: hesitation > 5.0s even with low frustration
        router.reset()
        router.ingest_telemetry(
            {"attentiveness": 1.0, "frustration": 0.10, "hesitation_seconds": 6.0},
            timestamp=2.0,
        )
        assert router.evaluate_pedagogical_state() == "CONFUSED_HINT"

    def test_confident_advance_default(self) -> None:
        router = AffectRouter(alpha=1.0)
        router.ingest_telemetry(
            {"attentiveness": 0.95, "frustration": 0.15, "hesitation_seconds": 1.0},
            timestamp=1.0,
        )
        assert router.evaluate_pedagogical_state() == "CONFIDENT_ADVANCE"
        directive = router.get_pedagogical_directive("CONFIDENT_ADVANCE")
        assert directive["tts_speed"] == 1.02
        assert directive["avatar_mode"] == "happy_attentive"


class TestProactiveSilenceIntervention:
    """Verifies proactive intervention trigger conditions, answer gating, and cooldown."""

    def test_proactive_help_trigger_conditions(self) -> None:
        router = AffectRouter()
        t = 100.0

        # Hesitation <= 6.5s -> no trigger
        router.hesitation_seconds = 6.0
        router.frustration = 0.65
        assert router.should_trigger_proactive_help(now_s=t, awaiting_answer=True) is False

        # Frustration <= 0.55 -> no trigger
        router.hesitation_seconds = 8.0
        router.frustration = 0.50
        assert router.should_trigger_proactive_help(now_s=t, awaiting_answer=True) is False

        # Not awaiting answer -> no trigger
        router.hesitation_seconds = 8.0
        router.frustration = 0.65
        assert router.should_trigger_proactive_help(now_s=t, awaiting_answer=False) is False

        # All conditions met: hesitation > 6.5, frustration > 0.55, awaiting_answer=True
        assert router.should_trigger_proactive_help(now_s=t, awaiting_answer=True) is True
        assert router.last_proactive_ts == t

    def test_proactive_cooldown_strictly_enforced(self) -> None:
        router = AffectRouter()
        t = 200.0

        router.hesitation_seconds = 9.0
        router.frustration = 0.70

        # First trigger succeeds
        assert router.should_trigger_proactive_help(now_s=t, awaiting_answer=True) is True

        # Attempt trigger at +5 seconds: blocked by 20-second cooldown
        assert router.should_trigger_proactive_help(now_s=t + 5.0, awaiting_answer=True) is False

        # Attempt trigger at +19.9 seconds: blocked
        assert router.should_trigger_proactive_help(now_s=t + 19.9, awaiting_answer=True) is False

        # Attempt trigger at +21 seconds: allowed
        assert router.should_trigger_proactive_help(now_s=t + 21.0, awaiting_answer=True) is True
        assert router.last_proactive_ts == t + 21.0


class TestFrontendTelemetryIngestion:
    """Verifies camelCase and snake_case telemetry payloads from frontend WebSocket."""

    def test_ingest_camelcase_frontend_telemetry(self) -> None:
        router = AffectRouter(alpha=1.0)
        payload = router.ingest_telemetry({
            "attentiveness": 0.88,
            "frustration": 0.22,
            "unmaskedFrustration": 0.51,
            "hesitationSeconds": 5.4,
            "isSpeaking": True,
            "speechMaskActive": True,
            "yaw": -8.0,
            "pitch": 3.0,
        })

        assert pytest.approx(router.unmasked_frustration, abs=1e-4) == 0.51
        assert pytest.approx(router.hesitation_seconds, abs=1e-4) == 5.4
        assert router.speech_mask_active is True
        assert router.is_speaking is True
        assert router.evaluate_pedagogical_state() == "CONFUSED_HINT"
        assert payload["pedagogical_state"] == "CONFUSED_HINT"


class TestVADSpeechMaskWeightRenormalization:
    """Verifies weight re-normalization: 0.0 FP on lower face only, but >=0.62 on genuine upper face strain during speech."""

    def test_lower_face_alone_produces_zero_false_positive_during_speech(self) -> None:
        router = AffectRouter(alpha=1.0)
        # Intense articulation frowning and pressing, but zero brow down / eye squint / hesitation
        bs = {
            "mouthFrownLeft": 0.85,
            "mouthFrownRight": 0.85,
            "mouthPressLeft": 0.80,
            "mouthPressRight": 0.80,
            "browDownLeft": 0.0,
            "browDownRight": 0.0,
            "eyeSquintLeft": 0.0,
            "eyeSquintRight": 0.0,
        }
        payload = router.ingest_blendshapes(
            blendshapes=bs,
            is_speaking=True,
            time_since_last_speech_ms=0.0,
            hesitation_seconds=0.0,
            timestamp=1.0,
        )
        assert router.speech_mask_active is True
        assert router.frustration == 0.0
        assert payload["frustration"] == 0.0
        assert router.unmasked_frustration >= 0.20

    def test_genuine_upper_face_strain_detected_during_speech(self) -> None:
        router = AffectRouter(alpha=1.0)
        # Real frustration while talking: furrowed brows + squint + hesitation > 3s
        bs = {
            "browDownLeft": 0.70,
            "browDownRight": 0.70,
            "eyeSquintLeft": 0.50,
            "eyeSquintRight": 0.50,
            "mouthFrownLeft": 0.60,
            "mouthFrownRight": 0.60,
        }
        payload = router.ingest_blendshapes(
            blendshapes=bs,
            is_speaking=True,
            time_since_last_speech_ms=0.0,
            hesitation_seconds=6.0,  # (6.0 - 3.0) / 8.0 = 0.375
            timestamp=1.0,
        )
        assert router.speech_mask_active is True
        # Upper face strain: 0.7*1.4 + 0.5*0.6 = 0.98 + 0.30 = 1.0 (clamped)
        # Re-normalized: 0.67 * 1.0 + 0.33 * 0.375 = 0.67 + 0.12375 = 0.79375
        assert router.frustration >= 0.62
        assert payload["frustration"] >= 0.62


class TestRollingAffectLatch:
    """Verifies that 4.0-second rolling affect latch preserves pre-speech frustration across spoken turns."""

    def test_latch_preserves_pre_speech_frustration(self) -> None:
        router = AffectRouter(alpha=1.0)
        t = 100.0

        # Feed sustained frustration before speech for 1.8 seconds (frustration >= 0.58)
        for _ in range(9):
            router.ingest_telemetry({
                "attentiveness": 1.0,
                "frustration": 0.70,
                "hesitation_seconds": 5.0,
            }, timestamp=t)
            t += 0.2

        # Student starts speaking, and lower-face mask / relaxation causes instantaneous frustration to drop
        router.ingest_telemetry({
            "attentiveness": 1.0,
            "frustration": 0.20,
            "is_speaking": True,
            "speech_mask_active": True,
        }, timestamp=t)

        # Before latching, instantaneous frustration is 0.20
        assert router.frustration == 0.20

        # Latch turn affect over the 4-second window
        state = router.latch_turn_affect()
        # The peak p90 frustration and sustained duration are restored
        assert router.frustration >= 0.65
        assert router.high_frustration_duration >= 2.1
        assert state == "FRUSTRATED_SCAFFOLD"

    def test_ingest_latched_telemetry_from_frontend(self) -> None:
        router = AffectRouter(alpha=1.0)
        router.ingest_telemetry({
            "attentiveness": 0.90,
            "frustration": 0.25,
            "latchedFrustration": 0.72,
            "latchedState": "FRUSTRATED_SCAFFOLD",
            "isSpeaking": True,
        })
        state = router.latch_turn_affect()
        assert router.frustration >= 0.70
        assert state == "FRUSTRATED_SCAFFOLD"


class TestMultimodalSignalFusion:
    """Verifies fuse_turn_signals across verbal cues, consecutive misses, and proactive triggers."""

    def test_strong_verbal_frustration_escalates_to_scaffold(self) -> None:
        router = AffectRouter(alpha=1.0)
        # Even if visual frustration is low
        router.frustration = 0.20
        state = router.fuse_turn_signals(
            student_text="I don't get this, it makes no sense at all",
            consecutive_misses=0,
            proactive=False,
        )
        assert router.frustration >= 0.72
        assert router.high_frustration_duration >= 2.5
        assert state == "FRUSTRATED_SCAFFOLD"

    def test_mild_verbal_confusion_routes_to_hint(self) -> None:
        router = AffectRouter(alpha=1.0)
        router.frustration = 0.15
        state = router.fuse_turn_signals(
            student_text="idk what to do next",
            consecutive_misses=0,
            proactive=False,
        )
        assert router.frustration >= 0.50
        assert state == "CONFUSED_HINT"

    def test_consecutive_misses_escalation(self) -> None:
        router = AffectRouter(alpha=1.0)

        # 1 miss -> CONFUSED_HINT
        state_1 = router.fuse_turn_signals(
            student_text="x = 5",
            consecutive_misses=1,
            proactive=False,
        )
        assert router.frustration >= 0.50
        assert state_1 == "CONFUSED_HINT"

        # 2 misses -> FRUSTRATED_SCAFFOLD
        state_2 = router.fuse_turn_signals(
            student_text="x = 10",
            consecutive_misses=2,
            proactive=False,
        )
        assert router.frustration >= 0.68
        assert router.high_frustration_duration >= 2.4
        assert state_2 == "FRUSTRATED_SCAFFOLD"

    def test_proactive_help_high_frustration_upgrades_to_scaffold(self) -> None:
        router = AffectRouter(alpha=1.0)
        router.frustration = 0.65
        state = router.fuse_turn_signals(
            student_text="",
            consecutive_misses=0,
            proactive=True,
        )
        assert router.high_frustration_duration >= 2.1
        assert state == "FRUSTRATED_SCAFFOLD"


