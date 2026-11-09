"""Local Text-to-Speech engine supporting Kokoro-ONNX, macOS say, and prosodic synth.

Generates 24kHz PCM16 audio bytes with curated blended warm voices, spoken math
normalization, sentence chunking, and speed regulation calibrated for affective
pedagogy.
"""

import io
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any
import wave

from .affect_router import clamp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "warm_mentor": {
        "id": "warm_mentor",
        "label": "Aria (Warm Mentor)",
        "blend": [("af_bella", 0.60), ("af_heart", 0.40)],
        "lang": "en-us",
        "macos_voice": "Samantha",
    },
    "bright_encourager": {
        "id": "bright_encourager",
        "label": "Sky (Upbeat)",
        "blend": [("af_sky", 0.65), ("af_bella", 0.35)],
        "lang": "en-us",
        "macos_voice": "Flo",
    },
    "calm_british": {
        "id": "calm_british",
        "label": "Emma (Calm Guide)",
        "blend": [("bf_emma", 0.70), ("af_bella", 0.30)],
        "lang": "en-gb",
        "macos_voice": "Daniel",
    },
    "friendly_coach": {
        "id": "friendly_coach",
        "label": "Puck (Friendly Coach)",
        "blend": [("am_puck", 0.65), ("am_liam", 0.35)],
        "lang": "en-us",
        "macos_voice": "Reed",
    },
}


def list_voices() -> list[dict[str, str]]:
    """Return all available voice presets with id and user-facing label."""
    return [{"id": v["id"], "label": v["label"]} for v in VOICE_PRESETS.values()]


def normalize_text_for_speech(text: str) -> str:
    """Normalize math and code expressions into natural conversational spoken words."""
    if not text:
        return ""

    s = text

    # Strip markdown bold, italics, backticks, and bullet prefixes
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"^[\*\-•]\s+", "", s, flags=re.MULTILINE)

    # Em-dashes and double hyphens become commas for smooth breath pauses
    s = s.replace("—", ", ").replace("--", ", ")

    # Common fractions
    fraction_map = [
        (r"\b3/4\b", "three fourths"),
        (r"\b2/3\b", "two thirds"),
        (r"\b1/3\b", "one third"),
        (r"\b1/2\b", "one half"),
        (r"\b2/5\b", "two fifths"),
        (r"\b6/12\b", "six twelfths"),
        (r"\b5/15\b", "five fifteenths"),
        (r"\b6/15\b", "six fifteenths"),
        (r"\b11/15\b", "eleven fifteenths"),
        (r"\b4/15\b", "four fifteenths"),
        (r"\b15/15\b", "fifteen fifteenths"),
        (r"\b7/8\b", "seven eighths"),
        (r"\b5/6\b", "five sixths"),
        (r"\b21/24\b", "twenty-one twenty-fourths"),
        (r"\b20/24\b", "twenty twenty-fourths"),
        (r"\b1/24\b", "one twenty-fourth"),
    ]
    for pattern, replacement in fraction_map:
        s = re.sub(pattern, replacement, s)

    # Python / code idioms
    s = re.sub(r"range\((\d+)\s*,\s*(\d+)\)", r"range \1 to \2", s)
    s = re.sub(r"len\((\w+)\)", r"length of \1", s)
    s = re.sub(r"\b(\w+)\[-1\]", r"\1 at negative 1", s)
    s = re.sub(r"<=", " less than or equal to ", s)
    s = re.sub(r">=", " greater than or equal to ", s)
    s = re.sub(r"==", " equals ", s)
    s = re.sub(r"\+=", " plus equals ", s)
    s = re.sub(r"\s+\*\s+", " times ", s)
    s = re.sub(r"(?<=\d|\))\s*\*\s*(?=\d|\()", " times ", s)
    s = re.sub(r"\b(\d+)x\b", r"\1 x", s)
    s = re.sub(r"(?<![<>=!])=(?![=])", " equals ", s)

    # Clean whitespace and repeated commas
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*,+", ",", s)
    return s.strip()


def split_into_speech_chunks(text: str) -> list[str]:
    """Split text on sentence boundaries, merging short fragments into <= 3 chunks."""
    raw = text.strip()
    if not raw:
        return []

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if not parts:
        return [raw]
    if len(parts) == 1:
        return parts

    merged: list[str] = []
    idx = 0
    while idx < len(parts):
        curr = parts[idx]
        while len(curr) < 18 and idx + 1 < len(parts):
            idx += 1
            curr = curr + " " + parts[idx]
        merged.append(curr)
        idx += 1

    # If the trailing chunk is very small, merge backward
    if len(merged) > 1 and len(merged[-1]) < 18:
        last = merged.pop()
        merged[-1] = merged[-1] + " " + last

    # Cap at most 3 chunks
    if len(merged) > 3:
        return merged[:2] + [" ".join(merged[2:])]
    return merged


class LocalTTSEngine:
    """Local multi-backend TTS engine with sub-second generation and speed modulation."""

    def __init__(self) -> None:
        self.models_dir = PROJECT_ROOT / "models"
        self.kokoro_model_path = self.models_dir / "kokoro-v1.0.onnx"
        self.voices_path = self.models_dir / "voices-v1.0.bin"

        self._kokoro_instance: Any = None
        self._kokoro_attempted: bool = False
        self._has_say = bool(shutil.which("say") and shutil.which("afconvert"))
        self._style_cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def list_voices(self) -> list[dict[str, str]]:
        """Return list of voice presets with identifiers and user-facing labels."""
        return list_voices()

    def split_into_speech_chunks(self, text: str) -> list[str]:
        """Split text on sentence boundaries, merging short fragments into <= 3 chunks."""
        return split_into_speech_chunks(text)

    def normalize_text_for_speech(self, text: str) -> str:
        """Normalize math, code syntax, and contractions for speech."""
        return normalize_text_for_speech(text)

    def get_engine_name(self) -> str:
        """Return the active or highest-priority available TTS engine name."""
        if self.kokoro_model_path.is_file() and self.voices_path.is_file():
            return "kokoro-onnx (blended)"
        if self._has_say:
            return "macos-say (afconvert-24kHz)"
        return "numpy-prosodic-synth (24kHz)"

    def _get_kokoro(self) -> Any:
        if self._kokoro_attempted:
            return self._kokoro_instance

        self._kokoro_attempted = True
        if self.kokoro_model_path.is_file() and self.voices_path.is_file():
            try:
                import onnxruntime as rt  # type: ignore
                from kokoro_onnx import Kokoro  # type: ignore

                sess_opts = rt.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess_opts.inter_op_num_threads = 1
                sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                available = rt.get_available_providers()
                providers = (
                    ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    if "CoreMLExecutionProvider" in available
                    else ["CPUExecutionProvider"]
                )
                sess = rt.InferenceSession(
                    str(self.kokoro_model_path), sess_options=sess_opts, providers=providers
                )
                self._kokoro_instance = Kokoro.from_session(sess, str(self.voices_path))
            except Exception:
                try:
                    from kokoro_onnx import Kokoro  # type: ignore

                    self._kokoro_instance = Kokoro(
                        model_path=str(self.kokoro_model_path),
                        voices_path=str(self.voices_path),
                    )
                except Exception:
                    self._kokoro_instance = None
        return self._kokoro_instance

    def _get_voice_style_or_name(self, kokoro: Any, voice_id: str) -> tuple[Any, str]:
        preset = VOICE_PRESETS.get(voice_id, VOICE_PRESETS["warm_mentor"])
        lang = preset.get("lang", "en-us")

        if voice_id in self._style_cache:
            return self._style_cache[voice_id], lang

        try:
            import numpy as np  # type: ignore

            if hasattr(kokoro, "get_voice_style"):
                blended = None
                total_weight = 0.0
                for v_name, weight in preset.get("blend", []):
                    try:
                        style = kokoro.get_voice_style(v_name)
                        if style is not None:
                            if blended is None:
                                blended = np.zeros_like(style, dtype=np.float32)
                            blended += float(weight) * style.astype(np.float32)
                            total_weight += float(weight)
                    except Exception:
                        pass

                if blended is not None and total_weight > 0.0:
                    blended /= total_weight
                    self._style_cache[voice_id] = blended
                    return blended, lang
        except Exception:
            pass

        # Fallback to primary single voice string
        blend = preset.get("blend", [("af_heart", 1.0)])
        primary_name = blend[0][0] if blend else "af_heart"
        return primary_name, lang

    def synthesize_speech(
        self,
        text: str,
        speed: float = 1.0,
        voice: str = "warm_mentor",
    ) -> dict[str, Any]:
        """Synthesize text into 24kHz PCM16 WAV bytes using the best available backend.

        Args:
            text: Spoken utterance text.
            speed: Target playback rate between 0.90 and 1.10.
            voice: Voice preset key from VOICE_PRESETS (default "warm_mentor").

        Returns:
            dict containing:
                wav_bytes: Complete WAV file payload as bytes.
                sample_rate: Integer sampling rate (24000).
                engine: String describing the TTS backend used.
                latency_ms: Synthesis time in milliseconds.
                speed: Effective clamped speed applied.
                voice: Voice preset identifier used.
        """
        start_time = time.perf_counter()
        normalized_text = normalize_text_for_speech(text) or "Let us continue."
        effective_speed = clamp(speed, 0.90, 1.10)
        preset = VOICE_PRESETS.get(voice, VOICE_PRESETS["warm_mentor"])

        # 1. Primary Engine: Kokoro ONNX with blended style vectors
        kokoro = self._get_kokoro()
        if kokoro is not None:
            try:
                voice_obj, lang = self._get_voice_style_or_name(kokoro, preset["id"])
                with self._lock:
                    try:
                        samples, sample_rate = kokoro.create(
                            normalized_text,
                            voice=voice_obj,
                            speed=effective_speed,
                            lang=lang,
                            clause_pause=0.12,
                            sentence_pause=0.22,
                        )
                    except TypeError:
                        samples, sample_rate = kokoro.create(
                            normalized_text,
                            voice=voice_obj,
                            speed=effective_speed,
                            lang=lang,
                        )

                wav_bytes = self._audio_array_to_wav_bytes(samples, sample_rate)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "wav_bytes": wav_bytes,
                    "sample_rate": int(sample_rate),
                    "engine": f"kokoro-onnx ({preset['id']})",
                    "latency_ms": round(elapsed_ms, 2),
                    "speed": effective_speed,
                    "voice": preset["id"],
                }
            except Exception:
                pass

        # 2. Secondary Engine: macOS native say + afconvert
        if self._has_say:
            try:
                wpm = int(round(185.0 * effective_speed))
                macos_voice = preset.get("macos_voice", "Samantha")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_aiff = Path(tmp_dir) / "output.aiff"
                    tmp_wav = Path(tmp_dir) / "output.wav"

                    subprocess.run(
                        [
                            "say",
                            "-v",
                            macos_voice,
                            "-r",
                            str(wpm),
                            "-o",
                            str(tmp_aiff),
                            normalized_text,
                        ],
                        check=True,
                        capture_output=True,
                        timeout=8.0,
                    )
                    subprocess.run(
                        [
                            "afconvert",
                            "-f",
                            "WAVE",
                            "-d",
                            "LEI16@24000",
                            str(tmp_aiff),
                            str(tmp_wav),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=8.0,
                    )

                    wav_bytes = tmp_wav.read_bytes()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return {
                        "wav_bytes": wav_bytes,
                        "sample_rate": 24000,
                        "engine": f"macos-say ({macos_voice})",
                        "latency_ms": round(elapsed_ms, 2),
                        "speed": effective_speed,
                        "voice": preset["id"],
                    }
            except Exception:
                pass

        # 3. Tertiary Engine: Prosodic multi-harmonic synthesis (24kHz fallback)
        wav_bytes = self._synthesize_prosodic_waveform(normalized_text, effective_speed)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "wav_bytes": wav_bytes,
            "sample_rate": 24000,
            "engine": "numpy-prosodic-synth",
            "latency_ms": round(elapsed_ms, 2),
            "speed": effective_speed,
            "voice": preset["id"],
        }

    def _audio_array_to_wav_bytes(self, samples: Any, sample_rate: int) -> bytes:
        """Encode audio samples to 16-bit PCM WAV bytes."""
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore

            bio = io.BytesIO()
            sf.write(bio, samples, sample_rate, format="WAV", subtype="PCM_16")
            return bio.getvalue()
        except ImportError:
            # Pure standard library fallback encoding
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm_bytes = bytearray()
                for s in samples:
                    val = max(-1.0, min(1.0, float(s)))
                    pcm_bytes.extend(struct.pack("<h", int(round(val * 32767.0))))
                wf.writeframes(pcm_bytes)
            return bio.getvalue()

    def _synthesize_prosodic_waveform(self, text: str, speed: float) -> bytes:
        """Synthesize pleasant multi-harmonic syllables with audio envelope at 24kHz."""
        sample_rate = 24000
        words = [w for w in text.split() if w]
        num_words = max(1, len(words))

        # Syllable duration modulated by speed (approx 160ms per syllable + 40ms pause)
        syllable_len = int(round((0.16 / speed) * sample_rate))
        pause_len = int(round((0.04 / speed) * sample_rate))

        total_frames = num_words * (syllable_len + pause_len)
        f0_base = 210.0  # Pleasant vocal pitch (Hz)

        try:
            import numpy as np  # type: ignore

            audio = np.zeros(total_frames, dtype=np.float32)
            for idx in range(num_words):
                start = idx * (syllable_len + pause_len)
                end = start + syllable_len

                # Hann window envelope for natural speech rise and fall
                t = np.linspace(0.0, 1.0, syllable_len, endpoint=False)
                envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))

                # Pitch contour variation across sentence
                progress = float(idx) / float(num_words)
                f0 = f0_base * (1.05 - 0.15 * progress)

                time_axis = np.arange(syllable_len) / sample_rate
                # Harmonic overtone series for rich vocal timbre
                signal = (
                    0.50 * np.sin(2.0 * np.pi * f0 * time_axis)
                    + 0.25 * np.sin(4.0 * np.pi * f0 * time_axis)
                    + 0.15 * np.sin(6.0 * np.pi * f0 * time_axis)
                    + 0.10 * np.sin(8.0 * np.pi * f0 * time_axis)
                )

                audio[start:end] = signal * envelope * 0.35

            return self._audio_array_to_wav_bytes(audio, sample_rate)

        except ImportError:
            # Standard library pure math synthesis if numpy is not yet imported
            bio = io.BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm_bytes = bytearray()

                for idx in range(num_words):
                    progress = float(idx) / float(num_words)
                    f0 = f0_base * (1.05 - 0.15 * progress)

                    for s_idx in range(syllable_len):
                        t = float(s_idx) / float(syllable_len)
                        envelope = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
                        time_sec = float(s_idx) / float(sample_rate)

                        sample = (
                            0.50 * math.sin(2.0 * math.pi * f0 * time_sec)
                            + 0.25 * math.sin(4.0 * math.pi * f0 * time_sec)
                            + 0.15 * math.sin(6.0 * math.pi * f0 * time_sec)
                            + 0.10 * math.sin(8.0 * math.pi * f0 * time_sec)
                        ) * envelope * 0.35

                        val = max(-32767, min(32767, int(round(sample * 32767.0))))
                        pcm_bytes.extend(struct.pack("<h", val))

                    # Pause frames
                    for _ in range(pause_len):
                        pcm_bytes.extend(b"\x00\x00")

                wf.writeframes(pcm_bytes)
            return bio.getvalue()
