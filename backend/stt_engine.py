"""Local Speech-to-Text engine supporting Apple Silicon MLX and Faster-Whisper.

Transcribes 16kHz mono audio streams with silence gating, root-mean-square (RMS)
energy thresholding, and sub-second lazy initialization.
"""

import io
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LocalSTTEngine:
    """Multi-backend local STT engine optimized for sub-second edge inference."""

    def __init__(self) -> None:
        self.is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        self._mlx_model_name = os.environ.get(
            "AITUTOR_MLX_MODEL", "mlx-community/whisper-base-mlx"
        )
        self._faster_model_name = os.environ.get(
            "AITUTOR_WHISPER_MODEL", "base.en"
        )

        self._mlx_whisper: Any = None
        self._faster_model: Any = None
        self._active_backend = "uninitialized"

        # Check if MLX is available on Apple Silicon
        if self.is_apple_silicon:
            try:
                import mlx_whisper  # type: ignore

                self._mlx_whisper = mlx_whisper
                self._active_backend = "mlx-whisper"
            except ImportError:
                self._mlx_whisper = None

    def warmup(self) -> None:
        """Eagerly warm up the selected whisper model weights."""
        if self.is_apple_silicon and self._mlx_whisper is not None:
            self._active_backend = "mlx-whisper"
            return

        if self._faster_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore

                model_dir = str(PROJECT_ROOT / "models" / "whisper")
                os.makedirs(model_dir, exist_ok=True)
                self._faster_model = WhisperModel(
                    self._faster_model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=model_dir,
                )
                self._active_backend = "faster-whisper"
            except Exception:
                self._faster_model = None
                self._active_backend = "mock-stt"

    def get_engine_name(self) -> str:
        """Return the active STT engine identifier for health probes."""
        if self.is_apple_silicon and self._mlx_whisper is not None:
            return f"mlx-whisper ({self._mlx_model_name})"
        if self._faster_model is not None:
            return f"faster-whisper ({self._faster_model_name})"
        return "faster-whisper (lazy-loaded)"

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> dict[str, Any]:
        """Transcribe WAV byte payload with silence detection and 16kHz resampling.

        Returns:
            dict containing:
                text: Transcribed string
                engine: Model/backend string
                latency_ms: Processing duration in milliseconds
                rms: Audio energy root-mean-square value
        """
        start_t = time.perf_counter()

        if not wav_bytes:
            return {
                "text": "",
                "engine": "empty-payload",
                "latency_ms": 0.0,
                "rms": 0.0,
            }

        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "text": "",
                "engine": "missing-audio-deps",
                "latency_ms": round(elapsed_ms, 2),
                "rms": 0.0,
            }

        try:
            data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "text": "",
                "engine": "wav-decode-error",
                "latency_ms": round(elapsed_ms, 2),
                "rms": 0.0,
            }

        # Convert multi-channel to mono
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)

        duration_s = float(len(data)) / float(sample_rate) if sample_rate > 0 else 0.0
        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) > 0 else 0.0

        # Silence / duration gating: ignore bursts under 150ms or quiet background noise
        if duration_s < 0.15 or rms < 0.002:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "text": "",
                "engine": "silence-gate",
                "latency_ms": round(elapsed_ms, 2),
                "rms": round(rms, 6),
            }

        # Resample to 16kHz if necessary using linear interpolation
        target_sr = 16000
        if sample_rate != target_sr and len(data) > 0:
            target_len = int(round(len(data) * float(target_sr) / float(sample_rate)))
            orig_indices = np.arange(len(data))
            new_indices = np.linspace(0.0, len(data) - 1, target_len, endpoint=True)
            resampled = np.interp(new_indices, orig_indices, data).astype(np.float32)
        else:
            resampled = data.astype(np.float32)

        # MLX Whisper execution on Apple Silicon
        if self.is_apple_silicon and self._mlx_whisper is not None:
            try:
                result = self._mlx_whisper.transcribe(
                    resampled,
                    path_or_hf_repo=self._mlx_model_name,
                    language="en",
                )
                text = str(result.get("text", "")).strip()
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                return {
                    "text": text,
                    "engine": f"mlx-whisper ({self._mlx_model_name})",
                    "latency_ms": round(elapsed_ms, 2),
                    "rms": round(rms, 6),
                }
            except Exception:
                # Fallback to faster-whisper if MLX encounters an error
                pass

        # Faster-Whisper lazy loading and inference
        if self._faster_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore

                model_dir = str(PROJECT_ROOT / "models" / "whisper")
                os.makedirs(model_dir, exist_ok=True)
                self._faster_model = WhisperModel(
                    self._faster_model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=model_dir,
                )
            except Exception:
                self._faster_model = None

        if self._faster_model is not None:
            try:
                segments, _ = self._faster_model.transcribe(
                    resampled,
                    language="en",
                    beam_size=1,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                return {
                    "text": text,
                    "engine": f"faster-whisper ({self._faster_model_name})",
                    "latency_ms": round(elapsed_ms, 2),
                    "rms": round(rms, 6),
                }
            except Exception:
                pass

        # Clean fallback when running in minimal test/mock environment
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "text": "",
            "engine": "stt-fallback",
            "latency_ms": round(elapsed_ms, 2),
            "rms": round(rms, 6),
        }
