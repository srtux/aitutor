# AiTutor

**AiTutor** is a 100% local, open-source multimodal AI tutor built for Apple Silicon macOS (and testable on Linux). It teaches middle/high school algebra and introductory Python via real-time speech-to-speech dialogue while tracking facial expressions and 3D head pose at 30–60 FPS to detect **Attentiveness** and **Frustration** completely offline.

---

## Key Capabilities

- **100% Offline Edge Execution**: Zero cloud APIs or external runtime CDNs. Audio, video, blendshapes, speech recognition, LLM inference, and neural speech synthesis execute locally on device.
- **Calibrated FACS Affective State Router**:
  - **1.5-Second Personal Neutral Baseline Calibration**: Captures each learner's resting brow, squint, and mouth geometry on session start (or via the **Calibrate** button) and scores $\Delta$ above personal baseline.
  - **VAD Speech Mask with Weight Re-Normalization**: Masks speech-induced lower-face jaw/lip movements (AU15 `mouthFrown` + AU24 `mouthPress`) during active speech and a `300 ms` post-speech hangover tail while re-normalizing upper-face strain (AU4 `browDown`, AU7 `eyeSquint`, AU9 `noseSneer`, attenuated AU1 `browInnerUp`) and hesitation weights so genuine mid-speech frustration is preserved without false positives.
  - **4-Second Pre-Utterance Affect Latch & Multimodal Fusion**: Latches the 90th-percentile pre-speech frustration and pre-onset hesitation across a 4-second rolling window, fused with verbal confusion cues and consecutive step misses.
- **Warm Streaming Neural Voices**:
  - **Blended Kokoro ONNX Voices**: Ships with 4 curated voice personas (`Aria — Warm Mentor` blending `0.60 af_bella + 0.40 af_heart`, `Sky — Bright Encourager`, `Emma — Calm British`, and `Puck — Friendly Coach`), with automatic fallback to macOS `say` + `afconvert`.
  - **Spoken Math & Code Normalizer**: Converts equations (`3x + 7 = 22`), fractions (`3/4`), and Python expressions (`range(1, 5)`, `<=`) into natural spoken phrasing before synthesis.
  - **Sentence-Chunked Streaming Audio**: Synthesizes and streams the opening sentence immediately over WebSocket (`tutor_turn` + `tutor_audio_chunk`) for sub-second time-to-first-audio while remaining sentences queue gaplessly in WebAudio.
- **Interactive Socratic Curriculum & Live Telemetry**:
  - Three built-in interactive lessons (*Linear Equations & Balancing*, *Python Loops & Off-by-One Bugs*, *Fraction Word Problems*), SVG robot embodiment with gaze tracking and WebAudio visemes, live ablation gauges (VAD-Masked vs. Unmasked Frustration), built-in Demo Simulator, and one-click CSV telemetry export.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
| Browser Client (frontend/)                                                        |
|                                                                                   |
|  [Webcam 30-60 FPS] -> MediaPipe FaceLandmarker (Vendored WASM + Local .task)     |
|    - 478 3D Landmarks + 4x4 Head Pose Matrix (Yaw, Pitch, Roll)                   |
|    - 52 FACS Blendshapes + 1.5s Personal Neutral Baseline Calibration             |
|                                                                                   |
|  [Mic + CoreAudio AEC] -> AudioWorklet VAD (vad_worklet.js)                       |
|    - Real-time RMS Energy + Zero-Crossing Rate -> Speech Mask + 300ms Hangover    |
|    - 4s Rolling Affect Latch + 16 kHz Mono PCM WAV Utterance Capture              |
+-----------------------------------------------------------------------------------+
                                         |
                         WebSocket (ws://127.0.0.1:8000/ws)
                         JSON Telemetry + Base64 16kHz WAV + Chunked Audio
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| FastAPI Edge Server (backend/)                                                    |
|                                                                                   |
|  1. STT Engine (stt_engine.py)                                                    |
|     - macOS Apple Silicon: mlx-whisper (mlx-community/whisper-base-mlx)           |
|     - Cross-platform fallback: faster-whisper (base.en, int8)                     |
|                                                                                   |
|  2. Multimodal Affect Router (affect_router.py)                                   |
|     - 4 Pedagogical Modes: CONFIDENT_ADVANCE, CONFUSED_HINT,                      |
|       FRUSTRATED_SCAFFOLD, DISTRACTED_REENGAGE + Proactive Silence Intervention   |
|     - Fuses FACS affect latch + lexical frustration cues + consecutive misses     |
|                                                                                   |
|  3. Socratic Tutor LLM (llm_client.py + curriculum.py)                            |
|     - Local Ollama (/api/chat): qwen2.5:7b-instruct or llama3.2:3b (JSON schema)  |
|     - Deterministic multi-variant Socratic fallback when Ollama is offline        |
|                                                                                   |
|  4. Streaming Dual-Engine TTS (tts_engine.py)                                     |
|     - Spoken math/code normalizer + sentence-level streaming chunks               |
|     - Primary: Kokoro ONNX v1.0 blended voice styles (0.95x - 1.03x speeds)       |
|     - Fallback: macOS native `say` + `afconvert`                                  |
+-----------------------------------------------------------------------------------+
```

---

## Affective State Formulation

### 1. 3D Head Pose & Attentiveness ($A \in [0, 1]$)
Euler angles (`yaw`, `pitch`, `roll`) are extracted from the 4x4 facial transformation matrix alongside gaze blendshapes:
- `headPenalty = clamp((|yaw| - 12) / 25, 0, 1) * 0.6 + clamp((|pitch| - 10) / 20, 0, 1) * 0.4`
- `gazePenalty = clamp((gazeAway - 0.35) / 0.45, 0, 1)`
- `rawAttentiveness = clamp(1.0 - max(headPenalty, 0.8 * gazePenalty), 0.0, 1.0)`, smoothed via EMA ($\alpha = 0.15$).

### 2. Calibrated VAD-Masked Frustration ($F \in [0, 1]$)
Given personal neutral baseline $\mu_i$ calibrated over 1.5s, each Action Unit computes $\Delta_i = \max(0, \text{AU}_i - \mu_i)$:
- **Upper-face strain** (always active): combines $\Delta_{\text{browDown}}$ (AU4), $\Delta_{\text{eyeSquint}}$ (AU7), $\Delta_{\text{noseSneer}}$ (AU9), and speech-attenuated $\Delta_{\text{browInnerUp}}$ (AU1).
- **Lower-face tension** (active only when `isSpeaking == false` and `timeSinceLastSpeech > 300ms`): combines $\Delta_{\text{mouthFrown}}$ (AU15) and $\Delta_{\text{mouthPress}}$ (AU24).
- **Weight re-normalization**:
  - Silent (`speechMaskActive == false`): `0.50 * upperFaceStrain + 0.25 * lowerFaceTension + 0.25 * hesitationFactor`
  - Speaking (`speechMaskActive == true`): `0.67 * upperFaceStrain + 0.33 * hesitationFactor`

### 3. Four-State Pedagogical Router

| Pedagogical Mode | Trigger Condition | Tutor Adaptation & Voice Pacing |
| :--- | :--- | :--- |
| `DISTRACTED_REENGAGE` | `attentiveness < 0.45` sustained $> 2.5\text{s}$ | Pause explanation; ask a 1-sentence gamified check-in (`1.03x`). |
| `FRUSTRATED_SCAFFOLD` | `frustration >= 0.62` sustained $> 2.0\text{s}$ (or latched $P_{90} \ge 0.58$, lexical frustration cue, or $\ge 2$ misses) | Empathize warmly, reframe with a concrete analogy + binary A/B choice, push visual whiteboard scaffold (`0.95x`). |
| `CONFUSED_HINT` | `0.45 <= frustration < 0.62` or `hesitation > 5.0s` (or confusion phrase) | Provide a concise Socratic nudge pointing to the exact term or variable (`0.98x`). |
| `CONFIDENT_ADVANCE` | `attentiveness >= 0.45` and `frustration < 0.45` | Affirm reasoning and advance to the next problem step (`1.02x`). |

---

## Quickstart (macOS Apple Silicon & Linux)

### Prerequisites
- **macOS 14+** (Apple Silicon M1/M2/M3/M4 recommended) or **Linux**
- **Python 3.11+**
- **Ollama** (optional but recommended for full open-ended generative dialogue; offline Socratic curriculum works out-of-the-box even without Ollama)

### 1. Optional: Start Local Ollama LLM
```bash
brew install ollama
ollama pull qwen2.5:7b-instruct
ollama serve
```

### 2. Clone & Launch AiTutor
```bash
git clone https://github.com/srtux/aitutor.git
cd aitutor
./scripts/start_mac.sh
```

`./scripts/start_mac.sh` automatically:
1. Creates `.venv/` and installs `requirements.txt` (`mlx-whisper` on Apple Silicon, `faster-whisper` elsewhere).
2. Runs `scripts/download_models.py` to fetch and verify local offline assets (`face_landmarker.task`, vendored `@mediapipe/tasks-vision` WASM runtime, `kokoro-v1.0.onnx`, and `voices-v1.0.bin`).
3. Checks local Ollama availability and starts the FastAPI server at **`http://127.0.0.1:8000`**.

---

## Verification, Ablation Benchmark & OSS Hygiene

```bash
# Run unit, WebSocket/API integration, and open-source hygiene tests
.venv/bin/pytest -v

# Run the 60-second (1,800-frame) VAD Speech Mask & latency ablation benchmark
.venv/bin/python scripts/ablation_benchmark.py

# Run standalone open-source hygiene check
.venv/bin/python scripts/oss_audit.py
```

Ablation outputs are written to `results/ablation_summary.md`, `results/ablation_summary.csv`, and `results/ablation_timeseries_60s.csv`.

---

## Project Structure

```text
aitutor/
├── backend/
│   ├── affect_router.py     # Calibrated FACS + lexical + miss-count 4-state router
│   ├── curriculum.py        # Interactive lessons, step evaluator & multi-variant Socratic bank
│   ├── llm_client.py        # Ollama JSON-mode client + warm pedagogical persona prompts
│   ├── main.py              # FastAPI REST & WebSocket server with streaming audio chunks
│   ├── stt_engine.py        # mlx-whisper (Apple Silicon) + faster-whisper fallback
│   └── tts_engine.py        # Blended Kokoro ONNX voices, math/code normalizer & macOS say
├── frontend/
│   ├── index.html           # Split-screen dark/light responsive UI
│   ├── style.css            # Design tokens & layout styles
│   ├── affect_engine.js     # 30-60 FPS Euler pose, FACS calibration, VAD mask & 4s latch
│   ├── avatar.js            # Expressive SVG robot head with gaze tracking & audio visemes
│   ├── vad_worklet.js       # AudioWorklet RMS + ZCR voice activity detector
│   └── app.js               # Session controller, WebRTC/WASM vision & gapless audio queue
├── scripts/
│   ├── start_mac.sh         # One-command setup, model verification & server launcher
│   ├── download_models.py   # Offline model & MediaPipe WASM asset downloader
│   ├── ablation_benchmark.py# Reproducible 60s VAD Speech Mask & TTFA ablation benchmark
│   └── oss_audit.py         # Open-source hygiene & portability verifier
├── tests/                   # Pytest unit, API/WebSocket & OSS audit suites
└── LICENSE                  # MIT License
```

## License

Released under the [MIT License](LICENSE).
