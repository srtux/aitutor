# AiTutor Offline Ablation Benchmark & Latency Analysis

Generated: 60-Second Multimodal Student Telemetry (1,800 frames @ 30 FPS)
Random Seed: 42

## 1. Affective Routing Pipeline Ablation Comparison

| Metric | Pipeline 1: Unfiltered Frame-by-Frame | Pipeline 2: Unmasked EMA Temporal | Pipeline 3: AiTutor VAD-Masked (Proposed) | Target Invariant |
| :--- | :---: | :---: | :---: | :---: |
| **Speech False-Positive Trigger Rate** | 26.2% (204/780) | 23.2% (181/780) | **0.0%** (0/780) | **0.0%** during active speech |
| **Silent Frustration Recall (28.5–36s)** | 100.0% | 100.0% | **100.0%** | **>= 95.0%** sustained struggle |
| **Premature Distraction Trigger (<2.5s turn)**| 98.0% (spurious) | 0.0% (filtered) | **0.0%** (filtered) | **0.0%** on transient glances |
| **Sustained Distraction Recall (53.0–56s)** | 100.0% | 100.0% | **100.0%** | **100.0%** after >2.5s head turn |
| **Distraction Detection Latency** | 0.00s (noisy/instant) | 2.80s | **2.80s** | Strict 2.5s hysteresis filter |
| **Proactive Silence Intervention Fire** | N/A (unsupported) | N/A (unsupported) | **t = 32.53s** | Fires at hesitation > 6.5s & frust > 0.55 |

### Key Findings:
1. **Speech Articulation Immunity**: In Pipeline 1 and 2, speaking articulation (jaw movement, lip pressing) falsely drives lower face strain (`mouthFrown` and `mouthPress` AU24) into the frustration classifier, producing **26.2%** and **23.2%** false-positive pedagogical interrupts. AiTutor's VAD speech mask + 300ms tail drops this to **0.0%**.
2. **Silent Struggle Sensitivity**: When the student gets silently stuck, the combination of AU4 brow furrow (`0.68–0.82`) and rising hesitation seconds smoothly sustains **100.0%** recall into `CONFUSED_HINT` and `FRUSTRATED_SCAFFOLD`.
3. **Transient Glance Rejection**: Pipeline 1 erroneously fires distraction alerts on brief 0.8s head glances (**98.0%** premature trigger rate), while Pipelines 2 and 3 enforce a strict 2.5s duration gate (**0.0%** premature rate) while maintaining **100.0%** recall once distraction is sustained.
4. **Proactive Intervention**: The silence trigger correctly fires at **t = 32.53s** (hesitation = 6.67s, frustration = 0.57), offering an empathetic nudge without the student needing to speak or press a button.

---

## 2. End-to-End Turn Latency Breakdown

| Subsystem Component | Local Engine Backend | Mean Latency | Median Latency | p95 Latency |
| :--- | :--- | :---: | :---: | :---: |
| **STT Audio Transcription (2.0s WAV)** | faster-whisper (base.en) | 537.9 ms | 520.5 ms | 578.2 ms |
| **Affective State Router (per-frame)** | AffectRouter (C-extension Python / math) | 0.009 ms | 0.008 ms | 0.009 ms |
| **LLM Time-to-First-Token (TTFT)** | offline-socratic-fallback | 28.4 ms | 14.7 ms | 70.4 ms |
| **LLM Turn Completion** | offline-socratic-fallback | 28.4 ms | 14.7 ms | 70.4 ms |
| **TTS Speech Synthesis (Audio generation)**| kokoro-onnx (blended) | 1923.0 ms | 1923.5 ms | 2012.6 ms |
