/**
 * app.js - Main Application Orchestrator for AiTutor.
 *
 * Coordinates:
 * 1. MediaPipe FaceLandmarker & AffectEngine pipeline
 * 2. AudioWorklet VAD, 16kHz WAV packaging & WebAudio TTS playback
 * 3. RobotAvatar state animations and mouth viseme equalizer
 * 4. WebSocket session synchronization & fallback offline curriculum
 * 5. Interactive Whiteboard, SVG hint diagrams & Socratic chat log
 */

import { AffectEngine, clamp, extractEulerAnglesFromMatrix } from "./affect_engine.js";
import { RobotAvatar } from "./avatar.js";

// Global Application State
const state = {
  theme: "dark",
  isSessionActive: false,
  isTutorSpeaking: false,
  isStudentSpeaking: false,
  awaitingStudentAnswer: true,
  currentTopicId: "linear_equations",
  currentModel: "qwen2.5:7b-instruct",
  currentVoice: "warm_mentor",
  pedagogicalState: "CONFIDENT_ADVANCE",
  hesitationTimerSeconds: 0.0,
  speechOnsetHesitationSeconds: 0.0,
  hesitationIntervalId: null,
  ws: null,
  wsReconnectAttempts: 0,
  lastTelemetrySendMs: 0,
  audioContext: null,
  audioSourceNode: null,
  vadNode: null,
  analyserNode: null,
  micStream: null,
  audioWorkletSupported: true,
  speechBuffer: [],
  speechDurationMs: 0,
  silenceDurationMs: 0,
  speechRmsThreshold: 0.016,
  fpsCounter: 0,
  fpsLastTime: performance.now(),
  currentFps: 0,
  meshEnabled: true,
  simulationMode: false,
  simParams: {
    yaw: 0.0,
    pitch: 0.0,
    browFurrow: 0.0,
    mouthTension: 0.0,
    hesitation: 0.0,
    isSpeaking: false,
  },
  // Gapless Sentence-Chunked WebAudio Queue
  audioQueue: {
    activeTurnId: null,
    chunks: [],
    isPlaying: false,
    currentSourceNode: null,
    streamFinalReceived: false,
  },
};

// Built-in Offline Curriculum Fallback Database
const OFFLINE_CURRICULUM = {
  linear_equations: {
    id: "linear_equations",
    title: "Balancing Linear Equations",
    subject: "Algebra I",
    problem: "3x + 7 = 22",
    steps: [
      { id: 1, text: "Isolate the variable term (3x) by subtracting 7 from both sides", status: "current" },
      { id: 2, text: "Simplify both sides: 3x = 15", status: "pending" },
      { id: 3, text: "Divide both sides by the coefficient (3): x = 5", status: "pending" },
    ],
    diagramType: "balance_scale",
    binaryChoices: [
      { id: "A", text: "Subtract 7 from both sides" },
      { id: "B", text: "Divide both sides by 3" },
    ],
    hint: "Notice how +7 is grouped with 3x on the left side of the equals sign. How can we isolate the term with x?",
    initialGreeting: "Welcome! I'm your local AI tutor. We have a balancing equation on the whiteboard: 3x + 7 = 22. What move would you like to try first to isolate the x term?",
  },
  python_loops: {
    id: "python_loops",
    title: "Python Loops & Off-by-One Bugs",
    subject: "Computer Science",
    problem: "for i in range(1, len(items) + 1): print(items[i])",
    steps: [
      { id: 1, text: "Identify array zero-based indexing boundaries in Python", status: "current" },
      { id: 2, text: "Examine range(1, len(items) + 1) upper index traversal", status: "pending" },
      { id: 3, text: "Correct range to range(len(items)) to prevent IndexError", status: "pending" },
    ],
    diagramType: "loop_trace",
    binaryChoices: [
      { id: "A", text: "Use range(len(items)) starting at index 0" },
      { id: "B", text: "Keep index 1 and use items[i - 1]" },
    ],
    hint: "Remember that Python lists start at index 0 and end at index len(items) - 1. What happens when i reaches len(items)?",
    initialGreeting: "Welcome to computer science! We're exploring an off-by-one bug in a Python loop. Look at the code on the board. Can you spot why items[i] raises an IndexError on the last pass?",
  },
  fraction_word_problems: {
    id: "fraction_word_problems",
    title: "Fraction Word Problems & Proportions",
    subject: "Middle Math",
    problem: "Maya read 2/3 of a 90-page book. How many pages remain?",
    steps: [
      { id: 1, text: "Calculate the fraction of pages remaining: 1 - 2/3 = 1/3", status: "current" },
      { id: 2, text: "Multiply remaining fraction by total pages: (1/3) * 90", status: "pending" },
      { id: 3, text: "Conclude with 30 pages remaining", status: "pending" },
    ],
    diagramType: "fraction_bars",
    binaryChoices: [
      { id: "A", text: "Calculate 1/3 of 90 directly" },
      { id: "B", text: "Find 2/3 of 90 (60) and subtract from 90" },
    ],
    hint: "If Maya has already finished 2 out of 3 equal parts of the book, what fraction of the book is still left to read?",
    initialGreeting: "Welcome! Let's solve this fraction story problem together. The problem is displayed on your whiteboard. What is your first instinct on how to approach this?",
  },
};

// DOM References
const dom = {};

// Module singletons
let avatar = null;
let affectEngine = null;

// Initialize Application on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initDOMReferences();
  initTheme();
  initRobotAvatar();
  initAffectEngine();
  initEventListeners();
  loadInitialCurriculum();
  initWebSocket();
  startVisionLoop();
});

function initDOMReferences() {
  dom.themeToggle = document.getElementById("theme-toggle");
  dom.modelSelect = document.getElementById("model-select");
  dom.topicSelect = document.getElementById("topic-select");
  dom.voiceSelect = document.getElementById("voice-select");
  dom.sessionBtn = document.getElementById("session-btn");
  dom.sessionBtnText = document.getElementById("session-btn-text");
  dom.sessionIconStart = document.getElementById("session-icon-start");
  dom.sessionIconStop = document.getElementById("session-icon-stop");
  dom.exportCsvBtn = document.getElementById("export-csv-btn");

  dom.avatarStage = document.getElementById("avatar-stage");
  dom.statePill = document.getElementById("state-pill");
  dom.ttsSpeedReadout = document.getElementById("tts-speed-readout");

  dom.video = document.getElementById("webcam-video");
  dom.canvas = document.getElementById("mesh-canvas");
  dom.cameraFallback = document.getElementById("camera-fallback");
  dom.calibrateBtn = document.getElementById("calibrate-btn");
  dom.meshToggleBtn = document.getElementById("mesh-toggle-btn");
  dom.fpsPill = document.getElementById("fps-pill");
  dom.vadStatusPill = document.getElementById("vad-status-pill");
  dom.vadStatusText = document.getElementById("vad-status-text");

  // Simulator controls
  dom.simulatorDrawer = document.getElementById("simulator-drawer");
  dom.simulatorToggleBtn = document.getElementById("simulator-toggle-btn");
  dom.simulatorBody = document.getElementById("simulator-body");
  dom.simYaw = document.getElementById("sim-yaw");
  dom.simYawVal = document.getElementById("sim-yaw-val");
  dom.simBrow = document.getElementById("sim-brow");
  dom.simBrowVal = document.getElementById("sim-brow-val");
  dom.simMouth = document.getElementById("sim-mouth");
  dom.simMouthVal = document.getElementById("sim-mouth-val");
  dom.simHesitation = document.getElementById("sim-hesitation");
  dom.simHesitationVal = document.getElementById("sim-hesitation-val");
  dom.simEnabled = document.getElementById("sim-enabled");
  dom.simSpeaking = document.getElementById("sim-speaking");

  // Telemetry display elements
  dom.poseVal = document.getElementById("pose-val");
  dom.attentivenessBar = document.getElementById("attentiveness-bar");
  dom.attentivenessVal = document.getElementById("attentiveness-val");
  dom.frustrationBar = document.getElementById("frustration-bar");
  dom.frustrationVal = document.getElementById("frustration-val");
  dom.unmaskedFrustrationBar = document.getElementById("unmasked-frustration-bar");
  dom.unmaskedFrustrationVal = document.getElementById("unmasked-frustration-val");
  dom.ablationPill = document.getElementById("ablation-pill");
  dom.hesitationTimer = document.getElementById("hesitation-timer");
  dom.routerModeBadge = document.getElementById("router-mode-badge");
  dom.latencyReadout = document.getElementById("latency-readout");

  // Whiteboard elements
  dom.subjectPill = document.getElementById("subject-pill");
  dom.problemTitle = document.getElementById("problem-title");
  dom.problemDisplay = document.getElementById("problem-display");
  dom.whiteboardSteps = document.getElementById("whiteboard-steps");
  dom.whiteboardDiagram = document.getElementById("whiteboard-diagram");
  dom.binaryChoicesContainer = document.getElementById("binary-choices-container");
  dom.socraticHintText = document.getElementById("socratic-hint-text");

  // Dialogue elements
  dom.chatLog = document.getElementById("chat-log");
  dom.chatForm = document.getElementById("chat-form");
  dom.textPrompt = document.getElementById("text-prompt");
  dom.sessionStatusBadge = document.getElementById("session-status-badge");
}

function initTheme() {
  const savedTheme = localStorage.getItem("aitutor_theme") || "dark";
  setTheme(savedTheme);
}

function setTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("aitutor_theme", theme);
}

function initRobotAvatar() {
  avatar = new RobotAvatar(dom.avatarStage);
  avatar.setState("CONFIDENT_ADVANCE");
}

function initAffectEngine() {
  affectEngine = new AffectEngine({
    alpha: 0.15,
    historyWindowS: 3.0,
  });

  affectEngine.onProactiveHelpTrigger = (telemetrySnapshot) => {
    handleProactiveHelpTrigger(telemetrySnapshot);
  };

  affectEngine.onTelemetryFrame = (frame) => {
    updateTelemetryUI(frame);
  };

  affectEngine.onCalibrationStatusChange = ({ isCalibrating, calibrated, progress }) => {
    if (!dom.calibrateBtn) return;
    if (isCalibrating) {
      dom.calibrateBtn.textContent = `Calibrating (${Math.round(progress * 100)}%)`;
      dom.calibrateBtn.classList.add("active");
    } else if (calibrated) {
      dom.calibrateBtn.textContent = "Calibrated";
      dom.calibrateBtn.classList.remove("active");
    } else {
      dom.calibrateBtn.textContent = "Calibrate";
      dom.calibrateBtn.classList.remove("active");
    }
  };

  // Attempt initializing offline MediaPipe FaceLandmarker
  affectEngine
    .initFaceLandmarker(
      "./vendor/mediapipe/vision_bundle.mjs",
      "./vendor/mediapipe/wasm",
      "./assets/face_landmarker.task"
    )
    .then((success) => {
      if (!success) {
        enableSimulatorMode();
      }
    })
    .catch(() => {
      enableSimulatorMode();
    });
}

function initEventListeners() {
  // Theme toggle
  dom.themeToggle.addEventListener("click", () => {
    const nextTheme = state.theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
  });

  // Model select
  dom.modelSelect.addEventListener("change", (e) => {
    state.currentModel = e.target.value;
  });

  // Topic select
  dom.topicSelect.addEventListener("change", (e) => {
    switchTopic(e.target.value);
  });

  // Voice select
  if (dom.voiceSelect) {
    dom.voiceSelect.addEventListener("change", (e) => {
      state.currentVoice = e.target.value;
    });
  }

  // Calibrate button
  if (dom.calibrateBtn) {
    dom.calibrateBtn.addEventListener("click", () => {
      if (affectEngine) {
        affectEngine.startCalibration();
      }
    });
  }

  // Start / Stop Session Button
  dom.sessionBtn.addEventListener("click", () => {
    if (state.isSessionActive) {
      stopSession();
    } else {
      startSession();
    }
  });

  // Export CSV button
  dom.exportCsvBtn.addEventListener("click", () => {
    if (affectEngine) {
      affectEngine.exportTelemetryCSV();
    }
  });

  // Mesh toggle button
  dom.meshToggleBtn.addEventListener("click", () => {
    state.meshEnabled = !state.meshEnabled;
    dom.meshToggleBtn.classList.toggle("active", state.meshEnabled);
    if (!state.meshEnabled && dom.canvas) {
      const ctx = dom.canvas.getContext("2d");
      ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
    }
  });

  // Collapsible Lab Affect Simulator Drawer
  dom.simulatorToggleBtn.addEventListener("click", () => {
    const isExpanded = dom.simulatorToggleBtn.getAttribute("aria-expanded") === "true";
    dom.simulatorToggleBtn.setAttribute("aria-expanded", String(!isExpanded));
    dom.simulatorBody.classList.toggle("collapsed", isExpanded);
  });

  // Simulator Sliders
  dom.simYaw.addEventListener("input", (e) => {
    state.simParams.yaw = parseFloat(e.target.value);
    dom.simYawVal.textContent = `${state.simParams.yaw}°`;
    enableSimulatorMode();
  });

  dom.simBrow.addEventListener("input", (e) => {
    state.simParams.browFurrow = parseFloat(e.target.value);
    dom.simBrowVal.textContent = state.simParams.browFurrow.toFixed(2);
    enableSimulatorMode();
  });

  dom.simMouth.addEventListener("input", (e) => {
    state.simParams.mouthTension = parseFloat(e.target.value);
    dom.simMouthVal.textContent = state.simParams.mouthTension.toFixed(2);
    enableSimulatorMode();
  });

  dom.simHesitation.addEventListener("input", (e) => {
    state.simParams.hesitation = parseFloat(e.target.value);
    state.hesitationTimerSeconds = state.simParams.hesitation;
    dom.simHesitationVal.textContent = `${state.simParams.hesitation.toFixed(1)}s`;
    enableSimulatorMode();
  });

  // Simulator Enabled Switch
  if (dom.simEnabled) {
    dom.simEnabled.addEventListener("change", (e) => {
      const isEnabled = Boolean(e.target.checked);
      if (isEnabled) {
        enableSimulatorMode();
      } else {
        state.simulationMode = false;
        affectEngine.simulationMode = false;
        if (!state.isSessionActive || !dom.video || dom.video.paused) {
          if (dom.cameraFallback) dom.cameraFallback.classList.remove("hidden");
        }
      }
    });
  }

  dom.simSpeaking.addEventListener("change", (e) => {
    state.simParams.isSpeaking = Boolean(e.target.checked);
    state.isStudentSpeaking = state.simParams.isSpeaking;
    updateVadStatusUI(state.simParams.isSpeaking);
    enableSimulatorMode();
  });

  // Text Chat Form
  dom.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = dom.textPrompt.value.trim();
    if (!text) return;
    submitStudentAnswer(text);
    dom.textPrompt.value = "";
  });
}

function enableSimulatorMode() {
  state.simulationMode = true;
  affectEngine.simulationMode = true;
  if (dom.simEnabled) {
    dom.simEnabled.checked = true;
  }
  // When simulator is active, hide camera fallback so synthetic mesh on #mesh-canvas is visible!
  if (dom.cameraFallback) {
    dom.cameraFallback.classList.add("hidden");
  }
}

// --------------------------------------------------------------------------
// Camera & Vision Loop
// --------------------------------------------------------------------------
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 360, frameRate: { ideal: 30 } },
    });
    dom.video.srcObject = stream;
    await dom.video.play();
    state.simulationMode = false;
    affectEngine.simulationMode = false;
    if (dom.simEnabled) dom.simEnabled.checked = false;
    if (dom.cameraFallback) dom.cameraFallback.classList.add("hidden");

    // Automatically trigger neutral face auto-calibration on webcam start
    if (affectEngine) {
      affectEngine.startCalibration();
    }
    return true;
  } catch (err) {
    console.warn("Webcam access unavailable or declined; using Lab Affect Simulator:", err);
    enableSimulatorMode();
    return false;
  }
}

function startVisionLoop() {
  const loop = (timestamp) => {
    // FPS Meter
    state.fpsCounter++;
    if (timestamp - state.fpsLastTime >= 1000) {
      state.currentFps = Math.round((state.fpsCounter * 1000) / (timestamp - state.fpsLastTime));
      dom.fpsPill.textContent = `${state.currentFps} FPS`;
      state.fpsCounter = 0;
      state.fpsLastTime = timestamp;
    }

    const currentHesitation = state.isStudentSpeaking
      ? state.speechOnsetHesitationSeconds
      : state.hesitationTimerSeconds;

    if (state.simulationMode) {
      // Run simulated affective frame
      const simulatedBlendshapes = {
        browDownLeft: state.simParams.browFurrow,
        browDownRight: state.simParams.browFurrow,
        eyeSquintLeft: state.simParams.browFurrow * 0.5,
        eyeSquintRight: state.simParams.browFurrow * 0.5,
        mouthFrownLeft: state.simParams.mouthTension,
        mouthFrownRight: state.simParams.mouthTension,
        mouthPressLeft: state.simParams.mouthTension * 0.8,
        mouthPressRight: state.simParams.mouthTension * 0.8,
      };

      const radYaw = (state.simParams.yaw * Math.PI) / 180.0;
      const radPitch = (state.simParams.pitch * Math.PI) / 180.0;
      const fakeMatrix = [
        Math.cos(radYaw), 0, Math.sin(radYaw), 0,
        0, 1, 0, 0,
        -Math.sin(radYaw), 0, Math.cos(radYaw), 0,
        0, 0, 0, 1,
      ];

      const frame = affectEngine.computeFrameMetrics({
        blendshapesMap: simulatedBlendshapes,
        matrix: fakeMatrix,
        isSpeaking: state.simParams.isSpeaking,
        hesitationSeconds: currentHesitation,
        nowMs: timestamp,
      });

      if (state.meshEnabled && dom.canvas) {
        affectEngine.drawSimulatedFace(dom.canvas, {
          yaw: state.simParams.yaw,
          pitch: state.simParams.pitch,
          browFurrow: state.simParams.browFurrow,
          mouthTension: state.simParams.mouthTension,
          isSpeaking: state.simParams.isSpeaking,
          speechMaskActive: frame.speechMaskActive,
        });
      }
    } else {
      // Process real webcam video frame
      if (dom.video && dom.video.readyState >= 2) {
        affectEngine.isSpeaking = state.isStudentSpeaking;
        affectEngine.hesitationSeconds = currentHesitation;
        affectEngine.processVideoFrame(
          dom.video,
          state.meshEnabled ? dom.canvas : null,
          timestamp
        );
      }
    }

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

// --------------------------------------------------------------------------
// Audio Pipeline: WebAudio, VAD Worklet & WAV Packaging
// --------------------------------------------------------------------------
async function setupAudioPipeline() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!state.audioContext) {
      state.audioContext = new AudioContextClass();
    }
    if (state.audioContext.state === "suspended") {
      await state.audioContext.resume();
    }

    // Connect AnalyserNode to avatar for viseme equalizer
    state.analyserNode = state.audioContext.createAnalyser();
    state.analyserNode.fftSize = 64;
    avatar.connectAudioAnalyser(state.analyserNode);

    // Request mono microphone stream with hardware AEC / noise suppression
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });

    state.audioSourceNode = state.audioContext.createMediaStreamSource(state.micStream);

    // Try registering AudioWorklet vad-processor
    try {
      await state.audioContext.audioWorklet.addModule("./vad_worklet.js");
      state.vadNode = new AudioWorkletNode(state.audioContext, "vad-processor");
      state.audioSourceNode.connect(state.vadNode);

      state.vadNode.port.onmessage = (event) => {
        handleVadWorkletMessage(event.data);
      };
      state.audioWorkletSupported = true;
    } catch (workletError) {
      console.warn("AudioWorklet unavailable; using ScriptProcessorNode fallback:", workletError);
      setupScriptProcessorVadFallback();
    }

    return true;
  } catch (err) {
    console.warn("Microphone setup unavailable or declined:", err);
    return false;
  }
}

function setupScriptProcessorVadFallback() {
  const bufferSize = 2048;
  const scriptNode = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
  state.audioSourceNode.connect(scriptNode);
  scriptNode.connect(state.audioContext.destination);

  scriptNode.onaudioprocess = (e) => {
    const channel = e.inputBuffer.getChannelData(0);
    let sumSquares = 0;
    let zeroCrossings = 0;
    for (let i = 0; i < channel.length; i++) {
      const s = channel[i];
      sumSquares += s * s;
      if (i > 0 && ((s >= 0 && channel[i - 1] < 0) || (s < 0 && channel[i - 1] >= 0))) {
        zeroCrossings++;
      }
    }
    const rms = Math.sqrt(sumSquares / channel.length);
    const zcr = zeroCrossings / channel.length;

    // Linear downsampling to 16kHz
    const ratio = state.audioContext.sampleRate / 16000;
    const outLength = Math.floor(channel.length / ratio);
    const samples16k = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const idx = Math.floor(i * ratio);
      samples16k[i] = channel[idx];
    }

    handleVadWorkletMessage({ rms, zcr, samples16k });
  };
}

function handleVadWorkletMessage({ rms, zcr, samples16k }) {
  if (!state.isSessionActive) return;

  const isVoiceActive = rms > state.speechRmsThreshold && zcr < 0.45;
  const chunkMs = (samples16k.length / 16000) * 1000;

  if (isVoiceActive) {
    if (!state.isStudentSpeaking) {
      // Freeze the pre-utterance hesitation duration when student begins speaking
      state.speechOnsetHesitationSeconds = state.hesitationTimerSeconds;
    }
    state.isStudentSpeaking = true;
    state.speechDurationMs += chunkMs;
    state.silenceDurationMs = 0;
    state.speechBuffer.push(samples16k);
    updateVadStatusUI(true);

    // Keep active timer at 0, but retain speechOnsetHesitationSeconds for affect_engine
    state.hesitationTimerSeconds = 0.0;
  } else {
    if (state.speechBuffer.length > 0) {
      state.silenceDurationMs += chunkMs;
      // Append brief hangover silence
      state.speechBuffer.push(samples16k);

      if (state.silenceDurationMs > 700) {
        // Speech segment finished
        if (state.speechDurationMs >= 250) {
          packageAndSendStudentAudio();
        } else {
          // Discard short pop/click noise
          state.speechBuffer = [];
          state.speechDurationMs = 0;
          state.silenceDurationMs = 0;
        }
        state.isStudentSpeaking = false;
        state.speechOnsetHesitationSeconds = 0.0;
        updateVadStatusUI(false);
      }
    } else {
      state.isStudentSpeaking = false;
      state.speechOnsetHesitationSeconds = 0.0;
      updateVadStatusUI(false);
    }
  }
}

function updateVadStatusUI(isSpeaking) {
  if (isSpeaking) {
    dom.vadStatusPill.className = "pill vad-pill vad-active";
    dom.vadStatusText.textContent = "VAD MASK ACTIVE";
  } else {
    dom.vadStatusPill.className = "pill vad-pill vad-idle";
    dom.vadStatusText.textContent = "MIC IDLE";
  }
}

function packageAndSendStudentAudio() {
  // Capture 4s pre-utterance latched affect telemetry before clearing speech
  const turnTelemetry = affectEngine ? affectEngine.getTurnAffectSnapshot() : null;

  // Combine all 16kHz float slices into a single Float32Array
  let totalLength = 0;
  for (const chunk of state.speechBuffer) {
    totalLength += chunk.length;
  }
  const combined = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of state.speechBuffer) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  // Clear buffer
  state.speechBuffer = [];
  state.speechDurationMs = 0;
  state.silenceDurationMs = 0;

  // Encode to RIFF PCM16 mono WAV ArrayBuffer
  const wavBuffer = float32ToPcm16Wav(combined, 16000);
  const base64Str = arrayBufferToBase64(wavBuffer);

  // Send over WebSocket to local STT engine
  sendWebSocketMessage({
    type: "student_audio",
    wav_base64: base64Str,
    topic_id: state.currentTopicId,
    model: state.currentModel,
    voice: state.currentVoice,
    stream_audio: true,
    telemetry: turnTelemetry || affectEngine.getTelemetrySnapshot(),
  });
}

function float32ToPcm16Wav(floatSamples, sampleRate = 16000) {
  const numSamples = floatSamples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  // "RIFF"
  view.setUint32(0, 0x52494646, false);
  view.setUint32(4, 36 + numSamples * 2, true);
  // "WAVE"
  view.setUint32(8, 0x57415645, false);
  // "fmt "
  view.setUint32(12, 0x666d7420, false);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM = 1
  view.setUint16(22, 1, true); // Mono = 1
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); // 16-bit
  // "data"
  view.setUint32(36, 0x64617461, false);
  view.setUint32(40, numSamples * 2, true);

  let byteOffset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1.0, Math.min(1.0, floatSamples[i]));
    const val = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(byteOffset, val, true);
    byteOffset += 2;
  }

  return buffer;
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  const chunkSize = 8192;
  for (let i = 0; i < len; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(i + chunkSize, len));
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// TTS Audio Playback Engine & Gapless Chunked Audio Queue
// --------------------------------------------------------------------------
function clearAudioQueue() {
  if (state.audioQueue.currentSourceNode) {
    try {
      state.audioQueue.currentSourceNode.stop();
      state.audioQueue.currentSourceNode.disconnect();
    } catch (e) {
      // Ignore if already stopped
    }
    state.audioQueue.currentSourceNode = null;
  }
  state.audioQueue.chunks = [];
  state.audioQueue.isPlaying = false;
  state.audioQueue.streamFinalReceived = false;
}

function queueAudioChunk(turnId, base64Wav, isFinal = false) {
  if (!base64Wav || !state.audioContext) return;

  // If a new turn started, flush prior queue
  if (state.audioQueue.activeTurnId !== turnId) {
    clearAudioQueue();
    state.audioQueue.activeTurnId = turnId;
  }

  state.audioQueue.chunks.push({ base64Wav, isFinal });
  if (isFinal) {
    state.audioQueue.streamFinalReceived = true;
  }

  if (!state.audioQueue.isPlaying) {
    playNextQueuedAudioChunk();
  }
}

async function playNextQueuedAudioChunk() {
  if (!state.audioQueue.chunks.length) {
    state.audioQueue.isPlaying = false;
    if (state.audioQueue.streamFinalReceived) {
      state.isTutorSpeaking = false;
      affectEngine.awaitingStudentAnswer = true;
      startHesitationTimer();
    }
    return;
  }

  state.audioQueue.isPlaying = true;
  const chunkItem = state.audioQueue.chunks.shift();

  try {
    const binaryStr = atob(chunkItem.base64Wav);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    const audioBuffer = await state.audioContext.decodeAudioData(bytes.buffer);
    const sourceNode = state.audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.playbackRate.value = 1.0;

    sourceNode.connect(state.analyserNode);
    state.analyserNode.connect(state.audioContext.destination);
    state.audioQueue.currentSourceNode = sourceNode;

    state.isTutorSpeaking = true;
    state.hesitationTimerSeconds = 0.0;
    affectEngine.awaitingStudentAnswer = false;

    sourceNode.onended = () => {
      state.audioQueue.currentSourceNode = null;
      playNextQueuedAudioChunk();
    };

    sourceNode.start(0);
  } catch (err) {
    console.warn("Chunk audio decode/playback error:", err);
    state.audioQueue.currentSourceNode = null;
    playNextQueuedAudioChunk();
  }
}

async function playTutorAudio(base64Wav, ttsSpeed = 1.0) {
  if (!base64Wav || !state.audioContext) return;
  clearAudioQueue();

  try {
    const binaryStr = atob(base64Wav);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    const audioBuffer = await state.audioContext.decodeAudioData(bytes.buffer);
    const sourceNode = state.audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.playbackRate.value = 1.0;

    // Route: source -> analyser -> destination
    sourceNode.connect(state.analyserNode);
    state.analyserNode.connect(state.audioContext.destination);
    state.audioQueue.currentSourceNode = sourceNode;

    state.isTutorSpeaking = true;
    state.hesitationTimerSeconds = 0.0;
    affectEngine.awaitingStudentAnswer = false;

    sourceNode.onended = () => {
      state.isTutorSpeaking = false;
      affectEngine.awaitingStudentAnswer = true;
      state.audioQueue.currentSourceNode = null;
      startHesitationTimer();
    };

    sourceNode.start(0);
  } catch (err) {
    console.warn("TTS audio playback failed:", err);
    state.isTutorSpeaking = false;
    affectEngine.awaitingStudentAnswer = true;
    state.audioQueue.currentSourceNode = null;
    startHesitationTimer();
  }
}

// --------------------------------------------------------------------------
// Hesitation Timer
// --------------------------------------------------------------------------
function startHesitationTimer() {
  if (state.hesitationIntervalId) {
    clearInterval(state.hesitationIntervalId);
  }
  state.hesitationTimerSeconds = 0.0;
  state.hesitationIntervalId = setInterval(() => {
    if (state.isSessionActive && !state.isTutorSpeaking && !state.isStudentSpeaking) {
      state.hesitationTimerSeconds += 0.1;
      dom.hesitationTimer.textContent = `${state.hesitationTimerSeconds.toFixed(1)}s`;
    }
  }, 100);
}

function stopHesitationTimer() {
  if (state.hesitationIntervalId) {
    clearInterval(state.hesitationIntervalId);
    state.hesitationIntervalId = null;
  }
  state.hesitationTimerSeconds = 0.0;
  dom.hesitationTimer.textContent = "0.0s";
}

// --------------------------------------------------------------------------
// UI Telemetry Sync
// --------------------------------------------------------------------------
function updateTelemetryUI(frame) {
  // Pose readout
  dom.poseVal.textContent = `Yaw: ${frame.yaw}° · Pitch: ${frame.pitch}°`;

  // Attentiveness bar
  const attPct = Math.round(frame.attentiveness * 100);
  dom.attentivenessBar.style.width = `${attPct}%`;
  dom.attentivenessVal.textContent = `${attPct}%`;

  // VAD-Masked Frustration bar
  const frustPct = Math.round(frame.frustration * 100);
  dom.frustrationBar.style.width = `${frustPct}%`;
  dom.frustrationVal.textContent = `${frustPct}%`;

  // Unmasked Frustration bar & ablation pill
  const unmaskedPct = Math.round(frame.unmaskedFrustration * 100);
  dom.unmaskedFrustrationBar.style.width = `${unmaskedPct}%`;
  dom.unmaskedFrustrationVal.textContent = `${unmaskedPct}%`;

  const suppressedDeltaPct = Math.max(0, unmaskedPct - frustPct);
  dom.ablationPill.textContent = `Suppressed: ${suppressedDeltaPct}%`;

  // Pedagogical State Transition
  if (frame.state !== state.pedagogicalState) {
    setPedagogicalState(frame.state);
  }

  // Update Gaze Tracking on Avatar from normalized face coordinate when present
  const normX = frame.faceX !== undefined ? (0.5 - frame.faceX) * 2.2 : -frame.yaw / 35.0;
  const normY = frame.faceY !== undefined ? (frame.faceY - 0.5) * 2.2 : frame.pitch / 25.0;
  avatar.setTargetGaze(normX, normY, frame.yaw);

  // Periodically stream telemetry frame to backend affect router (~5 FPS)
  const now = performance.now();
  if (
    state.isSessionActive &&
    state.ws &&
    state.ws.readyState === WebSocket.OPEN &&
    now - state.lastTelemetrySendMs >= 200
  ) {
    state.lastTelemetrySendMs = now;
    state.ws.send(JSON.stringify({ type: "telemetry", telemetry: frame }));
  }
}

function setPedagogicalState(newState) {
  state.pedagogicalState = newState;
  avatar.setState(newState);

  // Update State Pill Class
  dom.statePill.textContent = newState;
  dom.statePill.className = "pill state-pill";

  let speedText = "1.00x";
  let modeBadgeText = "CONFIDENT";

  switch (newState) {
    case "CONFIDENT_ADVANCE":
      dom.statePill.classList.add("state-confident");
      speedText = "1.02x";
      modeBadgeText = "CONFIDENT";
      break;
    case "CONFUSED_HINT":
      dom.statePill.classList.add("state-confused");
      speedText = "0.98x";
      modeBadgeText = "HINT";
      break;
    case "FRUSTRATED_SCAFFOLD":
      dom.statePill.classList.add("state-frustrated");
      speedText = "0.95x";
      modeBadgeText = "SCAFFOLD";
      break;
    case "DISTRACTED_REENGAGE":
      dom.statePill.classList.add("state-distracted");
      speedText = "1.03x";
      modeBadgeText = "REENGAGE";
      break;
  }

  dom.ttsSpeedReadout.textContent = speedText;
  dom.routerModeBadge.textContent = modeBadgeText;
}

// --------------------------------------------------------------------------
// Socratic Dialogue & Whiteboard Management
// --------------------------------------------------------------------------
function loadInitialCurriculum() {
  switchTopic(state.currentTopicId);
}

function switchTopic(topicId) {
  state.currentTopicId = topicId;
  const cur = OFFLINE_CURRICULUM[topicId] || OFFLINE_CURRICULUM.linear_equations;

  dom.subjectPill.textContent = cur.subject;
  dom.problemTitle.textContent = cur.title;
  dom.problemDisplay.textContent = cur.problem;
  dom.socraticHintText.textContent = cur.hint;

  renderWhiteboardSteps(cur.steps);
  renderVisualDiagram(cur.diagramType);
  renderBinaryChoices(cur.binaryChoices);

  // Clear chat log and render welcoming greeting
  dom.chatLog.innerHTML = "";
  appendChatMessage("TUTOR", cur.initialGreeting, "CONFIDENT_ADVANCE", false);
}

function renderWhiteboardSteps(steps) {
  dom.whiteboardSteps.innerHTML = "";
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const li = document.createElement("li");
    li.className = `step-item step-${step.status || "pending"}`;

    const stepNum = step.id || (i + 1);
    let badgeIcon = `${stepNum}`;
    if (step.status === "completed") {
      badgeIcon = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
    }

    let stepText = step.label || step.text || step.expr || "";
    if (step.label && step.expr && step.label !== step.expr) {
      stepText = `${step.label}: ${step.expr}`;
    }

    li.innerHTML = `
      <div class="step-left">
        <span class="step-badge">${badgeIcon}</span>
        <span class="step-text">${stepText}</span>
      </div>
      <span class="pill pill-mono">${step.status || "pending"}</span>
    `;
    dom.whiteboardSteps.appendChild(li);
  }
}

function renderVisualDiagram(diagramType) {
  dom.whiteboardDiagram.innerHTML = "";

  if (diagramType === "balance_scale") {
    // Crisp SVG Balance Scale (Strictly bounded within x=12..308, y=12..103)
    dom.whiteboardDiagram.innerHTML = `
      <svg viewBox="0 0 320 115" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Balance Scale Diagram">
        <!-- Base & Fulcrum -->
        <polygon points="160,76 148,102 172,102" fill="var(--bg-elevated, #334155)" stroke="var(--border-strong, #64748b)" stroke-width="1.5"/>
        <line x1="120" y1="102" x2="200" y2="102" stroke="var(--border-strong, #64748b)" stroke-width="2"/>
        <circle cx="160" cy="76" r="4" fill="var(--cyan-primary, #22d3ee)"/>

        <!-- Balance Beam -->
        <line x1="45" y1="48" x2="275" y2="48" stroke="var(--text-secondary, #94a3b8)" stroke-width="3" stroke-linecap="round"/>

        <!-- Left Pan Strings & Pan -->
        <line x1="45" y1="48" x2="22" y2="78" stroke="var(--border-strong, #64748b)" stroke-width="1.2"/>
        <line x1="45" y1="48" x2="68" y2="78" stroke="var(--border-strong, #64748b)" stroke-width="1.2"/>
        <ellipse cx="45" cy="80" rx="28" ry="5" fill="var(--bg-card-subtle, #1e293b)" stroke="var(--cyan-primary, #22d3ee)" stroke-width="1.5"/>
        <!-- Left Pan Weight Label -->
        <rect x="22" y="58" width="46" height="18" rx="4" fill="var(--bg-surface, #0f172a)" stroke="var(--blue-primary, #38bdf8)" stroke-width="1"/>
        <text x="45" y="71" fill="var(--text-primary, #f8fafc)" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">3x + 7</text>

        <!-- Equals Sign at Center with Backing Pill -->
        <rect x="148" y="38" width="24" height="20" rx="5" fill="var(--bg-surface, #0f172a)" stroke="#f59e0b" stroke-width="1.2"/>
        <text x="160" y="52" fill="#f59e0b" font-family="monospace" font-size="16" font-weight="bold" text-anchor="middle" dominant-baseline="central">=</text>

        <!-- Right Pan Strings & Pan -->
        <line x1="275" y1="48" x2="252" y2="78" stroke="var(--border-strong, #64748b)" stroke-width="1.2"/>
        <line x1="275" y1="48" x2="298" y2="78" stroke="var(--border-strong, #64748b)" stroke-width="1.2"/>
        <ellipse cx="275" cy="80" rx="28" ry="5" fill="var(--bg-card-subtle, #1e293b)" stroke="var(--cyan-primary, #22d3ee)" stroke-width="1.5"/>
        <!-- Right Pan Weight Label -->
        <rect x="258" y="58" width="34" height="18" rx="4" fill="var(--bg-surface, #0f172a)" stroke="var(--blue-primary, #38bdf8)" stroke-width="1"/>
        <text x="275" y="71" fill="var(--text-primary, #f8fafc)" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">22</text>
      </svg>
    `;
  } else if (diagramType === "loop_trace") {
    // Array blocks & pointer for Python Off-by-one bug (Strictly bounded within x=12..308, y=12..103)
    dom.whiteboardDiagram.innerHTML = `
      <svg viewBox="0 0 320 115" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Python Loop Array Bounds Diagram">
        <!-- Current Pointer Arrow -->
        <polygon points="117,28 112,18 122,18" fill="#f59e0b"/>
        <text x="117" y="15" fill="#f59e0b" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">i=1</text>

        <!-- Array Box 0 -->
        <rect x="14" y="36" width="62" height="34" rx="6" fill="var(--bg-card-subtle, #1e293b)" stroke="var(--blue-primary, #38bdf8)" stroke-width="1.5"/>
        <text x="45" y="58" fill="var(--text-primary, #f8fafc)" font-family="monospace" font-size="12" font-weight="bold" text-anchor="middle">"alpha"</text>
        <text x="45" y="88" fill="var(--text-muted, #64748b)" font-family="monospace" font-size="10" text-anchor="middle">idx 0</text>

        <!-- Array Box 1 -->
        <rect x="86" y="36" width="62" height="34" rx="6" fill="var(--bg-card-subtle, #1e293b)" stroke="var(--blue-primary, #38bdf8)" stroke-width="1.5"/>
        <text x="117" y="58" fill="var(--text-primary, #f8fafc)" font-family="monospace" font-size="12" font-weight="bold" text-anchor="middle">"beta"</text>
        <text x="117" y="88" fill="var(--text-muted, #64748b)" font-family="monospace" font-size="10" text-anchor="middle">idx 1</text>

        <!-- Array Box 2 -->
        <rect x="158" y="36" width="62" height="34" rx="6" fill="var(--bg-card-subtle, #1e293b)" stroke="var(--blue-primary, #38bdf8)" stroke-width="1.5"/>
        <text x="189" y="58" fill="var(--text-primary, #f8fafc)" font-family="monospace" font-size="12" font-weight="bold" text-anchor="middle">"gamma"</text>
        <text x="189" y="88" fill="var(--text-muted, #64748b)" font-family="monospace" font-size="10" text-anchor="middle">idx 2</text>

        <!-- Out of Bounds Warning Box 3 -->
        <rect x="230" y="36" width="76" height="34" rx="6" fill="var(--rose-bg, #450a0a)" stroke="var(--rose-primary, #f43f5e)" stroke-width="1.5" stroke-dasharray="4,3"/>
        <text x="268" y="57" fill="var(--rose-primary, #fca5a5)" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">IndexError</text>
        <text x="268" y="88" fill="var(--rose-primary, #f43f5e)" font-family="monospace" font-size="10" text-anchor="middle">idx 3 (out)</text>
      </svg>
    `;
  } else if (diagramType === "fraction_bars") {
    // Fraction Comparison Rectangles (Strictly bounded within x=12..308, y=12..103)
    dom.whiteboardDiagram.innerHTML = `
      <svg viewBox="0 0 320 115" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fraction Comparison Diagram">
        <!-- 90 Pages Total Bar -->
        <text x="20" y="22" fill="var(--text-secondary, #94a3b8)" font-family="sans-serif" font-size="11" font-weight="bold">Book Total: 90 Pages</text>

        <!-- 3 Segmented Part Bar -->
        <rect x="20" y="34" width="88" height="32" rx="4" fill="var(--blue-primary, #0284c7)" stroke="var(--cyan-primary, #38bdf8)" stroke-width="1.2"/>
        <text x="64" y="54" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">30p (1/3)</text>

        <rect x="114" y="34" width="88" height="32" rx="4" fill="var(--blue-primary, #0284c7)" stroke="var(--cyan-primary, #38bdf8)" stroke-width="1.2"/>
        <text x="158" y="54" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold" text-anchor="middle">30p (1/3)</text>

        <rect x="208" y="34" width="88" height="32" rx="4" fill="var(--bg-card-subtle, #1e293b)" stroke="#f59e0b" stroke-width="1.5"/>
        <text x="252" y="54" fill="#f59e0b" font-family="monospace" font-size="10" font-weight="bold" text-anchor="middle">30p (1/3 left)</text>

        <text x="110" y="86" fill="var(--blue-primary, #38bdf8)" font-family="sans-serif" font-size="11" text-anchor="middle">Read: 2/3 (60 pages)</text>
        <text x="252" y="86" fill="#f59e0b" font-family="sans-serif" font-size="11" text-anchor="middle">Left: 1/3</text>
      </svg>
    `;
  }
}

function renderBinaryChoices(choices) {
  if (!choices || choices.length === 0) {
    dom.binaryChoicesContainer.classList.add("hidden");
    dom.binaryChoicesContainer.innerHTML = "";
    return;
  }

  dom.binaryChoicesContainer.innerHTML = "";
  for (let i = 0; i < choices.length; i++) {
    const choice = choices[i];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "binary-choice-btn";
    const defaultKey = String.fromCharCode(65 + i); // 'A', 'B', 'C'
    const key = typeof choice === "object" ? (choice.id || choice.key || defaultKey) : defaultKey;
    const label = typeof choice === "object" ? (choice.text || choice.label || "") : String(choice);

    btn.innerHTML = `
      <span class="choice-key">${key}</span>
      <span class="choice-text">${label}</span>
    `;
    btn.addEventListener("click", () => {
      submitStudentAnswer(label);
    });
    dom.binaryChoicesContainer.appendChild(btn);
  }
  dom.binaryChoicesContainer.classList.remove("hidden");
}

function appendChatMessage(speaker, text, statePill = null, isProactive = false) {
  const turnDiv = document.createElement("div");
  turnDiv.className = `chat-turn turn-${speaker.toLowerCase()}`;

  const metaDiv = document.createElement("div");
  metaDiv.className = "turn-meta";

  const speakerSpan = document.createElement("span");
  speakerSpan.className = "turn-speaker";
  speakerSpan.textContent = speaker;
  metaDiv.appendChild(speakerSpan);

  const timeSpan = document.createElement("span");
  timeSpan.className = "turn-time";
  const now = new Date();
  timeSpan.textContent = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}`;
  metaDiv.appendChild(timeSpan);

  if (statePill) {
    const pill = document.createElement("span");
    pill.className = "pill pill-mono";
    pill.textContent = statePill;
    metaDiv.appendChild(pill);
  }

  if (isProactive) {
    const proPill = document.createElement("span");
    proPill.className = "pill pill-proactive";
    proPill.textContent = "PROACTIVE";
    metaDiv.appendChild(proPill);
  }

  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "turn-bubble";
  bubbleDiv.textContent = text;

  turnDiv.appendChild(metaDiv);
  turnDiv.appendChild(bubbleDiv);

  dom.chatLog.appendChild(turnDiv);
  dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
}

function submitStudentAnswer(answerText) {
  // Capture 4s pre-utterance latched affect telemetry before stopping hesitation timer
  const turnTelemetry = affectEngine ? affectEngine.getTurnAffectSnapshot() : null;

  appendChatMessage("STUDENT", answerText, (turnTelemetry?.latched_pedagogical_state || state.pedagogicalState));
  stopHesitationTimer();

  sendWebSocketMessage({
    type: "student_text",
    text: answerText,
    topic_id: state.currentTopicId,
    model: state.currentModel,
    voice: state.currentVoice,
    stream_audio: true,
    telemetry: turnTelemetry || affectEngine.getTelemetrySnapshot(),
  });
}

function handleProactiveHelpTrigger(telemetrySnapshot) {
  sendWebSocketMessage({
    type: "proactive_help_trigger",
    topic_id: state.currentTopicId,
    model: state.currentModel,
    voice: state.currentVoice,
    stream_audio: true,
    telemetry: telemetrySnapshot,
  });
}

// --------------------------------------------------------------------------
// WebSocket Session Client & Local Fallback
// --------------------------------------------------------------------------
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/session`;

  try {
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      state.wsReconnectAttempts = 0;
      dom.sessionStatusBadge.textContent = "Connected";
      dom.sessionStatusBadge.className = "pill pill-live";

      // Sync initial topics from backend REST API
      syncBackendCurriculum();
    };

    state.ws.onmessage = (event) => {
      handleWebSocketMessage(event.data);
    };

    state.ws.onclose = () => {
      dom.sessionStatusBadge.textContent = "Offline (Local Standalone)";
      dom.sessionStatusBadge.className = "pill pill-idle";
      scheduleWebSocketReconnect();
    };

    state.ws.onerror = (err) => {
      console.warn("WebSocket connection error; falling back to offline standalone mode:", err);
    };
  } catch (err) {
    console.warn("WebSocket initialization error:", err);
  }
}

function scheduleWebSocketReconnect() {
  if (state.wsReconnectAttempts < 5) {
    state.wsReconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, state.wsReconnectAttempts), 10000);
    setTimeout(() => {
      initWebSocket();
    }, delay);
  }
}

function sendWebSocketMessage(msgObj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(msgObj));
  } else {
    // Backend offline: run local mock Socratic response
    handleOfflineLocalResponse(msgObj);
  }
}

function handleWebSocketMessage(dataStr) {
  try {
    const msg = JSON.parse(dataStr);

    if (msg.type === "tutor_turn") {
      const stateDirective = msg.pedagogical_state || "CONFIDENT_ADVANCE";
      setPedagogicalState(stateDirective);

      // Add to transcript
      const text = msg.text || msg.spoken_response || "";
      appendChatMessage("TUTOR", text, stateDirective, Boolean(msg.proactive || msg.is_proactive));

      // Update whiteboard if provided
      if (msg.whiteboard) {
        if (msg.whiteboard.problem) dom.problemDisplay.textContent = msg.whiteboard.problem;
        const title = msg.whiteboard.title || msg.whiteboard.whiteboard_title;
        if (title) dom.problemTitle.textContent = title;
        if (msg.whiteboard.steps) renderWhiteboardSteps(msg.whiteboard.steps);
        const diag = msg.whiteboard.visual_diagram || msg.whiteboard.diagramType || msg.whiteboard.diagram;
        if (diag) renderVisualDiagram(diag);
        const choices = msg.whiteboard.binary_choices || msg.whiteboard.binaryChoices;
        if (choices) renderBinaryChoices(choices);
        const hint = msg.whiteboard.socratic_hint || msg.whiteboard.hint;
        if (hint) dom.socraticHintText.textContent = hint;
      }

      // Latency metric
      const lat = msg.latency || msg.metrics;
      if (lat) {
        const stt = lat.stt_ms !== undefined ? `${lat.stt_ms}ms` : "--";
        const llm = lat.llm_ttft_ms !== undefined ? `${lat.llm_ttft_ms}ms` : (lat.llm_total_ms !== undefined ? `${lat.llm_total_ms}ms` : "--");
        const tts = lat.tts_ms !== undefined ? `${lat.tts_ms}ms` : "--";
        dom.latencyReadout.textContent = `STT ${stt} · LLM ${llm} · TTS ${tts}`;
      }

      // Play Kokoro TTS Audio (via gapless chunk queue when turn_id / chunked_audio is present)
      const audioB64 = msg.audio_wav_base64 || msg.audio_base64;
      if (audioB64) {
        if (msg.chunked_audio || msg.turn_id) {
          const isSingleChunk = !msg.chunked_audio || (msg.total_chunks !== undefined && msg.total_chunks <= 1);
          queueAudioChunk(msg.turn_id || "default", audioB64, isSingleChunk);
        } else {
          const speed = msg.tts_speed || (stateDirective === "FRUSTRATED_SCAFFOLD" ? 0.95 : stateDirective === "CONFUSED_HINT" ? 0.98 : 1.0);
          playTutorAudio(audioB64, speed);
        }
      }
    } else if (msg.type === "tutor_audio_chunk") {
      // Warm streaming chunked audio packet
      if (msg.audio_wav_base64) {
        queueAudioChunk(msg.turn_id || "default", msg.audio_wav_base64, Boolean(msg.is_final));
      }
    } else if (msg.type === "transcript_student") {
      if (msg.text) {
        appendChatMessage("STUDENT", msg.text, msg.pedagogical_state || state.pedagogicalState);
        stopHesitationTimer();
      }
    } else if (msg.type === "whiteboard_update") {
      if (msg.whiteboard) {
        if (msg.whiteboard.steps) renderWhiteboardSteps(msg.whiteboard.steps);
        const diag = msg.whiteboard.visual_diagram || msg.whiteboard.diagramType || msg.whiteboard.diagram;
        if (diag) renderVisualDiagram(diag);
      }
    }
  } catch (err) {
    console.error("Error parsing WebSocket message:", err);
  }
}

async function syncBackendCurriculum() {
  // 1. Sync models from /api/health
  try {
    const healthRes = await fetch("/api/health");
    if (healthRes.ok) {
      const healthData = await healthRes.json();
      const ollama = healthData?.ollama;
      if (ollama) {
        const models = ollama.supported_models || ollama.available_models;
        if (Array.isArray(models) && models.length > 0) {
          dom.modelSelect.innerHTML = "";
          for (const m of models) {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            if (m === ollama.default_model) {
              opt.selected = true;
              state.currentModel = m;
            }
            dom.modelSelect.appendChild(opt);
          }
        }
      }
    }
  } catch (err) {
    // Keep default model options
  }

  // 2. Sync topics from /api/topics
  try {
    const res = await fetch("/api/topics");
    if (res.ok) {
      const data = await res.json();
      const topicsList = Array.isArray(data) ? data : (Array.isArray(data?.topics) ? data.topics : null);
      if (topicsList && topicsList.length > 0) {
        dom.topicSelect.innerHTML = "";
        for (const t of topicsList) {
          const opt = document.createElement("option");
          opt.value = t.id;
          opt.textContent = t.label || t.title;
          dom.topicSelect.appendChild(opt);
        }
      }
    }
  } catch (err) {
    // Keep offline fallback curriculum
  }
}

function handleOfflineLocalResponse(msgObj) {
  // If backend is not running, provide an offline Socratic response after a realistic delay
  setTimeout(() => {
    let reply = "Good thinking! Let's check the next step on the whiteboard.";
    let nextState = state.pedagogicalState;

    if (nextState === "FRUSTRATED_SCAFFOLD") {
      reply = "Take your time—this step is tricky! Let's simplify: look at the balance scale. If we take away 7 from both sides, what does 3x equal?";
    } else if (nextState === "CONFUSED_HINT") {
      reply = "Notice how 3 is multiplied by x. What is the opposite operation of multiplication?";
    } else if (nextState === "DISTRACTED_REENGAGE") {
      reply = "Hey there! Ready to solve the next part? Check out the balance scale on the board.";
    }

    appendChatMessage("TUTOR", reply, nextState, false);
    startHesitationTimer();
  }, 600);
}

// --------------------------------------------------------------------------
// Session Lifecycle (Start / Stop)
// --------------------------------------------------------------------------
async function startSession() {
  state.isSessionActive = true;
  dom.sessionBtn.classList.add("btn-active-session");
  dom.sessionBtnText.textContent = "Stop Session";
  dom.sessionIconStart.classList.add("hidden");
  dom.sessionIconStop.classList.remove("hidden");
  dom.sessionStatusBadge.textContent = "Session Active";
  dom.sessionStatusBadge.className = "pill pill-live";

  // Setup webcam and audio
  await setupWebcam();
  await setupAudioPipeline();

  startHesitationTimer();

  sendWebSocketMessage({
    type: "session_config",
    topic_id: state.currentTopicId,
    model: state.currentModel,
    voice: state.currentVoice,
    stream_audio: true,
    reset: true,
    greet: true,
  });
}

function stopSession() {
  state.isSessionActive = false;
  dom.sessionBtn.classList.remove("btn-active-session");
  dom.sessionBtnText.textContent = "Start Session";
  dom.sessionIconStart.classList.remove("hidden");
  dom.sessionIconStop.classList.add("hidden");
  dom.sessionStatusBadge.textContent = "Standby";
  dom.sessionStatusBadge.className = "pill pill-idle";

  stopHesitationTimer();
  clearAudioQueue();

  if (state.micStream) {
    state.micStream.getTracks().forEach((track) => track.stop());
    state.micStream = null;
  }

  // Re-show camera fallback when session stops and simulator is unchecked
  if (!dom.simEnabled || !dom.simEnabled.checked) {
    state.simulationMode = false;
    affectEngine.simulationMode = false;
    if (dom.cameraFallback) dom.cameraFallback.classList.remove("hidden");
  }

  sendWebSocketMessage({
    type: "stop_session",
  });
}
