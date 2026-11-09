/**
 * affect_engine.js - Local Edge Affective Telemetry & 4-State Pedagogical Router.
 * 
 * Computes 3D head pose Euler angles, attentiveness, VAD-masked frustration,
 * sustain timers, landmark mesh highlights, and proactive silence interventions.
 */

export function clamp(val, lo = 0.0, hi = 1.0) {
  if (val < lo) return lo;
  if (val > hi) return hi;
  return val;
}

export function extractEulerAnglesFromMatrix(m) {
  if (!m || m.length !== 16) {
    return { yaw: 0.0, pitch: 0.0, roll: 0.0 };
  }

  // OpenGL / MediaPipe column-major detection
  const colMajorTrans = Math.abs(m[12]) + Math.abs(m[13]) + Math.abs(m[14]);
  const rowMajorTrans = Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]);

  let r00, r10, r20, r01, r11, r21, r02, r12, r22;
  if (colMajorTrans > rowMajorTrans) {
    r00 = m[0]; r10 = m[1]; r20 = m[2];
    r01 = m[4]; r11 = m[5]; r21 = m[6];
    r02 = m[8]; r12 = m[9]; r22 = m[10];
  } else {
    r00 = m[0]; r01 = m[1]; r02 = m[2];
    r10 = m[4]; r11 = m[5]; r12 = m[6];
    r20 = m[8]; r21 = m[9]; r22 = m[10];
  }

  const pitch = Math.asin(clamp(-r12, -1.0, 1.0)) * 180.0 / Math.PI;
  const yaw = Math.atan2(r02, r22) * 180.0 / Math.PI;
  const roll = Math.atan2(r10, r11) * 180.0 / Math.PI;

  return { yaw, pitch, roll };
}

// MediaPipe 478 Landmark Special Subsets
export const EYEBROW_INDICES = new Set([
  70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
  300, 293, 334, 296, 336, 285, 295, 282, 283, 276
]);

export const LIP_INDICES = new Set([
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]);

export function median(arr) {
  if (!arr || arr.length === 0) return 0.0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
}

export class AffectEngine {
  constructor(options = {}) {
    this.alpha = options.alpha || 0.15;
    this.historyWindowMs = (options.historyWindowS || 3.0) * 1000.0;
    this.maxLogs = options.maxLogs || 54000;

    // Filter states
    this.attentiveness = 1.0;
    this.frustration = 0.0;
    this.unmaskedFrustration = 0.0;

    this.rawAttentiveness = 1.0;
    this.rawFrustration = 0.0;
    this.rawUnmaskedFrustration = 0.0;

    this.yaw = 0.0;
    this.pitch = 0.0;
    this.roll = 0.0;

    this.browFurrow = 0.0;
    this.eyeSquint = 0.0;
    this.browInnerUp = 0.0;
    this.noseSneer = 0.0;
    this.dBrow = 0.0;
    this.upperFaceStrain = 0.0;
    this.lowerFaceTension = 0.0;
    this.unmaskedLowerFaceTension = 0.0;

    this.isSpeaking = false;
    this.speechMaskActive = false;
    this.lastSpeechTimeMs = -10000;
    this.hesitationSeconds = 0.0;

    this.faceX = 0.5;
    this.faceY = 0.5;
    this.gazeAway = 0.0;

    this.lowAttentivenessDuration = 0.0;
    this.highFrustrationDuration = 0.0;

    this.lastTsMs = null;
    this.lastProactiveMs = 0;
    this.awaitingStudentAnswer = true;

    // 1.5s Neutral Face Auto-Calibration state
    this.calibrated = false;
    this.isCalibrating = true;
    this.calibrationSamples = [];
    this.calibrationTargetFrames = 40; // ~1.3s at 30 FPS
    this.baseline = {
      browFurrow: 0.0,
      eyeSquint: 0.0,
      browInnerUp: 0.0,
      noseSneer: 0.0,
      mouthFrown: 0.0,
      mouthPress: 0.0,
    };
    this.onCalibrationStatusChange = null;

    // 4.0-second rolling pre-utterance affect window
    this.turnWindow = [];

    // 3-second rolling window buffer
    this.history = [];

    // Complete session telemetry log for CSV export
    this.telemetryLogs = [];

    // MediaPipe Vision instance
    this.faceLandmarker = null;
    this.landmarkerReady = false;
    this.isInitializingLandmarker = false;
    this.simulationMode = false;

    // Callbacks
    this.onProactiveHelpTrigger = null;
    this.onTelemetryFrame = null;
  }

  startCalibration() {
    this.calibrated = false;
    this.isCalibrating = true;
    this.calibrationSamples = [];
    if (typeof this.onCalibrationStatusChange === "function") {
      this.onCalibrationStatusChange({ isCalibrating: true, calibrated: false, progress: 0 });
    }
  }

  reset() {
    this.attentiveness = 1.0;
    this.frustration = 0.0;
    this.unmaskedFrustration = 0.0;
    this.rawAttentiveness = 1.0;
    this.rawFrustration = 0.0;
    this.rawUnmaskedFrustration = 0.0;
    this.yaw = 0.0;
    this.pitch = 0.0;
    this.roll = 0.0;
    this.browFurrow = 0.0;
    this.eyeSquint = 0.0;
    this.browInnerUp = 0.0;
    this.noseSneer = 0.0;
    this.dBrow = 0.0;
    this.upperFaceStrain = 0.0;
    this.lowerFaceTension = 0.0;
    this.unmaskedLowerFaceTension = 0.0;
    this.faceX = 0.5;
    this.faceY = 0.5;
    this.gazeAway = 0.0;
    this.isSpeaking = false;
    this.speechMaskActive = false;
    this.lastSpeechTimeMs = -10000;
    this.hesitationSeconds = 0.0;
    this.lowAttentivenessDuration = 0.0;
    this.highFrustrationDuration = 0.0;
    this.lastTsMs = null;
    this.lastProactiveMs = 0;
    this.history = [];
    this.turnWindow = [];
  }

  async initFaceLandmarker(
    visionBundlePath = "./vendor/mediapipe/vision_bundle.mjs",
    wasmPath = "./vendor/mediapipe/wasm",
    modelAssetPath = "./assets/face_landmarker.task"
  ) {
    if (this.landmarkerReady || this.isInitializingLandmarker) return true;
    this.isInitializingLandmarker = true;

    try {
      const vision = await import(visionBundlePath);
      const FilesetResolver = vision.FilesetResolver || window.FilesetResolver;
      const FaceLandmarker = vision.FaceLandmarker || window.FaceLandmarker;

      if (!FilesetResolver || !FaceLandmarker) {
        throw new Error("FilesetResolver or FaceLandmarker not exported from vision bundle");
      }

      const fileset = await FilesetResolver.forVisionTasks(wasmPath);

      // Attempt WebGL / GPU delegate first; fallback to CPU in headless/virtualized environments
      try {
        this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath,
            delegate: "GPU",
          },
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
      } catch (gpuError) {
        console.warn("GPU delegate creation failed; falling back to CPU delegate:", gpuError);
        this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath,
            delegate: "CPU",
          },
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
      }

      this.landmarkerReady = true;
      this.isInitializingLandmarker = false;
      return true;
    } catch (err) {
      console.error("Failed to initialize MediaPipe FaceLandmarker:", err);
      this.isInitializingLandmarker = false;
      this.landmarkerReady = false;
      return false;
    }
  }

  processVideoFrame(videoElement, canvasElement, nowMs = performance.now()) {
    if (!this.landmarkerReady || !this.faceLandmarker || this.simulationMode) {
      return null;
    }

    if (!videoElement || videoElement.readyState < 2) {
      return null;
    }

    let result = null;
    try {
      result = this.faceLandmarker.detectForVideo(videoElement, nowMs);
    } catch (err) {
      console.warn("FaceLandmarker detection error:", err);
      return null;
    }

    if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
      // Clear canvas if no face
      if (canvasElement) {
        const ctx = canvasElement.getContext("2d");
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      }
      return null;
    }

    const landmarks = result.faceLandmarks[0];
    const blendshapesCategories = result.faceBlendshapes?.[0]?.categories || [];
    const blendshapesMap = {};
    for (let i = 0; i < blendshapesCategories.length; i++) {
      const cat = blendshapesCategories[i];
      blendshapesMap[cat.categoryName] = cat.score;
    }

    // 1.5s Neutral Face Auto-Calibration (Webcam frames only, never in simulationMode)
    if (this.isCalibrating && !this.isSpeaking) {
      const sBrow = ((blendshapesMap.browDownLeft || 0.0) + (blendshapesMap.browDownRight || 0.0)) / 2.0;
      const sSquint = ((blendshapesMap.eyeSquintLeft || 0.0) + (blendshapesMap.eyeSquintRight || 0.0)) / 2.0;
      const sInner = blendshapesMap.browInnerUp || 0.0;
      const sSneer = ((blendshapesMap.noseSneerLeft || 0.0) + (blendshapesMap.noseSneerRight || 0.0)) / 2.0;
      const sFrown = ((blendshapesMap.mouthFrownLeft || 0.0) + (blendshapesMap.mouthFrownRight || 0.0)) / 2.0;
      const sPress = ((blendshapesMap.mouthPressLeft || 0.0) + (blendshapesMap.mouthPressRight || 0.0)) / 2.0;

      this.calibrationSamples.push({
        browFurrow: sBrow,
        eyeSquint: sSquint,
        browInnerUp: sInner,
        noseSneer: sSneer,
        mouthFrown: sFrown,
        mouthPress: sPress,
      });

      const progress = clamp(this.calibrationSamples.length / this.calibrationTargetFrames, 0.0, 1.0);
      if (typeof this.onCalibrationStatusChange === "function") {
        this.onCalibrationStatusChange({ isCalibrating: true, calibrated: false, progress });
      }

      if (this.calibrationSamples.length >= this.calibrationTargetFrames) {
        this.baseline = {
          browFurrow: clamp(median(this.calibrationSamples.map((s) => s.browFurrow)), 0.0, 0.22),
          eyeSquint: clamp(median(this.calibrationSamples.map((s) => s.eyeSquint)), 0.0, 0.22),
          browInnerUp: clamp(median(this.calibrationSamples.map((s) => s.browInnerUp)), 0.0, 0.22),
          noseSneer: clamp(median(this.calibrationSamples.map((s) => s.noseSneer)), 0.0, 0.22),
          mouthFrown: clamp(median(this.calibrationSamples.map((s) => s.mouthFrown)), 0.0, 0.22),
          mouthPress: clamp(median(this.calibrationSamples.map((s) => s.mouthPress)), 0.0, 0.22),
        };
        this.calibrated = true;
        this.isCalibrating = false;
        if (typeof this.onCalibrationStatusChange === "function") {
          this.onCalibrationStatusChange({ isCalibrating: false, calibrated: true, progress: 1.0 });
        }
      }
    }

    const matrix = result.facialTransformationMatrixes?.[0]?.data || null;
    const faceX = landmarks[1]?.x ?? 0.5;
    const faceY = landmarks[1]?.y ?? 0.5;

    const frame = this.computeFrameMetrics({
      blendshapesMap,
      matrix,
      faceX,
      faceY,
      isSpeaking: this.isSpeaking,
      hesitationSeconds: this.hesitationSeconds,
      nowMs,
    });

    if (canvasElement && landmarks) {
      const isBrowHighlighted =
        (this.calibrated ? this.dBrow >= 0.10 : this.browFurrow >= 0.32) ||
        this.upperFaceStrain >= 0.45;
      this.drawMesh(
        canvasElement,
        landmarks,
        isBrowHighlighted ? 0.35 : 0.0,
        this.speechMaskActive
      );
    }

    return frame;
  }

  computeFrameMetrics({
    blendshapesMap = {},
    matrix = null,
    faceX = null,
    faceY = null,
    isSpeaking = false,
    hesitationSeconds = 0.0,
    nowMs = performance.now(),
  }) {
    // 1. 3D Head Pose
    if (matrix) {
      const euler = extractEulerAnglesFromMatrix(matrix);
      this.yaw = euler.yaw;
      this.pitch = euler.pitch;
      this.roll = euler.roll;
    }

    this.faceX = faceX !== null ? faceX : clamp(0.5 - this.yaw / 70.0, 0.05, 0.95);
    this.faceY = faceY !== null ? faceY : clamp(0.5 + this.pitch / 50.0, 0.05, 0.95);

    // 2. Attentiveness
    const gazeAway = Math.max(
      blendshapesMap.eyeLookOutLeft || 0.0,
      blendshapesMap.eyeLookOutRight || 0.0,
      blendshapesMap.eyeLookInLeft || 0.0,
      blendshapesMap.eyeLookInRight || 0.0,
      blendshapesMap.eyeLookUpLeft || 0.0,
      blendshapesMap.eyeLookDownLeft || 0.0
    );
    this.gazeAway = gazeAway;

    const headPenalty =
      clamp((Math.abs(this.yaw) - 12.0) / 25.0, 0.0, 1.0) * 0.6 +
      clamp((Math.abs(this.pitch) - 10.0) / 20.0, 0.0, 1.0) * 0.4;
    const gazePenalty = clamp((gazeAway - 0.35) / 0.45, 0.0, 1.0);
    this.rawAttentiveness = clamp(1.0 - Math.max(headPenalty, gazePenalty * 0.8), 0.0, 1.0);

    // 3. VAD-Masked Frustration with Calibrated Baseline Subtraction
    const rawBrowFurrow =
      ((blendshapesMap.browDownLeft || 0.0) + (blendshapesMap.browDownRight || 0.0)) / 2.0;
    const rawEyeSquint =
      ((blendshapesMap.eyeSquintLeft || 0.0) + (blendshapesMap.eyeSquintRight || 0.0)) / 2.0;
    const rawBrowInnerUp = blendshapesMap.browInnerUp || 0.0;
    const rawNoseSneer =
      ((blendshapesMap.noseSneerLeft || 0.0) + (blendshapesMap.noseSneerRight || 0.0)) / 2.0;
    const rawMouthFrown =
      ((blendshapesMap.mouthFrownLeft || 0.0) + (blendshapesMap.mouthFrownRight || 0.0)) / 2.0;
    const rawMouthPress =
      ((blendshapesMap.mouthPressLeft || 0.0) + (blendshapesMap.mouthPressRight || 0.0)) / 2.0;

    this.browFurrow = rawBrowFurrow;
    this.eyeSquint = rawEyeSquint;
    this.browInnerUp = rawBrowInnerUp;
    this.noseSneer = rawNoseSneer;

    if (isSpeaking) {
      this.lastSpeechTimeMs = nowMs;
    }
    const timeSinceLastSpeechMs = nowMs - this.lastSpeechTimeMs;
    this.speechMaskActive = Boolean(isSpeaking || timeSinceLastSpeechMs <= 300.0);
    this.isSpeaking = Boolean(isSpeaking);
    this.hesitationSeconds = Math.max(0.0, Number(hesitationSeconds) || 0.0);

    // Delas above resting baseline (clamped at 0)
    const dBrow = Math.max(0.0, rawBrowFurrow - this.baseline.browFurrow);
    const dSquint = Math.max(0.0, rawEyeSquint - this.baseline.eyeSquint);
    const dInner = Math.max(0.0, rawBrowInnerUp - this.baseline.browInnerUp);
    const dSneer = Math.max(0.0, rawNoseSneer - this.baseline.noseSneer);
    const dFrown = Math.max(0.0, rawMouthFrown - this.baseline.mouthFrown);
    const dPress = Math.max(0.0, rawMouthPress - this.baseline.mouthPress);
    this.dBrow = dBrow;

    // Attenuate browInnerUp during speech (AU1 moves when talking)
    const innerWeight = this.speechMaskActive ? 0.15 : 0.75;
    this.upperFaceStrain = clamp(
      dBrow * 2.6 + dSquint * 0.95 + dSneer * 1.2 + dInner * innerWeight,
      0.0,
      1.0
    );

    this.unmaskedLowerFaceTension = clamp(dFrown * 1.85 + dPress * 1.65, 0.0, 1.0);
    this.lowerFaceTension = this.speechMaskActive ? 0.0 : this.unmaskedLowerFaceTension;
    const hesitationFactor = clamp((this.hesitationSeconds - 3.0) / 8.0, 0.0, 1.0);

    // Weight re-normalization:
    // When speech mask active: weights sum to 1.0 (0.67 + 0.33)
    // When inactive: weights sum to 1.0 (0.50 + 0.25 + 0.25)
    if (this.speechMaskActive) {
      this.rawFrustration = clamp(
        0.67 * this.upperFaceStrain + 0.33 * hesitationFactor,
        0.0,
        1.0
      );
    } else {
      this.rawFrustration = clamp(
        0.50 * this.upperFaceStrain + 0.25 * this.lowerFaceTension + 0.25 * hesitationFactor,
        0.0,
        1.0
      );
    }

    this.rawUnmaskedFrustration = clamp(
      0.50 * this.upperFaceStrain + 0.25 * this.unmaskedLowerFaceTension + 0.25 * hesitationFactor,
      0.0,
      1.0
    );

    // 4. Exponential Moving Average Smoothing (alpha = 0.15)
    this.attentiveness = (1.0 - this.alpha) * this.attentiveness + this.alpha * this.rawAttentiveness;
    this.frustration = (1.0 - this.alpha) * this.frustration + this.alpha * this.rawFrustration;
    this.unmaskedFrustration =
      (1.0 - this.alpha) * this.unmaskedFrustration + this.alpha * this.rawUnmaskedFrustration;

    // 5. Update Continuous Sustain Durations (dt)
    let dt = 0.033;
    if (this.lastTsMs !== null) {
      dt = Math.max(0.0, (nowMs - this.lastTsMs) / 1000.0);
      if (dt > 1.0) dt = 0.1; // Guard against tab sleep / suspension
    }
    this.lastTsMs = nowMs;

    if (this.attentiveness < 0.45) {
      this.lowAttentivenessDuration += dt;
    } else {
      this.lowAttentivenessDuration = 0.0;
    }

    if (this.frustration >= 0.62) {
      this.highFrustrationDuration += dt;
    } else {
      this.highFrustrationDuration = 0.0;
    }

    // 6. Rolling 3-second history maintainer
    this.history.push({ tsMs: nowMs, attentiveness: this.attentiveness, frustration: this.frustration });
    const cutoff = nowMs - this.historyWindowMs;
    while (this.history.length > 0 && this.history[0].tsMs < cutoff) {
      this.history.shift();
    }

    // 4.0-second rolling pre-utterance turn window
    this.turnWindow.push({
      tsMs: nowMs,
      attentiveness: this.attentiveness,
      frustration: this.frustration,
      unmaskedFrustration: this.unmaskedFrustration,
      hesitationSeconds: this.hesitationSeconds,
      upperFaceStrain: this.upperFaceStrain,
      lowerFaceTension: this.lowerFaceTension,
      isSpeaking: this.isSpeaking,
      speechMaskActive: this.speechMaskActive,
      dBrow: this.dBrow,
    });
    const turnCutoff = nowMs - 4000.0;
    while (this.turnWindow.length > 0 && this.turnWindow[0].tsMs < turnCutoff) {
      this.turnWindow.shift();
    }

    // 7. Four-State Pedagogical Router
    const pedagogicalState = this.evaluatePedagogicalState();

    // 8. Proactive Silence Intervention Check (>= 0.58 frustration)
    if (
      this.awaitingStudentAnswer &&
      !this.isSpeaking &&
      this.hesitationSeconds > 6.5 &&
      this.frustration >= 0.58 &&
      nowMs - this.lastProactiveMs >= 20000.0
    ) {
      this.lastProactiveMs = nowMs;
      if (typeof this.onProactiveHelpTrigger === "function") {
        this.onProactiveHelpTrigger(this.getTelemetrySnapshot(pedagogicalState, nowMs));
      }
    }

    const frameRecord = {
      timestamp: new Date().toISOString(),
      tsMs: Math.round(nowMs),
      attentiveness: Number(this.attentiveness.toFixed(4)),
      rawAttentiveness: Number(this.rawAttentiveness.toFixed(4)),
      frustration: Number(this.frustration.toFixed(4)),
      unmaskedFrustration: Number(this.unmaskedFrustration.toFixed(4)),
      suppressedDelta: Number(Math.max(0, this.unmaskedFrustration - this.frustration).toFixed(4)),
      yaw: Number(this.yaw.toFixed(2)),
      pitch: Number(this.pitch.toFixed(2)),
      roll: Number(this.roll.toFixed(2)),
      faceX: Number(this.faceX.toFixed(4)),
      faceY: Number(this.faceY.toFixed(4)),
      gazeAway: Number(this.gazeAway.toFixed(4)),
      browFurrow: Number(this.browFurrow.toFixed(4)),
      eyeSquint: Number(this.eyeSquint.toFixed(4)),
      upperFaceStrain: Number(this.upperFaceStrain.toFixed(4)),
      lowerFaceTension: Number(this.lowerFaceTension.toFixed(4)),
      unmaskedLowerFaceTension: Number(this.unmaskedLowerFaceTension.toFixed(4)),
      hesitationSeconds: Number(this.hesitationSeconds.toFixed(2)),
      isSpeaking: this.isSpeaking,
      speechMaskActive: this.speechMaskActive,
      lowAttentivenessDuration: Number(this.lowAttentivenessDuration.toFixed(2)),
      highFrustrationDuration: Number(this.highFrustrationDuration.toFixed(2)),
      state: pedagogicalState,
    };

    // Store in exportable telemetry array
    this.telemetryLogs.push(frameRecord);
    if (this.telemetryLogs.length > this.maxLogs) {
      this.telemetryLogs.shift();
    }

    if (typeof this.onTelemetryFrame === "function") {
      this.onTelemetryFrame(frameRecord);
    }

    return frameRecord;
  }

  evaluatePedagogicalState() {
    // 1. Distracted Re-engage: attentiveness < 0.45 continuously for > 2.5s
    if (this.lowAttentivenessDuration > 2.5) {
      return "DISTRACTED_REENGAGE";
    }

    // 2. Frustrated Scaffold: frustration >= 0.62 continuously for > 2.0s
    if (this.highFrustrationDuration > 2.0) {
      return "FRUSTRATED_SCAFFOLD";
    }

    // 3. Confused Hint: moderate frustration, short spike, or extended hesitation
    if (
      (this.frustration >= 0.45 && this.frustration < 0.62) ||
      (this.frustration >= 0.62 && this.highFrustrationDuration <= 2.0) ||
      this.hesitationSeconds > 5.0
    ) {
      return "CONFUSED_HINT";
    }

    // 4. Confident Advance: default state
    return "CONFIDENT_ADVANCE";
  }

  getTurnAffectSnapshot(nowMs = performance.now()) {
    const windowFrames = this.turnWindow.length > 0 ? this.turnWindow : [
      {
        attentiveness: this.attentiveness,
        frustration: this.frustration,
        unmaskedFrustration: this.unmaskedFrustration,
        hesitationSeconds: this.hesitationSeconds,
      },
    ];

    const frustList = windowFrames.map((f) => f.frustration).sort((a, b) => a - b);
    const p90Idx = Math.floor(frustList.length * 0.9);
    const p90Frustration = frustList[Math.min(p90Idx, frustList.length - 1)] || this.frustration;

    const attentList = windowFrames.map((f) => f.attentiveness).sort((a, b) => a - b);
    const p10Idx = Math.floor(attentList.length * 0.1);
    const p10Attentiveness = attentList[Math.min(p10Idx, attentList.length - 1)] || this.attentiveness;

    const frustratedTimeS = (windowFrames.filter((f) => f.frustration >= 0.58).length * 0.033);

    // Latch pedagogical state across the window:
    // If high frustration sustained >= 1.5s or p90 >= 0.62 -> FRUSTRATED_SCAFFOLD
    // Else if low attentiveness p10 < 0.45 or duration >= 2.0s -> DISTRACTED_REENGAGE
    // Else if p90 >= 0.46 or max hesitation >= 5.0s -> CONFUSED_HINT
    // Else -> CONFIDENT_ADVANCE
    let latchedState = "CONFIDENT_ADVANCE";
    const maxHesitation = Math.max(...windowFrames.map((f) => f.hesitationSeconds || 0.0));

    if (this.highFrustrationDuration >= 1.5 || p90Frustration >= 0.62 || frustratedTimeS >= 1.5) {
      latchedState = "FRUSTRATED_SCAFFOLD";
    } else if (this.lowAttentivenessDuration >= 2.0 || p10Attentiveness < 0.45) {
      latchedState = "DISTRACTED_REENGAGE";
    } else if (p90Frustration >= 0.46 || maxHesitation >= 5.0) {
      latchedState = "CONFUSED_HINT";
    }

    return {
      timestamp: new Date().toISOString(),
      tsMs: Math.round(nowMs),
      p90_frustration: Number(p90Frustration.toFixed(4)),
      p10_attentiveness: Number(p10Attentiveness.toFixed(4)),
      frustration: Number(p90Frustration.toFixed(4)),
      attentiveness: Number(p10Attentiveness.toFixed(4)),
      unmasked_frustration: Number(this.unmaskedFrustration.toFixed(4)),
      frustrated_duration_s: Number(frustratedTimeS.toFixed(2)),
      high_frustration_duration: Number(this.highFrustrationDuration.toFixed(2)),
      low_attentiveness_duration: Number(this.lowAttentivenessDuration.toFixed(2)),
      hesitation_seconds: Number(maxHesitation.toFixed(2)),
      latched_pedagogical_state: latchedState,
      pedagogical_state: latchedState,
      calibrated: this.calibrated,
    };
  }

  getTelemetrySnapshot(state = null, nowMs = performance.now()) {
    return {
      timestamp: new Date().toISOString(),
      tsMs: Math.round(nowMs),
      attentiveness: this.attentiveness,
      frustration: this.frustration,
      unmaskedFrustration: this.unmaskedFrustration,
      yaw: this.yaw,
      pitch: this.pitch,
      roll: this.roll,
      faceX: this.faceX,
      faceY: this.faceY,
      gazeAway: this.gazeAway,
      browFurrow: this.browFurrow,
      speechMaskActive: this.speechMaskActive,
      isSpeaking: this.isSpeaking,
      hesitationSeconds: this.hesitationSeconds,
      highFrustrationDuration: this.highFrustrationDuration,
      lowAttentivenessDuration: this.lowAttentivenessDuration,
      calibrated: this.calibrated,
      isCalibrating: this.isCalibrating,
      state: state || this.evaluatePedagogicalState(),
    };
  }

  drawMesh(canvas, landmarks, browFurrow, speechMaskActive) {
    if (!canvas || !landmarks) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const isBrowSpike = browFurrow >= 0.32;
    const isMaskActive = speechMaskActive;

    // Draw baseline subtle facial contour points
    ctx.fillStyle = "rgba(34, 211, 238, 0.45)";
    for (let i = 0; i < landmarks.length; i++) {
      if (EYEBROW_INDICES.has(i) || LIP_INDICES.has(i)) {
        continue;
      }
      const pt = landmarks[i];
      const x = pt.x * width;
      const y = pt.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Eyebrow landmarks: highlight bright amber when browFurrow >= 0.32 (FACS AU4)
    if (isBrowSpike) {
      ctx.save();
      ctx.fillStyle = "#f59e0b";
      ctx.shadowColor = "rgba(245, 158, 11, 0.85)";
      ctx.shadowBlur = 6;
      for (const idx of EYEBROW_INDICES) {
        const pt = landmarks[idx];
        if (pt) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 2.6, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.restore();
    } else {
      ctx.fillStyle = "rgba(34, 211, 238, 0.75)";
      for (const idx of EYEBROW_INDICES) {
        const pt = landmarks[idx];
        if (pt) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 1.5, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // Lip landmarks: highlight bright emerald green when speechMaskActive is true
    if (isMaskActive) {
      ctx.save();
      ctx.fillStyle = "#10b981";
      ctx.shadowColor = "rgba(16, 185, 129, 0.85)";
      ctx.shadowBlur = 6;
      for (const idx of LIP_INDICES) {
        const pt = landmarks[idx];
        if (pt) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 2.4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.restore();
    } else {
      ctx.fillStyle = "rgba(34, 211, 238, 0.7)";
      for (const idx of LIP_INDICES) {
        const pt = landmarks[idx];
        if (pt) {
          ctx.beginPath();
          ctx.arc(pt.x * width, pt.y * height, 1.4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
  }

  drawSimulatedFace(canvas, params = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const yaw = params.yaw || 0.0;
    const pitch = params.pitch || 0.0;
    const browFurrow = params.browFurrow || 0.0;
    const mouthTension = params.mouthTension || 0.0;
    const isSpeaking = Boolean(params.isSpeaking);
    const speechMaskActive = Boolean(params.speechMaskActive);

    const cx = width / 2 + (yaw / 40.0) * (width * 0.18);
    const cy = height / 2 + (pitch / 30.0) * (height * 0.15);

    ctx.save();
    // Subtle background mesh grid
    ctx.strokeStyle = "rgba(34, 211, 238, 0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(cx - 65, cy - 85, 130, 170);

    // Oval face wireframe
    ctx.strokeStyle = "rgba(34, 211, 238, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, cy, 60, 80, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Eyes
    ctx.fillStyle = "rgba(34, 211, 238, 0.85)";
    ctx.beginPath();
    ctx.arc(cx - 24, cy - 14, 4, 0, 2 * Math.PI);
    ctx.arc(cx + 24, cy - 14, 4, 0, 2 * Math.PI);
    ctx.fill();

    // Eyebrows (sloped by browFurrow)
    const browColor = browFurrow >= 0.32 ? "#f59e0b" : "rgba(34, 211, 238, 0.85)";
    const browAngle = (browFurrow * 18 * Math.PI) / 180;
    ctx.strokeStyle = browColor;
    ctx.lineWidth = browFurrow >= 0.32 ? 3.0 : 2.0;

    // Left Brow
    ctx.beginPath();
    ctx.moveTo(cx - 38, cy - 28 + Math.sin(browAngle) * 8);
    ctx.lineTo(cx - 12, cy - 28 - Math.sin(browAngle) * 8);
    ctx.stroke();

    // Right Brow
    ctx.beginPath();
    ctx.moveTo(cx + 12, cy - 28 - Math.sin(browAngle) * 8);
    ctx.lineTo(cx + 38, cy - 28 + Math.sin(browAngle) * 8);
    ctx.stroke();

    // Nose line
    ctx.strokeStyle = "rgba(34, 211, 238, 0.4)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 6);
    ctx.lineTo(cx, cy + 14);
    ctx.lineTo(cx + 5, cy + 18);
    ctx.stroke();

    // Mouth
    const lipColor = speechMaskActive ? "#10b981" : "rgba(34, 211, 238, 0.85)";
    ctx.strokeStyle = lipColor;
    ctx.fillStyle = speechMaskActive ? "rgba(16, 185, 129, 0.25)" : "transparent";
    ctx.lineWidth = speechMaskActive ? 2.8 : 1.8;

    ctx.beginPath();
    if (isSpeaking) {
      // Open animated talking mouth
      const openH = 6 + Math.abs(Math.sin(performance.now() * 0.015)) * 10;
      ctx.ellipse(cx, cy + 42, 18, openH, 0, 0, 2 * Math.PI);
    } else {
      // Frown curve pulled down by mouthTension
      const curveY = cy + 42 + mouthTension * 10;
      ctx.moveTo(cx - 20, cy + 42);
      ctx.quadraticCurveTo(cx, curveY, cx + 20, cy + 42);
    }
    ctx.stroke();
    if (isSpeaking) ctx.fill();

    ctx.restore();
  }

  exportTelemetryCSV() {
    if (!this.telemetryLogs || this.telemetryLogs.length === 0) {
      return null;
    }

    const headers = [
      "timestamp",
      "tsMs",
      "attentiveness",
      "rawAttentiveness",
      "frustration",
      "unmaskedFrustration",
      "suppressedDelta",
      "yaw",
      "pitch",
      "roll",
      "browFurrow",
      "eyeSquint",
      "upperFaceStrain",
      "lowerFaceTension",
      "unmaskedLowerFaceTension",
      "hesitationSeconds",
      "isSpeaking",
      "speechMaskActive",
      "lowAttentivenessDuration",
      "highFrustrationDuration",
      "state",
    ];

    const rows = [headers.join(",")];
    for (let i = 0; i < this.telemetryLogs.length; i++) {
      const log = this.telemetryLogs[i];
      const line = [
        log.timestamp,
        log.tsMs,
        log.attentiveness,
        log.rawAttentiveness,
        log.frustration,
        log.unmaskedFrustration,
        log.suppressedDelta,
        log.yaw,
        log.pitch,
        log.roll,
        log.browFurrow,
        log.eyeSquint,
        log.upperFaceStrain,
        log.lowerFaceTension,
        log.unmaskedLowerFaceTension,
        log.hesitationSeconds,
        log.isSpeaking ? 1 : 0,
        log.speechMaskActive ? 1 : 0,
        log.lowAttentivenessDuration,
        log.highFrustrationDuration,
        `"${log.state}"`,
      ].join(",");
      rows.push(line);
    }

    const csvContent = rows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const nowStr = new Date().toISOString().replace(/[:.]/g, "-");
    link.setAttribute("href", url);
    link.setAttribute("download", `aitutor_telemetry_${nowStr}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    return csvContent;
  }
}
