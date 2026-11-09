/**
 * avatar.js - Expressive Vector SVG Robot Tutor Embodiment.
 *
 * Renders an animated SVG Robot Head inside #avatar-stage:
 * 1. Dark titanium visor shell with status beacon and ear pods
 * 2. Glowing cyan OLED eyes that smoothly track student webcam gaze or yaw
 * 3. Articulated eyebrows morphing across the 4 pedagogical states
 * 4. Procedural head tilt (CONFUSED_HINT), gentle empathetic nod (FRUSTRATED_SCAFFOLD),
 *    and articulated waving hand actuator (DISTRACTED_REENGAGE)
 * 5. 9-bar OLED mouth viseme equalizer driven by WebAudio AnalyserNode
 */

export class RobotAvatar {
  constructor(container) {
    this.container = typeof container === "string" ? document.querySelector(container) : container;
    if (!this.container) {
      throw new Error("RobotAvatar container element not found");
    }

    this.state = "CONFIDENT_ADVANCE"; // CONFIDENT_ADVANCE | CONFUSED_HINT | FRUSTRATED_SCAFFOLD | DISTRACTED_REENGAGE
    this.analyserNode = null;
    this.frequencyData = new Uint8Array(32);
    this.animFrameId = null;

    // Smooth interpolation targets & current values
    this.gaze = { x: 0.0, y: 0.0, targetX: 0.0, targetY: 0.0 };
    this.headTilt = { current: 0.0, target: 0.0 };
    this.headNod = { current: 0.0, target: 0.0 };
    this.eyeScale = { current: 1.0, target: 1.0 };
    this.waveHand = { angle: 0.0, raise: 0.0, targetRaise: 0.0 };

    // Brow state targets
    this.brows = {
      leftAngle: -8,
      leftY: -2,
      rightAngle: 8,
      rightY: -2,
      targetLeftAngle: -8,
      targetLeftY: -2,
      targetRightAngle: 8,
      targetRightY: -2,
      color: "#22d3ee",
      targetColor: "#22d3ee",
    };

    // Mouth equalizer 9 bars
    this.mouthBars = [4, 5, 6, 8, 10, 8, 6, 5, 4];

    this.renderSvg();
    this.cacheDOMElements();
    this.startAnimationLoop();
  }

  renderSvg() {
    this.container.innerHTML = `
      <svg id="robot-avatar-svg" viewBox="0 0 320 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AI Tutor Robot Avatar">
        <defs>
          <!-- Shell Metallic Gradients -->
          <linearGradient id="shell-metal" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#334155"/>
            <stop offset="45%" stop-color="#1e293b"/>
            <stop offset="100%" stop-color="#0f172a"/>
          </linearGradient>

          <linearGradient id="visor-glass" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#090d16"/>
            <stop offset="100%" stop-color="#020617"/>
          </linearGradient>

          <radialGradient id="eye-glow-cyan" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#67e8f9" stop-opacity="0.9"/>
            <stop offset="60%" stop-color="#06b6d4" stop-opacity="0.5"/>
            <stop offset="100%" stop-color="#0891b2" stop-opacity="0"/>
          </radialGradient>

          <!-- Glow Filters -->
          <filter id="cyan-bloom" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3.5" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>

        <!-- Top Antenna Beacon Stem -->
        <line x1="160" y1="36" x2="160" y2="18" stroke="#475569" stroke-width="3" stroke-linecap="round"/>
        <!-- Top Beacon Light -->
        <circle id="beacon-light" cx="160" cy="16" r="6" fill="#22d3ee" filter="url(#cyan-bloom)"/>

        <!-- Articulated Right Waving Hand Element -->
        <g id="wave-hand-group" transform="translate(288, 140) rotate(15)">
          <rect x="-8" y="-28" width="16" height="32" rx="6" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
          <circle cx="0" cy="-32" r="9" fill="#334155" stroke="#64748b" stroke-width="1.5"/>
          <line x1="-5" y1="-32" x2="5" y2="-32" stroke="#22d3ee" stroke-width="2"/>
        </g>

        <!-- Left Ear Pod -->
        <g id="left-ear-pod">
          <rect x="42" y="86" width="18" height="52" rx="7" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
          <line x1="46" y1="102" x2="56" y2="102" stroke="#64748b" stroke-width="1.5"/>
          <line x1="46" y1="112" x2="56" y2="112" stroke="#64748b" stroke-width="1.5"/>
          <line x1="46" y1="122" x2="56" y2="122" stroke="#64748b" stroke-width="1.5"/>
        </g>

        <!-- Right Ear Pod -->
        <g id="right-ear-pod">
          <rect x="260" y="86" width="18" height="52" rx="7" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
          <line x1="264" y1="102" x2="274" y2="102" stroke="#64748b" stroke-width="1.5"/>
          <line x1="264" y1="112" x2="274" y2="112" stroke="#64748b" stroke-width="1.5"/>
          <line x1="264" y1="122" x2="274" y2="122" stroke="#64748b" stroke-width="1.5"/>
        </g>

        <!-- Main Robot Head Group (Procedural Tilt & Nod applied here) -->
        <g id="head-assembly" transform="translate(0, 0)">
          <!-- Titanium Outer Shell -->
          <rect x="58" y="34" width="204" height="162" rx="38" fill="url(#shell-metal)" stroke="#475569" stroke-width="2"/>

          <!-- Subtle Visor Brow Edge Accent -->
          <path d="M 80 48 Q 160 42 240 48" stroke="#64748b" stroke-width="1.5" fill="none" opacity="0.6"/>

          <!-- OLED Visor Screen -->
          <rect x="74" y="48" width="172" height="134" rx="26" fill="url(#visor-glass)" stroke="#1e293b" stroke-width="1.5"/>

          <!-- Ambient Screen Scanline Grid (subtle decorative) -->
          <line x1="84" y1="80" x2="236" y2="80" stroke="rgba(34,211,238,0.06)" stroke-width="1"/>
          <line x1="84" y1="114" x2="236" y2="114" stroke="rgba(34,211,238,0.06)" stroke-width="1"/>
          <line x1="84" y1="148" x2="236" y2="148" stroke="rgba(34,211,238,0.06)" stroke-width="1"/>

          <!-- Left Eyebrow -->
          <g id="left-brow-group" transform="translate(112, 70) rotate(-8)">
            <line x1="-22" y1="0" x2="22" y2="0" stroke="#22d3ee" stroke-width="3.5" stroke-linecap="round" filter="url(#cyan-bloom)"/>
          </g>

          <!-- Right Eyebrow -->
          <g id="right-brow-group" transform="translate(208, 70) rotate(8)">
            <line x1="-22" y1="0" x2="22" y2="0" stroke="#22d3ee" stroke-width="3.5" stroke-linecap="round" filter="url(#cyan-bloom)"/>
          </g>

          <!-- Left Eye Assembly -->
          <g id="left-eye-group" transform="translate(112, 98)">
            <!-- Eye Socket Glow Aperture -->
            <rect x="-24" y="-18" width="48" height="36" rx="14" fill="#0f172a" stroke="rgba(34,211,238,0.3)" stroke-width="1.5"/>
            <!-- Glowing Inner Iris -->
            <rect x="-18" y="-13" width="36" height="26" rx="10" fill="url(#eye-glow-cyan)" filter="url(#cyan-bloom)"/>
            <!-- Pupil + Specular Highlight -->
            <g id="left-pupil" transform="translate(0, 0)">
              <circle cx="0" cy="0" r="7.5" fill="#22d3ee"/>
              <circle cx="2.5" cy="-2.5" r="2.5" fill="#ffffff"/>
            </g>
          </g>

          <!-- Right Eye Assembly -->
          <g id="right-eye-group" transform="translate(208, 98)">
            <!-- Eye Socket Glow Aperture -->
            <rect x="-24" y="-18" width="48" height="36" rx="14" fill="#0f172a" stroke="rgba(34,211,238,0.3)" stroke-width="1.5"/>
            <!-- Glowing Inner Iris -->
            <rect x="-18" y="-13" width="36" height="26" rx="10" fill="url(#eye-glow-cyan)" filter="url(#cyan-bloom)"/>
            <!-- Pupil + Specular Highlight -->
            <g id="right-pupil" transform="translate(0, 0)">
              <circle cx="0" cy="0" r="7.5" fill="#22d3ee"/>
              <circle cx="2.5" cy="-2.5" r="2.5" fill="#ffffff"/>
            </g>
          </g>

          <!-- 9-Bar OLED Mouth Viseme Equalizer -->
          <g id="mouth-bars-group" transform="translate(112, 154)">
            <rect id="mbar-0" x="0" y="-4" width="6" height="8" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-1" x="12" y="-5" width="6" height="10" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-2" x="24" y="-7" width="6" height="14" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-3" x="36" y="-9" width="6" height="18" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-4" x="48" y="-11" width="6" height="22" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-5" x="60" y="-9" width="6" height="18" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-6" x="72" y="-7" width="6" height="14" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-7" x="84" y="-5" width="6" height="10" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
            <rect id="mbar-8" x="96" y="-4" width="6" height="8" rx="3" fill="#22d3ee" filter="url(#cyan-bloom)"/>
          </g>
        </g>
      </svg>
    `;
  }

  cacheDOMElements() {
    this.elHead = this.container.querySelector("#head-assembly");
    this.elBeacon = this.container.querySelector("#beacon-light");
    this.elLeftBrow = this.container.querySelector("#left-brow-group");
    this.elRightBrow = this.container.querySelector("#right-brow-group");
    this.elLeftBrowLine = this.container.querySelector("#left-brow-group line");
    this.elRightBrowLine = this.container.querySelector("#right-brow-group line");
    this.elLeftEye = this.container.querySelector("#left-eye-group");
    this.elRightEye = this.container.querySelector("#right-eye-group");
    this.elLeftPupil = this.container.querySelector("#left-pupil");
    this.elRightPupil = this.container.querySelector("#right-pupil");
    this.elWaveHand = this.container.querySelector("#wave-hand-group");

    this.elMouthBars = [];
    for (let i = 0; i < 9; i++) {
      this.elMouthBars.push(this.container.querySelector(`#mbar-${i}`));
    }
  }

  connectAudioAnalyser(analyser) {
    this.analyserNode = analyser;
  }

  setState(newState) {
    this.state = newState;

    switch (newState) {
      case "CONFIDENT_ADVANCE":
        this.brows.targetLeftAngle = -8;
        this.brows.targetLeftY = -2;
        this.brows.targetRightAngle = 8;
        this.brows.targetRightY = -2;
        this.brows.targetColor = "#22d3ee";
        this.headTilt.target = 0.0;
        this.eyeScale.target = 1.0;
        this.waveHand.targetRaise = 0.0;
        if (this.elBeacon) this.elBeacon.setAttribute("fill", "#22d3ee");
        break;

      case "CONFUSED_HINT":
        // Asymmetric curious brow lift + 7deg procedural head tilt
        this.brows.targetLeftAngle = -18;
        this.brows.targetLeftY = -8;
        this.brows.targetRightAngle = 4;
        this.brows.targetRightY = 3;
        this.brows.targetColor = "#38bdf8";
        this.headTilt.target = 7.0; // 7 deg curious tilt
        this.eyeScale.target = 1.05;
        this.waveHand.targetRaise = 0.0;
        if (this.elBeacon) this.elBeacon.setAttribute("fill", "#38bdf8");
        break;

      case "FRUSTRATED_SCAFFOLD":
        // Empathetic brow slope + warm amber/teal accent glow + procedural nod
        this.brows.targetLeftAngle = 12;
        this.brows.targetLeftY = 4;
        this.brows.targetRightAngle = -12;
        this.brows.targetRightY = 4;
        this.brows.targetColor = "#f59e0b"; // warm amber
        this.headTilt.target = 0.0;
        this.eyeScale.target = 0.95;
        this.waveHand.targetRaise = 0.0;
        if (this.elBeacon) this.elBeacon.setAttribute("fill", "#f59e0b");
        break;

      case "DISTRACTED_REENGAGE":
        // Wide alert eyes + raised brows + waving actuator hand
        this.brows.targetLeftAngle = -5;
        this.brows.targetLeftY = -9;
        this.brows.targetRightAngle = 5;
        this.brows.targetRightY = -9;
        this.brows.targetColor = "#38bdf8";
        this.headTilt.target = 0.0;
        this.eyeScale.target = 1.18; // alert wide eyes
        this.waveHand.targetRaise = 1.0; // hand raised and waving
        if (this.elBeacon) this.elBeacon.setAttribute("fill", "#38bdf8");
        break;

      default:
        break;
    }
  }

  setTargetGaze(normalizedX, normalizedY, yawDegrees = 0.0) {
    if (this.state === "DISTRACTED_REENGAGE") {
      // In distracted re-engage mode, look deliberately toward the student's yaw direction!
      const yawDirectionX = Math.max(-1.0, Math.min(1.0, yawDegrees / 30.0));
      this.gaze.targetX = yawDirectionX * 10.0;
      this.gaze.targetY = 0.0;
    } else {
      // Smoothly track webcam face coordinate (-1 to +1)
      const clampedX = Math.max(-1.0, Math.min(1.0, normalizedX));
      const clampedY = Math.max(-1.0, Math.min(1.0, normalizedY));
      this.gaze.targetX = clampedX * 8.0;
      this.gaze.targetY = clampedY * 6.0;
    }
  }

  startAnimationLoop() {
    const render = (time) => {
      this.update(time);
      this.animFrameId = requestAnimationFrame(render);
    };
    this.animFrameId = requestAnimationFrame(render);
  }

  update(time) {
    const tSec = time * 0.001;

    // 1. Lerp Gaze Pupils
    const lerpFactor = 0.12;
    this.gaze.x += (this.gaze.targetX - this.gaze.x) * lerpFactor;
    this.gaze.y += (this.gaze.targetY - this.gaze.y) * lerpFactor;

    if (this.elLeftPupil) {
      this.elLeftPupil.setAttribute("transform", `translate(${this.gaze.x.toFixed(2)}, ${this.gaze.y.toFixed(2)})`);
    }
    if (this.elRightPupil) {
      this.elRightPupil.setAttribute("transform", `translate(${this.gaze.x.toFixed(2)}, ${this.gaze.y.toFixed(2)})`);
    }

    // 2. Lerp Eyebrows
    const browLerp = 0.1;
    this.brows.leftAngle += (this.brows.targetLeftAngle - this.brows.leftAngle) * browLerp;
    this.brows.leftY += (this.brows.targetLeftY - this.brows.leftY) * browLerp;
    this.brows.rightAngle += (this.brows.targetRightAngle - this.brows.rightAngle) * browLerp;
    this.brows.rightY += (this.brows.targetRightY - this.brows.rightY) * browLerp;

    if (this.elLeftBrow) {
      this.elLeftBrow.setAttribute(
        "transform",
        `translate(112, ${(70 + this.brows.leftY).toFixed(2)}) rotate(${this.brows.leftAngle.toFixed(2)})`
      );
      this.elLeftBrowLine.setAttribute("stroke", this.brows.targetColor);
    }
    if (this.elRightBrow) {
      this.elRightBrow.setAttribute(
        "transform",
        `translate(208, ${(70 + this.brows.rightY).toFixed(2)}) rotate(${this.brows.rightAngle.toFixed(2)})`
      );
      this.elRightBrowLine.setAttribute("stroke", this.brows.targetColor);
    }

    // 3. Eye Scale
    this.eyeScale.current += (this.eyeScale.target - this.eyeScale.current) * 0.08;
    if (this.elLeftEye && this.elRightEye) {
      const s = this.eyeScale.current.toFixed(3);
      this.elLeftEye.setAttribute("transform", `translate(112, 98) scale(${s}) translate(-112, -98) translate(112, 98)`);
      this.elRightEye.setAttribute("transform", `translate(208, 98) scale(${s}) translate(-208, -98) translate(208, 98)`);
    }

    // 4. Head Tilt & Empathetic Nod
    this.headTilt.current += (this.headTilt.target - this.headTilt.current) * 0.08;

    let nodY = 0.0;
    if (this.state === "FRUSTRATED_SCAFFOLD") {
      // Gentle sinusoidal nod
      nodY = Math.sin(tSec * 3.5) * 4.0;
    }

    if (this.elHead) {
      this.elHead.setAttribute(
        "transform",
        `translate(160, 115) rotate(${this.headTilt.current.toFixed(2)}) translate(-160, -115) translate(0, ${nodY.toFixed(2)})`
      );
    }

    // 5. Waving Hand Actuator (in DISTRACTED_REENGAGE)
    this.waveHand.raise += (this.waveHand.targetRaise - this.waveHand.raise) * 0.1;
    if (this.elWaveHand) {
      if (this.waveHand.raise > 0.05) {
        const waveAngle = Math.sin(tSec * 9.0) * 22.0 * this.waveHand.raise;
        const handY = 140 - this.waveHand.raise * 55.0;
        this.elWaveHand.setAttribute(
          "transform",
          `translate(290, ${handY.toFixed(1)}) rotate(${waveAngle.toFixed(1)})`
        );
        this.elWaveHand.style.opacity = "1";
      } else {
        this.elWaveHand.setAttribute("transform", "translate(288, 140) rotate(15)");
        this.elWaveHand.style.opacity = "0.4";
      }
    }

    // 6. 9-Bar Mouth Viseme Equalizer
    let isTutorSpeaking = false;
    if (this.analyserNode) {
      try {
        this.analyserNode.getByteFrequencyData(this.frequencyData);
        // Check energy
        let sum = 0;
        for (let i = 0; i < 9; i++) {
          sum += this.frequencyData[i];
        }
        if (sum > 60) {
          isTutorSpeaking = true;
        }
      } catch (err) {
        isTutorSpeaking = false;
      }
    }

    for (let i = 0; i < 9; i++) {
      let targetH = 6;
      if (isTutorSpeaking) {
        // Map bin frequency to bar height (6 to 28px)
        const freqVal = this.frequencyData[i] || 0;
        targetH = 6 + (freqVal / 255.0) * 22.0;
      } else {
        // Idle gentle breathing wave
        targetH = 5 + Math.sin(tSec * 2.5 + i * 0.6) * 2.5;
      }

      this.mouthBars[i] += (targetH - this.mouthBars[i]) * 0.25;
      const h = Math.max(4, this.mouthBars[i]);
      const el = this.elMouthBars[i];
      if (el) {
        el.setAttribute("height", h.toFixed(1));
        el.setAttribute("y", (-h / 2).toFixed(1));
        el.setAttribute("fill", isTutorSpeaking ? "#38bdf8" : this.brows.targetColor);
      }
    }
  }

  destroy() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }
}
