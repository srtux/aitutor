/**
 * vad_worklet.js - High-performance AudioWorkletProcessor for Edge VAD.
 *
 * Runs on the dedicated audio rendering thread:
 * 1. Computes frame-by-frame RMS audio energy
 * 2. Computes Zero-Crossing Rate (ZCR) to distinguish speech from hiss/clicks
 * 3. Downsamples mono PCM stream to hermetic 16,000 Hz float buffer
 * 4. Transmits telemetry and 16kHz audio slices to main thread via MessagePort
 */

class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.resamplePhase = 0.0;
    this.lastSample = 0.0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    // Mono channel extraction
    const channel = input[0];
    if (!channel || channel.length === 0) {
      return true;
    }

    const length = channel.length;
    let sumSquares = 0.0;
    let zeroCrossings = 0;

    let prevSample = this.lastSample;
    for (let i = 0; i < length; i++) {
      const s = channel[i];
      sumSquares += s * s;

      if ((s >= 0.0 && prevSample < 0.0) || (s < 0.0 && prevSample >= 0.0)) {
        zeroCrossings++;
      }
      prevSample = s;
    }
    this.lastSample = channel[length - 1];

    const rms = Math.sqrt(sumSquares / length);
    const zcr = length > 0 ? zeroCrossings / length : 0.0;

    // Linear downsampling to 16,000 Hz
    let samples16k;
    const currentSampleRate = sampleRate; // Global AudioWorklet property

    if (Math.abs(currentSampleRate - this.targetSampleRate) < 1.0) {
      // Direct pass-through if context is already 16kHz
      samples16k = new Float32Array(channel);
    } else {
      const step = currentSampleRate / this.targetSampleRate;
      const outSamples = [];

      while (this.resamplePhase < length) {
        const index0 = Math.floor(this.resamplePhase);
        const index1 = Math.min(index0 + 1, length - 1);
        const frac = this.resamplePhase - index0;

        const val0 = channel[index0];
        const val1 = channel[index1];
        outSamples.push(val0 + frac * (val1 - val0));

        this.resamplePhase += step;
      }
      this.resamplePhase -= length;
      samples16k = new Float32Array(outSamples);
    }

    this.port.postMessage({
      rms,
      zcr,
      samples16k,
    });

    return true;
  }
}

registerProcessor("vad-processor", VadProcessor);
