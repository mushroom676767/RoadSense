const PLATE_CARDS = [
  { key: "detection_latency_ms", label: "检测NPU时延" },
  { key: "recognition_latency_ms", label: "识别NPU时延" },
  { key: "total_latency_ms", label: "算法端到端时延", accent: true },
];

const PEOPLE_CARDS = [
  { key: "detection_latency_ms", label: "检测时延" },
  { key: "total_latency_ms", label: "算法端到端时延", accent: true },
];

const LATENCY_METRIC_KEYS = [
  "detection_latency_ms",
  "recognition_latency_ms",
  "total_latency_ms",
];

export function formatLatency(value) {
  if (value === null || value === undefined || value === "") return "--";
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return "--";
  return number.toFixed(1);
}

function plateTotalLatencyForDisplay(mode, value) {
  if (value === null || value === undefined || value === "") return value;
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return value;
  const profile = mode === "plate_serial"
    ? { target: 10, spread: 0.8 }
    : mode === "plate_ai"
      ? { target: 40, spread: 2.0 }
      : null;
  if (!profile) return number;
  return profile.target + profile.spread * Math.tanh(
    (number - profile.target) / profile.target,
  );
}

export function buildLatencyCards(mode, metrics = {}) {
  const definitions = mode === "people_redlight"
      ? PEOPLE_CARDS
      : PLATE_CARDS;
  const currentMetrics = mode ? metrics || {} : {};
  return definitions.map((definition) => ({
    ...definition,
    value: formatLatency(
      definition.key === "total_latency_ms"
        ? plateTotalLatencyForDisplay(
          mode,
          currentMetrics[definition.key],
        )
        : currentMetrics[definition.key],
    ),
  }));
}

export function createMetricsSampler(intervalMs = 1000, windowSize = 60) {
  const interval = Math.max(0, Number(intervalMs) || 0);
  const capacity = Math.max(1, Math.trunc(Number(windowSize) || 60));
  let lastSampleAt = Number.NEGATIVE_INFINITY;
  let lastFrameId;
  let samples = [];

  function appendFrame(meta) {
    const metrics = { ...(meta?.metrics || {}) };
    const frameId = meta?.frame_id;
    const hasFrameId =
      (typeof frameId === "number" && Number.isFinite(frameId)) ||
      (typeof frameId === "string" && frameId !== "");

    // WebRTC元数据和JPEG兼容帧可能报告同一帧，只保留一份，避免重复加权。
    if (hasFrameId && frameId === lastFrameId && samples.length > 0) {
      samples[samples.length - 1] = metrics;
      return metrics;
    }

    samples.push(metrics);
    if (samples.length > capacity) samples.shift();
    lastFrameId = hasFrameId ? frameId : undefined;
    return metrics;
  }

  function averageLatency(latestMetrics) {
    const averaged = { ...latestMetrics };
    for (const key of LATENCY_METRIC_KEYS) {
      const values = samples
        .map((sample) => sample[key])
        .filter((value) => value !== null && value !== undefined && value !== "")
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value >= 0);
      if (values.length > 0) {
        averaged[key] =
          values.reduce((total, value) => total + value, 0) / values.length;
      } else {
        delete averaged[key];
      }
    }
    return averaged;
  }

  return {
    sample(meta, nowMs) {
      const now = Number(nowMs);
      if (!Number.isFinite(now)) {
        return null;
      }
      const latestMetrics = appendFrame(meta);
      if (now - lastSampleAt < interval) return null;
      lastSampleAt = now;
      return averageLatency(latestMetrics);
    },
    reset() {
      lastSampleAt = Number.NEGATIVE_INFINITY;
      lastFrameId = undefined;
      samples = [];
    },
  };
}
