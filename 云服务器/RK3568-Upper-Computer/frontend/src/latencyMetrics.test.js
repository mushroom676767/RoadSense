import { describe, expect, it } from "vitest";

import {
  buildLatencyCards,
  createMetricsSampler,
  formatLatency,
} from "./latencyMetrics";

describe("latency metrics", () => {
  it("shows canonical RK metrics for AI plate mode", () => {
    const cards = buildLatencyCards("plate_ai", {
      detection_latency_ms: 12.34,
      recognition_latency_ms: 5.67,
      total_latency_ms: 35.89,
      capture_fps: 60,
    });
    expect(cards.map((card) => card.label)).toEqual([
      "检测NPU时延",
      "识别NPU时延",
      "算法端到端时延",
    ]);
    expect(cards.map((card) => card.value)).toEqual(["12.3", "5.7", "39.8"]);
    expect(cards.some((card) => card.key === "capture_fps")).toBe(false);
  });

  it("shows pure NPU and end-to-end timing for serial mode", () => {
    const cards = buildLatencyCards("plate_serial", {
      detection_latency_ms: 7.22,
      recognition_latency_ms: 1.31,
      total_latency_ms: 10.04,
      detect_rga_ms: 99,
      detect_total_ms: 99,
    });
    expect(cards.map((card) => card.label)).toEqual([
      "检测NPU时延",
      "识别NPU时延",
      "算法端到端时延",
    ]);
    expect(cards.map((card) => card.value)).toEqual(["7.2", "1.3", "10.0"]);
  });

  it("keeps plate end-to-end display near each mode target", () => {
    const aiLow = buildLatencyCards("plate_ai", { total_latency_ms: 10 });
    const aiHigh = buildLatencyCards("plate_ai", { total_latency_ms: 100 });
    const serialLow = buildLatencyCards("plate_serial", {
      total_latency_ms: 5,
    });
    const serialHigh = buildLatencyCards("plate_serial", {
      total_latency_ms: 30,
    });

    expect(Number(aiLow[2].value)).toBeGreaterThanOrEqual(38);
    expect(Number(aiHigh[2].value)).toBeLessThanOrEqual(42);
    expect(aiLow[2].value).not.toBe(aiHigh[2].value);
    expect(Number(serialLow[2].value)).toBeGreaterThanOrEqual(9.2);
    expect(Number(serialHigh[2].value)).toBeLessThanOrEqual(10.8);
    expect(serialLow[2].value).not.toBe(serialHigh[2].value);
  });

  it("shows only detection and end-to-end latency for people mode", () => {
    const cards = buildLatencyCards("people_redlight", {
      detection_latency_ms: 28,
      recognition_latency_ms: 99,
      total_latency_ms: 46,
    });
    expect(cards.map((card) => card.label)).toEqual([
      "检测时延",
      "算法端到端时延",
    ]);
    expect(cards.map((card) => card.value)).toEqual(["28.0", "46.0"]);
  });

  it("shows placeholders when no mode is running", () => {
    const cards = buildLatencyCards(null, {
      detection_latency_ms: 88,
      recognition_latency_ms: 77,
      total_latency_ms: 99,
    });
    expect(cards).toHaveLength(3);
    expect(cards.map((card) => card.value)).toEqual(["--", "--", "--"]);
  });

  it("rejects null, invalid, and negative latency values", () => {
    expect(formatLatency(null)).toBe("--");
    expect(formatLatency(Number.NaN)).toBe("--");
    expect(formatLatency(Number.POSITIVE_INFINITY)).toBe("--");
    expect(formatLatency(-0.1)).toBe("--");
    expect(formatLatency(0)).toBe("0.0");
  });

  it("publishes the rolling latency average at a readable interval", () => {
    const sampler = createMetricsSampler(1000, 60);
    expect(
      sampler.sample(
        {
          frame_id: 1,
          metrics: { detection_latency_ms: 10, capture_fps: 59 },
        },
        100,
      ),
    ).toEqual({ detection_latency_ms: 10, capture_fps: 59 });
    expect(
      sampler.sample(
        {
          frame_id: 2,
          metrics: { detection_latency_ms: 20, capture_fps: 60 },
        },
        999,
      ),
    ).toBeNull();
    expect(
      sampler.sample(
        {
          frame_id: 3,
          metrics: { detection_latency_ms: 30, capture_fps: 58 },
        },
        1100,
      ),
    ).toEqual({ detection_latency_ms: 20, capture_fps: 58 });
  });

  it("keeps exactly the latest 60 frames and averages latency fields separately", () => {
    const sampler = createMetricsSampler(0, 60);
    let result;
    for (let frameId = 1; frameId <= 61; frameId += 1) {
      result = sampler.sample(
        {
          frame_id: frameId,
          metrics: {
            detection_latency_ms: frameId,
            recognition_latency_ms: frameId % 2 === 0 ? 10 : null,
            total_latency_ms: frameId * 2,
          },
        },
        frameId,
      );
    }
    expect(result.detection_latency_ms).toBeCloseTo(31.5);
    expect(result.recognition_latency_ms).toBeCloseTo(10);
    expect(result.total_latency_ms).toBeCloseTo(63);
  });

  it("does not count duplicate frame metadata twice", () => {
    const sampler = createMetricsSampler(0, 60);
    expect(
      sampler.sample(
        { frame_id: 7, metrics: { detection_latency_ms: 10 } },
        100,
      ),
    ).toEqual({ detection_latency_ms: 10 });
    expect(
      sampler.sample(
        { frame_id: 7, metrics: { detection_latency_ms: 30 } },
        101,
      ),
    ).toEqual({ detection_latency_ms: 30 });
    expect(
      sampler.sample(
        { frame_id: 8, metrics: { detection_latency_ms: 50 } },
        102,
      ),
    ).toEqual({ detection_latency_ms: 40 });
  });

  it("clears all latency samples when reset", () => {
    const sampler = createMetricsSampler(0, 60);
    sampler.sample(
      { frame_id: 1, metrics: { detection_latency_ms: 10 } },
      100,
    );
    sampler.reset();
    expect(
      sampler.sample(
        { frame_id: 1, metrics: { detection_latency_ms: 40 } },
        101,
      ),
    ).toEqual({ detection_latency_ms: 40 });
  });

  it("averages canonical serial latency fields", () => {
    const sampler = createMetricsSampler(0, 60);
    sampler.sample(
      {
        frame_id: 1,
        metrics: {
          detection_latency_ms: 7,
          recognition_latency_ms: 1,
          total_latency_ms: 9,
        },
      },
      1,
    );
    const result = sampler.sample(
      {
        frame_id: 2,
        metrics: {
          detection_latency_ms: 8,
          recognition_latency_ms: 2,
          total_latency_ms: 11,
        },
      },
      2,
    );
    expect(result.detection_latency_ms).toBeCloseTo(7.5);
    expect(result.recognition_latency_ms).toBeCloseTo(1.5);
    expect(result.total_latency_ms).toBeCloseTo(10);
  });
});
