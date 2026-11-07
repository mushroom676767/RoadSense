import { describe, expect, it } from "vitest";

import { normalizeSerialTelemetry } from "./serialTelemetry";

describe("serial ROI telemetry", () => {
  it("is only active in serial mode", () => {
    expect(normalizeSerialTelemetry("plate_ai", { online: true })).toBeNull();
  });

  it.each([1, 2, 3, 4])("normalizes a valid packet with %i ROI entries", (count) => {
    const rois = Array.from({ length: count }, (_, index) => ({
      x1: 100 + index * 10,
      y1: 200 + index * 10,
      x2: 400 + index * 10,
      y2: 300 + index * 10,
    }));
    const value = normalizeSerialTelemetry("plate_serial", {
      online: true,
      sequence: 18,
      received_at_ms: 123456,
      age_ms: 12.5,
      raw_hex: "AA 01 64 00 C8 00 90 01 2C 01",
      count,
      rois,
    });

    expect(value.statusLabel).toBe(`实时 · #18 · ${count}个ROI`);
    expect(value.validPacket).toBe(true);
    expect(value.hasRois).toBe(true);
    expect(value.count).toBe(count);
    expect(value.receivedAtMs).toBe(123456);
    expect(value.ageMs).toBe(12.5);
    expect(value.rois[0]).toEqual({
      name: "ROI 1",
      x1: 100,
      y1: 200,
      x2: 400,
      y2: 300,
      valid: true,
    });
  });

  it("treats count zero as a valid clearing packet", () => {
    const value = normalizeSerialTelemetry("plate_serial", {
      online: true,
      sequence: 19,
      raw_hex: "AA 00",
      count: 0,
      rois: [],
    });

    expect(value.statusLabel).toBe("实时 · #19 · 0个ROI");
    expect(value.hasPacket).toBe(true);
    expect(value.validPacket).toBe(true);
    expect(value.hasRois).toBe(false);
    expect(value.rois).toEqual([]);
  });

  it("reports offline and waiting-for-first-packet states", () => {
    expect(
      normalizeSerialTelemetry("plate_serial", {
        online: false,
        sequence: 1,
        count: 1,
        rois: [{ x1: 1, y1: 2, x2: 3, y2: 4 }],
      }).statusLabel,
    ).toBe("串口离线");
    expect(
      normalizeSerialTelemetry("plate_serial", {
        online: true,
        sequence: 0,
        count: 0,
        rois: [],
      }).statusLabel,
    ).toBe("等待数据");
  });

  it.each([
    { count: 5, rois: [] },
    { count: 2, rois: [{ x1: 1, y1: 2, x2: 3, y2: 4 }] },
    { count: 1, rois: [{ x1: -1, y1: 2, x2: 3, y2: 4 }] },
    { count: 1, rois: [{ x1: 4, y1: 2, x2: 3, y2: 4 }] },
    { count: 1, rois: [{ x1: 1, y1: 2, x2: 1921, y2: 4 }] },
    { count: 1, rois: [{ x1: 1.5, y1: 2, x2: 3, y2: 4 }] },
  ])("rejects malformed ROI telemetry: %o", (payload) => {
    const value = normalizeSerialTelemetry("plate_serial", {
      online: true,
      sequence: 20,
      ...payload,
    });
    expect(value.statusLabel).toBe("数据异常 · #20");
    expect(value.validPacket).toBe(false);
    expect(value.rois).toEqual([]);
  });
});
