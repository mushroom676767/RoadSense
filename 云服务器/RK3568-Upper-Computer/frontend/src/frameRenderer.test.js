import { describe, expect, it, vi } from "vitest";
import {
  drawVideoFrame,
  jpegBlob,
  orderLandmarks,
  overlayBoundingBox,
  overlayColor,
  overlayLabel,
  usesLandmarkPolygon,
} from "./frameRenderer";

describe("frame renderer", () => {
  it("uses mode-aware overlay colors", () => {
    expect(
      overlayColor({ type: "plate", source: "rk-roi" }, "plate_serial"),
    ).toBe("#2dd4ff");
    expect(
      overlayColor(
        { type: "plate", source: "rk-roi", color: "#123456" },
        "plate_serial",
      ),
    ).toBe("#2dd4ff");
    expect(overlayColor({ type: "plate", source: "rk" }, "plate_ai")).toBe("#2dd4ff");
    expect(overlayColor({ type: "person" }, "people_redlight")).toBe("#47e59b");
    expect(overlayColor({ type: "redlight_violation" }, "people_redlight")).toBe("#ff4d5e");
    expect(
      overlayColor({ type: "person", status: "violation", color: "#47e59b" }, "people_redlight"),
    ).toBe("#ff4d5e");
    expect(
      overlayColor({ type: "person", color: "#123456" }, "people_redlight"),
    ).toBe("#47e59b");
  });

  it("uses the same concise label for every plate source", () => {
    expect(
      overlayLabel(
        {
          type: "plate",
          label: "粤B12345",
          rec_score: 0.934,
          source: "rk",
        },
        "plate_ai",
      ),
    ).toBe("粤B12345 · 93%");
    expect(
      overlayLabel(
        {
          type: "plate",
          label: "粤B12345",
          rec_score: 0.934,
          source: "rk-roi",
        },
        "plate_serial",
      ),
    ).toBe("粤B12345 · 93%");
  });

  it("omits missing plate fields and hides text without a plate number", () => {
    expect(
      overlayLabel(
        { type: "plate", label: "粤B12345", rec_score: 0.934 },
        "plate_ai",
      ),
    ).toBe("粤B12345 · 93%");
    expect(
      overlayLabel(
        { type: "plate", label: "粤B12345" },
        "plate_serial",
      ),
    ).toBe("粤B12345");
    expect(
      overlayLabel(
        { type: "plate", rec_score: 0.934 },
        "plate_serial",
      ),
    ).toBe("");
  });

  it("accepts binary jpeg views", async () => {
    const blob = jpegBlob(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]));
    expect(blob.type).toBe("image/jpeg");
    expect(blob.size).toBe(4);
  });

  it("orders plate landmarks around the perimeter without crossed edges", () => {
    const topLeft = [10, 10];
    const topRight = [90, 12];
    const bottomRight = [86, 40];
    const bottomLeft = [14, 42];

    expect(
      orderLandmarks([topLeft, topRight, bottomLeft, bottomRight]),
    ).toEqual([topLeft, topRight, bottomRight, bottomLeft]);
    expect(
      orderLandmarks([bottomRight, topLeft, bottomLeft, topRight]),
    ).toEqual([topLeft, topRight, bottomRight, bottomLeft]);
  });

  it("uses bbox rectangles for every mode and source", () => {
    expect(
      usesLandmarkPolygon({ source: "rk-roi" }, "plate_serial"),
    ).toBe(false);
    expect(
      usesLandmarkPolygon({ source: "rk" }, "plate_ai"),
    ).toBe(false);
    expect(
      usesLandmarkPolygon({ type: "person" }, "people_redlight"),
    ).toBe(false);
  });

  it("uses the restored full-frame Pose bbox in serial mode", () => {
    expect(
      overlayBoundingBox(
        {
          bbox: [425, 257, 623, 297],
          landmarks: [
            [425, 257],
            [623, 257],
            [623, 297],
            [425, 310],
          ],
        },
        "plate_serial",
      ),
    ).toEqual([425, 257, 623, 297]);
  });

  it("falls back to the landmark extent when a serial bbox is absent", () => {
    expect(
      overlayBoundingBox(
        {
          landmarks: [
            [425, 257],
            [623, 257],
            [623, 297],
            [425, 310],
          ],
        },
        "plate_serial",
      ),
    ).toEqual([425, 257, 623, 310]);
  });

  it("draws decoded video pixels before overlays on the same canvas", () => {
    const calls = [];
    const context = {
      drawImage: vi.fn(() => calls.push("video")),
      save: vi.fn(() => calls.push("overlay")),
      restore: vi.fn(),
      strokeRect: vi.fn(),
      measureText: vi.fn(() => ({ width: 80 })),
      fillRect: vi.fn(),
      fillText: vi.fn(),
    };
    drawVideoFrame(
      context,
      { tagName: "VIDEO" },
      [{ type: "plate", bbox: [10, 10, 100, 50], label: "粤B12345" }],
      "plate_ai",
      1920,
      1080,
    );
    expect(calls.slice(0, 2)).toEqual(["video", "overlay"]);
    expect(context.drawImage).toHaveBeenCalledWith(
      { tagName: "VIDEO" },
      0,
      0,
      1920,
      1080,
    );
  });
});
