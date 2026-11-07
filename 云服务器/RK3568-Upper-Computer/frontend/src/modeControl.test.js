import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const source = readFileSync(resolve(process.cwd(), "src/App.vue"), "utf8");

describe("reliable device mode controls", () => {
  it("waits for the agent generation acknowledgement before updating UI state", () => {
    expect(source).toContain("async function waitForModeAck(");
    expect(source).toContain(
      "await waitForModeAck(body.generation, true, body.mode)",
    );
    expect(source).toContain(
      "await waitForModeAck(body.generation, false)",
    );
  });

  it("prevents duplicate stop requests and no longer exposes force terminate", () => {
    expect(source).toContain("if (stoppingMode.value) return");
    expect(source).toContain(':disabled="stoppingMode"');
    expect(source).not.toContain("terminatePrograms");
    expect(source).not.toContain("/api/mode/terminate");
    expect(source).not.toContain("终止程序");
  });

  it("uses the board default HDMI input without exposing a people source selector", () => {
    expect(source).toContain('source: "hdmi"');
    expect(source).not.toContain("sourceOptions");
    expect(source).not.toContain("以太网（UDP）");
  });
});
