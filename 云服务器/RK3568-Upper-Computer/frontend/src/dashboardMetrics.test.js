import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const source = readFileSync(resolve(process.cwd(), "src/App.vue"), "utf8");

describe("dashboard metrics", () => {
  it("renders mode-specific RK latency cards without local fallbacks", () => {
    const section = source.match(
      /<section class="metrics-grid">([\s\S]*?)<\/section>/,
    )?.[1];

    expect(section).toBeTruthy();
    expect(section).toContain('v-for="card in latencyCards"');
    expect(section).not.toContain("采集帧率");
    expect(section).not.toContain("detect_latency_ms");
    expect(section).not.toContain("rec_latency_ms");
    expect(section).not.toContain("end_to_end_latency_ms");
  });

  it("removes the compact status panel", () => {
    expect(source).not.toContain('<article class="insight-card compact">');
  });
});
