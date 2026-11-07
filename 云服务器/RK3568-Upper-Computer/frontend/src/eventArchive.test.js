import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const source = readFileSync(resolve(process.cwd(), "src/App.vue"), "utf8");

describe("event archive", () => {
  it("lets administrators clear all archived events after confirmation", () => {
    expect(source).toContain("async function clearEvents()");
    expect(source).toContain(
      'api("/api/events/clear", { method: "POST" })',
    );
    expect(source).toContain('v-if="canControl"');
    expect(source).toContain('@click="clearEvents"');
    expect(source).toContain("将永久删除全部事件记录和事件截图");
  });

  it("refreshes the visible archive when ingest reports a saved event", () => {
    expect(source).toContain("scheduleEventArchiveRefresh(frameMeta.value)");
    expect(source).toContain('if (view.value !== "events") return');
    expect(source).toContain("loadEvents(1)");
  });
});
