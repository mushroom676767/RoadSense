import { describe, expect, it } from "vitest";

import { createSavedEventTracker } from "./eventArchiveRefresh";


describe("event archive refresh tracker", () => {
  it("refreshes once for new saved event ids and ignores duplicates", () => {
    const tracker = createSavedEventTracker();

    expect(tracker.consume([10])).toBe(true);
    expect(tracker.consume([10])).toBe(false);
    expect(tracker.consume([10, 11])).toBe(true);
    expect(tracker.consume([11])).toBe(false);
  });

  it("ignores malformed ids and can reset after clearing the archive", () => {
    const tracker = createSavedEventTracker();

    expect(tracker.consume([null, "12", -1, 0])).toBe(false);
    expect(tracker.consume([12])).toBe(true);
    tracker.reset();
    expect(tracker.consume([12])).toBe(true);
  });
});
