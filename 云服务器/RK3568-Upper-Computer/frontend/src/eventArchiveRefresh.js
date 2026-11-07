export function createSavedEventTracker(limit = 256) {
  const seen = new Set();
  const order = [];

  return {
    consume(ids) {
      if (!Array.isArray(ids)) return false;
      let changed = false;
      for (const id of ids) {
        if (!Number.isInteger(id) || id <= 0 || seen.has(id)) continue;
        seen.add(id);
        order.push(id);
        changed = true;
      }
      while (order.length > limit) {
        seen.delete(order.shift());
      }
      return changed;
    },
    reset() {
      seen.clear();
      order.length = 0;
    },
  };
}
