const FRAME_WIDTH = 1920;
const FRAME_HEIGHT = 1080;
const MAX_ROIS = 4;

function integerCoordinate(value) {
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

function normalizeRoi(value, index) {
  const x1 = integerCoordinate(value?.x1);
  const y1 = integerCoordinate(value?.y1);
  const x2 = integerCoordinate(value?.x2);
  const y2 = integerCoordinate(value?.y2);
  const valid =
    x1 !== null &&
    y1 !== null &&
    x2 !== null &&
    y2 !== null &&
    x1 >= 0 &&
    y1 >= 0 &&
    x1 < x2 &&
    y1 < y2 &&
    x2 <= FRAME_WIDTH &&
    y2 <= FRAME_HEIGHT;
  return { name: `ROI ${index + 1}`, x1, y1, x2, y2, valid };
}

export function normalizeSerialTelemetry(mode, serialData) {
  if (mode !== "plate_serial") return null;

  const data = serialData && typeof serialData === "object" ? serialData : {};
  const online = data.online === true;
  const sequence = Math.max(0, Math.trunc(Number(data.sequence) || 0));
  const countValue = Number(data.count);
  const count = Number.isInteger(countValue) ? countValue : null;
  const rois = Array.isArray(data.rois)
    ? data.rois.slice(0, MAX_ROIS + 1).map(normalizeRoi)
    : [];
  const hasPacket = sequence > 0;
  const validPacket =
    hasPacket &&
    count !== null &&
    count >= 0 &&
    count <= MAX_ROIS &&
    rois.length === count &&
    rois.every((roi) => roi.valid);
  const ageValue = Number(data.age_ms);
  const ageMs = Number.isFinite(ageValue) && ageValue >= 0 ? ageValue : null;
  const receivedAtValue = Number(data.received_at_ms);
  const receivedAtMs = Number.isFinite(receivedAtValue)
    ? Math.trunc(receivedAtValue)
    : null;

  let statusLabel = "等待数据";
  if (!online) {
    statusLabel = "串口离线";
  } else if (hasPacket && !validPacket) {
    statusLabel = `数据异常 · #${sequence}`;
  } else if (validPacket) {
    statusLabel = `实时 · #${sequence} · ${count}个ROI`;
  }

  return {
    online,
    sequence,
    count: validPacket ? count : 0,
    hasPacket,
    validPacket,
    hasRois: validPacket && count > 0,
    rawHex: typeof data.raw_hex === "string" ? data.raw_hex.trim() : "",
    receivedAtMs,
    ageMs,
    rois: validPacket ? rois : [],
    statusLabel,
  };
}
