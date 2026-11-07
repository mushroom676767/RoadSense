export function overlayColor(overlay, mode) {
  if (overlay.type === "plate") return "#2dd4ff";
  if (
    overlay.type === "redlight_violation" ||
    ["violation", "redlight", "闯红灯"].includes(overlay.status)
  ) {
    return "#ff4d5e";
  }
  if (overlay.type === "person") return "#47e59b";
  if (typeof overlay.color === "string" && overlay.color) {
    return overlay.color;
  }
  return "#2dd4ff";
}

export function overlayLabel(overlay, mode) {
  const isPlate =
    overlay.type === "plate" ||
    mode === "plate_ai" ||
    mode === "plate_serial";
  if (isPlate && !overlay.label) return "";

  const parts = [];
  if (overlay.label) parts.push(overlay.label);
  const score =
    overlay.rec_score ?? overlay.score ?? overlay.confidence ?? null;
  if (score != null && Number.isFinite(Number(score))) {
    parts.push(`${(Number(score) * 100).toFixed(0)}%`);
  }
  return parts.join(" · ");
}

export function orderLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 4) return [];
  const points = landmarks.slice(0, 4).map((point) => [
    Number(point?.[0]),
    Number(point?.[1]),
  ]);
  if (points.some((point) => !point.every(Number.isFinite))) return [];

  const centerX = points.reduce((sum, point) => sum + point[0], 0) / 4;
  const centerY = points.reduce((sum, point) => sum + point[1], 0) / 4;
  points.sort(
    (left, right) =>
      Math.atan2(left[1] - centerY, left[0] - centerX) -
      Math.atan2(right[1] - centerY, right[0] - centerX),
  );

  const topLeftIndex = points.reduce(
    (best, point, index) =>
      point[0] + point[1] < points[best][0] + points[best][1]
        ? index
        : best,
    0,
  );
  return [
    ...points.slice(topLeftIndex),
    ...points.slice(0, topLeftIndex),
  ];
}

export function usesLandmarkPolygon(_overlay, _mode) {
  // 上位机统一采用模型/发布结果中的bbox矩形；四角点仅作为业务数据保留。
  return false;
}

export function overlayBoundingBox(overlay, mode) {
  if (Array.isArray(overlay?.bbox) && overlay.bbox.length >= 4) {
    const values = overlay.bbox.slice(0, 4).map(Number);
    if (values.every(Number.isFinite)) {
      return [
        Math.min(values[0], values[2]),
        Math.min(values[1], values[3]),
        Math.max(values[0], values[2]),
        Math.max(values[1], values[3]),
      ];
    }
  }
  if (mode === "plate_serial") {
    const points = orderLandmarks(overlay?.landmarks);
    if (points.length === 4) {
      const xs = points.map((point) => point[0]);
      const ys = points.map((point) => point[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }
  }
  return null;
}

export function drawOverlays(context, overlays, mode, width, height) {
  const scale = Math.max(width / 1920, height / 1080);
  context.save();
  context.lineJoin = "round";
  context.lineCap = "round";
  for (const overlay of overlays || []) {
    if (!overlay || typeof overlay !== "object") continue;
    const color = overlayColor(overlay, mode);
    context.strokeStyle = color;
    context.lineWidth = Math.max(2, 4 * scale);

    let anchor = null;
    const points = usesLandmarkPolygon(overlay, mode)
      ? orderLandmarks(overlay.landmarks)
      : [];
    if (points.length === 4) {
      context.beginPath();
      context.moveTo(points[0][0], points[0][1]);
      for (let index = 1; index < points.length; index += 1) {
        context.lineTo(points[index][0], points[index][1]);
      }
      context.closePath();
      context.stroke();
      anchor = points[0];
    } else {
      const bbox = overlayBoundingBox(overlay, mode);
      if (!bbox) continue;
      const [x1, y1, x2, y2] = bbox;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      anchor = [x1, y1];
    }

    const label = overlayLabel(overlay, mode);
    if (!anchor || !label) continue;
    const fontSize = Math.max(18, Math.round(28 * scale));
    context.font = `600 ${fontSize}px "Microsoft YaHei", sans-serif`;
    const padding = Math.max(5, Math.round(7 * scale));
    const textWidth = context.measureText(label).width;
    const x = Math.max(0, Math.min(width - textWidth - padding * 2, anchor[0]));
    const y = Math.max(fontSize + padding * 2, anchor[1]);
    context.fillStyle = "rgba(3, 10, 20, .82)";
    context.fillRect(
      x,
      y - fontSize - padding * 2,
      textWidth + padding * 2,
      fontSize + padding * 2,
    );
    context.fillStyle = color;
    context.fillText(label, x + padding, y - padding);
  }
  context.restore();
}

export function drawVideoFrame(
  context,
  video,
  overlays,
  mode,
  width,
  height,
) {
  context.drawImage(video, 0, 0, width, height);
  drawOverlays(context, overlays, mode, width, height);
}

export function jpegBlob(value) {
  if (value instanceof Blob) return value;
  if (value instanceof ArrayBuffer) return new Blob([value], { type: "image/jpeg" });
  if (ArrayBuffer.isView(value)) {
    const slice = value.buffer.slice(
      value.byteOffset,
      value.byteOffset + value.byteLength,
    );
    return new Blob([slice], { type: "image/jpeg" });
  }
  throw new TypeError("Unsupported JPEG payload");
}
