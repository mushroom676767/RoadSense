export function preferH264(transceiver) {
  if (!transceiver?.setCodecPreferences || !globalThis.RTCRtpReceiver) return false;
  const codecs = RTCRtpReceiver.getCapabilities?.("video")?.codecs || [];
  const h264 = codecs.filter(
    (codec) => String(codec.mimeType).toLowerCase() === "video/h264",
  );
  if (!h264.length) return false;
  transceiver.setCodecPreferences(h264);
  return true;
}

export function waitForIceGatheringComplete(peer, timeoutMs = 6000) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", changed);
      // Half-trickle WHEP can proceed with candidates gathered so far.
      resolve();
    }, timeoutMs);
    function changed() {
      if (peer.iceGatheringState !== "complete") return;
      clearTimeout(timeout);
      peer.removeEventListener("icegatheringstatechange", changed);
      resolve();
    }
    peer.addEventListener("icegatheringstatechange", changed);
  });
}

export function retryDelayMs(failedAttempts) {
  const schedule = [250, 500, 1000, 2000];
  const index = Math.max(0, Math.min(schedule.length - 1, failedAttempts));
  return schedule[index];
}

export function mediaReadyForPlayback(stream) {
  return Boolean(
    stream?.online &&
    stream?.board_online &&
    stream?.cloud_publisher_online
  );
}

export function shouldIncludeJpeg(videoTransport, decodedMediaFrame) {
  return videoTransport === "jpeg" || !decodedMediaFrame;
}

export function shouldUseHlsFallback(
  failedAttempts,
  noRtpMs,
  { maxWebRtcFailures = 2, noRtpTimeoutMs = 10_000 } = {},
) {
  return (
    Number(failedAttempts) >= maxWebRtcFailures ||
    Number(noRtpMs) >= noRtpTimeoutMs
  );
}

export function createMetadataRing(maxItems = 180) {
  const entries = [];
  const limit = Math.max(2, Number(maxItems) || 180);
  return {
    push(meta, receivedAtMs) {
      if (!meta || typeof meta !== "object") return;
      entries.push({
        meta,
        receivedAtMs: Number(receivedAtMs) || 0,
      });
      if (entries.length > limit) {
        entries.splice(0, entries.length - limit);
      }
    },
    select(nowMs, playbackDelayMs = 0) {
      if (!entries.length) return null;
      const target = Number(nowMs) - Math.max(0, Number(playbackDelayMs) || 0);
      for (let index = entries.length - 1; index >= 0; index -= 1) {
        if (entries[index].receivedAtMs <= target) {
          return entries[index].meta;
        }
      }
      return entries[0].meta;
    },
    clear() {
      entries.length = 0;
    },
    get size() {
      return entries.length;
    },
  };
}

export function iceTransportPolicyForAttempt(failedAttempts) {
  // Try the direct path first, use TURN for one recovery attempt, then
  // re-probe the direct path. A transient failure must not pin the browser to
  // a slower relay for the rest of the session.
  return Math.max(0, failedAttempts) % 2 === 1 ? "relay" : "all";
}

export function createInboundVideoHealth(startedAt = 0) {
  return {
    startedAt,
    lastProgressAt: startedAt,
    bytesReceived: 0,
    packetsReceived: 0,
    framesDecoded: 0,
    receivedMedia: false,
  };
}

function forEachReport(reports, callback) {
  if (!reports) return;
  if (typeof reports.forEach === "function") {
    reports.forEach(callback);
    return;
  }
  for (const report of reports) callback(report);
}

export function sampleInboundVideoHealth(
  health,
  reports,
  now,
  {
    startupTimeoutMs = 10_000,
    stallTimeoutMs = 10_000,
  } = {},
) {
  let bytesReceived = 0;
  let packetsReceived = 0;
  let framesDecoded = 0;
  let foundInboundVideo = false;

  forEachReport(reports, (report) => {
    const mediaKind = report?.kind || report?.mediaType;
    if (
      report?.type !== "inbound-rtp" ||
      report?.isRemote ||
      mediaKind !== "video"
    ) {
      return;
    }
    foundInboundVideo = true;
    bytesReceived += Number(report.bytesReceived) || 0;
    packetsReceived += Number(report.packetsReceived) || 0;
    framesDecoded += Number(report.framesDecoded) || 0;
  });

  const countersReset =
    foundInboundVideo &&
    (
      bytesReceived < health.bytesReceived ||
      packetsReceived < health.packetsReceived ||
      framesDecoded < health.framesDecoded
    );
  const progressed =
    foundInboundVideo &&
    (
      countersReset ||
      bytesReceived > health.bytesReceived ||
      packetsReceived > health.packetsReceived ||
      framesDecoded > health.framesDecoded
    );
  if (progressed) {
    health.lastProgressAt = now;
    health.receivedMedia = true;
  }
  if (foundInboundVideo) {
    health.bytesReceived = bytesReceived;
    health.packetsReceived = packetsReceived;
    health.framesDecoded = framesDecoded;
  }

  return {
    foundInboundVideo,
    progressed,
    receivedMedia: health.receivedMedia,
    startupTimedOut:
      !health.receivedMedia &&
      now - health.startedAt >= startupTimeoutMs,
    stalled:
      health.receivedMedia &&
      now - health.lastProgressAt >= stallTimeoutMs,
  };
}

export function configureLowLatencyReceiver(receiver) {
  if (!receiver) return;
  try {
    if ("playoutDelayHint" in receiver) receiver.playoutDelayHint = 0;
  } catch {
    // Browser-specific low-latency hints are best effort.
  }
  try {
    if ("jitterBufferTarget" in receiver) receiver.jitterBufferTarget = 0;
  } catch {
    // Ignore unsupported or read-only implementations.
  }
}

export function lowLatencyHlsConfig(xhrSetup) {
  return {
    lowLatencyMode: true,
    liveSyncDuration: 0.8,
    liveMaxLatencyDuration: 2,
    maxLiveSyncPlaybackRate: 1.2,
    maxBufferLength: 2,
    maxMaxBufferLength: 4,
    backBufferLength: 0,
    xhrSetup,
  };
}
