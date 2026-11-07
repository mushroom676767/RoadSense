import { afterEach, describe, expect, it } from "vitest";

import {
  configureLowLatencyReceiver,
  lowLatencyHlsConfig,
  createMetadataRing,
  createInboundVideoHealth,
  iceTransportPolicyForAttempt,
  mediaReadyForPlayback,
  preferH264,
  retryDelayMs,
  sampleInboundVideoHealth,
  shouldIncludeJpeg,
  shouldUseHlsFallback,
  waitForIceGatheringComplete,
} from "./webrtcClient";


const originalReceiver = globalThis.RTCRtpReceiver;

afterEach(() => {
  globalThis.RTCRtpReceiver = originalReceiver;
});

describe("WebRTC H.264 negotiation", () => {
  it("restricts the video transceiver to H.264 codecs", () => {
    globalThis.RTCRtpReceiver = {
      getCapabilities: () => ({
        codecs: [
          { mimeType: "video/VP8" },
          { mimeType: "video/H264", clockRate: 90000 },
        ],
      }),
    };
    let selected = [];
    const supported = preferH264({
      setCodecPreferences: (codecs) => {
        selected = codecs;
      },
    });
    expect(supported).toBe(true);
    expect(selected).toHaveLength(1);
    expect(selected[0].mimeType).toBe("video/H264");
  });

  it("waits until ICE gathering is complete", async () => {
    const peer = new EventTarget();
    peer.iceGatheringState = "gathering";
    const waiting = waitForIceGatheringComplete(peer, 1000);
    peer.iceGatheringState = "complete";
    peer.dispatchEvent(new Event("icegatheringstatechange"));
    await expect(waiting).resolves.toBeUndefined();
  });

  it("continues with gathered ICE candidates when gathering times out", async () => {
    const peer = new EventTarget();
    peer.iceGatheringState = "gathering";
    await expect(waitForIceGatheringComplete(peer, 1)).resolves.toBeUndefined();
  });

  it("uses bounded fast reconnect backoff", () => {
    expect([0, 1, 2, 3, 4, 9].map(retryDelayMs)).toEqual([
      250, 500, 1000, 2000, 2000, 2000,
    ]);
  });

  it("waits for metadata, board capture and cloud publisher before playback", () => {
    expect(
      mediaReadyForPlayback({
        online: true,
        board_online: true,
        cloud_publisher_online: true,
      }),
    ).toBe(true);
    expect(
      mediaReadyForPlayback({
        online: true,
        board_online: true,
        cloud_publisher_online: false,
      }),
    ).toBe(false);
    expect(
      mediaReadyForPlayback({
        online: false,
        board_online: true,
        cloud_publisher_online: true,
      }),
    ).toBe(false);
    expect(mediaReadyForPlayback(null)).toBe(false);
  });

  it("keeps JPEG until decoded media has produced a real frame", () => {
    expect(shouldIncludeJpeg("webrtc", false)).toBe(true);
    expect(shouldIncludeJpeg("hls", false)).toBe(true);
    expect(shouldIncludeJpeg("webrtc", true)).toBe(false);
    expect(shouldIncludeJpeg("hls", true)).toBe(false);
    expect(shouldIncludeJpeg("jpeg", true)).toBe(true);
  });

  it("falls back to HLS after repeated failures or ten seconds without RTP", () => {
    expect(shouldUseHlsFallback(1, 9999)).toBe(false);
    expect(shouldUseHlsFallback(2, 0)).toBe(true);
    expect(shouldUseHlsFallback(0, 10_000)).toBe(true);
  });

  it("selects metadata matching the delayed playback timeline", () => {
    const ring = createMetadataRing(3);
    ring.push({ frame_id: 1 }, 1000);
    ring.push({ frame_id: 2 }, 1200);
    ring.push({ frame_id: 3 }, 1400);
    ring.push({ frame_id: 4 }, 1600);
    expect(ring.size).toBe(3);
    expect(ring.select(1700, 400).frame_id).toBe(2);
    expect(ring.select(1700, 0).frame_id).toBe(4);
    ring.clear();
    expect(ring.select(1700, 0)).toBeNull();
  });

  it("alternates direct and TURN paths during recovery", () => {
    expect(iceTransportPolicyForAttempt(0)).toBe("all");
    expect(iceTransportPolicyForAttempt(1)).toBe("relay");
    expect(iceTransportPolicyForAttempt(2)).toBe("all");
    expect(iceTransportPolicyForAttempt(3)).toBe("relay");
  });

  it("uses inbound RTP progress instead of render callbacks for health", () => {
    const health = createInboundVideoHealth(1000);
    const report = new Map([
      [
        "video",
        {
          type: "inbound-rtp",
          kind: "video",
          bytesReceived: 4000,
          packetsReceived: 20,
          framesDecoded: 3,
        },
      ],
    ]);

    expect(sampleInboundVideoHealth(health, report, 2000).progressed).toBe(true);
    expect(
      sampleInboundVideoHealth(health, report, 9000).stalled,
    ).toBe(false);
    expect(
      sampleInboundVideoHealth(health, report, 12_001).stalled,
    ).toBe(true);
  });

  it("allows ten seconds for the first inbound RTP packet", () => {
    const health = createInboundVideoHealth(500);
    expect(
      sampleInboundVideoHealth(health, [], 10_499).startupTimedOut,
    ).toBe(false);
    expect(
      sampleInboundVideoHealth(health, [], 10_500).startupTimedOut,
    ).toBe(true);
  });

  it("accepts an RTP counter reset after the publisher changes SSRC", () => {
    const health = createInboundVideoHealth(1000);
    health.bytesReceived = 9000;
    health.packetsReceived = 90;
    health.framesDecoded = 30;
    health.receivedMedia = true;
    health.lastProgressAt = 2000;

    const result = sampleInboundVideoHealth(
      health,
      [
        {
          type: "inbound-rtp",
          mediaType: "video",
          bytesReceived: 1200,
          packetsReceived: 10,
          framesDecoded: 3,
        },
      ],
      8000,
    );

    expect(result.progressed).toBe(true);
    expect(result.stalled).toBe(false);
    expect(health.bytesReceived).toBe(1200);
  });

  it("applies low-latency receiver hints when supported", () => {
    const receiver = { playoutDelayHint: 1, jitterBufferTarget: 500 };
    configureLowLatencyReceiver(receiver);
    expect(receiver.playoutDelayHint).toBe(0);
    expect(receiver.jitterBufferTarget).toBe(0);
  });

  it("keeps LL-HLS close to the live edge and catches up when behind", () => {
    const xhrSetup = () => {};
    const config = lowLatencyHlsConfig(xhrSetup);
    expect(config.lowLatencyMode).toBe(true);
    expect(config.liveSyncDuration).toBe(0.8);
    expect(config.liveMaxLatencyDuration).toBe(2);
    expect(config.maxLiveSyncPlaybackRate).toBe(1.2);
    expect(config.maxBufferLength).toBe(2);
    expect(config.xhrSetup).toBe(xhrSetup);
  });
});
