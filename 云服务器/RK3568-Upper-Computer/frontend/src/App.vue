<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import Hls from "hls.js";
import { io } from "socket.io-client";

import { drawOverlays, drawVideoFrame, jpegBlob } from "./frameRenderer";
import { createSavedEventTracker } from "./eventArchiveRefresh";
import {
  buildLatencyCards,
  createMetricsSampler,
} from "./latencyMetrics";
import { normalizeSerialTelemetry } from "./serialTelemetry";
import {
  configureLowLatencyReceiver,
  createMetadataRing,
  createInboundVideoHealth,
  iceTransportPolicyForAttempt,
  lowLatencyHlsConfig,
  mediaReadyForPlayback,
  preferH264,
  retryDelayMs,
  sampleInboundVideoHealth,
  shouldIncludeJpeg,
  shouldUseHlsFallback,
  waitForIceGatheringComplete,
} from "./webrtcClient";
import contestLogo from "./assets/contest-logo.png";

const token = ref(localStorage.getItem("rk3568_token") || "");
const username = ref(localStorage.getItem("rk3568_user") || "");
const role = ref("");
const allowRegister = ref(false);
const loginTab = ref("login");
const loginLoading = ref(false);
const loginForm = reactive({ username: "", password: "" });
const registerForm = reactive({ username: "", password: "", confirm: "" });

const view = ref("live");
const modes = ref({});
const currentMode = ref(null);
const running = ref(false);
const switchingMode = ref("");
const stoppingMode = ref(false);
const connected = ref(false);
const streamOnline = ref(false);
const agentOnline = ref(false);
const boardOnline = ref(false);
const cloudPublisherOnline = ref(false);
const logs = ref("");
const logElement = ref(null);

const canvas = ref(null);
const webRtcVideo = ref(null);
const hlsVideo = ref(null);
const hasFrame = ref(false);
const frameMeta = ref({});
const displayedMetrics = ref({});
// 每帧累计时延样本，界面每秒刷新一次最近60帧的滑动平均值。
const metricsSampler = createMetricsSampler(1000, 60);
const videoTransport = ref("jpeg");
const webRtcState = ref("idle");
const frameFrozen = ref(false);
let socket = null;
let pendingFrame = null;
let decodingFrame = false;
let peerConnection = null;
let peerId = "";
let webRtcGeneration = 0;
let videoFrameCallback = 0;
let videoFrameUsesAnimation = false;
let videoFrameTarget = null;
let webRtcStartPromise = null;
let webRtcRetryAt = 0;
let refreshInFlight = false;
let heartbeatInFlight = false;
let mediaHealthInFlight = false;
let preferredVideoTransport = "webrtc";
let lastMediaWarningAt = 0;
let webRtcFailedAttempts = 0;
let webRtcConnectedAt = 0;
let inboundVideoHealth = createInboundVideoHealth();
let disconnectedTimer = 0;
let hlsPlayer = null;
let nativeHlsActive = false;
let hlsRetryAt = 0;
let hlsFallbackBaseUrl = "";
let sourceContinuity = false;
let decodedMediaFrame = false;
let lastMetadataAt = 0;
const metadataRing = createMetadataRing(240);
const WEBRTC_FALLBACK_RETRY_MS = 5_000;
const WEBRTC_WARNING_INTERVAL_MS = 30_000;
const WEBRTC_STABLE_RESET_MS = 20_000;
const DISCONNECTED_GRACE_MS = 5_000;

const transportLabel = computed(() => {
  if (!streamOnline.value) return "等待板端视频源";
  if (videoTransport.value === "hls") {
    return webRtcState.value === "hls"
      ? "H.264 / LL-HLS"
      : "LL-HLS 恢复中";
  }
  if (videoTransport.value === "webrtc") {
    if (webRtcState.value === "connected") return "H.264 / WebRTC";
    if (webRtcState.value === "no-media") return "WebRTC 无媒体";
    if (webRtcState.value === "waiting-source") return "等待视频源恢复";
    return "WebRTC 连接中";
  }
  return "JPEG 兼容模式";
});
const canControl = computed(() => role.value === "admin");

function applyInfrastructureStatus(status = {}) {
  if ("agent_online" in status) {
    agentOnline.value = Boolean(status.agent_online);
  }
  if ("board_online" in status) {
    boardOnline.value = Boolean(status.board_online);
  }
  if ("cloud_publisher_online" in status) {
    cloudPublisherOnline.value = Boolean(status.cloud_publisher_online);
  }
}

function currentMediaReady() {
  return mediaReadyForPlayback({
    online: streamOnline.value,
    board_online: boardOnline.value,
    cloud_publisher_online: cloudPublisherOnline.value,
  });
}

const events = ref([]);
const eventTotal = ref(0);
const eventPage = ref(1);
const eventPageSize = 12;
const eventFilters = reactive({ mode: "", event_type: "", query: "" });
const eventLoading = ref(false);
const eventClearing = ref(false);
const selectedEvent = ref(null);
const selectedSnapshot = ref("");
const savedEventTracker = createSavedEventTracker();
let eventRefreshTimer = 0;

const modeOrder = ["plate_serial", "plate_ai", "people_redlight"];
const modeIcon = {
  plate_serial: "串口",
  plate_ai: "车牌",
  people_redlight: "行人",
};

const metrics = computed(() => displayedMetrics.value);
const latencyCards = computed(() =>
  buildLatencyCards(currentMode.value, metrics.value),
);
const overlays = computed(() => frameMeta.value?.overlays || []);
const plateResults = computed(() =>
  overlays.value.filter((item) => item.type === "plate" && item.label),
);
const violationCount = computed(() =>
  overlays.value.filter(
    (item) =>
      item.type === "redlight_violation" ||
      ["violation", "redlight", "闯红灯"].includes(item.status),
  ).length,
);
const serialTelemetry = computed(() =>
  normalizeSerialTelemetry(currentMode.value, frameMeta.value?.serial_data),
);

function updateFrameMeta(meta) {
  frameMeta.value = meta || {};
  lastMetadataAt = performance.now();
  metadataRing.push(frameMeta.value, lastMetadataAt);
  scheduleEventArchiveRefresh(frameMeta.value);
  const sampled = metricsSampler.sample(frameMeta.value, performance.now());
  if (sampled !== null) displayedMetrics.value = sampled;
}

function scheduleEventArchiveRefresh(meta) {
  if (!savedEventTracker.consume(meta?.saved_event_ids)) return;
  if (view.value !== "events") return;
  if (eventRefreshTimer) window.clearTimeout(eventRefreshTimer);
  eventRefreshTimer = window.setTimeout(() => {
    eventRefreshTimer = 0;
    loadEvents(1);
  }, 250);
}

function resetFrameMeta() {
  frameMeta.value = {};
  displayedMetrics.value = {};
  metricsSampler.reset();
  metadataRing.clear();
  lastMetadataAt = 0;
}

function metric(...keys) {
  for (const key of keys) {
    const value = metrics.value[key];
    if (value !== undefined && value !== null) {
      return Number(value).toFixed(1);
    }
  }
  return "--";
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (token.value) headers.set("Authorization", `Bearer ${token.value}`);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    logout();
    throw new Error("登录已失效，请重新登录");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.error || `请求失败 (${response.status})`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return response;
}

function subscribeCurrentStream(streamId = currentMode.value) {
  if (!streamId || !socket?.connected) return;
  socket.emit("subscribe", {
    stream_id: streamId,
    // Keep a real JPEG background until WebRTC/HLS has rendered its first
    // frame. Metadata must never outrun video and leave only annotations.
    include_jpeg: shouldIncludeJpeg(
      videoTransport.value,
      decodedMediaFrame,
    ),
    jpeg_fps: 5,
  });
}

function playbackMetadata() {
  let delayMs = 100;
  if (videoTransport.value === "hls") {
    const latency = Number(hlsPlayer?.latency);
    delayMs = Number.isFinite(latency) ? latency * 1000 : 1500;
  }
  const selected = metadataRing.select(performance.now(), delayMs);
  if (
    !streamOnline.value &&
    lastMetadataAt &&
    performance.now() - lastMetadataAt > 1500
  ) {
    return { ...(selected || frameMeta.value), overlays: [] };
  }
  return selected || frameMeta.value;
}

function activeVideoElement() {
  return videoTransport.value === "hls"
    ? hlsVideo.value
    : webRtcVideo.value;
}

function recordDisplayedFrame() {
  hasFrame.value = true;
  frameFrozen.value = false;
}

function renderWebRtcFrame() {
  const video = activeVideoElement();
  const target = canvas.value;
  if (!["webrtc", "hls"].includes(videoTransport.value)) {
    return;
  }
  if (!video || !target || !video.videoWidth || !video.videoHeight) {
    scheduleWebRtcFrame();
    return;
  }
  if (target.width !== video.videoWidth || target.height !== video.videoHeight) {
    target.width = video.videoWidth;
    target.height = video.videoHeight;
  }
  const context = target.getContext("2d");
  const meta = playbackMetadata();
  drawVideoFrame(
    context,
    video,
    meta?.overlays || [],
    meta?.mode,
    target.width,
    target.height,
  );
  if (!decodedMediaFrame) {
    decodedMediaFrame = true;
    // H.264 is now proven usable. Return to metadata-only delivery so the
    // JPEG safety path has no steady-state bandwidth cost.
    subscribeCurrentStream();
  }
  recordDisplayedFrame();
  scheduleWebRtcFrame();
}

function freezeDisplayedVideo() {
  if (!hasFrame.value || frameFrozen.value) return;
  // The canvas already contains the complete last decoded frame and overlay.
  // Keeping it untouched freezes an opaque image during transport recovery.
  frameFrozen.value = true;
}

function scheduleWebRtcFrame() {
  const video = activeVideoElement();
  if (!video || videoFrameCallback) return;
  videoFrameTarget = video;
  if (video.requestVideoFrameCallback) {
    videoFrameUsesAnimation = false;
    videoFrameCallback = video.requestVideoFrameCallback(() => {
      videoFrameCallback = 0;
      videoFrameTarget = null;
      renderWebRtcFrame();
    });
  } else {
    videoFrameUsesAnimation = true;
    videoFrameCallback = requestAnimationFrame(() => {
      videoFrameCallback = 0;
      videoFrameTarget = null;
      renderWebRtcFrame();
    });
  }
}

async function closeWebRtc(notifyServer = true, preserveFrame = false) {
  if (preserveFrame) freezeDisplayedVideo();
  webRtcGeneration += 1;
  const oldPeer = peerConnection;
  const oldPeerId = peerId;
  peerConnection = null;
  peerId = "";
  if (
    videoFrameCallback &&
    videoFrameTarget === webRtcVideo.value
  ) {
    if (
      !videoFrameUsesAnimation &&
      videoFrameTarget?.cancelVideoFrameCallback
    ) {
      videoFrameTarget.cancelVideoFrameCallback(videoFrameCallback);
    } else {
      cancelAnimationFrame(videoFrameCallback);
    }
    videoFrameCallback = 0;
    videoFrameUsesAnimation = false;
    videoFrameTarget = null;
  }
  if (disconnectedTimer) window.clearTimeout(disconnectedTimer);
  disconnectedTimer = 0;
  if (webRtcVideo.value) webRtcVideo.value.srcObject = null;
  if (oldPeer) oldPeer.close();
  webRtcConnectedAt = 0;
  inboundVideoHealth = createInboundVideoHealth();
  if (notifyServer && oldPeerId && token.value) {
    api("/api/webrtc/close", {
      method: "POST",
      body: JSON.stringify({ peer_id: oldPeerId }),
    }).catch(() => {});
  }
}

function stopHls() {
  nativeHlsActive = false;
  if (videoFrameCallback && videoFrameTarget === hlsVideo.value) {
    if (
      !videoFrameUsesAnimation &&
      videoFrameTarget?.cancelVideoFrameCallback
    ) {
      videoFrameTarget.cancelVideoFrameCallback(videoFrameCallback);
    } else {
      cancelAnimationFrame(videoFrameCallback);
    }
    videoFrameCallback = 0;
    videoFrameTarget = null;
  }
  if (hlsPlayer) {
    hlsPlayer.destroy();
    hlsPlayer = null;
  }
  if (hlsVideo.value) {
    hlsVideo.value.pause();
    hlsVideo.value.removeAttribute("src");
    hlsVideo.value.load();
  }
}

function hlsStreamUrl(streamId) {
  if (!hlsFallbackBaseUrl || !streamId) return "";
  return `${hlsFallbackBaseUrl}/${encodeURIComponent(streamId)}/index.m3u8`;
}

async function startHlsFallback(message = "") {
  const streamId = currentMode.value;
  const url = hlsStreamUrl(streamId);
  if (!url || !hlsVideo.value) {
    return useJpegFallback(message || "LL-HLS不可用", false);
  }
  freezeDisplayedVideo();
  stopHls();
  videoTransport.value = "hls";
  webRtcState.value = "hls-connecting";
  subscribeCurrentStream(streamId);

  if (!Hls.isSupported()) {
    if (!hlsVideo.value.canPlayType("application/vnd.apple.mpegurl")) {
      return useJpegFallback(message || "当前浏览器不支持LL-HLS", false);
    }
    nativeHlsActive = true;
    hlsVideo.value.src = url;
    hlsVideo.value.onloadeddata = () => {
      webRtcState.value = "hls";
      frameFrozen.value = false;
      recordDisplayedFrame();
      scheduleWebRtcFrame();
    };
    await hlsVideo.value.play().catch(() => {});
    return;
  }

  const player = new Hls(lowLatencyHlsConfig((xhr) => {
    xhr.setRequestHeader("Authorization", `Bearer ${token.value}`);
  }));
  hlsPlayer = player;
  player.attachMedia(hlsVideo.value);
  player.on(Hls.Events.MEDIA_ATTACHED, () => {
    if (player === hlsPlayer) player.loadSource(url);
  });
  player.on(Hls.Events.MANIFEST_PARSED, () => {
    if (player !== hlsPlayer) return;
    hlsVideo.value?.play().catch(() => {});
  });
  player.on(Hls.Events.ERROR, (_event, data) => {
    if (player !== hlsPlayer || !data?.fatal) return;
    webRtcState.value = "hls-recovering";
    hlsRetryAt = performance.now() + 1000;
    stopHls();
    void useJpegFallback("LL-HLS播放失败", false);
  });
  hlsVideo.value.onloadeddata = () => {
    if (player !== hlsPlayer) return;
    webRtcState.value = "hls";
    frameFrozen.value = false;
    recordDisplayedFrame();
    scheduleWebRtcFrame();
  };
  if (
    message &&
    performance.now() - lastMediaWarningAt >= WEBRTC_WARNING_INTERVAL_MS
  ) {
    ElMessage.warning(`${message}，已切换稳定视频通道`);
    lastMediaWarningAt = performance.now();
  }
}

async function useJpegFallback(message = "", allowHls = true) {
  if (allowHls && sourceContinuity && hlsFallbackBaseUrl) {
    await startHlsFallback(message);
    return;
  }
  await closeWebRtc(true, true);
  videoTransport.value = "jpeg";
  webRtcState.value = "fallback";
  webRtcRetryAt = performance.now() + WEBRTC_FALLBACK_RETRY_MS;
  subscribeCurrentStream();
  if (
    message &&
    performance.now() - lastMediaWarningAt >= WEBRTC_WARNING_INTERVAL_MS
  ) {
    ElMessage.warning(`${message}，正在快速恢复视频连接`);
    lastMediaWarningAt = performance.now();
  }
}

function clearDisplayedVideo() {
  pendingFrame = null;
  decodedMediaFrame = false;
  hasFrame.value = false;
  frameFrozen.value = false;
  const target = canvas.value;
  if (target) {
    target.getContext("2d").clearRect(0, 0, target.width, target.height);
  }
}

async function handleWebRtcMediaEnded(generation) {
  if (generation !== webRtcGeneration) return;
  await recoverWebRtc(generation, "WebRTC媒体已中断");
}

async function recoverWebRtc(generation, message = "") {
  if (generation !== webRtcGeneration) return;
  const delay = retryDelayMs(webRtcFailedAttempts);
  webRtcFailedAttempts += 1;
  await closeWebRtc(true, videoTransport.value !== "hls");
  webRtcConnectedAt = 0;
  if (
    videoTransport.value !== "hls" &&
    sourceContinuity &&
    shouldUseHlsFallback(webRtcFailedAttempts, 0)
  ) {
    await startHlsFallback(message);
    webRtcRetryAt = performance.now() + delay;
    return;
  }
  if (!streamOnline.value) {
    if (videoTransport.value !== "hls") videoTransport.value = "webrtc";
    webRtcState.value = "waiting-source";
    webRtcRetryAt = performance.now() + delay;
    subscribeCurrentStream();
    return;
  }
  webRtcState.value = "no-media";
  webRtcRetryAt = performance.now() + delay;
  if (videoTransport.value !== "hls") videoTransport.value = "webrtc";
  subscribeCurrentStream();
  if (
    message &&
    performance.now() - lastMediaWarningAt >= WEBRTC_WARNING_INTERVAL_MS
  ) {
    ElMessage.warning(`${message}，正在重新连接`);
    lastMediaWarningAt = performance.now();
  }
}

function scheduleDisconnectedRecovery(peer, generation) {
  if (disconnectedTimer) window.clearTimeout(disconnectedTimer);
  disconnectedTimer = window.setTimeout(() => {
    disconnectedTimer = 0;
    if (
      generation === webRtcGeneration &&
      (
        ["disconnected", "failed"].includes(peer.connectionState) ||
        ["disconnected", "failed"].includes(peer.iceConnectionState)
      )
    ) {
      recoverWebRtc(generation, "WebRTC网络连接中断");
    }
  }, DISCONNECTED_GRACE_MS);
}

async function startWebRtcNow(streamId) {
  if (!streamId || !token.value) return;
  const probingFromHls = videoTransport.value === "hls";
  if (!probingFromHls) decodedMediaFrame = false;
  await closeWebRtc(true, !probingFromHls);
  if (hasFrame.value && !probingFromHls) frameFrozen.value = true;
  const generation = webRtcGeneration;
  try {
    const configResponse = await api("/api/webrtc/config");
    const rtcConfig = await configResponse.json();
    preferredVideoTransport = rtcConfig.transport || "jpeg";
    sourceContinuity = Boolean(rtcConfig.source_continuity);
    hlsFallbackBaseUrl = String(rtcConfig.hls_fallback_url || "");
    if (rtcConfig.transport !== "webrtc" || !rtcConfig.available) {
      await useJpegFallback("服务器未启用 WebRTC");
      return;
    }
    if (!streamOnline.value && !sourceContinuity) {
      if (!probingFromHls) videoTransport.value = "webrtc";
      webRtcState.value = "waiting-source";
      subscribeCurrentStream(streamId);
      return;
    }

    if (!probingFromHls) videoTransport.value = "webrtc";
    webRtcState.value = probingFromHls ? "hls-probing" : "connecting";
    subscribeCurrentStream(streamId);

    const hasTurn = (rtcConfig.iceServers || []).some((server) => {
      const urls = Array.isArray(server.urls) ? server.urls : [server.urls];
      return urls.some((url) => /^turns?:/i.test(String(url || "")));
    });
    const peer = new RTCPeerConnection({
      iceServers: rtcConfig.iceServers || [],
      bundlePolicy: "max-bundle",
      iceTransportPolicy: hasTurn
        ? iceTransportPolicyForAttempt(webRtcFailedAttempts)
        : "all",
    });
    peerConnection = peer;
    const transceiver = peer.addTransceiver("video", { direction: "recvonly" });
    configureLowLatencyReceiver(transceiver.receiver);
    if (!preferH264(transceiver)) {
      throw new Error("当前浏览器不支持 H.264 WebRTC");
    }

    peer.ontrack = async ({ track, streams }) => {
      if (generation !== webRtcGeneration || track.kind !== "video") return;
      const video = webRtcVideo.value;
      if (!video) return;
      track.onended = () => handleWebRtcMediaEnded(generation);
      video.srcObject = streams[0] || new MediaStream([track]);
      video.onloadeddata = () => {
        if (generation !== webRtcGeneration) return;
        if (videoTransport.value === "webrtc") {
          hasFrame.value = true;
          scheduleWebRtcFrame();
        }
      };
      await video.play().catch(() => {});
      scheduleWebRtcFrame();
    };
    peer.onconnectionstatechange = () => {
      if (generation !== webRtcGeneration) return;
      webRtcState.value = peer.connectionState;
      if (peer.connectionState === "connected") {
        if (videoTransport.value === "hls") {
          webRtcState.value = "hls";
        } else {
          videoTransport.value = "webrtc";
        }
        webRtcConnectedAt = performance.now();
        inboundVideoHealth = createInboundVideoHealth(webRtcConnectedAt);
        subscribeCurrentStream(streamId);
      } else if (peer.connectionState === "disconnected") {
        scheduleDisconnectedRecovery(peer, generation);
      } else if (peer.connectionState === "failed") {
        recoverWebRtc(generation, "WebRTC点对点连接失败");
      }
    };
    peer.oniceconnectionstatechange = () => {
      if (generation !== webRtcGeneration) return;
      if (peer.iceConnectionState === "failed") {
        recoverWebRtc(generation, "ICE连接失败");
      } else if (peer.iceConnectionState === "disconnected") {
        scheduleDisconnectedRecovery(peer, generation);
      }
    };

    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGatheringComplete(peer);
    const answerResponse = await api("/api/webrtc/offer", {
      method: "POST",
      body: JSON.stringify({
        stream_id: streamId,
        type: peer.localDescription.type,
        sdp: peer.localDescription.sdp,
      }),
    });
    const answer = await answerResponse.json();
    if (generation !== webRtcGeneration) {
      peer.close();
      return;
    }
    peerId = answer.peer_id;
    await peer.setRemoteDescription({
      type: answer.type,
      sdp: answer.sdp,
    });
  } catch (error) {
    if (
      generation === webRtcGeneration &&
      error?.body?.code === "stream_offline"
    ) {
      await closeWebRtc();
      if (videoTransport.value !== "hls") videoTransport.value = "webrtc";
      webRtcState.value = "waiting-source";
      webRtcRetryAt = performance.now() + 1000;
      subscribeCurrentStream(streamId);
    } else if (generation === webRtcGeneration) {
      await recoverWebRtc(
        generation,
        error.message || "WebRTC初始化失败",
      );
    }
  }
}

async function startWebRtc(streamId, force = false) {
  if (!streamId || !token.value) return;
  if (webRtcStartPromise) return webRtcStartPromise;
  if (!force && performance.now() < webRtcRetryAt) return;
  const promise = startWebRtcNow(streamId);
  webRtcStartPromise = promise;
  try {
    await promise;
  } finally {
    if (webRtcStartPromise === promise) webRtcStartPromise = null;
  }
}

async function login() {
  loginLoading.value = true;
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(loginForm),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "登录失败");
    token.value = body.token;
    username.value = body.username;
    role.value = body.role;
    localStorage.setItem("rk3568_token", body.token);
    localStorage.setItem("rk3568_user", body.username);
    await initializeDashboard();
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    loginLoading.value = false;
  }
}

async function register() {
  if (registerForm.password !== registerForm.confirm) {
    ElMessage.warning("两次输入的密码不一致");
    return;
  }
  try {
    const response = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(registerForm),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "注册失败");
    loginForm.username = registerForm.username;
    loginTab.value = "login";
    ElMessage.success("注册成功，请登录");
  } catch (error) {
    ElMessage.error(error.message);
  }
}

function logout() {
  closeWebRtc();
  stopHls();
  token.value = "";
  username.value = "";
  role.value = "";
  localStorage.removeItem("rk3568_token");
  localStorage.removeItem("rk3568_user");
  if (socket) socket.disconnect();
  socket = null;
  connected.value = false;
  webRtcFailedAttempts = 0;
  savedEventTracker.reset();
}

async function initializeDashboard() {
  try {
    const [userResponse, modeResponse, statusResponse] = await Promise.all([
      api("/api/user"),
      api("/api/modes"),
      api("/api/mode/status"),
    ]);
    const user = await userResponse.json();
    username.value = user.username;
    role.value = user.role;
    if (!user.can_control && view.value === "control") view.value = "live";
    modes.value = await modeResponse.json();
    const status = await statusResponse.json();
    applyInfrastructureStatus(status);
    currentMode.value = status.current_mode;
    running.value = status.running;
    connectSocket();
    await refreshStreams();
  } catch (error) {
    ElMessage.error(error.message);
  }
}

function connectSocket() {
  if (socket) socket.disconnect();
  socket = io(window.location.origin, {
    auth: { token: token.value },
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionDelayMax: 3000,
  });
  socket.on("connect", () => {
    connected.value = true;
    refreshStreams();
  });
  socket.on("disconnect", () => {
    connected.value = false;
  });
  socket.on("connect_error", (error) => {
    if (/auth|token|unauthorized/i.test(error.message)) logout();
  });
  socket.on("mode_status", (status) => {
    applyInfrastructureStatus(status);
    const previous = currentMode.value;
    const statusMode = status.mode || status.current_mode;
    running.value = Boolean(status.running);
    currentMode.value = status.running ? statusMode : null;
    if (previous !== currentMode.value) resetFrameMeta();
    if (previous && previous !== statusMode) {
      stopHls();
      socket.emit("unsubscribe", { stream_id: previous });
    }
    if (status.running) {
      if (
        currentMediaReady() &&
        (previous !== statusMode || !peerConnection)
      ) {
        startWebRtc(statusMode, true);
      } else {
        subscribeCurrentStream(statusMode);
      }
    } else {
      closeWebRtc();
      stopHls();
      webRtcFailedAttempts = 0;
      resetFrameMeta();
      clearDisplayedVideo();
      streamOnline.value = false;
    }
    if (status.error) ElMessage.error(status.error);
  });
  socket.on("agent_status", (status) => {
    applyInfrastructureStatus(status);
  });
  socket.on("log_output", ({ text }) => {
    logs.value = `${logs.value}${text}`.slice(-80000);
    nextTick(() => {
      if (logElement.value) {
        logElement.value.scrollTop = logElement.value.scrollHeight;
      }
    });
  });
  socket.on("log_snapshot", ({ text }) => {
    logs.value = String(text || "").slice(-80000);
    nextTick(() => {
      if (logElement.value) {
        logElement.value.scrollTop = logElement.value.scrollHeight;
      }
    });
  });
  socket.on("frame", (payload, acknowledge) => {
    pendingFrame = payload;
    if (typeof acknowledge === "function") acknowledge();
    pumpFrames();
  });
  socket.on("frame_meta", (payload, acknowledge) => {
    updateFrameMeta(payload?.meta);
    streamOnline.value = true;
    if (
      preferredVideoTransport === "webrtc" &&
      !peerConnection &&
      currentMode.value &&
      currentMediaReady()
    ) {
      startWebRtc(currentMode.value);
    }
    if (typeof acknowledge === "function") acknowledge();
  });
}

async function pumpFrames() {
  if (decodingFrame) return;
  decodingFrame = true;
  try {
    while (pendingFrame) {
      const payload = pendingFrame;
      pendingFrame = null;
      const blob = jpegBlob(payload.jpeg);
      const bitmap = await createImageBitmap(blob);
      if (pendingFrame) {
        bitmap.close();
        continue;
      }
      const target = canvas.value;
      if (target) {
        if (target.width !== bitmap.width || target.height !== bitmap.height) {
          target.width = bitmap.width;
          target.height = bitmap.height;
        }
        const context = target.getContext("2d");
        context.drawImage(bitmap, 0, 0);
        frameFrozen.value = false;
        updateFrameMeta(payload.meta);
        drawOverlays(
          context,
          frameMeta.value.overlays || [],
          frameMeta.value.mode,
          target.width,
          target.height,
        );
        recordDisplayedFrame();
      }
      bitmap.close();
    }
  } catch (error) {
    console.error("frame decode failed", error);
  } finally {
    decodingFrame = false;
    if (pendingFrame) pumpFrames();
  }
}

async function waitForModeAck(
  generation,
  expectedRunning,
  expectedMode = null,
  timeoutMs = 15000,
) {
  const targetGeneration = Number(generation);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await api("/api/mode/status");
    const status = await response.json();
    applyInfrastructureStatus(status);
    const ack = status.last_ack || {};
    const ackGeneration = Number(ack.generation);
    if (
      Number.isFinite(ackGeneration) &&
      ackGeneration >= targetGeneration
    ) {
      if (!ack.success) {
        throw new Error(ack.error || "板端执行失败");
      }
      const acknowledged = ack.status || {};
      const runningMatches =
        Boolean(acknowledged.running) === Boolean(expectedRunning);
      const modeMatches =
        !expectedRunning ||
        !expectedMode ||
        acknowledged.current_mode === expectedMode;
      if (runningMatches && modeMatches) return status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 250));
  }
  throw new Error("等待板端执行确认超时，请检查设备 Agent 和网络连接");
}

async function switchMode(modeId) {
  switchingMode.value = modeId;
  logs.value = "";
  hasFrame.value = false;
  decodedMediaFrame = false;
  streamOnline.value = false;
  resetFrameMeta();
  try {
    const response = await api("/api/mode/start", {
      method: "POST",
      body: JSON.stringify({
        mode: modeId,
        source: "hdmi",
      }),
    });
    const body = await response.json();
    await waitForModeAck(body.generation, true, body.mode);
    currentMode.value = body.mode;
    running.value = true;
    view.value = "live";
    await refreshStreams();
    ElMessage.success(`${body.name} 已在板端启动`);
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    switchingMode.value = "";
  }
}

async function stopMode() {
  if (stoppingMode.value) return;
  stoppingMode.value = true;
  try {
    const response = await api("/api/mode/stop", { method: "POST" });
    const body = await response.json();
    await waitForModeAck(body.generation, false);
    running.value = false;
    currentMode.value = null;
    streamOnline.value = false;
    await closeWebRtc();
    stopHls();
    ElMessage.success("板端程序已停止");
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    stoppingMode.value = false;
  }
}

async function checkWebRtcMediaHealth(now = performance.now()) {
  const peer = peerConnection;
  const generation = webRtcGeneration;
  if (
    mediaHealthInFlight ||
    !["webrtc", "hls"].includes(videoTransport.value) ||
    peer?.connectionState !== "connected" ||
    document.visibilityState === "hidden"
  ) {
    return false;
  }

  mediaHealthInFlight = true;
  try {
    const reports = await peer.getStats();
    if (
      generation !== webRtcGeneration ||
      peer !== peerConnection ||
      peer.connectionState !== "connected"
    ) {
      return false;
    }
    const health = sampleInboundVideoHealth(
      inboundVideoHealth,
      reports,
      now,
    );
    if (!health.startupTimedOut && !health.stalled) {
      if (
        videoTransport.value === "hls" &&
        health.receivedMedia &&
        webRtcConnectedAt > 0 &&
        now - webRtcConnectedAt >= WEBRTC_STABLE_RESET_MS
      ) {
        videoTransport.value = "webrtc";
        webRtcState.value = "connected";
        frameFrozen.value = false;
        stopHls();
        subscribeCurrentStream();
        scheduleWebRtcFrame();
      }
      return false;
    }
    if (
      videoTransport.value !== "hls" &&
      sourceContinuity &&
      shouldUseHlsFallback(
        webRtcFailedAttempts,
        now - inboundVideoHealth.lastProgressAt,
      )
    ) {
      await startHlsFallback("WebRTC连续10秒未收到媒体");
      webRtcRetryAt = now + retryDelayMs(webRtcFailedAttempts);
      return true;
    }
    await recoverWebRtc(
      generation,
      "视频源在线，但WebRTC没有收到媒体",
    );
    return true;
  } catch {
    // getStats can fail transiently while ICE changes candidate pairs. The
    // connection state handlers remain responsible for hard failures.
    return false;
  } finally {
    mediaHealthInFlight = false;
  }
}

async function refreshStreams() {
  if (!token.value || refreshInFlight) return;
  refreshInFlight = true;
  try {
    const response = await api("/api/streams");
    const data = await response.json();
    const stream = currentMode.value ? data[currentMode.value] : null;
    if (stream) applyInfrastructureStatus(stream);
    const online = Boolean(stream?.online);
    const mediaReady = mediaReadyForPlayback(stream);
    if (!online) {
      const hadVideo = streamOnline.value || hasFrame.value || peerConnection;
      streamOnline.value = false;
      if (hadVideo) {
        freezeDisplayedVideo();
      }
      if (currentMode.value) webRtcState.value = "waiting-source";
      if (!sourceContinuity) return;
    }

    if (online) streamOnline.value = true;
    const now = performance.now();
    if (
      webRtcConnectedAt > 0 &&
      now - webRtcConnectedAt >= WEBRTC_STABLE_RESET_MS
    ) {
      webRtcFailedAttempts = 0;
    }
    if (await checkWebRtcMediaHealth(now)) {
      return;
    }

    if (
      preferredVideoTransport === "webrtc" &&
      !peerConnection &&
      currentMode.value &&
      mediaReady &&
      now >= webRtcRetryAt
    ) {
      await startWebRtc(currentMode.value);
    }
    if (
      videoTransport.value === "hls" &&
      !hlsPlayer &&
      !nativeHlsActive &&
      currentMode.value &&
      mediaReady &&
      now >= hlsRetryAt
    ) {
      await startHlsFallback();
    }
  } catch {
    streamOnline.value = false;
  } finally {
    refreshInFlight = false;
  }
}

async function loadEvents(page = 1) {
  eventLoading.value = true;
  try {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(eventPageSize),
    });
    for (const [key, value] of Object.entries(eventFilters)) {
      if (value) params.set(key, value);
    }
    const response = await api(`/api/events?${params}`);
    const body = await response.json();
    events.value = body.items;
    eventTotal.value = body.total;
    eventPage.value = body.page;
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    eventLoading.value = false;
  }
}

async function showEvent(event) {
  if (selectedSnapshot.value) URL.revokeObjectURL(selectedSnapshot.value);
  selectedSnapshot.value = "";
  try {
    const [detailResponse, imageResponse] = await Promise.all([
      api(`/api/events/${event.id}`),
      api(`/api/events/${event.id}/snapshot`),
    ]);
    selectedEvent.value = await detailResponse.json();
    selectedSnapshot.value = URL.createObjectURL(await imageResponse.blob());
  } catch (error) {
    ElMessage.error(error.message);
  }
}

function closeEvent() {
  selectedEvent.value = null;
  if (selectedSnapshot.value) URL.revokeObjectURL(selectedSnapshot.value);
  selectedSnapshot.value = "";
}

async function exportCsv() {
  try {
    const response = await api("/api/events/export.csv");
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "rk3568-events.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error(error.message);
  }
}

async function clearEvents() {
  try {
    await ElMessageBox.confirm(
      "将永久删除全部事件记录和事件截图，是否继续？",
      "清空事件档案",
      {
        confirmButtonText: "确认清空",
        cancelButtonText: "取消",
        type: "warning",
      },
    );
  } catch {
    return;
  }

  eventClearing.value = true;
  try {
    const response = await api("/api/events/clear", { method: "POST" });
    const body = await response.json();
    savedEventTracker.reset();
    closeEvent();
    await loadEvents(1);
    ElMessage.success(`已清空 ${body.deleted || 0} 条事件记录`);
    if (body.snapshot_delete_failures) {
      ElMessage.warning(
        `${body.snapshot_delete_failures} 张历史截图删除失败，请检查文件权限`,
      );
    }
  } catch (error) {
    ElMessage.error(error.message);
  } finally {
    eventClearing.value = false;
  }
}

function formatTime(milliseconds) {
  return new Date(milliseconds).toLocaleString("zh-CN", { hour12: false });
}

let streamTimer;
let heartbeatTimer;

async function sendWebRtcHeartbeat() {
  const currentPeerId = peerId;
  if (!currentPeerId || !token.value || heartbeatInFlight) return;
  heartbeatInFlight = true;
  try {
    const response = await api("/api/webrtc/heartbeat", {
      method: "POST",
      body: JSON.stringify({ peer_id: currentPeerId }),
    });
    const body = await response.json();
    if (!body.alive && currentPeerId === peerId) {
      await recoverWebRtc(webRtcGeneration, "WebRTC会话已失效");
    }
  } catch {
    // A transient API failure must not tear down a healthy media session.
  } finally {
    heartbeatInFlight = false;
  }
}

function restoreWebRtcAfterInterruption() {
  if (!token.value || !currentMode.value) return;
  if (document.visibilityState === "hidden") return;
  webRtcRetryAt = 0;
  if (
    !peerConnection ||
    ["failed", "closed", "disconnected"].includes(
      peerConnection.connectionState,
    )
  ) {
    startWebRtc(currentMode.value, true);
  } else {
    scheduleWebRtcFrame();
  }
}

onMounted(async () => {
  try {
    const response = await fetch("/api/config");
    allowRegister.value = (await response.json()).allow_register;
  } catch {
    allowRegister.value = false;
  }
  if (token.value) await initializeDashboard();
  streamTimer = window.setInterval(refreshStreams, 1500);
  heartbeatTimer = window.setInterval(sendWebRtcHeartbeat, 10_000);
  document.addEventListener("visibilitychange", restoreWebRtcAfterInterruption);
  window.addEventListener("online", restoreWebRtcAfterInterruption);
});

onBeforeUnmount(() => {
  closeWebRtc();
  stopHls();
  if (socket) socket.disconnect();
  window.clearInterval(streamTimer);
  window.clearInterval(heartbeatTimer);
  if (eventRefreshTimer) window.clearTimeout(eventRefreshTimer);
  document.removeEventListener(
    "visibilitychange",
    restoreWebRtcAfterInterruption,
  );
  window.removeEventListener("online", restoreWebRtcAfterInterruption);
  closeEvent();
});
</script>

<template>
  <main v-if="!token" class="login-page">
    <section class="login-visual">
      <div class="contest-identity">
        <img :src="contestLogo" alt="全国大学生集成电路创新创业大赛标志" />
        <div>
          <span>第十届</span>
          <strong>全国大学生集成电路创新创业大赛</strong>
          <small>紫光同创赛题</small>
        </div>
      </div>
      <h1 class="hero-slogan">
        <span>集十年匠心</span>
        <span>创芯辰大海</span>
      </h1>
      <p class="login-copy">
        基于紫光FPGA和RK3568的视频流智能交通识别系统
      </p>
      <div class="feature-strip">
        <span>全高清实时视频</span><span>智能推理</span><span>事件追踪</span>
      </div>
    </section>
    <section class="login-panel">
      <div class="login-card">
        <h2>管理中心</h2>
        <p class="muted">登录后连接 RK3568 边缘设备</p>
        <div v-if="allowRegister" class="auth-tabs">
          <button :class="{ active: loginTab === 'login' }" @click="loginTab = 'login'">登录</button>
          <button :class="{ active: loginTab === 'register' }" @click="loginTab = 'register'">注册</button>
        </div>
        <form v-if="loginTab === 'login'" @submit.prevent="login">
          <label>用户名<input v-model.trim="loginForm.username" autocomplete="username" required /></label>
          <label>密码<input v-model="loginForm.password" type="password" autocomplete="current-password" required /></label>
          <button class="primary wide" :disabled="loginLoading">
            {{ loginLoading ? "正在验证…" : "进入管理平台" }}
          </button>
        </form>
        <form v-else @submit.prevent="register">
          <label>用户名<input v-model.trim="registerForm.username" required minlength="3" /></label>
          <label>密码<input v-model="registerForm.password" type="password" required minlength="8" /></label>
          <label>确认密码<input v-model="registerForm.confirm" type="password" required /></label>
          <button class="primary wide">创建账号</button>
        </form>
      </div>
    </section>
  </main>

  <div v-else class="shell">
    <aside class="sidebar">
      <div class="side-brand">
        <img :src="contestLogo" alt="" />
        <span><strong>第十届集创赛</strong><small>紫光同创赛题</small></span>
      </div>
      <nav>
        <button :class="{ active: view === 'live' }" @click="view = 'live'"><i>01</i>实时态势</button>
        <button v-if="canControl" :class="{ active: view === 'control' }" @click="view = 'control'"><i>02</i>设备控制</button>
        <button :class="{ active: view === 'events' }" @click="view = 'events'; loadEvents(1)"><i>03</i>事件档案</button>
      </nav>
      <div class="side-status">
        <div><span :class="['status-dot', connected && 'online']"></span>管理服务 {{ connected ? "在线" : "离线" }}</div>
        <div><span :class="['status-dot', agentOnline && 'online']"></span>设备代理 {{ agentOnline ? "在线" : "离线" }}</div>
        <div><span :class="['status-dot', boardOnline && 'online']"></span>RK板端 {{ boardOnline ? "在线" : "离线" }}</div>
        <div><span :class="['status-dot', cloudPublisherOnline && 'online']"></span>SRT上行 {{ cloudPublisherOnline ? "在线" : "恢复中" }}</div>
        <div><span :class="['status-dot', streamOnline && 'online']"></span>元数据 {{ streamOnline ? "在线" : "等待" }}</div>
      </div>
      <button class="logout" @click="logout">
        <span class="logout-user">{{ username }}</span>
        <span class="logout-text">退出</span>
      </button>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <div class="page-heading">
          <p class="eyebrow">第十届全国大学生集成电路创新创业大赛 · 紫光同创赛题</p>
          <h2>{{ view === "live" ? "实时态势" : view === "control" ? "设备控制" : "事件档案" }}</h2>
          <p class="project-name">
            基于紫光FPGA和RK3568的视频流智能交通识别系统
            <span>集十年匠心　创芯辰大海</span>
          </p>
        </div>
        <div class="run-state">
          <span :class="['status-dot', running && 'online']"></span>
          {{ running ? modes[currentMode]?.name || currentMode : "当前未运行任务" }}
        </div>
      </header>

      <div v-if="view === 'live'" class="page live-page">
        <section class="metrics-grid">
          <article
            v-for="card in latencyCards"
            :key="card.key"
            :class="{ accent: card.accent }"
          >
            <span>{{ card.label }}</span>
            <strong>{{ card.value }}</strong>
            <em>毫秒</em>
          </article>
        </section>

        <section class="live-grid">
          <div class="video-card">
            <div class="video-head">
              <div class="video-title">
                <span class="live-pill">实时</span>
                <strong>高清视频输入 · 1920×1080</strong>
              </div>
              <span>{{ transportLabel }} · 帧序号 {{ frameMeta.frame_id ?? "--" }}</span>
            </div>
            <div class="canvas-stage">
              <video
                ref="webRtcVideo"
                class="media-source"
                autoplay
                muted
                playsinline
              ></video>
              <video
                ref="hlsVideo"
                class="media-source"
                autoplay
                muted
                playsinline
              ></video>
              <canvas
                ref="canvas"
              ></canvas>
              <div v-if="!hasFrame" class="empty-video">
                <div class="radar"></div>
                <strong>等待板端视频流</strong>
                <span>{{ canControl ? "请在“设备控制”中启动一个模式" : "等待管理员启动识别任务" }}</span>
              </div>
            </div>
          </div>

          <aside class="insight-column">
            <article class="insight-card">
              <div class="card-title">
                <span>{{ currentMode === "plate_serial" ? "实时串口数据" : "当前识别" }}</span>
                <em v-if="serialTelemetry" :class="{ offline: !serialTelemetry.online }">
                  {{ serialTelemetry.statusLabel }}
                </em>
                <em v-else>{{ overlays.length }} 个目标</em>
              </div>
              <div v-if="serialTelemetry" class="serial-telemetry">
                <template v-if="serialTelemetry.validPacket">
                  <div class="serial-raw">
                    <span>
                      原始报文
                      <small v-if="serialTelemetry.ageMs !== null">
                        {{ serialTelemetry.ageMs.toFixed(0) }}ms前
                      </small>
                    </span>
                    <code>{{ serialTelemetry.rawHex || "未提供原始字节" }}</code>
                  </div>
                  <div v-if="serialTelemetry.hasRois" class="serial-rois">
                    <div v-for="roi in serialTelemetry.rois" :key="roi.name">
                      <span>{{ roi.name }}</span>
                      <strong>X1 {{ roi.x1 }}</strong>
                      <strong>Y1 {{ roi.y1 }}</strong>
                      <strong>X2 {{ roi.x2 }}</strong>
                      <strong>Y2 {{ roi.y2 }}</strong>
                    </div>
                  </div>
                  <p v-else class="empty-copy">FPGA当前未上报候选ROI</p>
                </template>
                <p v-else class="empty-copy">
                  {{ !serialTelemetry.online
                    ? "串口离线，正在等待板端重连"
                    : serialTelemetry.hasPacket
                      ? "串口ROI数据格式异常"
                      : "等待FPGA发送ROI数据" }}
                </p>
                <div v-if="plateResults.length" class="plate-results serial-results">
                  <div v-for="plate in plateResults" :key="plate.label" class="plate-result">
                    <strong>{{ plate.label }}</strong>
                    <span>{{ ((plate.rec_score ?? plate.score ?? 0) * 100).toFixed(0) }}%</span>
                  </div>
                </div>
                <p v-else-if="serialTelemetry.validPacket" class="empty-copy">暂无识别结果</p>
              </div>
              <div v-else-if="plateResults.length" class="plate-results">
                <div v-for="plate in plateResults" :key="plate.label" class="plate-result">
                  <strong>{{ plate.label }}</strong>
                  <span>{{ ((plate.rec_score ?? plate.score ?? 0) * 100).toFixed(0) }}%</span>
                </div>
              </div>
              <div v-else-if="currentMode === 'people_redlight'" class="people-summary">
                <strong>{{ overlays.length }}</strong><span>检测目标</span>
                <strong class="danger">{{ violationCount }}</strong><span>违规目标</span>
              </div>
              <p v-else class="empty-copy">暂无识别结果</p>
            </article>
            <button v-if="canControl" class="primary wide" @click="view = 'control'">切换运行模式</button>
          </aside>
        </section>
      </div>

      <div v-else-if="view === 'control' && canControl" class="page control-page">
        <section class="mode-grid">
          <article v-for="modeId in modeOrder" :key="modeId" :class="['mode-card', currentMode === modeId && running && 'active']">
            <div class="mode-number">{{ modeIcon[modeId] }}</div>
            <span class="mode-state">{{ currentMode === modeId && running ? "运行中" : "就绪" }}</span>
            <h3>{{ modes[modeId]?.name }}</h3>
            <p>{{ modes[modeId]?.desc }}</p>
            <button class="primary wide" :disabled="Boolean(switchingMode)" @click="switchMode(modeId)">
              {{ switchingMode === modeId ? "正在切换…" : currentMode === modeId && running ? "重新启动" : "启动模式" }}
            </button>
          </article>
        </section>
        <section class="log-panel">
          <div class="card-title"><span>板端运行日志</span><button @click="logs = ''">清空</button></div>
          <pre ref="logElement">{{ logs || "等待板端程序启动…" }}</pre>
          <button
            v-if="running"
            class="secondary stop-program"
            :disabled="stoppingMode"
            @click="stopMode"
          >
            {{ stoppingMode ? "正在停止…" : "停止当前程序" }}
          </button>
        </section>
      </div>

      <div v-else class="page events-page">
        <section class="event-toolbar">
          <select v-model="eventFilters.mode">
            <option value="">全部模式</option>
            <option v-for="(mode, key) in modes" :key="key" :value="key">{{ mode.name }}</option>
          </select>
          <select v-model="eventFilters.event_type">
            <option value="">全部事件</option>
            <option value="plate">车牌识别</option>
            <option value="redlight_violation">闯红灯</option>
          </select>
          <input v-model.trim="eventFilters.query" placeholder="搜索车牌号或标签" @keyup.enter="loadEvents(1)" />
          <button class="primary" @click="loadEvents(1)">查询</button>
          <button class="secondary" @click="exportCsv">导出表格</button>
          <button
            v-if="canControl"
            class="danger-button"
            :disabled="eventClearing || eventTotal === 0"
            @click="clearEvents"
          >
            {{ eventClearing ? "正在清空…" : "清空档案" }}
          </button>
        </section>
        <section class="event-table">
          <div class="table-head"><span>时间</span><span>模式 / 类型</span><span>识别内容</span><span>置信度</span><span></span></div>
          <div v-if="eventLoading" class="table-empty">正在读取事件…</div>
          <button v-for="event in events" v-else :key="event.id" class="event-row" @click="showEvent(event)">
            <span>{{ formatTime(event.occurred_at_ms) }}</span>
            <span><strong>{{ modes[event.mode]?.name || event.mode }}</strong><small>{{ event.event_type }}</small></span>
            <span class="event-label">{{ event.label || "未命名事件" }}</span>
            <span>{{ (event.confidence * 100).toFixed(1) }}%</span>
            <span>查看截图 →</span>
          </button>
          <div v-if="!eventLoading && !events.length" class="table-empty">暂无符合条件的事件记录</div>
        </section>
        <el-pagination
          v-if="eventTotal > eventPageSize"
          background
          layout="prev, pager, next"
          :total="eventTotal"
          :page-size="eventPageSize"
          :current-page="eventPage"
          @current-change="loadEvents"
        />
      </div>
    </section>
  </div>

  <div v-if="selectedEvent" class="modal-backdrop" @click.self="closeEvent">
    <article class="event-modal">
      <button class="modal-close" @click="closeEvent">×</button>
      <p class="eyebrow">事件编号 {{ selectedEvent.id }}</p>
      <h2>{{ selectedEvent.label || "事件详情" }}</h2>
      <img v-if="selectedSnapshot" :src="selectedSnapshot" alt="事件截图" />
      <div class="event-detail-grid">
        <span>发生时间<strong>{{ formatTime(selectedEvent.occurred_at_ms) }}</strong></span>
        <span>事件类型<strong>{{ selectedEvent.event_type }}</strong></span>
        <span>所属模式<strong>{{ modes[selectedEvent.mode]?.name || selectedEvent.mode }}</strong></span>
        <span>置信度<strong>{{ (selectedEvent.confidence * 100).toFixed(1) }}%</strong></span>
      </div>
    </article>
  </div>
</template>
