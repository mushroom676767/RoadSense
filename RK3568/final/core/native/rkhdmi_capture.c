/*
 * HDMI采集Native后端：V4L2(MJPEG) -> MPP -> NV12 DMA-BUF。
 *
 * 线程和缓冲生命周期：
 * - capture_thread_main只负责DQBUF、校验MJPEG、复制到预分配Packet槽并QBUF；
 * - Python读取线程从Packet池选择最新压缩帧，MPP解码到固定Frame槽；
 * - 每个Frame槽通过generation token和references计数，DRM/RGA/识别全部释放后
 *   才允许解码器复用；
 * - latest-wins只丢弃过期任务，不允许复用仍被扫描或处理的DMA-BUF；
 * - Python调用者必须在close前清空外部Frame引用；close随后停止采集线程并按
 *   V4L2、MPP frame/packet、buffer group的既定顺序销毁本上下文资源。
 *
 * 本文件开头的公开结构和导出函数是ctypes ABI。修改字段顺序、整数宽度或返回
 * 约定会破坏Python调用，纯重构不得调整。
 */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <linux/videodev2.h>

#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>

/* MODULE_TAG 供 mpp_buffer_get / mpp_buffer_group_get_internal 宏展开使用 */
#ifndef MODULE_TAG
#define MODULE_TAG "rkhdmi_capture"
#endif

#define RKHDMI_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

#define RKHDMI_BUFFER_COUNT 4
#define RKHDMI_FRAME_POOL 6
#define RKHDMI_PACKET_POOL 4
#define RKHDMI_MAX_CONSECUTIVE_V4L2_TIMEOUTS 1

struct rkhdmi_stats {
    uint64_t frames;
    uint64_t captured;
    uint64_t drops;
    uint64_t errors;
    uint64_t timeouts;
    uint64_t bytes;
    double v4l2_total_ms;
    double v4l2_max_ms;
    double mpp_total_ms;
    double mpp_max_ms;
    double rga_total_ms;
    double rga_max_ms;
    uint64_t decoded;
    uint64_t output_frames;
    uint64_t compressed_drops;
    uint64_t result_drops;
    uint64_t decode_errors;
    double result_total_ms;
    double result_max_ms;
    double output_total_ms;
    double output_max_ms;
};

struct rkhdmi_v4l2_buffer {
    void *start;
    size_t length;
    int dma_fd;
};

struct rkhdmi_frame_info {
    uint64_t token;
    int32_t dma_fd;
    int32_t width;
    int32_t height;
    int32_t hor_stride;
    int32_t ver_stride;
    uint32_t format;
    uint32_t sequence;
    uint64_t timestamp_ns;
    uint64_t decoded_timestamp_ns;
};

struct rkhdmi_frame_slot {
    MppFrame frame;
    MppBuffer buffer;
    uint32_t generation;
    int references;
    int valid;
    int width;
    int height;
    int hor_stride;
    int ver_stride;
    MppFrameFormat format;
    uint32_t sequence;
    uint64_t timestamp_ns;
    uint64_t dequeued_timestamp_ns;
    uint64_t decoded_timestamp_ns;
};

struct rkhdmi_packet_slot {
    MppBuffer buffer;
    size_t capacity;
    size_t bytes;
    uint32_t sequence;
    uint64_t timestamp_ns;
    uint64_t dequeued_timestamp_ns;
    int ready;
    int decoding;
    int v4l2_index;
};

enum rkhdmi_decoder_kind {
    RKHDMI_DECODER_MPP = 0,
};

struct rkhdmi_capture {
    int fd;
    int streaming;
    int width;
    int height;
    int fps;
    uint32_t pixfmt;
    struct rkhdmi_v4l2_buffer buffers[RKHDMI_BUFFER_COUNT];
    unsigned int buffer_count;
    uint32_t last_sequence;
    int have_sequence;
    int expbuf_enabled;

    pthread_t capture_thread;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int sync_ready;
    int capture_thread_started;
    int stop_capture;
    uint32_t decoded_sequence;
    int consecutive_decode_errors;
    char async_error[256];

    enum rkhdmi_decoder_kind decoder;
    MppCtx mpp_ctx;
    MppApi *mpi;
    /* MJPEG 硬解走官方 advanced 模式：预分配一个输出 MppFrame + 缓冲，
     * 每个 packet 通过 meta 携带该帧，一次 put 一次 get 拿回同一帧，
     * 不依赖 info_change / 内部缓冲队列，规避 MPP 1.5.0 MJPEG 的不稳定路径。 */
    MppBufferGroup mpp_frm_grp;   /* 输出帧缓冲组（预分配，advanced 模式外部缓冲） */
    MppBufferGroup mpp_pkt_grp;   /* 输入包缓冲组（承载 JPEG 码流） */
    struct rkhdmi_frame_slot frame_slots[RKHDMI_FRAME_POOL];
    struct rkhdmi_packet_slot packet_slots[RKHDMI_PACKET_POOL];
    int mpp_frm_w;
    int mpp_frm_h;

    uint8_t *jpeg_tap;
    size_t jpeg_tap_capacity;
    size_t jpeg_tap_bytes;
    uint32_t jpeg_tap_sequence;
    uint64_t jpeg_tap_timestamp_ns;
    uint64_t jpeg_tap_interval_ns;
    uint64_t jpeg_tap_last_ns;
    int jpeg_tap_enabled;

    struct rkhdmi_stats stats;
};

void rkhdmi_capture_close(void *handle);

/* 函数职责：把格式化错误写入调用方缓冲，缓冲为空时安全忽略。 */
static void set_err(char *err, size_t err_len, const char *fmt, ...)
{
    if (!err || err_len == 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

/* 函数职责：读取单调时钟毫秒值，供阶段耗时统计使用。 */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* 函数职责：读取单调时钟纳秒值，避免系统时间调整影响时延。 */
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* 函数职责：把V4L2时间戳换算为统一的单调时钟纳秒值。 */
static uint64_t capture_timestamp_ns(const struct v4l2_buffer *buffer)
{
    if (buffer &&
        (buffer->flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) ==
            V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC &&
        (buffer->timestamp.tv_sec > 0 || buffer->timestamp.tv_usec > 0)) {
        return (uint64_t)buffer->timestamp.tv_sec * 1000000000ull +
               (uint64_t)buffer->timestamp.tv_usec * 1000ull;
    }
    /*
     * 某些UVC驱动不提供单调时钟采集时间戳，此时使用DQBUF完成时刻，
     * 它是板卡侧最早能够观测到该帧的兜底时间点。
     */
    return now_ns();
}

/* 函数职责：累计非采集线程阶段的耗时统计。 */
static void stat_time(double *total, double *max_value, double value)
{
    *total += value;
    if (value > *max_value) {
        *max_value = value;
    }
}

/* 函数职责：在已持有采集锁时累计耗时，避免重复加锁。 */
static void stat_time_capture_locked(struct rkhdmi_capture *ctx,
                                     double *total, double *max_value,
                                     double value)
{
    (void)ctx;
    stat_time(total, max_value, value);
}

/* 函数职责：自动重试被信号中断的ioctl调用。 */
static int xioctl(int fd, unsigned long request, void *arg)
{
    int ret;
    do {
        ret = ioctl(fd, request, arg);
    } while (ret < 0 && errno == EINTR);
    return ret;
}

/* 函数职责：返回当前解码后端名称，供诊断日志使用。 */
static const char *decoder_backend_name(enum rkhdmi_decoder_kind decoder)
{
    (void)decoder;
    return "native-mpp";
}

/* 函数职责：把文本像素格式解析为V4L2 FourCC值。 */
static uint32_t fourcc_from_text(const char *fourcc)
{
    if (!fourcc || !fourcc[0]) {
        return V4L2_PIX_FMT_MJPEG;
    }
    if (strcasecmp(fourcc, "MJPG") == 0 || strcasecmp(fourcc, "MJPEG") == 0) {
        return V4L2_PIX_FMT_MJPEG;
    }
    return 0;
}

/* 函数职责：配置V4L2 MJPEG格式、帧率与MMAP缓冲并开始Streaming。 */
static int setup_v4l2(struct rkhdmi_capture *ctx, const char *device,
                      const char *fourcc, char *err, size_t err_len)
{
    const char *dev = device && device[0] ? device : "/dev/video9";
    ctx->pixfmt = fourcc_from_text(fourcc);
    if (ctx->pixfmt != V4L2_PIX_FMT_MJPEG) {
        set_err(err, err_len,
                "%s HDMI capture currently supports MJPG only, got '%s'",
                decoder_backend_name(ctx->decoder),
                fourcc ? fourcc : "");
        return -1;
    }

    ctx->fd = open(dev, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (ctx->fd < 0) {
        set_err(err, err_len, "open %s failed: %s", dev, strerror(errno));
        return -1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(ctx->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        set_err(err, err_len, "VIDIOC_QUERYCAP failed: %s", strerror(errno));
        return -1;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(cap.capabilities & V4L2_CAP_STREAMING)) {
        set_err(err, err_len, "%s is not a streaming video capture device", dev);
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)ctx->width;
    fmt.fmt.pix.height = (uint32_t)ctx->height;
    fmt.fmt.pix.pixelformat = ctx->pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
        set_err(err, err_len, "VIDIOC_S_FMT %dx%d MJPG failed: %s",
                ctx->width, ctx->height, strerror(errno));
        return -1;
    }
    if (fmt.fmt.pix.pixelformat != ctx->pixfmt) {
        set_err(err, err_len, "device did not accept MJPG format");
        return -1;
    }
    ctx->width = (int)fmt.fmt.pix.width;
    ctx->height = (int)fmt.fmt.pix.height;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = (uint32_t)ctx->fps;
    if (xioctl(ctx->fd, VIDIOC_S_PARM, &parm) < 0) {
        set_err(err, err_len, "VIDIOC_S_PARM %dfps failed: %s",
                ctx->fps, strerror(errno));
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = RKHDMI_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        set_err(err, err_len, "VIDIOC_REQBUFS failed: %s", strerror(errno));
        return -1;
    }
    if (req.count < 2) {
        set_err(err, err_len, "insufficient V4L2 mmap buffers: %u", req.count);
        return -1;
    }

    ctx->buffer_count = req.count;
    for (unsigned int i = 0; i < RKHDMI_BUFFER_COUNT; ++i) {
        ctx->buffers[i].dma_fd = -1;
    }
    int all_exported = 1;
    for (unsigned int i = 0; i < ctx->buffer_count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(ctx->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            set_err(err, err_len, "VIDIOC_QUERYBUF[%u] failed: %s",
                    i, strerror(errno));
            return -1;
        }
        ctx->buffers[i].length = buf.length;
        ctx->buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, ctx->fd, buf.m.offset);
        if (ctx->buffers[i].start == MAP_FAILED) {
            ctx->buffers[i].start = NULL;
            set_err(err, err_len, "mmap V4L2 buffer[%u] failed: %s",
                    i, strerror(errno));
            return -1;
        }
        struct v4l2_exportbuffer exported;
        memset(&exported, 0, sizeof(exported));
        exported.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        exported.index = i;
        exported.flags = O_CLOEXEC;
        if (xioctl(ctx->fd, VIDIOC_EXPBUF, &exported) < 0) {
            all_exported = 0;
        } else {
            ctx->buffers[i].dma_fd = exported.fd;
        }
    }
    if (!all_exported) {
        for (unsigned int i = 0; i < ctx->buffer_count; ++i) {
            if (ctx->buffers[i].dma_fd >= 0) {
                close(ctx->buffers[i].dma_fd);
                ctx->buffers[i].dma_fd = -1;
            }
        }
    }

    for (unsigned int i = 0; i < ctx->buffer_count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            set_err(err, err_len, "VIDIOC_QBUF[%u] failed: %s",
                    i, strerror(errno));
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
        set_err(err, err_len, "VIDIOC_STREAMON failed: %s", strerror(errno));
        return -1;
    }
    ctx->streaming = 1;
    return 0;
}

/* 函数职责：在锁内选择可写Packet槽；没有空槽时按latest-wins回收最旧待解码包。 */
static struct rkhdmi_packet_slot *select_capture_packet_locked(
    struct rkhdmi_capture *ctx)
{
    struct rkhdmi_packet_slot *selected = NULL;
    for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *slot = &ctx->packet_slots[i];
        if (slot->decoding) {
            continue;
        }
        if (!slot->ready) {
            return slot;
        }
        if (!selected ||
            (int32_t)(slot->sequence - selected->sequence) < 0) {
            selected = slot;
        }
    }
    if (selected && selected->ready) {
        ctx->stats.compressed_drops++;
    }
    return selected;
}

/* 函数职责：在锁内记录采集线程错误并唤醒等待者。 */
static void set_async_error_locked(struct rkhdmi_capture *ctx, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->async_error, sizeof(ctx->async_error), fmt, ap);
    va_end(ap);
}

/* 函数职责：循环执行DQBUF、复制压缩帧到Packet槽和QBUF；不负责MPP解码。 */
static void *capture_thread_main(void *arg)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)arg;
    int consecutive_timeouts = 0;

    while (1) {
        pthread_mutex_lock(&ctx->lock);
        int stop = ctx->stop_capture;
        pthread_mutex_unlock(&ctx->lock);
        if (stop) {
            break;
        }

        struct pollfd pfd;
        pfd.fd = ctx->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;

        double t_v4l2 = now_ms();
        int pret = poll(&pfd, 1, 1000);
        if (pret < 0) {
            if (errno == EINTR) {
                continue;
            }
            pthread_mutex_lock(&ctx->lock);
            ctx->stats.errors++;
            set_async_error_locked(ctx, "poll V4L2 failed: %s", strerror(errno));
            ctx->stop_capture = 1;
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        if (pret == 0) {
            pthread_mutex_lock(&ctx->lock);
            ctx->stats.timeouts++;
            consecutive_timeouts++;
            if (
                consecutive_timeouts >=
                RKHDMI_MAX_CONSECUTIVE_V4L2_TIMEOUTS
            ) {
                ctx->stats.errors++;
                set_async_error_locked(
                    ctx,
                    "V4L2 frame timeout for %d consecutive seconds",
                    consecutive_timeouts);
                ctx->stop_capture = 1;
                pthread_cond_broadcast(&ctx->cond);
            }
            pthread_mutex_unlock(&ctx->lock);
            if (
                consecutive_timeouts >=
                RKHDMI_MAX_CONSECUTIVE_V4L2_TIMEOUTS
            ) {
                break;
            }
            continue;
        }
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            pthread_mutex_lock(&ctx->lock);
            ctx->stats.errors++;
            set_async_error_locked(
                ctx, "poll V4L2 device event 0x%x", pfd.revents);
            ctx->stop_capture = 1;
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }

        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) {
                continue;
            }
            pthread_mutex_lock(&ctx->lock);
            ctx->stats.errors++;
            set_async_error_locked(ctx, "VIDIOC_DQBUF failed: %s", strerror(errno));
            ctx->stop_capture = 1;
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        consecutive_timeouts = 0;

        /* DQBUF返回表示压缩帧已经完整进入板卡内存；端到端处理时延从这里起算。 */
        uint64_t dequeued_timestamp_ns = now_ns();
        double v4l2_ms = now_ms() - t_v4l2;
        int should_stop = 0;
        int requeue_now = 0;
        pthread_mutex_lock(&ctx->lock);
        stat_time_capture_locked(ctx, &ctx->stats.v4l2_total_ms,
                                 &ctx->stats.v4l2_max_ms, v4l2_ms);
        ctx->stats.captured++;
        if (ctx->have_sequence && buf.sequence > ctx->last_sequence + 1) {
            ctx->stats.drops += (uint64_t)(buf.sequence - ctx->last_sequence - 1);
            ctx->stats.compressed_drops += (uint64_t)(buf.sequence - ctx->last_sequence - 1);
        }
        ctx->last_sequence = buf.sequence;
        ctx->have_sequence = 1;
        ctx->stats.bytes += buf.bytesused;

        if (buf.index >= ctx->buffer_count || !ctx->buffers[buf.index].start) {
            ctx->stats.errors++;
            set_async_error_locked(ctx, "invalid V4L2 buffer index %u", buf.index);
            ctx->stop_capture = 1;
            pthread_cond_broadcast(&ctx->cond);
            should_stop = 1;
        } else {
            uint64_t capture_ns = capture_timestamp_ns(&buf);
            size_t jpeg_bytes = buf.bytesused;
            const uint8_t *jpeg_src =
                (const uint8_t *)ctx->buffers[buf.index].start;
            while (jpeg_bytes >= 2 &&
                   !(jpeg_src[jpeg_bytes - 2] == 0xff &&
                     jpeg_src[jpeg_bytes - 1] == 0xd9)) {
                jpeg_bytes--;
            }
            int valid_jpeg =
                jpeg_bytes >= 4 &&
                jpeg_src[0] == 0xff &&
                jpeg_src[1] == 0xd8;
            if (!valid_jpeg) {
                /*
                 * USB/UVC复位时，内核可能在报告ENODEV前先返回截断缓冲。
                 * 该数据不能送入MPP，否则解析器会连续报错并可能污染解码器状态；
                 * 因此这里只丢弃当前帧，后续采集仍可恢复。
                 */
                ctx->stats.errors++;
                ctx->stats.decode_errors++;
                ctx->stats.compressed_drops++;
                requeue_now = 1;
            } else {
                if (ctx->jpeg_tap_enabled &&
                    jpeg_bytes <= ctx->jpeg_tap_capacity &&
                    (ctx->jpeg_tap_last_ns == 0 ||
                     capture_ns - ctx->jpeg_tap_last_ns >=
                         ctx->jpeg_tap_interval_ns)) {
                    memcpy(ctx->jpeg_tap, jpeg_src, jpeg_bytes);
                    ctx->jpeg_tap_bytes = jpeg_bytes;
                    ctx->jpeg_tap_sequence = buf.sequence;
                    ctx->jpeg_tap_timestamp_ns = capture_ns;
                    ctx->jpeg_tap_last_ns = capture_ns;
                }
                struct rkhdmi_packet_slot *packet = ctx->expbuf_enabled
                    ? &ctx->packet_slots[buf.index]
                    : select_capture_packet_locked(ctx);
                void *packet_ptr =
                    packet && packet->buffer
                        ? mpp_buffer_get_ptr(packet->buffer)
                        : NULL;
                if (!packet || (ctx->expbuf_enabled &&
                                (packet->ready || packet->decoding))) {
                    ctx->stats.errors++;
                    set_async_error_locked(
                        ctx, "no reusable MPP packet buffer");
                    ctx->stop_capture = 1;
                    pthread_cond_broadcast(&ctx->cond);
                    should_stop = 1;
                } else if (jpeg_bytes > packet->capacity) {
                    ctx->stats.errors++;
                    set_async_error_locked(
                        ctx,
                        "MJPEG frame %zu bytes exceeds MPP packet capacity %zu",
                        jpeg_bytes, packet->capacity);
                    ctx->stop_capture = 1;
                    pthread_cond_broadcast(&ctx->cond);
                    should_stop = 1;
                } else if (ctx->expbuf_enabled) {
                    packet->bytes = jpeg_bytes;
                    packet->sequence = buf.sequence;
                    packet->timestamp_ns = capture_ns;
                    packet->dequeued_timestamp_ns = dequeued_timestamp_ns;
                    packet->ready = 1;
                    pthread_cond_broadcast(&ctx->cond);
                } else if (!packet_ptr) {
                    ctx->stats.errors++;
                    set_async_error_locked(
                        ctx, "MPP packet buffer has no CPU address");
                    ctx->stop_capture = 1;
                    pthread_cond_broadcast(&ctx->cond);
                    should_stop = 1;
                } else {
                    /*
                     * 这是兼容路径唯一一次压缩帧复制：
                     * 从V4L2 MMAP缓冲写入持久MPP Packet缓冲。
                     */
                    memcpy(
                        packet_ptr,
                        ctx->buffers[buf.index].start,
                        jpeg_bytes);
                    packet->bytes = jpeg_bytes;
                    packet->sequence = buf.sequence;
                    packet->timestamp_ns = capture_ns;
                    packet->dequeued_timestamp_ns = dequeued_timestamp_ns;
                    packet->ready = 1;
                    pthread_cond_broadcast(&ctx->cond);
                }
            }
        }
        pthread_mutex_unlock(&ctx->lock);

        if ((!ctx->expbuf_enabled || requeue_now) &&
            xioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            pthread_mutex_lock(&ctx->lock);
            ctx->stats.errors++;
            set_async_error_locked(ctx, "VIDIOC_QBUF failed: %s", strerror(errno));
            ctx->stop_capture = 1;
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        if (should_stop) {
            break;
        }
    }
    return NULL;
}

/* 函数职责：创建V4L2采集线程并等待其进入可用状态。 */
static int start_capture_thread(struct rkhdmi_capture *ctx, char *err, size_t err_len)
{
    int ret = pthread_create(&ctx->capture_thread, NULL, capture_thread_main, ctx);
    if (ret != 0) {
        set_err(err, err_len, "pthread_create capture thread failed: %s", strerror(ret));
        return -1;
    }
    ctx->capture_thread_started = 1;
    return 0;
}

/* 函数职责：初始化MPP MJPEG解码器、外部缓冲组和固定Frame槽池。 */
static int setup_mpp(struct rkhdmi_capture *ctx, char *err, size_t err_len)
{
    MPP_RET ret = mpp_create(&ctx->mpp_ctx, &ctx->mpi);
    if (ret != MPP_OK) {
        set_err(err, err_len, "mpp_create failed: %d", ret);
        return -1;
    }

    /* MJPEG: 每个 UVC buffer 已是一整张完整 JPEG，必须关闭 split 模式。
     * split=1 是给 H.264/H.265 裸流找帧边界用的，喂整帧 JPEG 时解析器会一直
     * 等不到边界 → decode_put_packet 永远 BUFFER_FULL 或只吐 info_change。 */
    RK_U32 parser_split = 0;
    ret = ctx->mpi->control(ctx->mpp_ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &parser_split);
    if (ret != MPP_OK) {
        set_err(err, err_len, "MPP_DEC_SET_PARSER_SPLIT_MODE failed: %d", ret);
        return -1;
    }

    RK_U32 parser_fast = 1;
    ret = ctx->mpi->control(ctx->mpp_ctx, MPP_DEC_SET_PARSER_FAST_MODE, &parser_fast);
    if (ret != MPP_OK) {
        set_err(err, err_len, "MPP_DEC_SET_PARSER_FAST_MODE failed: %d", ret);
        return -1;
    }

    ret = mpp_init(ctx->mpp_ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
    if (ret != MPP_OK) {
        set_err(err, err_len, "mpp_init MJPEG decoder failed: %d", ret);
        return -1;
    }

    /* 必须设置 I/O 超时模式，否则 decode_put_packet 默认非阻塞，
     * 冷启动首帧内部 slot 未就绪时立刻返回 BUFFER_FULL。
     * 输入设为阻塞：put 等到有空位再返回；
     * 输出设为 100ms 有界超时：get 取不到帧也不会把采集线程卡死。 */
    RK_S64 input_timeout = MPP_POLL_BLOCK;   /* -1 = 阻塞 */
    ret = ctx->mpi->control(ctx->mpp_ctx, MPP_SET_INPUT_TIMEOUT, &input_timeout);
    if (ret != MPP_OK) {
        set_err(err, err_len, "MPP_SET_INPUT_TIMEOUT failed: %d", ret);
        return -1;
    }
    RK_S64 output_timeout = 100;             /* 100ms 有界超时 */
    ret = ctx->mpi->control(ctx->mpp_ctx, MPP_SET_OUTPUT_TIMEOUT, &output_timeout);
    if (ret != MPP_OK) {
        set_err(err, err_len, "MPP_SET_OUTPUT_TIMEOUT failed: %d", ret);
        return -1;
    }

    /* NOTE: 官方 mpi_dec_test 对 MJPEG 强制走 advanced 模式
     * (cmd->simple = (type != MPP_VIDEO_CodingMJPEG))：设置输出格式必须在
     * 送第一帧之前完成。生产链路固定输出 YUV420SP（NV12）。 */
    MppFrameFormat out_fmt = MPP_FMT_YUV420SP;
    ret = ctx->mpi->control(ctx->mpp_ctx, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt);
    if (ret != MPP_OK) {
        set_err(err, err_len, "MPP_DEC_SET_OUTPUT_FORMAT failed: %d", ret);
        return -1;
    }

    /*
     * advanced 模式资源预分配（对照官方 dec_decode 的 else 分支）：
     *   - 输入包缓冲组：承载 JPEG 码流，mpp_packet_init_with_buffer 使用
     *   - 输出帧缓冲组 + 一个 MppFrame + 一个底层缓冲
     * MJPEG 可能输出 YUV420(3/2·wh) 或 YUV422(2·wh)，按 16 对齐并预留
     * 2·wh（取较大者）保证任意子采样都放得下，避免运行期再触发 info_change。
     */
    int hor_stride = RKHDMI_ALIGN(ctx->width, 16);
    int ver_stride = RKHDMI_ALIGN(ctx->height, 16);
    size_t frm_size = (size_t)hor_stride * ver_stride * 2;

    ret = mpp_buffer_group_get_internal(&ctx->mpp_pkt_grp, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        set_err(err, err_len, "mpp_buffer_group_get_internal(pkt) failed: %d", ret);
        return -1;
    }
    ret = mpp_buffer_group_get_internal(&ctx->mpp_frm_grp, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        set_err(err, err_len, "mpp_buffer_group_get_internal(frm) failed: %d", ret);
        return -1;
    }

    for (int i = 0; i < RKHDMI_FRAME_POOL; ++i) {
        struct rkhdmi_frame_slot *slot = &ctx->frame_slots[i];
        ret = mpp_frame_init(&slot->frame);
        if (ret != MPP_OK || !slot->frame) {
            set_err(err, err_len, "mpp_frame_init slot %d failed: %d", i, ret);
            return -1;
        }
        ret = mpp_buffer_get(ctx->mpp_frm_grp, &slot->buffer, frm_size);
        if (ret != MPP_OK || !slot->buffer) {
            set_err(err, err_len,
                    "mpp_buffer_get output slot %d (%zu bytes) failed: %d",
                    i, frm_size, ret);
            return -1;
        }
        mpp_frame_set_buffer(slot->frame, slot->buffer);
        slot->generation = 1;
    }
    ctx->mpp_frm_w = ctx->width;
    ctx->mpp_frm_h = ctx->height;

    /*
     * 优先把全部V4L2压缩DMA-BUF导入MPP。只有decode_get_frame返回后才
     * 重新QBUF，因此采集端不会覆盖仍在解码中的JPEG Packet。
     */
    const char *expbuf_env = getenv("RKHDMI_ENABLE_EXPBUF");
    int expbuf_requested =
        expbuf_env && expbuf_env[0] && strcmp(expbuf_env, "0") != 0;
    int imported =
        expbuf_requested && ctx->buffer_count == RKHDMI_PACKET_POOL;
    for (int i = 0; imported && i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *packet = &ctx->packet_slots[i];
        if (ctx->buffers[i].dma_fd < 0) {
            imported = 0;
            break;
        }
        MppBufferInfo info;
        memset(&info, 0, sizeof(info));
        info.type = MPP_BUFFER_TYPE_EXT_DMA;
        info.fd = ctx->buffers[i].dma_fd;
        info.size = ctx->buffers[i].length;
        info.index = i;
        ret = mpp_buffer_import(&packet->buffer, &info);
        if (ret != MPP_OK || !packet->buffer) {
            imported = 0;
            break;
        }
        packet->capacity = ctx->buffers[i].length;
        packet->v4l2_index = i;
    }
    if (!imported) {
        for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
            struct rkhdmi_packet_slot *packet = &ctx->packet_slots[i];
            if (packet->buffer) {
                mpp_buffer_put(packet->buffer);
                packet->buffer = NULL;
            }
            packet->capacity = 0;
            packet->v4l2_index = -1;
        }
    }
    ctx->expbuf_enabled = imported;
    if (imported) {
        fprintf(stderr,
                "V4L2 EXPBUF -> MPP packet zero-copy enabled (%u buffers)\n",
                ctx->buffer_count);
        return 0;
    }

    /* 兼容路径每帧只执行一次V4L2 MMAP到持久MPP Packet的复制。 */
    size_t packet_capacity = (size_t)hor_stride * ver_stride;
    for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *packet = &ctx->packet_slots[i];
        ret = mpp_buffer_get(ctx->mpp_pkt_grp, &packet->buffer, packet_capacity);
        if (ret != MPP_OK || !packet->buffer) {
            set_err(err, err_len,
                    "mpp_buffer_get input packet slot %d (%zu bytes) failed: %d",
                    i, packet_capacity, ret);
            return -1;
        }
        packet->capacity = packet_capacity;
        packet->v4l2_index = -1;
    }
    fprintf(stderr,
            "V4L2 EXPBUF %s; using one-copy MPP packet pool\n",
            expbuf_requested ? "import unavailable" :
                               "available but production-disabled");
    return 0;
}

/*
 * MJPEG 硬解，advanced 模式（对照官方 test/mpi_dec_test.c 的 dec_advanced）。
 *
 * 关键点：官方 dec_decode 对 MJPEG 强制 cmd->simple = 0，即：
 *   1. 用一个预分配、带底层缓冲的 MppFrame 作为输出目标；
 *   2. 把 JPEG 码流塞进一个来自缓冲组的 MppBuffer，用
 *      mpp_packet_init_with_buffer 建包；
 *   3. 通过 packet 的 meta 设置 KEY_OUTPUT_FRAME，把输出帧交给解码器；
 *   4. 一次 decode_put_packet + 一次 decode_get_frame 拿回同一帧。
 * 全程不触发 info_change、不依赖解码器内部输出队列，因此不会出现
 * 老 simple 路径那种 BUFFER_FULL 死循环或 MPP 内部段错误。
 */
static int decode_mjpeg_to_slot(struct rkhdmi_capture *ctx,
                                struct rkhdmi_packet_slot *input,
                                struct rkhdmi_frame_slot *slot,
                                char *err, size_t err_len)
{
    MppPacket packet = NULL;
    MppFrame frame_out = NULL;
    MppMeta meta = NULL;
    int result = -1;

    if (!input || !input->buffer || !slot || !slot->frame || !slot->buffer) {
        set_err(err, err_len, "MPP advanced decoder not initialized");
        return -1;
    }
    if (input->bytes > input->capacity) {
        set_err(err, err_len, "MPP input JPEG %zu bytes exceeds buffer %zu",
                input->bytes, input->capacity);
        return -1;
    }

    double t_mpp = now_ms();

    MPP_RET ret = mpp_packet_init_with_buffer(&packet, input->buffer);
    if (ret != MPP_OK || !packet) {
        stat_time(&ctx->stats.mpp_total_ms, &ctx->stats.mpp_max_ms, now_ms() - t_mpp);
        set_err(err, err_len, "mpp_packet_init_with_buffer failed: %d", ret);
        return -1;
    }
    mpp_packet_set_length(packet, input->bytes);

    /* 把预分配的输出帧通过 meta 交给解码器 */
    meta = mpp_packet_get_meta(packet);
    if (meta) {
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, slot->frame);
    }

    ret = ctx->mpi->decode_put_packet(ctx->mpp_ctx, packet);
    if (ret != MPP_OK) {
        stat_time(&ctx->stats.mpp_total_ms, &ctx->stats.mpp_max_ms, now_ms() - t_mpp);
        set_err(err, err_len, "MPP MJPEG decode_put_packet failed: %d", ret);
        goto cleanup;
    }

    ret = ctx->mpi->decode_get_frame(ctx->mpp_ctx, &frame_out);
    double mpp_ms = now_ms() - t_mpp;
    stat_time(&ctx->stats.mpp_total_ms, &ctx->stats.mpp_max_ms, mpp_ms);
    if (ret != MPP_OK || !frame_out) {
        set_err(err, err_len, "MPP MJPEG decode_get_frame failed: ret %d frame %p",
                ret, frame_out);
        goto cleanup;
    }

    if (mpp_frame_get_errinfo(frame_out) || mpp_frame_get_discard(frame_out)) {
        set_err(err, err_len, "MPP decoded frame marked invalid err=%u discard=%u",
                mpp_frame_get_errinfo(frame_out), mpp_frame_get_discard(frame_out));
        goto cleanup;
    }

    MppBuffer output_buffer = mpp_frame_get_buffer(frame_out);
    MppFrameFormat output_format = mpp_frame_get_fmt(frame_out);
    if (!output_buffer || output_buffer != slot->buffer ||
        MPP_FRAME_FMT_IS_FBC(output_format)) {
        set_err(err, err_len,
                "MPP returned an unexpected or compressed output frame");
        goto cleanup;
    }
    slot->width = (int)mpp_frame_get_width(frame_out);
    slot->height = (int)mpp_frame_get_height(frame_out);
    slot->hor_stride = (int)mpp_frame_get_hor_stride(frame_out);
    slot->ver_stride = (int)mpp_frame_get_ver_stride(frame_out);
    slot->format = output_format;
    if ((output_format & MPP_FRAME_FMT_MASK) != MPP_FMT_YUV420SP) {
        set_err(err, err_len, "expected NV12 output, got MPP format 0x%x",
                output_format);
        goto cleanup;
    }
    result = 1;

cleanup:
    /* Frame和输入缓冲归池管理，只有本地Packet对象需要在此释放。 */
    if (packet) {
        mpp_packet_deinit(&packet);
    }
    return result;
}

/* 函数职责：创建统一采集上下文并显式选择MPP解码后端。 */
void *rkhdmi_capture_open_decoder(const char *device, int width, int height, int fps,
                                  const char *fourcc, const char *decoder,
                                  char *err, size_t err_len)
{
    if (width <= 0 || height <= 0 || fps <= 0) {
        set_err(err, err_len, "invalid capture mode %dx%d@%d", width, height, fps);
        return NULL;
    }
    /* 只支持 MPP 硬解。保留 decoder 形参仅为兼容旧 ABI；非空时必须是 "mpp"。 */
    if (decoder && decoder[0] != '\0' && strcasecmp(decoder, "mpp") != 0) {
        set_err(err, err_len,
                "unsupported native HDMI decoder '%s' (only MPP is supported)", decoder);
        return NULL;
    }

    struct rkhdmi_capture *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    ctx->fd = -1;
    if (pthread_mutex_init(&ctx->lock, NULL) != 0) {
        set_err(err, err_len, "pthread_mutex_init failed");
        free(ctx);
        return NULL;
    }
    if (pthread_cond_init(&ctx->cond, NULL) != 0) {
        set_err(err, err_len, "pthread_cond_init failed");
        pthread_mutex_destroy(&ctx->lock);
        free(ctx);
        return NULL;
    }
    ctx->sync_ready = 1;
    ctx->decoded_sequence = UINT32_MAX;
    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    ctx->decoder = RKHDMI_DECODER_MPP;

    if (setup_v4l2(ctx, device, fourcc, err, err_len) != 0) {
        rkhdmi_capture_close(ctx);
        return NULL;
    }
    if (setup_mpp(ctx, err, err_len) != 0) {
        rkhdmi_capture_close(ctx);
        return NULL;
    }
    if (start_capture_thread(ctx, err, err_len) != 0) {
        rkhdmi_capture_close(ctx);
        return NULL;
    }
    return ctx;
}

/* 函数职责：使用默认MPP后端打开HDMI采集上下文。 */
void *rkhdmi_capture_open(const char *device, int width, int height, int fps,
                          const char *fourcc, char *err, size_t err_len)
{
    return rkhdmi_capture_open_decoder(device, width, height, fps, fourcc,
                                       "mpp", err, err_len);
}

/* 函数职责：配置JPEG旁路帧率与预分配缓冲，供管理端推流使用。 */
int rkhdmi_capture_enable_jpeg_tap(void *handle, int max_fps,
                                  char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || max_fps < 0) {
        set_err(err, err_len, "invalid JPEG tap arguments");
        return -1;
    }

    pthread_mutex_lock(&ctx->lock);
    if (max_fps == 0) {
        ctx->jpeg_tap_enabled = 0;
        ctx->jpeg_tap_bytes = 0;
        free(ctx->jpeg_tap);
        ctx->jpeg_tap = NULL;
        ctx->jpeg_tap_capacity = 0;
        pthread_mutex_unlock(&ctx->lock);
        return 0;
    }

    size_t capacity = 0;
    for (unsigned int i = 0; i < ctx->buffer_count; ++i) {
        if (ctx->buffers[i].length > capacity) {
            capacity = ctx->buffers[i].length;
        }
    }
    if (capacity == 0) {
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len, "V4L2 has no JPEG buffer capacity");
        return -1;
    }
    if (ctx->jpeg_tap_capacity < capacity) {
        uint8_t *resized = realloc(ctx->jpeg_tap, capacity);
        if (!resized) {
            pthread_mutex_unlock(&ctx->lock);
            set_err(err, err_len, "allocate JPEG tap buffer (%zu) failed",
                    capacity);
            return -1;
        }
        ctx->jpeg_tap = resized;
        ctx->jpeg_tap_capacity = capacity;
    }
    ctx->jpeg_tap_bytes = 0;
    ctx->jpeg_tap_sequence = 0;
    ctx->jpeg_tap_timestamp_ns = 0;
    ctx->jpeg_tap_last_ns = 0;
    ctx->jpeg_tap_interval_ns = 1000000000ull / (uint64_t)max_fps;
    ctx->jpeg_tap_enabled = 1;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：读取最新JPEG旁路帧；序号未更新时返回无新帧状态。 */
int rkhdmi_capture_read_jpeg(void *handle, uint32_t last_sequence,
                             uint8_t *dst, size_t dst_capacity,
                             size_t *out_bytes, uint32_t *out_sequence,
                             uint64_t *out_timestamp_ns,
                             char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || !dst || !out_bytes || !out_sequence || !out_timestamp_ns) {
        set_err(err, err_len, "invalid JPEG tap read arguments");
        return -1;
    }

    pthread_mutex_lock(&ctx->lock);
    if (!ctx->jpeg_tap_enabled) {
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len, "JPEG tap is not enabled");
        return -1;
    }
    if (ctx->jpeg_tap_bytes == 0 ||
        ctx->jpeg_tap_sequence == last_sequence) {
        pthread_mutex_unlock(&ctx->lock);
        return 0;
    }
    if (ctx->jpeg_tap_bytes > dst_capacity) {
        size_t required = ctx->jpeg_tap_bytes;
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len,
                "JPEG tap destination too small: need %zu, have %zu",
                required, dst_capacity);
        return -1;
    }
    memcpy(dst, ctx->jpeg_tap, ctx->jpeg_tap_bytes);
    *out_bytes = ctx->jpeg_tap_bytes;
    *out_sequence = ctx->jpeg_tap_sequence;
    *out_timestamp_ns = ctx->jpeg_tap_timestamp_ns;
    pthread_mutex_unlock(&ctx->lock);
    return 1;
}

/* 函数职责：由槽位索引和generation生成防止陈旧引用误用的不透明token。 */
static uint64_t slot_token(int index, const struct rkhdmi_frame_slot *slot)
{
    return ((uint64_t)slot->generation << 8) | (uint64_t)(index + 1);
}

/* 函数职责：在锁内校验token并定位仍属于该generation的Frame槽。 */
static struct rkhdmi_frame_slot *slot_for_token_locked(
    struct rkhdmi_capture *ctx, uint64_t token, int *out_index)
{
    int index = (int)(token & 0xffu) - 1;
    uint32_t generation = (uint32_t)(token >> 8);
    if (index < 0 || index >= RKHDMI_FRAME_POOL || generation == 0) {
        return NULL;
    }
    struct rkhdmi_frame_slot *slot = &ctx->frame_slots[index];
    if (!slot->valid || slot->generation != generation || slot->references <= 0) {
        return NULL;
    }
    if (out_index) {
        *out_index = index;
    }
    return slot;
}

/* 函数职责：在锁内选择序号最新且尚未解码的Packet槽。 */
static struct rkhdmi_packet_slot *latest_packet_locked(
    struct rkhdmi_capture *ctx)
{
    struct rkhdmi_packet_slot *latest = NULL;
    for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *slot = &ctx->packet_slots[i];
        if (!slot->ready || slot->decoding) {
            continue;
        }
        if (!latest || (int32_t)(slot->sequence - latest->sequence) > 0) {
            latest = slot;
        }
    }
    return latest;
}

/* 函数职责：等待并占用最新Packet槽，同时丢弃更旧的待解码压缩帧。 */
static int acquire_latest_packet(struct rkhdmi_capture *ctx,
                                 struct rkhdmi_packet_slot **out,
                                 uint32_t *sequence,
                                 char *err, size_t err_len)
{
    pthread_mutex_lock(&ctx->lock);
    struct rkhdmi_packet_slot *selected = latest_packet_locked(ctx);
    while (!ctx->stop_capture && !selected) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += 1;
        int wait_ret = pthread_cond_timedwait(&ctx->cond, &ctx->lock, &deadline);
        if (wait_ret == ETIMEDOUT) {
            ctx->stats.timeouts++;
            pthread_mutex_unlock(&ctx->lock);
            return 0;
        }
        selected = latest_packet_locked(ctx);
    }
    if (ctx->async_error[0]) {
        set_err(err, err_len, "%s", ctx->async_error);
        pthread_mutex_unlock(&ctx->lock);
        return -1;
    }
    if (ctx->stop_capture) {
        set_err(err, err_len, "native HDMI capture thread stopped");
        pthread_mutex_unlock(&ctx->lock);
        return -1;
    }
    if (!selected) {
        set_err(err, err_len, "native HDMI packet pool stopped");
        pthread_mutex_unlock(&ctx->lock);
        return -1;
    }

    selected->ready = 0;
    selected->decoding = 1;
    *sequence = selected->sequence;
    *out = selected;
    ctx->decoded_sequence = selected->sequence;

    /* latest-wins：立即丢弃过期压缩包，禁止其继续增加显示时延。 */
    for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *slot = &ctx->packet_slots[i];
        if (slot != selected && slot->ready) {
            slot->ready = 0;
            ctx->stats.compressed_drops++;
            if (ctx->expbuf_enabled) {
                struct v4l2_buffer buffer;
                memset(&buffer, 0, sizeof(buffer));
                buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buffer.memory = V4L2_MEMORY_MMAP;
                buffer.index = (uint32_t)slot->v4l2_index;
                if (xioctl(ctx->fd, VIDIOC_QBUF, &buffer) < 0) {
                    set_async_error_locked(
                        ctx, "VIDIOC_QBUF stale EXPBUF[%d] failed: %s",
                        slot->v4l2_index, strerror(errno));
                    ctx->stop_capture = 1;
                }
            }
        }
    }
    pthread_mutex_unlock(&ctx->lock);
    return 1;
}

/* 函数职责：标记Packet槽解码完成并唤醒等待空槽的采集线程。 */
static void release_packet(struct rkhdmi_capture *ctx,
                           struct rkhdmi_packet_slot *packet)
{
    pthread_mutex_lock(&ctx->lock);
    packet->decoding = 0;
    if (ctx->expbuf_enabled) {
        struct v4l2_buffer buffer;
        memset(&buffer, 0, sizeof(buffer));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = (uint32_t)packet->v4l2_index;
        if (xioctl(ctx->fd, VIDIOC_QBUF, &buffer) < 0) {
            set_async_error_locked(
                ctx, "VIDIOC_QBUF decoded EXPBUF[%d] failed: %s",
                packet->v4l2_index, strerror(errno));
            ctx->stop_capture = 1;
        }
    }
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->lock);
}

/* 函数职责：选择引用计数为零的Frame槽供下一次MPP解码写入。 */
static struct rkhdmi_frame_slot *reserve_frame_slot(
    struct rkhdmi_capture *ctx, int *out_index)
{
    struct rkhdmi_frame_slot *selected = NULL;
    pthread_mutex_lock(&ctx->lock);
    for (int i = 0; i < RKHDMI_FRAME_POOL; ++i) {
        struct rkhdmi_frame_slot *slot = &ctx->frame_slots[i];
        if (slot->references == 0) {
            slot->generation++;
            if (slot->generation == 0) {
                slot->generation = 1;
            }
            slot->references = 1;
            slot->valid = 0;
            selected = slot;
            *out_index = i;
            break;
        }
    }
    if (!selected) {
        ctx->stats.result_drops++;
    }
    pthread_mutex_unlock(&ctx->lock);
    return selected;
}

/* 函数职责：解码最新Packet到固定NV12槽并返回带一份引用的不透明帧token。 */
int rkhdmi_capture_read_nv12(void *handle, struct rkhdmi_frame_info *info,
                             char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || !info) {
        set_err(err, err_len, "invalid rkhdmi_capture_read_nv12 arguments");
        return -1;
    }
    struct rkhdmi_packet_slot *packet = NULL;
    uint32_t sequence = 0;
    int packet_result =
        acquire_latest_packet(ctx, &packet, &sequence, err, err_len);
    if (packet_result != 1) {
        return packet_result;
    }
    int slot_index = -1;
    uint64_t source_timestamp_ns = packet->timestamp_ns;
    uint64_t dequeued_timestamp_ns = packet->dequeued_timestamp_ns;
    struct rkhdmi_frame_slot *slot = reserve_frame_slot(ctx, &slot_index);
    if (!slot) {
        release_packet(ctx, packet);
        return 0;
    }

    int ret = decode_mjpeg_to_slot(ctx, packet, slot, err, err_len);
    release_packet(ctx, packet);
    if (ret != 1) {
        int reset_result = MPP_OK;
        pthread_mutex_lock(&ctx->lock);
        slot->references = 0;
        slot->valid = 0;
        ctx->stats.errors++;
        ctx->stats.decode_errors++;
        ctx->consecutive_decode_errors++;
        int rebuild = ctx->consecutive_decode_errors >= 10;
        pthread_cond_broadcast(&ctx->cond);
        pthread_mutex_unlock(&ctx->lock);
        if (rebuild) {
            reset_result = ctx->mpi->reset(ctx->mpp_ctx);
            pthread_mutex_lock(&ctx->lock);
            if (reset_result == MPP_OK) {
                ctx->consecutive_decode_errors = 0;
            } else {
                set_async_error_locked(
                    ctx, "MPP decoder reset failed after 10 errors: %d",
                    reset_result);
            }
            pthread_mutex_unlock(&ctx->lock);
        }
        /*
         * 单个损坏MJPEG帧属于可恢复错误：丢弃该帧并保持视频服务运行；
         * 只有解码器重置失败才升级为致命错误。
         */
        return reset_result == MPP_OK ? 0 : -1;
    }

    pthread_mutex_lock(&ctx->lock);
    ctx->consecutive_decode_errors = 0;
    slot->sequence = sequence;
    slot->decoded_timestamp_ns = now_ns();
    slot->timestamp_ns = source_timestamp_ns
                             ? source_timestamp_ns
                             : slot->decoded_timestamp_ns;
    slot->dequeued_timestamp_ns = dequeued_timestamp_ns
                                      ? dequeued_timestamp_ns
                                      : slot->timestamp_ns;
    slot->valid = 1;
    ctx->stats.decoded++;
    ctx->stats.frames++;
    ctx->stats.output_frames++;
    uint64_t token = slot_token(slot_index, slot);
    memset(info, 0, sizeof(*info));
    info->token = token;
    info->dma_fd = mpp_buffer_get_fd(slot->buffer);
    info->width = slot->width;
    info->height = slot->height;
    info->hor_stride = slot->hor_stride;
    info->ver_stride = slot->ver_stride;
    info->format = V4L2_PIX_FMT_NV12;
    info->sequence = sequence;
    info->timestamp_ns = slot->timestamp_ns;
    info->decoded_timestamp_ns = slot->decoded_timestamp_ns;
    pthread_mutex_unlock(&ctx->lock);
    return 1;
}

/* 函数职责：返回DQBUF完成时间戳；该时刻排除摄像头采集等待但保留MPP及后续处理。 */
int rkhdmi_capture_get_frame_dequeued_timestamp_ns(
    void *handle, uint64_t token, uint64_t *out_timestamp_ns,
    char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || token == 0 || !out_timestamp_ns) {
        set_err(err, err_len, "invalid frame dequeue timestamp request");
        return -1;
    }
    pthread_mutex_lock(&ctx->lock);
    struct rkhdmi_frame_slot *slot = slot_for_token_locked(ctx, token, NULL);
    if (!slot) {
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len, "stale NV12 frame token");
        return -1;
    }
    *out_timestamp_ns = slot->dequeued_timestamp_ns
                            ? slot->dequeued_timestamp_ns
                            : slot->timestamp_ns;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：按token增加NV12 Frame槽引用计数，允许异步消费者继续持有。 */
int rkhdmi_capture_retain_frame(void *handle, uint64_t token,
                                char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || token == 0) {
        set_err(err, err_len, "invalid frame retain request");
        return -1;
    }
    pthread_mutex_lock(&ctx->lock);
    struct rkhdmi_frame_slot *slot = slot_for_token_locked(ctx, token, NULL);
    if (!slot) {
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len, "stale NV12 frame token");
        return -1;
    }
    slot->references++;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：按token减少NV12 Frame槽引用计数，归零后允许解码器复用。 */
int rkhdmi_capture_release_frame(void *handle, uint64_t token,
                                 char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || token == 0) {
        set_err(err, err_len, "invalid frame release request");
        return -1;
    }
    pthread_mutex_lock(&ctx->lock);
    struct rkhdmi_frame_slot *slot = slot_for_token_locked(ctx, token, NULL);
    if (!slot) {
        pthread_mutex_unlock(&ctx->lock);
        set_err(err, err_len, "stale or already released NV12 frame token");
        return -1;
    }
    slot->references--;
    if (slot->references == 0) {
        pthread_cond_broadcast(&ctx->cond);
    }
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：在锁保护下复制或重置采集、解码和转换统计。 */
int rkhdmi_capture_get_stats(void *handle, struct rkhdmi_stats *stats,
                             int reset, char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || !stats) {
        set_err(err, err_len, "invalid rkhdmi_capture_get_stats arguments");
        return -1;
    }
    pthread_mutex_lock(&ctx->lock);
    *stats = ctx->stats;
    if (reset) {
        memset(&ctx->stats, 0, sizeof(ctx->stats));
    }
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：停止采集线程后返回最终统计，底层资源仍由close释放。 */
int rkhdmi_capture_stop_and_get_stats(void *handle, struct rkhdmi_stats *stats,
                                      char *err, size_t err_len)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx || !stats) {
        set_err(err, err_len, "invalid rkhdmi_capture_stop_and_get_stats arguments");
        return -1;
    }
    pthread_mutex_lock(&ctx->lock);
    ctx->stop_capture = 1;
    *stats = ctx->stats;
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

/* 函数职责：先停止采集线程，再按V4L2、MPP帧、Packet和缓冲组顺序释放上下文。 */
void rkhdmi_capture_close(void *handle)
{
    struct rkhdmi_capture *ctx = (struct rkhdmi_capture *)handle;
    if (!ctx) {
        return;
    }

    if (ctx->sync_ready) {
        pthread_mutex_lock(&ctx->lock);
        ctx->stop_capture = 1;
        pthread_cond_broadcast(&ctx->cond);
        pthread_mutex_unlock(&ctx->lock);
    }
    if (ctx->streaming && ctx->fd >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
        ctx->streaming = 0;
    }
    if (ctx->capture_thread_started) {
        pthread_join(ctx->capture_thread, NULL);
        ctx->capture_thread_started = 0;
    }
    for (unsigned int i = 0; i < ctx->buffer_count; i++) {
        if (ctx->buffers[i].start) {
            munmap(ctx->buffers[i].start, ctx->buffers[i].length);
            ctx->buffers[i].start = NULL;
        }
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
        ctx->fd = -1;
    }
    if (ctx->mpp_ctx) {
        mpp_destroy(ctx->mpp_ctx);
        ctx->mpp_ctx = NULL;
        ctx->mpi = NULL;
    }
    /* advanced 模式预分配资源：先销毁 ctx 再释放帧/缓冲/缓冲组
     * （顺序对照官方 dec_decode 的 MPP_TEST_OUT 清理段） */
    for (int i = 0; i < RKHDMI_FRAME_POOL; ++i) {
        struct rkhdmi_frame_slot *slot = &ctx->frame_slots[i];
        if (slot->frame) {
            mpp_frame_deinit(&slot->frame);
            slot->frame = NULL;
        }
        if (slot->buffer) {
            mpp_buffer_put(slot->buffer);
            slot->buffer = NULL;
        }
    }
    for (int i = 0; i < RKHDMI_PACKET_POOL; ++i) {
        struct rkhdmi_packet_slot *packet = &ctx->packet_slots[i];
        if (packet->buffer) {
            mpp_buffer_put(packet->buffer);
            packet->buffer = NULL;
        }
    }
    for (unsigned int i = 0; i < ctx->buffer_count; ++i) {
        if (ctx->buffers[i].dma_fd >= 0) {
            close(ctx->buffers[i].dma_fd);
            ctx->buffers[i].dma_fd = -1;
        }
    }
    if (ctx->mpp_frm_grp) {
        mpp_buffer_group_put(ctx->mpp_frm_grp);
        ctx->mpp_frm_grp = NULL;
    }
    if (ctx->mpp_pkt_grp) {
        mpp_buffer_group_put(ctx->mpp_pkt_grp);
        ctx->mpp_pkt_grp = NULL;
    }
    free(ctx->jpeg_tap);
    ctx->jpeg_tap = NULL;
    if (ctx->sync_ready) {
        pthread_cond_destroy(&ctx->cond);
        pthread_mutex_destroy(&ctx->lock);
        ctx->sync_ready = 0;
    }
    free(ctx);
}
