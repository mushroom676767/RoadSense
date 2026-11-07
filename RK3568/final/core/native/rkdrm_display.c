#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <rga/im2d.h>
#include <rga/rga.h>

#define RKDRM_PREFERRED_WIDTH 1920
#define RKDRM_PREFERRED_HEIGHT 1080
#define RKDRM_PREFERRED_REFRESH 60

struct rkdrm_bo {
    uint32_t handle;
    uint32_t fb_id;
    uint32_t pitch;
    uint64_t size;
    void *map;
    int prime_fd;
};

struct rkdrm_display {
    int fd;
    uint32_t conn_id;
    uint32_t crtc_id;
    uint32_t width;
    uint32_t height;
    drmModeModeInfo mode;
    drmModeCrtcPtr old_crtc;
    struct rkdrm_bo bo[2];
    int front;
    int pending_front;
    int flip_pending;
};

void rkdrm_display_close(void *handle);

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

/* 函数职责：查找编码器当前或可用的CRTC。 */
static int find_crtc_for_encoder(drmModeResPtr res, drmModeEncoderPtr enc, uint32_t *crtc_id)
{
    if (enc->crtc_id) {
        *crtc_id = enc->crtc_id;
        return 0;
    }
    for (int i = 0; i < res->count_crtcs; i++) {
        if (enc->possible_crtcs & (1 << i)) {
            *crtc_id = res->crtcs[i];
            return 0;
        }
    }
    return -1;
}

/* 函数职责：为已连接Connector选择可驱动它的CRTC。 */
static int find_crtc_for_connector(int fd, drmModeResPtr res, drmModeConnectorPtr conn,
                                   uint32_t *crtc_id)
{
    if (conn->encoder_id) {
        drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoder_id);
        if (enc) {
            int ret = find_crtc_for_encoder(res, enc, crtc_id);
            drmModeFreeEncoder(enc);
            if (ret == 0) {
                return 0;
            }
        }
    }

    for (int i = 0; i < conn->count_encoders; i++) {
        drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoders[i]);
        if (!enc) {
            continue;
        }
        int ret = find_crtc_for_encoder(res, enc, crtc_id);
        drmModeFreeEncoder(enc);
        if (ret == 0) {
            return 0;
        }
    }
    return -1;
}

/* 函数职责：选择连接器与显示模式并保存原CRTC配置。 */
static int setup_connector(struct rkdrm_display *ctx, char *err, size_t err_len)
{
    drmModeResPtr res = drmModeGetResources(ctx->fd);
    if (!res) {
        set_err(err, err_len, "drmModeGetResources failed: %s", strerror(errno));
        return -1;
    }

    int found = 0;
    for (int i = 0; i < res->count_connectors && !found; i++) {
        drmModeConnectorPtr conn = drmModeGetConnector(ctx->fd, res->connectors[i]);
        if (!conn) {
            continue;
        }
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            uint32_t crtc_id = 0;
            if (find_crtc_for_connector(ctx->fd, res, conn, &crtc_id) == 0) {
                int mode_index = 0;
                for (int m = 0; m < conn->count_modes; m++) {
                    drmModeModeInfo *candidate = &conn->modes[m];
                    int refresh = candidate->vrefresh ? candidate->vrefresh : 0;
                    if ((int)candidate->hdisplay == RKDRM_PREFERRED_WIDTH &&
                        (int)candidate->vdisplay == RKDRM_PREFERRED_HEIGHT &&
                        refresh == RKDRM_PREFERRED_REFRESH) {
                        mode_index = m;
                        break;
                    }
                }
                ctx->conn_id = conn->connector_id;
                ctx->crtc_id = crtc_id;
                ctx->mode = conn->modes[mode_index];
                ctx->width = ctx->mode.hdisplay;
                ctx->height = ctx->mode.vdisplay;
                found = 1;
            }
        }
        drmModeFreeConnector(conn);
    }

    drmModeFreeResources(res);
    if (!found) {
        set_err(err, err_len, "no connected DRM connector with a usable CRTC");
        return -1;
    }
    return 0;
}

/* 函数职责：创建Legacy显示所需的Dumb Buffer和Framebuffer。 */
static int create_bo(struct rkdrm_display *ctx, struct rkdrm_bo *bo, char *err, size_t err_len)
{
    struct drm_mode_create_dumb create_req;
    memset(&create_req, 0, sizeof(create_req));
    create_req.width = ctx->width;
    create_req.height = ctx->height;
    create_req.bpp = 32;

    if (drmIoctl(ctx->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        set_err(err, err_len, "DRM_IOCTL_MODE_CREATE_DUMB failed: %s", strerror(errno));
        return -1;
    }

    bo->handle = create_req.handle;
    bo->pitch = create_req.pitch;
    bo->size = create_req.size;
    bo->prime_fd = -1;

    uint32_t handles[4] = {bo->handle, 0, 0, 0};
    uint32_t pitches[4] = {bo->pitch, 0, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};
    if (drmModeAddFB2(ctx->fd, ctx->width, ctx->height, DRM_FORMAT_XRGB8888,
                      handles, pitches, offsets, &bo->fb_id, 0) != 0) {
        set_err(err, err_len, "drmModeAddFB2 failed: %s", strerror(errno));
        return -1;
    }

    struct drm_mode_map_dumb map_req;
    memset(&map_req, 0, sizeof(map_req));
    map_req.handle = bo->handle;
    if (drmIoctl(ctx->fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
        set_err(err, err_len, "DRM_IOCTL_MODE_MAP_DUMB failed: %s", strerror(errno));
        return -1;
    }

    bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, map_req.offset);
    if (bo->map == MAP_FAILED) {
        bo->map = NULL;
        set_err(err, err_len, "mmap dumb buffer failed: %s", strerror(errno));
        return -1;
    }
    memset(bo->map, 0, bo->size);

    if (drmPrimeHandleToFD(ctx->fd, bo->handle, DRM_CLOEXEC, &bo->prime_fd) != 0) {
        set_err(err, err_len, "drmPrimeHandleToFD failed: %s", strerror(errno));
        return -1;
    }

    return 0;
}

/* 函数职责：解除映射并销毁Legacy显示缓冲。 */
static void destroy_bo(struct rkdrm_display *ctx, struct rkdrm_bo *bo)
{
    if (bo->prime_fd >= 0) {
        close(bo->prime_fd);
        bo->prime_fd = -1;
    }
    if (bo->map) {
        munmap(bo->map, bo->size);
        bo->map = NULL;
    }
    if (bo->fb_id) {
        drmModeRmFB(ctx->fd, bo->fb_id);
        bo->fb_id = 0;
    }
    if (bo->handle) {
        struct drm_mode_destroy_dumb destroy_req;
        memset(&destroy_req, 0, sizeof(destroy_req));
        destroy_req.handle = bo->handle;
        drmIoctl(ctx->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
        bo->handle = 0;
    }
}

/* 函数职责：处理DRM Page Flip完成事件并释放上一扫描帧引用。 */
static void page_flip_handler(int fd, unsigned int frame, unsigned int sec,
                              unsigned int usec, void *data)
{
    (void)fd;
    (void)frame;
    (void)sec;
    (void)usec;
    struct rkdrm_display *ctx = (struct rkdrm_display *)data;
    ctx->front = ctx->pending_front;
    ctx->flip_pending = 0;
}

/* 函数职责：等待Legacy后端尚未完成的Page Flip。 */
static int wait_for_pending_flip(struct rkdrm_display *ctx, char *err, size_t err_len)
{
    drmEventContext evctx;
    memset(&evctx, 0, sizeof(evctx));
    evctx.version = DRM_EVENT_CONTEXT_VERSION;
    evctx.page_flip_handler = page_flip_handler;

    while (ctx->flip_pending) {
        struct pollfd pfd;
        pfd.fd = ctx->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int ret = poll(&pfd, 1, 1000);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_err(err, err_len, "poll page flip failed: %s", strerror(errno));
            return -1;
        }
        if (ret == 0) {
            set_err(err, err_len, "page flip timed out");
            return -1;
        }
        if (drmHandleEvent(ctx->fd, &evctx) != 0) {
            set_err(err, err_len, "drmHandleEvent failed: %s", strerror(errno));
            return -1;
        }
    }
    return 0;
}

/* 函数职责：初始化Legacy DRM显示后端并创建双缓冲。 */
void *rkdrm_display_open(const char *device, int *out_width, int *out_height,
                         char *err, size_t err_len)
{
    const char *dev = device && device[0] ? device : "/dev/dri/card0";
    struct rkdrm_display *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    ctx->fd = -1;
    ctx->bo[0].prime_fd = -1;
    ctx->bo[1].prime_fd = -1;

    ctx->fd = open(dev, O_RDWR | O_CLOEXEC);
    if (ctx->fd < 0) {
        set_err(err, err_len, "open %s failed: %s", dev, strerror(errno));
        rkdrm_display_close(ctx);
        return NULL;
    }

    if (setup_connector(ctx, err, err_len) != 0) {
        rkdrm_display_close(ctx);
        return NULL;
    }

    ctx->old_crtc = drmModeGetCrtc(ctx->fd, ctx->crtc_id);

    if (create_bo(ctx, &ctx->bo[0], err, err_len) != 0 ||
        create_bo(ctx, &ctx->bo[1], err, err_len) != 0) {
        rkdrm_display_close(ctx);
        return NULL;
    }

    if (drmModeSetCrtc(ctx->fd, ctx->crtc_id, ctx->bo[0].fb_id, 0, 0,
                       &ctx->conn_id, 1, &ctx->mode) != 0) {
        set_err(err, err_len, "drmModeSetCrtc failed: %s", strerror(errno));
        rkdrm_display_close(ctx);
        return NULL;
    }

    ctx->front = 0;
    ctx->pending_front = 0;
    ctx->flip_pending = 0;
    if (out_width) {
        *out_width = (int)ctx->width;
    }
    if (out_height) {
        *out_height = (int)ctx->height;
    }
    set_err(err, err_len, "");
    return ctx;
}

/* 函数职责：把BGR帧转换并提交到Legacy DRM Page Flip。 */
int rkdrm_display_show(void *handle, const uint8_t *bgr, int width, int height,
                       int stride, char *err, size_t err_len)
{
    struct rkdrm_display *ctx = (struct rkdrm_display *)handle;
    if (!ctx || !bgr) {
        set_err(err, err_len, "invalid display handle or image pointer");
        return -1;
    }
    if (width <= 0 || height <= 0 || stride <= 0) {
        set_err(err, err_len, "invalid image dimensions");
        return -1;
    }
    if (stride != width * 3) {
        set_err(err, err_len, "BGR input must be contiguous: stride=%d width*3=%d",
                stride, width * 3);
        return -1;
    }

    if (ctx->flip_pending && wait_for_pending_flip(ctx, err, err_len) != 0) {
        return -1;
    }

    int back = 1 - ctx->front;
    struct rkdrm_bo *dst_bo = &ctx->bo[back];

    rga_buffer_t src = wrapbuffer_virtualaddr_t((void *)bgr, width, height,
                                                width, height, RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_fd_t(dst_bo->prime_fd, (int)ctx->width, (int)ctx->height,
                                       (int)(dst_bo->pitch / 4), (int)ctx->height,
                                       RK_FORMAT_BGRX_8888);

    /* 源图已经匹配显示分辨率时走1080p60快速路径：跳过RGA缩放，
     * 只执行格式转换拷贝。imcopy比imresize的双线性采样开销更低；
     * 仅在确实需要缩放时使用imresize。 */
    IM_STATUS status;
    if (width == (int)ctx->width && height == (int)ctx->height) {
        status = imcopy_t(src, dst, 1);
    } else {
        status = imresize_t(src, dst, 0, 0, INTER_LINEAR, 1);
    }
    if (status <= 0) {
        set_err(err, err_len, "RGA blit failed: status=%d", status);
        return -1;
    }

    if (drmModePageFlip(ctx->fd, ctx->crtc_id, dst_bo->fb_id,
                        DRM_MODE_PAGE_FLIP_EVENT, ctx) != 0) {
        set_err(err, err_len, "drmModePageFlip failed: %s", strerror(errno));
        return -1;
    }

    ctx->pending_front = back;
    ctx->flip_pending = 1;
    set_err(err, err_len, "");
    return 0;
}

/* 函数职责：等待Flip并恢复原CRTC，随后释放Legacy显示资源。 */
void rkdrm_display_close(void *handle)
{
    struct rkdrm_display *ctx = (struct rkdrm_display *)handle;
    if (!ctx) {
        return;
    }

    if (ctx->fd >= 0 && ctx->flip_pending) {
        char ignored[128];
        wait_for_pending_flip(ctx, ignored, sizeof(ignored));
    }

    if (ctx->fd >= 0 && ctx->old_crtc) {
        drmModeSetCrtc(ctx->fd, ctx->old_crtc->crtc_id, ctx->old_crtc->buffer_id,
                       ctx->old_crtc->x, ctx->old_crtc->y, &ctx->conn_id, 1,
                       &ctx->old_crtc->mode);
        drmModeFreeCrtc(ctx->old_crtc);
        ctx->old_crtc = NULL;
    }

    if (ctx->fd >= 0) {
        destroy_bo(ctx, &ctx->bo[0]);
        destroy_bo(ctx, &ctx->bo[1]);
        close(ctx->fd);
        ctx->fd = -1;
    }
    free(ctx);
}
