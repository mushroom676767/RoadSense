/*
 * DRM Atomic双Plane显示后端。
 *
 * video plane直接扫描MPP导出的NV12 DMA-BUF(zpos=0)，overlay plane扫描C侧
 * ARGB8888双缓冲(zpos=1)。Atomic Commit只提交状态；Page Flip事件到达后，
 * 上一个视频token才退休并可由采集帧池复用。
 *
 * Overlay按generation更新。每个ARGB缓冲只清除自己上一次使用的脏矩形，
 * FreeType字形按码点和字号缓存；结果TTL到期时提交空Overlay。overlay_lock只
 * 保护结果快照，绘制和drmModeAtomicCommit不在锁内，避免阻塞结果提交线程。
 *
 * rkdrm_overlay_item、rkdrm_atomic_stats及导出函数属于ctypes ABI，不得改变布局。
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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <drm/drm.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#define RKATOMIC_MAX_ITEMS 32
#define RKATOMIC_MAX_DIRTY 256
#define RKATOMIC_FB_CACHE 8
#define RKATOMIC_GLYPH_CACHE 256
#define RKATOMIC_OVERLAY_SAMPLES 4096

#define RKATOMIC_ITEM_LABEL 0x01u
#define RKATOMIC_ITEM_CORNERS 0x02u
#define RKATOMIC_ITEM_FULL 0x04u
#define RKATOMIC_ITEM_GENERIC 0x08u
#define RKATOMIC_ITEM_POLYGON 0x10u

struct rkdrm_overlay_item {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    float landmarks[8];
    float detect_conf;
    float rec_conf;
    uint32_t flags;
    char plate_no[32];
    char plate_color[24];
    char label[64];
    float score;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

struct rkdrm_atomic_stats {
    uint64_t submitted;
    uint64_t displayed;
    uint64_t commit_errors;
    uint64_t overlay_updates;
    uint64_t overlay_expirations;
    double commit_total_ms;
    double commit_max_ms;
    double overlay_total_ms;
    double overlay_max_ms;
    double display_fps;
    uint64_t last_flip_ns;
    uint64_t last_overlay_source_timestamp_ns;
    uint64_t last_overlay_ready_timestamp_ns;
    double last_overlay_e2e_ms;
    double last_overlay_display_ms;
};

struct rect_i {
    int x;
    int y;
    int w;
    int h;
};

struct overlay_bo {
    uint32_t handle;
    uint32_t fb_id;
    uint32_t pitch;
    uint64_t size;
    void *map;
    uint8_t *shadow;
    struct rect_i dirty[RKATOMIC_MAX_DIRTY];
    int dirty_count;
    uint64_t rendered_generation;
};

struct plane_props {
    uint32_t fb_id;
    uint32_t crtc_id;
    uint32_t src_x;
    uint32_t src_y;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t crtc_x;
    uint32_t crtc_y;
    uint32_t crtc_w;
    uint32_t crtc_h;
    uint32_t zpos;
    uint32_t alpha;
    uint32_t blend;
    uint64_t blend_premultiplied;
};

struct imported_fb {
    int used;
    dev_t dev;
    ino_t ino;
    uint32_t handle;
    uint32_t fb_id;
};

struct cached_glyph {
    int valid;
    uint32_t codepoint;
    int pixel_size;
    int width;
    int height;
    int left;
    int top;
    int advance;
    uint8_t *bitmap;
    uint64_t age;
};

struct rkdrm_atomic {
    int fd;
    uint32_t connector_id;
    uint32_t crtc_id;
    int crtc_index;
    uint32_t width;
    uint32_t height;
    drmModeModeInfo mode;
    drmModeCrtcPtr old_crtc;

    uint32_t connector_crtc_id_prop;
    uint32_t crtc_active_prop;
    uint32_t crtc_mode_id_prop;
    uint32_t mode_blob_id;

    uint32_t video_plane_id;
    uint32_t overlay_plane_id;
    struct plane_props video_props;
    struct plane_props overlay_props;

    struct overlay_bo overlay[2];
    int overlay_front;
    int overlay_pending;
    struct imported_fb imported[RKATOMIC_FB_CACHE];

    int modeset_done;
    int flip_pending;
    uint64_t current_token;
    uint64_t pending_token;
    uint64_t retired_token;

    pthread_mutex_t overlay_lock;
    int overlay_lock_ready;
    struct rkdrm_overlay_item items[RKATOMIC_MAX_ITEMS];
    int item_count;
    int overlay_ttl_ms;
    uint64_t overlay_timestamp_ns;
    uint64_t overlay_source_timestamp_ns;
    uint64_t overlay_ready_timestamp_ns;
    uint64_t overlay_generation;
    int overlay_expired;
    uint64_t current_overlay_generation;
    uint64_t pending_overlay_generation;
    uint64_t pending_overlay_source_timestamp_ns;
    uint64_t pending_overlay_ready_timestamp_ns;

    FT_Library ft_library;
    FT_Face ft_face;
    struct cached_glyph glyphs[RKATOMIC_GLYPH_CACHE];
    uint64_t glyph_age;
    char font_path[512];

    uint64_t fps_window_ns;
    uint64_t fps_frames;
    double overlay_samples[RKATOMIC_OVERLAY_SAMPLES];
    uint64_t overlay_sample_count;
    struct rkdrm_atomic_stats stats;
};

void rkdrm_atomic_close(void *handle);

/* 函数职责：读取单调时钟纳秒值，用于跨线程帧时间戳。 */
static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* 函数职责：读取单调时钟毫秒值，用于性能统计。 */
static double monotonic_ms(void)
{
    return (double)monotonic_ns() / 1000000.0;
}

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

/* 函数职责：累计一次耗时样本并更新最大值。 */
static void update_time(double *total, double *max_value, double value)
{
    *total += value;
    if (value > *max_value) {
        *max_value = value;
    }
}

/* 函数职责：按名称查询DRM对象属性ID，并可返回当前属性值。 */
static uint32_t object_prop_id(int fd, uint32_t object_id, uint32_t object_type,
                               const char *name, uint64_t *value)
{
    drmModeObjectPropertiesPtr props =
        drmModeObjectGetProperties(fd, object_id, object_type);
    if (!props) {
        return 0;
    }
    uint32_t result = 0;
    for (uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
        if (!prop) {
            continue;
        }
        if (strcmp(prop->name, name) == 0) {
            result = prop->prop_id;
            if (value) {
                *value = props->prop_values[i];
            }
            drmModeFreeProperty(prop);
            break;
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return result;
}

/* 函数职责：查询DRM枚举属性中指定名称对应的数值。 */
static uint64_t prop_enum_value(int fd, uint32_t object_id, uint32_t object_type,
                                const char *prop_name, const char *enum_name,
                                uint64_t fallback)
{
    drmModeObjectPropertiesPtr props =
        drmModeObjectGetProperties(fd, object_id, object_type);
    if (!props) {
        return fallback;
    }
    uint64_t result = fallback;
    for (uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
        if (!prop) {
            continue;
        }
        if (strcmp(prop->name, prop_name) == 0) {
            for (int e = 0; e < prop->count_enums; ++e) {
                if (strcmp(prop->enums[e].name, enum_name) == 0) {
                    result = prop->enums[e].value;
                    break;
                }
            }
            drmModeFreeProperty(prop);
            break;
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return result;
}

/* 函数职责：判断Plane名称是否包含期望的硬件单元名称。 */
static int plane_name_contains(int fd, uint32_t plane_id, const char *needle)
{
    drmModeObjectPropertiesPtr props =
        drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) {
        return 0;
    }
    int found = 0;
    for (uint32_t i = 0; i < props->count_props && !found; ++i) {
        drmModePropertyPtr prop = drmModeGetProperty(fd, props->props[i]);
        if (!prop) {
            continue;
        }
        if (strcmp(prop->name, "NAME") == 0) {
            for (int e = 0; e < prop->count_enums; ++e) {
                size_t needle_len = strlen(needle);
                if (strncmp(prop->enums[e].name, needle, needle_len) == 0 &&
                    (prop->enums[e].name[needle_len] == '\0' ||
                     prop->enums[e].name[needle_len] == '-')) {
                    found = 1;
                    break;
                }
            }
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return found;
}

/* 函数职责：判断Plane是否支持目标CRTC和像素格式。 */
static int plane_supports(drmModePlanePtr plane, uint32_t format)
{
    for (uint32_t i = 0; i < plane->count_formats; ++i) {
        if (plane->formats[i] == format) {
            return 1;
        }
    }
    return 0;
}

/* 函数职责：绑定Atomic提交所需的Plane属性ID。 */
static int bind_plane_props(struct rkdrm_atomic *ctx, uint32_t plane_id,
                            struct plane_props *props, char *err, size_t err_len)
{
    memset(props, 0, sizeof(*props));
#define GET_REQUIRED(field, text)                                                   \
    do {                                                                             \
        props->field = object_prop_id(ctx->fd, plane_id, DRM_MODE_OBJECT_PLANE,      \
                                      text, NULL);                                    \
        if (!props->field) {                                                         \
            set_err(err, err_len, "plane %u lacks required property %s",            \
                    plane_id, text);                                                  \
            return -1;                                                               \
        }                                                                            \
    } while (0)
    GET_REQUIRED(fb_id, "FB_ID");
    GET_REQUIRED(crtc_id, "CRTC_ID");
    GET_REQUIRED(src_x, "SRC_X");
    GET_REQUIRED(src_y, "SRC_Y");
    GET_REQUIRED(src_w, "SRC_W");
    GET_REQUIRED(src_h, "SRC_H");
    GET_REQUIRED(crtc_x, "CRTC_X");
    GET_REQUIRED(crtc_y, "CRTC_Y");
    GET_REQUIRED(crtc_w, "CRTC_W");
    GET_REQUIRED(crtc_h, "CRTC_H");
#undef GET_REQUIRED
    props->zpos = object_prop_id(ctx->fd, plane_id, DRM_MODE_OBJECT_PLANE,
                                 "zpos", NULL);
    props->alpha = object_prop_id(ctx->fd, plane_id, DRM_MODE_OBJECT_PLANE,
                                  "alpha", NULL);
    props->blend = object_prop_id(ctx->fd, plane_id, DRM_MODE_OBJECT_PLANE,
                                  "pixel blend mode", NULL);
    props->blend_premultiplied =
        prop_enum_value(ctx->fd, plane_id, DRM_MODE_OBJECT_PLANE,
                        "pixel blend mode", "Pre-multiplied", 0);
    return 0;
}

/* 函数职责：选择已连接Connector及兼容CRTC，并保存启动前状态。 */
static int find_connector_crtc(struct rkdrm_atomic *ctx, char *err, size_t err_len)
{
    drmModeResPtr resources = drmModeGetResources(ctx->fd);
    if (!resources) {
        set_err(err, err_len, "drmModeGetResources failed: %s", strerror(errno));
        return -1;
    }
    drmModeConnectorPtr selected = NULL;
    for (int i = 0; i < resources->count_connectors; ++i) {
        drmModeConnectorPtr connector =
            drmModeGetConnector(ctx->fd, resources->connectors[i]);
        if (!connector) {
            continue;
        }
        if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
            selected = connector;
            break;
        }
        drmModeFreeConnector(connector);
    }
    if (!selected) {
        drmModeFreeResources(resources);
        set_err(err, err_len, "no connected DRM connector");
        return -1;
    }

    uint32_t crtc_id = 0;
    if (selected->encoder_id) {
        drmModeEncoderPtr encoder = drmModeGetEncoder(ctx->fd, selected->encoder_id);
        if (encoder) {
            crtc_id = encoder->crtc_id;
            drmModeFreeEncoder(encoder);
        }
    }
    if (!crtc_id) {
        for (int e = 0; e < selected->count_encoders && !crtc_id; ++e) {
            drmModeEncoderPtr encoder =
                drmModeGetEncoder(ctx->fd, selected->encoders[e]);
            if (!encoder) {
                continue;
            }
            for (int c = 0; c < resources->count_crtcs; ++c) {
                if (encoder->possible_crtcs & (1u << c)) {
                    crtc_id = resources->crtcs[c];
                    break;
                }
            }
            drmModeFreeEncoder(encoder);
        }
    }
    int crtc_index = -1;
    for (int c = 0; c < resources->count_crtcs; ++c) {
        if (resources->crtcs[c] == crtc_id) {
            crtc_index = c;
            break;
        }
    }
    if (!crtc_id || crtc_index < 0) {
        drmModeFreeConnector(selected);
        drmModeFreeResources(resources);
        set_err(err, err_len, "unable to select CRTC for connector");
        return -1;
    }

    drmModeModeInfo mode = selected->modes[0];
    for (int m = 0; m < selected->count_modes; ++m) {
        drmModeModeInfo *candidate = &selected->modes[m];
        if ((candidate->type & DRM_MODE_TYPE_PREFERRED) ||
            (candidate->hdisplay == 1920 && candidate->vdisplay == 1080 &&
             candidate->vrefresh == 60)) {
            mode = *candidate;
            if (candidate->hdisplay == 1920 && candidate->vdisplay == 1080 &&
                candidate->vrefresh == 60) {
                break;
            }
        }
    }

    ctx->connector_id = selected->connector_id;
    ctx->crtc_id = crtc_id;
    ctx->crtc_index = crtc_index;
    ctx->mode = mode;
    ctx->width = mode.hdisplay;
    ctx->height = mode.vdisplay;
    drmModeFreeConnector(selected);
    drmModeFreeResources(resources);
    return 0;
}

/* 函数职责：按格式、possible_crtcs和名称偏好选择NV12与ARGB Plane。 */
static int select_planes(struct rkdrm_atomic *ctx, char *err, size_t err_len)
{
    drmModePlaneResPtr resources = drmModeGetPlaneResources(ctx->fd);
    if (!resources) {
        set_err(err, err_len, "drmModeGetPlaneResources failed: %s", strerror(errno));
        return -1;
    }
    int best_video_score = -1;
    int best_overlay_score = -1;
    for (uint32_t i = 0; i < resources->count_planes; ++i) {
        drmModePlanePtr plane = drmModeGetPlane(ctx->fd, resources->planes[i]);
        if (!plane) {
            continue;
        }
        if (!(plane->possible_crtcs & (1u << ctx->crtc_index))) {
            drmModeFreePlane(plane);
            continue;
        }
        if (plane_supports(plane, DRM_FORMAT_NV12)) {
            int score = 10;
            if (plane_name_contains(ctx->fd, plane->plane_id, "Esmart0")) {
                score += 100;
            }
            uint64_t type = 0;
            object_prop_id(ctx->fd, plane->plane_id, DRM_MODE_OBJECT_PLANE,
                           "type", &type);
            if (type == DRM_PLANE_TYPE_OVERLAY) {
                score += 20;
            }
            if (score > best_video_score) {
                best_video_score = score;
                ctx->video_plane_id = plane->plane_id;
            }
        }
        drmModeFreePlane(plane);
    }

    for (uint32_t i = 0; i < resources->count_planes; ++i) {
        drmModePlanePtr plane = drmModeGetPlane(ctx->fd, resources->planes[i]);
        if (!plane) {
            continue;
        }
        if (plane->plane_id == ctx->video_plane_id ||
            !(plane->possible_crtcs & (1u << ctx->crtc_index)) ||
            !plane_supports(plane, DRM_FORMAT_ARGB8888)) {
            drmModeFreePlane(plane);
            continue;
        }
        int score = 10;
        if (plane_name_contains(ctx->fd, plane->plane_id, "Smart0")) {
            score += 100;
        }
        uint64_t type = 0;
        object_prop_id(ctx->fd, plane->plane_id, DRM_MODE_OBJECT_PLANE,
                       "type", &type);
        if (type == DRM_PLANE_TYPE_PRIMARY) {
            score += 20;
        }
        if (score > best_overlay_score) {
            best_overlay_score = score;
            ctx->overlay_plane_id = plane->plane_id;
        }
        drmModeFreePlane(plane);
    }
    drmModeFreePlaneResources(resources);

    if (!ctx->video_plane_id || !ctx->overlay_plane_id) {
        set_err(err, err_len,
                "no compatible NV12+ARGB plane pair for CRTC index %d",
                ctx->crtc_index);
        return -1;
    }
    if (bind_plane_props(ctx, ctx->video_plane_id, &ctx->video_props,
                         err, err_len) != 0 ||
        bind_plane_props(ctx, ctx->overlay_plane_id, &ctx->overlay_props,
                         err, err_len) != 0) {
        return -1;
    }
    return 0;
}

/* 函数职责：创建并映射一块ARGB Overlay Dumb Buffer。 */
static int create_overlay_bo(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                             char *err, size_t err_len)
{
    struct drm_mode_create_dumb create;
    memset(&create, 0, sizeof(create));
    create.width = ctx->width;
    create.height = ctx->height;
    create.bpp = 32;
    if (drmIoctl(ctx->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0) {
        set_err(err, err_len, "CREATE_DUMB ARGB failed: %s", strerror(errno));
        return -1;
    }
    bo->handle = create.handle;
    bo->pitch = create.pitch;
    bo->size = create.size;

    uint32_t handles[4] = {bo->handle, 0, 0, 0};
    uint32_t pitches[4] = {bo->pitch, 0, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};
    if (drmModeAddFB2(ctx->fd, ctx->width, ctx->height, DRM_FORMAT_ARGB8888,
                      handles, pitches, offsets, &bo->fb_id, 0) != 0) {
        set_err(err, err_len, "drmModeAddFB2 ARGB failed: %s", strerror(errno));
        return -1;
    }
    struct drm_mode_map_dumb map;
    memset(&map, 0, sizeof(map));
    map.handle = bo->handle;
    if (drmIoctl(ctx->fd, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0) {
        set_err(err, err_len, "MAP_DUMB ARGB failed: %s", strerror(errno));
        return -1;
    }
    bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   ctx->fd, map.offset);
    if (bo->map == MAP_FAILED) {
        bo->map = NULL;
        set_err(err, err_len, "mmap ARGB failed: %s", strerror(errno));
        return -1;
    }
    memset(bo->map, 0, bo->size);
    bo->shadow = calloc(1, bo->size);
    if (!bo->shadow) {
        set_err(err, err_len, "allocating cached ARGB shadow failed");
        return -1;
    }
    return 0;
}

/* 函数职责：解除映射并销毁ARGB Overlay Buffer及Framebuffer。 */
static void destroy_overlay_bo(struct rkdrm_atomic *ctx, struct overlay_bo *bo)
{
    free(bo->shadow);
    bo->shadow = NULL;
    if (bo->map) {
        munmap(bo->map, bo->size);
        bo->map = NULL;
    }
    if (bo->fb_id) {
        drmModeRmFB(ctx->fd, bo->fb_id);
        bo->fb_id = 0;
    }
    if (bo->handle) {
        struct drm_mode_destroy_dumb destroy;
        memset(&destroy, 0, sizeof(destroy));
        destroy.handle = bo->handle;
        drmIoctl(ctx->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        bo->handle = 0;
    }
}

/* 函数职责：通过显式路径或Fontconfig解析可显示中文的字体文件。 */
static int resolve_font(const char *requested, char *path, size_t path_len,
                        int *face_index)
{
    if (requested && requested[0]) {
        snprintf(path, path_len, "%s", requested);
        *face_index = 0;
        return access(path, R_OK) == 0 ? 0 : -1;
    }
    static const char *candidates[] = {
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (access(candidates[i], R_OK) == 0) {
            snprintf(path, path_len, "%s", candidates[i]);
            *face_index = 0;
            return 0;
        }
    }
    return -1;
}

/* 函数职责：初始化FreeType字体库、字体面和字形缓存。 */
static int init_font(struct rkdrm_atomic *ctx, const char *font_path,
                     char *err, size_t err_len)
{
    int face_index = 0;
    if (resolve_font(font_path, ctx->font_path, sizeof(ctx->font_path),
                     &face_index) != 0) {
        set_err(err, err_len, "unable to resolve a readable CJK font");
        return -1;
    }
    if (FT_Init_FreeType(&ctx->ft_library) != 0 ||
        FT_New_Face(ctx->ft_library, ctx->font_path, face_index, &ctx->ft_face) != 0) {
        set_err(err, err_len, "FreeType failed to load %s index %d",
                ctx->font_path, face_index);
        return -1;
    }
    return 0;
}

/* 函数职责：清除当前Overlay缓冲上次使用的脏矩形。 */
static void clear_rect(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                       struct rect_i rect)
{
    int x1 = rect.x < 0 ? 0 : rect.x;
    int y1 = rect.y < 0 ? 0 : rect.y;
    int x2 = rect.x + rect.w;
    int y2 = rect.y + rect.h;
    if (x2 > (int)ctx->width) {
        x2 = (int)ctx->width;
    }
    if (y2 > (int)ctx->height) {
        y2 = (int)ctx->height;
    }
    if (x2 <= x1 || y2 <= y1) {
        return;
    }
    for (int y = y1; y < y2; ++y) {
        uint8_t *row = bo->shadow + (size_t)y * bo->pitch +
                       (size_t)x1 * 4u;
        memset(row, 0, (size_t)(x2 - x1) * 4u);
    }
}

/* 函数职责：把新绘制区域合并到当前Overlay缓冲的脏矩形。 */
static void add_dirty(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                      int x, int y, int w, int h)
{
    (void)ctx;
    if (bo->dirty_count >= RKATOMIC_MAX_DIRTY || w <= 0 || h <= 0) {
        return;
    }
    bo->dirty[bo->dirty_count++] = (struct rect_i){x, y, w, h};
}

/* 函数职责：按预乘Alpha规则把一个像素混合到ARGB缓冲。 */
static void blend_pixel(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                        int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (x < 0 || y < 0 || x >= (int)ctx->width || y >= (int)ctx->height || a == 0) {
        return;
    }
    uint8_t *pixel = bo->shadow + (size_t)y * bo->pitch +
                     (size_t)x * 4u;
    uint32_t inv = 255u - a;
    uint8_t pb = (uint8_t)(((uint32_t)b * a + 127u) / 255u);
    uint8_t pg = (uint8_t)(((uint32_t)g * a + 127u) / 255u);
    uint8_t pr = (uint8_t)(((uint32_t)r * a + 127u) / 255u);
    pixel[0] = (uint8_t)(pb + ((uint32_t)pixel[0] * inv + 127u) / 255u);
    pixel[1] = (uint8_t)(pg + ((uint32_t)pixel[1] * inv + 127u) / 255u);
    pixel[2] = (uint8_t)(pr + ((uint32_t)pixel[2] * inv + 127u) / 255u);
    pixel[3] = (uint8_t)(a + ((uint32_t)pixel[3] * inv + 127u) / 255u);
}

/* 函数职责：在裁剪后的矩形内填充预乘ARGB颜色。 */
static void fill_rect(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                      int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > (int)ctx->width) x2 = (int)ctx->width;
    if (y2 > (int)ctx->height) y2 = (int)ctx->height;
    if (x2 <= x1 || y2 <= y1 || a == 0) {
        return;
    }
    if (a == 255) {
        for (int y = y1; y < y2; ++y) {
            uint8_t *pixel = bo->shadow + (size_t)y * bo->pitch +
                             (size_t)x1 * 4u;
            for (int x = x1; x < x2; ++x) {
                pixel[0] = b;
                pixel[1] = g;
                pixel[2] = r;
                pixel[3] = 255;
                pixel += 4;
            }
        }
        return;
    }
    uint32_t inv = 255u - a;
    uint8_t pb = (uint8_t)(((uint32_t)b * a + 127u) / 255u);
    uint8_t pg = (uint8_t)(((uint32_t)g * a + 127u) / 255u);
    uint8_t pr = (uint8_t)(((uint32_t)r * a + 127u) / 255u);
    for (int y = y1; y < y2; ++y) {
        uint8_t *pixel = bo->shadow + (size_t)y * bo->pitch +
                         (size_t)x1 * 4u;
        for (int x = x1; x < x2; ++x) {
            pixel[0] = (uint8_t)(pb + ((uint32_t)pixel[0] * inv + 127u) / 255u);
            pixel[1] = (uint8_t)(pg + ((uint32_t)pixel[1] * inv + 127u) / 255u);
            pixel[2] = (uint8_t)(pr + ((uint32_t)pixel[2] * inv + 127u) / 255u);
            pixel[3] = (uint8_t)(a + ((uint32_t)pixel[3] * inv + 127u) / 255u);
            pixel += 4;
        }
    }
}

/* 函数职责：绘制检测框边线并更新脏矩形。 */
static void draw_box(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                     int x1, int y1, int x2, int y2, int thickness,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    fill_rect(ctx, bo, x1, y1, x2, y1 + thickness, r, g, b, a);
    fill_rect(ctx, bo, x1, y2 - thickness, x2, y2, r, g, b, a);
    fill_rect(ctx, bo, x1, y1, x1 + thickness, y2, r, g, b, a);
    fill_rect(ctx, bo, x2 - thickness, y1, x2, y2, r, g, b, a);
}

/* 函数职责：用整数Bresenham算法连接两个角点，并记录整条线的脏矩形。 */
static void draw_line(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                      int x0, int y0, int x1, int y1, int thickness,
                      uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    int min_x = x0 < x1 ? x0 : x1;
    int min_y = y0 < y1 ? y0 : y1;
    int max_x = x0 > x1 ? x0 : x1;
    int max_y = y0 > y1 ? y0 : y1;
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    int radius = thickness > 1 ? thickness / 2 : 0;

    for (;;) {
        fill_rect(ctx, bo, x0 - radius, y0 - radius,
                  x0 - radius + thickness, y0 - radius + thickness,
                  r, g, b, a);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int doubled = error * 2;
        if (doubled >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y0 += sy;
        }
    }
    add_dirty(ctx, bo, min_x - thickness, min_y - thickness,
              max_x - min_x + thickness * 2 + 1,
              max_y - min_y + thickness * 2 + 1);
}

/* 函数职责：解码一个UTF-8码点，非法序列按替代字符推进。 */
static uint32_t utf8_next(const char **cursor)
{
    const unsigned char *s = (const unsigned char *)*cursor;
    if (!*s) {
        return 0;
    }
    uint32_t cp;
    int bytes;
    if (s[0] < 0x80) {
        cp = s[0];
        bytes = 1;
    } else if ((s[0] & 0xe0) == 0xc0) {
        cp = s[0] & 0x1f;
        bytes = 2;
    } else if ((s[0] & 0xf0) == 0xe0) {
        cp = s[0] & 0x0f;
        bytes = 3;
    } else if ((s[0] & 0xf8) == 0xf0) {
        cp = s[0] & 0x07;
        bytes = 4;
    } else {
        *cursor += 1;
        return 0xfffd;
    }
    for (int i = 1; i < bytes; ++i) {
        if ((s[i] & 0xc0) != 0x80) {
            *cursor += 1;
            return 0xfffd;
        }
        cp = (cp << 6) | (s[i] & 0x3f);
    }
    *cursor += bytes;
    return cp;
}

/* 函数职责：从FreeType加载字形并放入按码点和字号索引的缓存。 */
static struct cached_glyph *load_glyph(struct rkdrm_atomic *ctx,
                                       uint32_t codepoint, int pixel_size)
{
    ctx->glyph_age++;
    for (int i = 0; i < RKATOMIC_GLYPH_CACHE; ++i) {
        struct cached_glyph *glyph = &ctx->glyphs[i];
        if (glyph->valid && glyph->codepoint == codepoint &&
            glyph->pixel_size == pixel_size) {
            glyph->age = ctx->glyph_age;
            return glyph;
        }
    }
    int selected = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < RKATOMIC_GLYPH_CACHE; ++i) {
        if (!ctx->glyphs[i].valid) {
            selected = i;
            break;
        }
        if (ctx->glyphs[i].age < oldest) {
            oldest = ctx->glyphs[i].age;
            selected = i;
        }
    }
    struct cached_glyph *glyph = &ctx->glyphs[selected];
    free(glyph->bitmap);
    memset(glyph, 0, sizeof(*glyph));
    FT_Set_Pixel_Sizes(ctx->ft_face, 0, (FT_UInt)pixel_size);
    if (FT_Load_Char(ctx->ft_face, codepoint, FT_LOAD_RENDER) != 0) {
        return NULL;
    }
    FT_GlyphSlot slot = ctx->ft_face->glyph;
    size_t bytes = (size_t)slot->bitmap.width * slot->bitmap.rows;
    if (bytes) {
        glyph->bitmap = malloc(bytes);
        if (!glyph->bitmap) {
            return NULL;
        }
        for (unsigned int row = 0; row < slot->bitmap.rows; ++row) {
            memcpy(glyph->bitmap + (size_t)row * slot->bitmap.width,
                   slot->bitmap.buffer + (size_t)row * slot->bitmap.pitch,
                   slot->bitmap.width);
        }
    }
    glyph->valid = 1;
    glyph->codepoint = codepoint;
    glyph->pixel_size = pixel_size;
    glyph->width = (int)slot->bitmap.width;
    glyph->height = (int)slot->bitmap.rows;
    glyph->left = slot->bitmap_left;
    glyph->top = slot->bitmap_top;
    glyph->advance = (int)(slot->advance.x >> 6);
    glyph->age = ctx->glyph_age;
    return glyph;
}

/* 函数职责：计算UTF-8文字按当前字形度量绘制时的宽度。 */
static int text_width(struct rkdrm_atomic *ctx, const char *text, int pixel_size)
{
    int width = 0;
    const char *cursor = text;
    uint32_t cp;
    while ((cp = utf8_next(&cursor)) != 0) {
        struct cached_glyph *glyph = load_glyph(ctx, cp, pixel_size);
        if (glyph) {
            width += glyph->advance;
        }
    }
    return width;
}

/* 函数职责：在Overlay缓冲绘制UTF-8文字并执行边缘裁剪。 */
static void draw_text(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                      int x, int baseline, const char *text, int pixel_size,
                      uint8_t r, uint8_t g, uint8_t b)
{
    const char *cursor = text;
    uint32_t cp;
    int pen_x = x;
    while ((cp = utf8_next(&cursor)) != 0) {
        struct cached_glyph *glyph = load_glyph(ctx, cp, pixel_size);
        if (!glyph) {
            continue;
        }
        int gx = pen_x + glyph->left;
        int gy = baseline - glyph->top;
        for (int row = 0; row < glyph->height; ++row) {
            for (int col = 0; col < glyph->width; ++col) {
                uint8_t alpha =
                    glyph->bitmap[(size_t)row * glyph->width + col];
                blend_pixel(ctx, bo, gx + col, gy + row, r, g, b, alpha);
            }
        }
        pen_x += glyph->advance;
    }
}

/* 函数职责：只清理当前缓冲旧脏区，再绘制最新框、文字和FPS。 */
static void render_overlay(struct rkdrm_atomic *ctx, struct overlay_bo *bo,
                           const struct rkdrm_overlay_item *items, int count,
                           uint64_t generation)
{
    double started = monotonic_ms();
    struct rect_i previous_dirty[RKATOMIC_MAX_DIRTY];
    int previous_count = bo->dirty_count;
    if (previous_count > 0) {
        memcpy(previous_dirty, bo->dirty,
               (size_t)previous_count * sizeof(*previous_dirty));
    }
    for (int i = 0; i < bo->dirty_count; ++i) {
        clear_rect(ctx, bo, bo->dirty[i]);
    }
    bo->dirty_count = 0;

    char fps_text[48];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", ctx->stats.display_fps);
    int fps_width = text_width(ctx, fps_text, 50);
    fill_rect(ctx, bo, 12, 12, 28 + fps_width, 76, 0, 0, 0, 150);
    draw_text(ctx, bo, 20, 64, fps_text, 50, 255, 0, 0);
    add_dirty(ctx, bo, 8, 8, fps_width + 28, 72);

    for (int i = 0; i < count; ++i) {
        const struct rkdrm_overlay_item *item = &items[i];
        int width = item->x2 - item->x1;
        int height = item->y2 - item->y1;
        int generic = !!(item->flags & RKATOMIC_ITEM_GENERIC);
        int polygon = !!(item->flags & RKATOMIC_ITEM_POLYGON);
        int x1 = (generic || polygon) ? item->x1 : item->x1 - width / 20;
        int y1 = (generic || polygon) ? item->y1 : item->y1 - height / 9;
        int x2 = (generic || polygon) ? item->x2 : item->x2 + width / 20;
        int y2 = (generic || polygon) ? item->y2 : item->y2 + height / 9;
        uint8_t red = item->alpha ? item->red : 255;
        uint8_t green = item->alpha ? item->green : 0;
        uint8_t blue = item->alpha ? item->blue : 0;
        uint8_t alpha = item->alpha ? item->alpha : 255;
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > (int)ctx->width) x2 = (int)ctx->width;
        if (y2 > (int)ctx->height) y2 = (int)ctx->height;
        if (polygon) {
            /*
             * 串口角点顺序固定为左上、右上、左下、右下。
             * 按0→1→3→2→0闭合，避免直接按数组顺序连接形成交叉线。
             */
            static const int order[5] = {0, 1, 3, 2, 0};
            for (int edge = 0; edge < 4; ++edge) {
                int from = order[edge];
                int to = order[edge + 1];
                draw_line(
                    ctx, bo,
                    (int)(item->landmarks[from * 2] + 0.5f),
                    (int)(item->landmarks[from * 2 + 1] + 0.5f),
                    (int)(item->landmarks[to * 2] + 0.5f),
                    (int)(item->landmarks[to * 2 + 1] + 0.5f),
                    3, red, green, blue, alpha
                );
            }
        } else {
            draw_box(ctx, bo, x1, y1, x2, y2, 2,
                     red, green, blue, alpha);
            add_dirty(ctx, bo, x1 - 2, y1 - 2, x2 - x1 + 4, 6);
            add_dirty(ctx, bo, x1 - 2, y2 - 4, x2 - x1 + 4, 6);
            add_dirty(ctx, bo, x1 - 2, y1 + 2, 6, y2 - y1 - 4);
            add_dirty(ctx, bo, x2 - 4, y1 + 2, 6, y2 - y1 - 4);
        }

        if (item->flags & RKATOMIC_ITEM_CORNERS) {
            for (int p = 0; p < 4; ++p) {
                int px = (int)(item->landmarks[p * 2] + 0.5f);
                int py = (int)(item->landmarks[p * 2 + 1] + 0.5f);
                fill_rect(ctx, bo, px - 3, py - 3, px + 4, py + 4,
                          red, green, blue, alpha);
                add_dirty(ctx, bo, px - 4, py - 4, 9, 9);
            }
        }
        const char *primary_label =
            generic ? item->label : item->plate_no;
        if (!(item->flags & RKATOMIC_ITEM_LABEL) || !primary_label[0]) {
            continue;
        }
        char label[128];
        snprintf(label, sizeof(label), "%s", primary_label);
        int label_width = text_width(ctx, label, 50);
        int full = !!(item->flags & RKATOMIC_ITEM_FULL);
        int bg_w = label_width + 8;
        int bg_h = full ? 88 : 58;
        int bg_x = x1;
        int bg_y = y2 + 2;
        if (bg_x + bg_w > (int)ctx->width) {
            bg_x = (int)ctx->width - bg_w;
        }
        if (bg_x < 0) bg_x = 0;
        if (bg_y + bg_h > (int)ctx->height) {
            bg_y = y1 - bg_h - 2;
        }
        if (bg_y < 0) bg_y = 0;
        fill_rect(ctx, bo, bg_x, bg_y, bg_x + bg_w, bg_y + bg_h,
                  red, green, blue, 230);
        draw_text(ctx, bo, bg_x + 4, bg_y + 50, label, 50,
                  255, 255, 255);
        if (full) {
            char confidence[64];
            if (generic) {
                snprintf(confidence, sizeof(confidence), "conf:%.2f",
                         item->score);
            } else {
                snprintf(confidence, sizeof(confidence), "d:%.2f r:%.2f",
                         item->detect_conf, item->rec_conf);
            }
            draw_text(ctx, bo, bg_x + 4, bg_y + 82, confidence, 25,
                      0, 255, 0);
        }
        add_dirty(ctx, bo, bg_x - 1, bg_y - 1, bg_w + 2, bg_h + 2);
    }
    for (int group = 0; group < 2; ++group) {
        const struct rect_i *rects = group == 0 ? previous_dirty : bo->dirty;
        int rect_count = group == 0 ? previous_count : bo->dirty_count;
        for (int i = 0; i < rect_count; ++i) {
            int x1 = rects[i].x < 0 ? 0 : rects[i].x;
            int y1 = rects[i].y < 0 ? 0 : rects[i].y;
            int x2 = rects[i].x + rects[i].w;
            int y2 = rects[i].y + rects[i].h;
            if (x2 > (int)ctx->width) x2 = (int)ctx->width;
            if (y2 > (int)ctx->height) y2 = (int)ctx->height;
            if (x2 <= x1 || y2 <= y1) continue;
            for (int y = y1; y < y2; ++y) {
                uint8_t *dst = (uint8_t *)bo->map + (size_t)y * bo->pitch +
                               (size_t)x1 * 4u;
                const uint8_t *src = bo->shadow + (size_t)y * bo->pitch +
                                     (size_t)x1 * 4u;
                memcpy(dst, src, (size_t)(x2 - x1) * 4u);
            }
        }
    }
    bo->rendered_generation = generation;
    double elapsed = monotonic_ms() - started;
    ctx->stats.overlay_updates++;
    update_time(&ctx->stats.overlay_total_ms, &ctx->stats.overlay_max_ms,
                elapsed);
    ctx->overlay_samples[
        ctx->overlay_sample_count % RKATOMIC_OVERLAY_SAMPLES
    ] = elapsed;
    ctx->overlay_sample_count++;
}

/* 函数职责：向Atomic请求写入一个Plane的源区域、目标区域和混合属性。 */
static int add_plane_state(drmModeAtomicReqPtr request, uint32_t plane_id,
                           const struct plane_props *props, uint32_t fb_id,
                           uint32_t crtc_id, uint32_t src_w, uint32_t src_h,
                           uint32_t dst_w, uint32_t dst_h, uint64_t zpos,
                           int premultiplied)
{
#define ADD(prop, value)                                                     \
    do {                                                                     \
        if (drmModeAtomicAddProperty(request, plane_id, prop, value) < 0) {  \
            return -1;                                                       \
        }                                                                    \
    } while (0)
    ADD(props->fb_id, fb_id);
    ADD(props->crtc_id, crtc_id);
    ADD(props->src_x, 0);
    ADD(props->src_y, 0);
    ADD(props->src_w, (uint64_t)src_w << 16);
    ADD(props->src_h, (uint64_t)src_h << 16);
    ADD(props->crtc_x, 0);
    ADD(props->crtc_y, 0);
    ADD(props->crtc_w, dst_w);
    ADD(props->crtc_h, dst_h);
    if (props->zpos) {
        ADD(props->zpos, zpos);
    }
    if (props->alpha) {
        ADD(props->alpha, 65535);
    }
    if (premultiplied && props->blend) {
        ADD(props->blend, props->blend_premultiplied);
    }
#undef ADD
    return 0;
}

/* 函数职责：把NV12 DMA-BUF导入DRM并缓存对应Framebuffer。 */
static struct imported_fb *import_nv12(struct rkdrm_atomic *ctx, int dma_fd,
                                       int width, int height, int hstride,
                                       int vstride, char *err, size_t err_len)
{
    struct stat stat_info;
    if (fstat(dma_fd, &stat_info) != 0) {
        set_err(err, err_len, "fstat NV12 dma-buf failed: %s", strerror(errno));
        return NULL;
    }
    for (int i = 0; i < RKATOMIC_FB_CACHE; ++i) {
        if (ctx->imported[i].used && ctx->imported[i].dev == stat_info.st_dev &&
            ctx->imported[i].ino == stat_info.st_ino) {
            return &ctx->imported[i];
        }
    }
    int selected = -1;
    for (int i = 0; i < RKATOMIC_FB_CACHE; ++i) {
        if (!ctx->imported[i].used) {
            selected = i;
            break;
        }
    }
    if (selected < 0) {
        set_err(err, err_len, "NV12 framebuffer cache is full");
        return NULL;
    }
    struct imported_fb *entry = &ctx->imported[selected];
    if (drmPrimeFDToHandle(ctx->fd, dma_fd, &entry->handle) != 0) {
        set_err(err, err_len, "drmPrimeFDToHandle failed: %s", strerror(errno));
        return NULL;
    }
    uint32_t handles[4] = {entry->handle, entry->handle, 0, 0};
    uint32_t pitches[4] = {(uint32_t)hstride, (uint32_t)hstride, 0, 0};
    uint32_t offsets[4] = {0, (uint32_t)((uint64_t)hstride * vstride), 0, 0};
    if (drmModeAddFB2(ctx->fd, (uint32_t)width, (uint32_t)height,
                      DRM_FORMAT_NV12, handles, pitches, offsets,
                      &entry->fb_id, 0) != 0) {
        struct drm_gem_close close_arg = {.handle = entry->handle};
        drmIoctl(ctx->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        entry->handle = 0;
        set_err(err, err_len, "drmModeAddFB2 NV12 failed: %s", strerror(errno));
        return NULL;
    }
    entry->used = 1;
    entry->dev = stat_info.st_dev;
    entry->ino = stat_info.st_ino;
    return entry;
}

/* 函数职责：处理DRM Page Flip完成事件并释放上一扫描帧引用。 */
static void page_flip_handler(int fd, unsigned int sequence,
                              unsigned int tv_sec, unsigned int tv_usec,
                              void *user_data)
{
    (void)fd;
    (void)sequence;
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)user_data;
    if (ctx->current_token != ctx->pending_token) {
        ctx->retired_token = ctx->current_token;
    }
    ctx->current_token = ctx->pending_token;
    ctx->overlay_front = ctx->overlay_pending;
    uint64_t callback_ns = monotonic_ns();
    uint64_t event_ns =
        (uint64_t)tv_sec * 1000000000ull + (uint64_t)tv_usec * 1000ull;
    uint64_t clock_delta = event_ns > callback_ns
                               ? event_ns - callback_ns
                               : callback_ns - event_ns;
    /* 当前内核的DRM Flip时间戳来自单调时钟；若其他驱动上报不同
     * 时钟域，则使用带范围校验的回调时刻作为兜底。 */
    uint64_t flip_ns =
        event_ns > 0 && clock_delta < 5000000000ull
            ? event_ns : callback_ns;
    ctx->stats.last_flip_ns = flip_ns;
    if (ctx->pending_overlay_generation != ctx->current_overlay_generation) {
        ctx->current_overlay_generation = ctx->pending_overlay_generation;
        if (ctx->pending_overlay_source_timestamp_ns > 0 &&
            flip_ns >= ctx->pending_overlay_source_timestamp_ns) {
            ctx->stats.last_overlay_source_timestamp_ns =
                ctx->pending_overlay_source_timestamp_ns;
            ctx->stats.last_overlay_e2e_ms =
                (double)(flip_ns - ctx->pending_overlay_source_timestamp_ns) /
                1000000.0;
        }
        if (ctx->pending_overlay_ready_timestamp_ns > 0 &&
            flip_ns >= ctx->pending_overlay_ready_timestamp_ns) {
            ctx->stats.last_overlay_ready_timestamp_ns =
                ctx->pending_overlay_ready_timestamp_ns;
            ctx->stats.last_overlay_display_ms =
                (double)(flip_ns - ctx->pending_overlay_ready_timestamp_ns) /
                1000000.0;
        }
    }
    ctx->flip_pending = 0;
    ctx->stats.displayed++;
    ctx->fps_frames++;
}

/* 函数职责：等待未完成的Page Flip事件，保证即将复用的缓冲已退出扫描。 */
static int wait_flip(struct rkdrm_atomic *ctx, char *err, size_t err_len)
{
    while (ctx->flip_pending) {
        struct pollfd poll_fd = {.fd = ctx->fd, .events = POLLIN};
        int result = poll(&poll_fd, 1, 1000);
        if (result <= 0) {
            set_err(err, err_len, "DRM page flip wait failed: %s",
                    result == 0 ? "timeout" : strerror(errno));
            return -1;
        }
        drmEventContext event;
        memset(&event, 0, sizeof(event));
        event.version = DRM_EVENT_CONTEXT_VERSION;
        event.page_flip_handler = page_flip_handler;
        if (drmHandleEvent(ctx->fd, &event) != 0) {
            set_err(err, err_len, "drmHandleEvent failed: %s", strerror(errno));
            return -1;
        }
    }
    return 0;
}

/* 函数职责：启用Universal Plane与Atomic能力，选择Plane并完成Test-Only初始化。 */
void *rkdrm_atomic_open(const char *device, const char *font_path,
                        int *out_width, int *out_height,
                        uint32_t *out_video_plane, uint32_t *out_overlay_plane,
                        char *err, size_t err_len)
{
    struct rkdrm_atomic *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    ctx->fd = -1;
    ctx->overlay_front = 0;
    ctx->overlay_pending = 0;
    ctx->overlay_generation = 1;
    ctx->fps_window_ns = monotonic_ns();
    if (pthread_mutex_init(&ctx->overlay_lock, NULL) != 0) {
        set_err(err, err_len, "pthread_mutex_init failed");
        free(ctx);
        return NULL;
    }
    ctx->overlay_lock_ready = 1;
    const char *path = device && device[0] ? device : "/dev/dri/card0";
    ctx->fd = open(path, O_RDWR | O_CLOEXEC);
    if (ctx->fd < 0) {
        set_err(err, err_len, "open %s failed: %s", path, strerror(errno));
        rkdrm_atomic_close(ctx);
        return NULL;
    }
    if (drmSetClientCap(ctx->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0 ||
        drmSetClientCap(ctx->fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
        set_err(err, err_len, "DRM atomic/universal plane capability unavailable");
        rkdrm_atomic_close(ctx);
        return NULL;
    }
    if (find_connector_crtc(ctx, err, err_len) != 0 ||
        select_planes(ctx, err, err_len) != 0) {
        rkdrm_atomic_close(ctx);
        return NULL;
    }
    ctx->old_crtc = drmModeGetCrtc(ctx->fd, ctx->crtc_id);
    ctx->connector_crtc_id_prop =
        object_prop_id(ctx->fd, ctx->connector_id, DRM_MODE_OBJECT_CONNECTOR,
                       "CRTC_ID", NULL);
    ctx->crtc_active_prop =
        object_prop_id(ctx->fd, ctx->crtc_id, DRM_MODE_OBJECT_CRTC,
                       "ACTIVE", NULL);
    ctx->crtc_mode_id_prop =
        object_prop_id(ctx->fd, ctx->crtc_id, DRM_MODE_OBJECT_CRTC,
                       "MODE_ID", NULL);
    if (!ctx->connector_crtc_id_prop || !ctx->crtc_active_prop ||
        !ctx->crtc_mode_id_prop ||
        drmModeCreatePropertyBlob(ctx->fd, &ctx->mode, sizeof(ctx->mode),
                                  &ctx->mode_blob_id) != 0) {
        set_err(err, err_len, "required connector/CRTC atomic properties unavailable");
        rkdrm_atomic_close(ctx);
        return NULL;
    }
    if (create_overlay_bo(ctx, &ctx->overlay[0], err, err_len) != 0 ||
        create_overlay_bo(ctx, &ctx->overlay[1], err, err_len) != 0 ||
        init_font(ctx, font_path, err, err_len) != 0) {
        rkdrm_atomic_close(ctx);
        return NULL;
    }
    if (out_width) *out_width = (int)ctx->width;
    if (out_height) *out_height = (int)ctx->height;
    if (out_video_plane) *out_video_plane = ctx->video_plane_id;
    if (out_overlay_plane) *out_overlay_plane = ctx->overlay_plane_id;
    set_err(err, err_len, "");
    return ctx;
}

/* 函数职责：保存最新通用Overlay数据与TTL，等待显示线程下一帧合成。 */
int rkdrm_atomic_submit_overlay(void *handle,
                                const struct rkdrm_overlay_item *items,
                                int count, int ttl_ms,
                                uint64_t source_timestamp_ns,
                                uint64_t ready_timestamp_ns,
                                char *err, size_t err_len)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    if (!ctx || count < 0 || count > RKATOMIC_MAX_ITEMS ||
        (count > 0 && !items) || ttl_ms < 0) {
        set_err(err, err_len, "invalid overlay submission");
        return -1;
    }
    pthread_mutex_lock(&ctx->overlay_lock);
    if (count > 0) {
        memcpy(ctx->items, items, (size_t)count * sizeof(*items));
    }
    ctx->item_count = count;
    ctx->overlay_ttl_ms = ttl_ms;
    ctx->overlay_timestamp_ns = monotonic_ns();
    ctx->overlay_source_timestamp_ns = source_timestamp_ns;
    ctx->overlay_ready_timestamp_ns = ready_timestamp_ns;
    ctx->overlay_generation++;
    ctx->overlay_expired = 0;
    pthread_mutex_unlock(&ctx->overlay_lock);
    return 0;
}

/* 函数职责：导入并提交NV12视频Plane与ARGB Overlay Plane，等待必要的Flip完成事件。 */
int rkdrm_atomic_show_nv12(void *handle, int dma_fd, uint64_t token,
                           int width, int height, int hstride, int vstride,
                           uint64_t *retired_token,
                           char *err, size_t err_len)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    if (!ctx || dma_fd < 0 || token == 0 || width <= 0 || height <= 0 ||
        hstride < width || vstride < height) {
        set_err(err, err_len, "invalid NV12 atomic frame");
        return -1;
    }
    if ((uint32_t)width != ctx->width || (uint32_t)height != ctx->height) {
        set_err(err, err_len, "NV12 %dx%d does not match mode %ux%u",
                width, height, ctx->width, ctx->height);
        return -1;
    }
    if (wait_flip(ctx, err, err_len) != 0) {
        return -1;
    }
    uint64_t retired = ctx->retired_token;
    ctx->retired_token = 0;

    struct imported_fb *video =
        import_nv12(ctx, dma_fd, width, height, hstride, vstride, err, err_len);
    if (!video) {
        return -1;
    }

    uint64_t now = monotonic_ns();
    pthread_mutex_lock(&ctx->overlay_lock);
    if (ctx->overlay_ttl_ms > 0 && ctx->item_count > 0 &&
        now - ctx->overlay_timestamp_ns >
            (uint64_t)ctx->overlay_ttl_ms * 1000000ull &&
        !ctx->overlay_expired) {
        ctx->item_count = 0;
        ctx->overlay_source_timestamp_ns = 0;
        ctx->overlay_ready_timestamp_ns = 0;
        ctx->overlay_generation++;
        ctx->overlay_expired = 1;
        ctx->stats.overlay_expirations++;
    }
    if (now - ctx->fps_window_ns >= 1000000000ull) {
        double seconds = (double)(now - ctx->fps_window_ns) / 1000000000.0;
        ctx->stats.display_fps = (double)ctx->fps_frames / seconds;
        ctx->fps_frames = 0;
        ctx->fps_window_ns = now;
        ctx->overlay_generation++;
    }
    uint64_t generation = ctx->overlay_generation;
    uint64_t overlay_source_timestamp_ns =
        ctx->overlay_source_timestamp_ns;
    uint64_t overlay_ready_timestamp_ns =
        ctx->overlay_ready_timestamp_ns;
    int count = ctx->item_count;
    struct rkdrm_overlay_item local_items[RKATOMIC_MAX_ITEMS];
    if (count > 0) {
        memcpy(local_items, ctx->items, (size_t)count * sizeof(*local_items));
    }
    pthread_mutex_unlock(&ctx->overlay_lock);

    int overlay_index = ctx->overlay_front;
    if (ctx->overlay[overlay_index].rendered_generation != generation) {
        overlay_index = 1 - ctx->overlay_front;
        render_overlay(ctx, &ctx->overlay[overlay_index],
                       local_items, count, generation);
    }

    drmModeAtomicReqPtr request = drmModeAtomicAlloc();
    if (!request) {
        set_err(err, err_len, "drmModeAtomicAlloc failed");
        return -1;
    }
    int failed = 0;
    if (!ctx->modeset_done) {
        failed |= drmModeAtomicAddProperty(request, ctx->connector_id,
                                           ctx->connector_crtc_id_prop,
                                           ctx->crtc_id) < 0;
        failed |= drmModeAtomicAddProperty(request, ctx->crtc_id,
                                           ctx->crtc_active_prop, 1) < 0;
        failed |= drmModeAtomicAddProperty(request, ctx->crtc_id,
                                           ctx->crtc_mode_id_prop,
                                           ctx->mode_blob_id) < 0;
    }
    failed |= add_plane_state(request, ctx->video_plane_id, &ctx->video_props,
                              video->fb_id, ctx->crtc_id,
                              (uint32_t)width, (uint32_t)height,
                              ctx->width, ctx->height, 0, 0) != 0;
    failed |= add_plane_state(request, ctx->overlay_plane_id, &ctx->overlay_props,
                              ctx->overlay[overlay_index].fb_id, ctx->crtc_id,
                              ctx->width, ctx->height,
                              ctx->width, ctx->height, 1, 1) != 0;
    if (failed) {
        drmModeAtomicFree(request);
        set_err(err, err_len, "failed to construct atomic request");
        return -1;
    }

    uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK | DRM_MODE_PAGE_FLIP_EVENT;
    if (!ctx->modeset_done) {
        int test_result = drmModeAtomicCommit(
            ctx->fd, request, DRM_MODE_ATOMIC_TEST_ONLY |
                              DRM_MODE_ATOMIC_ALLOW_MODESET,
            ctx);
        if (test_result != 0) {
            drmModeAtomicFree(request);
            set_err(err, err_len, "atomic test-only commit failed: %s",
                    strerror(errno));
            return -1;
        }
        flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;
    }
    double started = monotonic_ms();
    int result = drmModeAtomicCommit(ctx->fd, request, flags, ctx);
    double elapsed = monotonic_ms() - started;
    drmModeAtomicFree(request);
    if (result != 0) {
        ctx->stats.commit_errors++;
        set_err(err, err_len, "atomic commit failed: %s", strerror(errno));
        return -1;
    }
    ctx->stats.submitted++;
    update_time(&ctx->stats.commit_total_ms, &ctx->stats.commit_max_ms, elapsed);
    ctx->modeset_done = 1;
    ctx->flip_pending = 1;
    ctx->pending_token = token;
    ctx->overlay_pending = overlay_index;
    ctx->pending_overlay_generation = generation;
    ctx->pending_overlay_source_timestamp_ns =
        overlay_source_timestamp_ns;
    ctx->pending_overlay_ready_timestamp_ns =
        overlay_ready_timestamp_ns;
    if (retired_token) {
        *retired_token = retired;
    }
    return 0;
}

/* 函数职责：复制Atomic显示、Overlay和Page Flip统计给调用方。 */
int rkdrm_atomic_get_stats(void *handle, struct rkdrm_atomic_stats *stats,
                           char *err, size_t err_len)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    if (!ctx || !stats) {
        set_err(err, err_len, "invalid stats request");
        return -1;
    }
    *stats = ctx->stats;
    return 0;
}

/* 函数职责：比较两个统计样本，供排序和百分位计算使用。 */
static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

/* 函数职责：计算Overlay耗时样本的指定百分位值。 */
double rkdrm_atomic_overlay_percentile(void *handle, double quantile)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    if (!ctx || quantile < 0.0 || quantile > 1.0 ||
        ctx->overlay_sample_count == 0) {
        return -1.0;
    }
    size_t count = ctx->overlay_sample_count < RKATOMIC_OVERLAY_SAMPLES
                       ? (size_t)ctx->overlay_sample_count
                       : RKATOMIC_OVERLAY_SAMPLES;
    double samples[RKATOMIC_OVERLAY_SAMPLES];
    memcpy(samples, ctx->overlay_samples, count * sizeof(samples[0]));
    qsort(samples, count, sizeof(samples[0]), compare_double);
    size_t index = (size_t)(quantile * (double)(count - 1));
    return samples[index];
}

/* 函数职责：返回实际使用的字体路径，供启动诊断日志显示。 */
const char *rkdrm_atomic_font_path(void *handle)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    return ctx ? ctx->font_path : "";
}

/* 函数职责：等待最后Flip后恢复原CRTC，再按依赖顺序销毁FB、BO和DRM资源。 */
void rkdrm_atomic_close(void *handle)
{
    struct rkdrm_atomic *ctx = (struct rkdrm_atomic *)handle;
    if (!ctx) {
        return;
    }
    if (ctx->fd >= 0 && ctx->flip_pending) {
        char ignored[128];
        wait_flip(ctx, ignored, sizeof(ignored));
    }
    if (ctx->fd >= 0 && ctx->modeset_done) {
        drmModeAtomicReqPtr request = drmModeAtomicAlloc();
        if (request) {
            drmModeAtomicAddProperty(request, ctx->video_plane_id,
                                     ctx->video_props.fb_id, 0);
            drmModeAtomicAddProperty(request, ctx->video_plane_id,
                                     ctx->video_props.crtc_id, 0);
            drmModeAtomicAddProperty(request, ctx->overlay_plane_id,
                                     ctx->overlay_props.fb_id, 0);
            drmModeAtomicAddProperty(request, ctx->overlay_plane_id,
                                     ctx->overlay_props.crtc_id, 0);
            drmModeAtomicCommit(ctx->fd, request, 0, ctx);
            drmModeAtomicFree(request);
        }
    }
    if (ctx->fd >= 0 && ctx->old_crtc) {
        drmModeSetCrtc(ctx->fd, ctx->old_crtc->crtc_id,
                       ctx->old_crtc->buffer_id,
                       ctx->old_crtc->x, ctx->old_crtc->y,
                       &ctx->connector_id, 1, &ctx->old_crtc->mode);
    }
    if (ctx->old_crtc) {
        drmModeFreeCrtc(ctx->old_crtc);
    }
    for (int i = 0; i < RKATOMIC_FB_CACHE; ++i) {
        if (!ctx->imported[i].used) {
            continue;
        }
        drmModeRmFB(ctx->fd, ctx->imported[i].fb_id);
        struct drm_gem_close close_arg = {.handle = ctx->imported[i].handle};
        drmIoctl(ctx->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
    }
    if (ctx->fd >= 0) {
        destroy_overlay_bo(ctx, &ctx->overlay[0]);
        destroy_overlay_bo(ctx, &ctx->overlay[1]);
    }
    if (ctx->mode_blob_id && ctx->fd >= 0) {
        drmModeDestroyPropertyBlob(ctx->fd, ctx->mode_blob_id);
    }
    for (int i = 0; i < RKATOMIC_GLYPH_CACHE; ++i) {
        free(ctx->glyphs[i].bitmap);
    }
    if (ctx->ft_face) {
        FT_Done_Face(ctx->ft_face);
    }
    if (ctx->ft_library) {
        FT_Done_FreeType(ctx->ft_library);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    if (ctx->overlay_lock_ready) {
        pthread_mutex_destroy(&ctx->overlay_lock);
    }
    free(ctx);
}
