// RGA模型预处理与NV12 ROI提取实现
//
// RGA 硬件前处理，替代车牌工程里 OpenCV 的 CPU 前处理。
// 借鉴 rknpu_rk3568/src/preprocess.cc 的 improcess 用法。
//
// 导出给Python(ctypes)的能力分三组:
//   rga_letterbox_rgb*    — BGR → 任意尺寸RGB NHWC letterbox；
//   rga_letterbox_nv12*   — NV12 DMA-BUF → RGB NHWC letterbox，不落地1080p BGR；
//   rga_nv12_roi_bgr_fd   — 从NV12 DMA-BUF提取偶数对齐的小型BGR ROI。
//
// 说明: RGA 输出的是 packed(交错) RGB。RKNN toolkit2 的 inference(data_format='nhwc')
// 正好接收 packed NHWC，因此可省掉 Python 侧的 transpose(2,0,1)+copy。
// RGA 无任意四点透视 API，故识别的warpPerspective仍由cv2承担，但只处理小ROI。
//
// Build: 见 Makefile 的 librga_preprocess.so target。

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <rga/im2d.h>
#include <rga/rga.h>

#ifndef MODULE_TAG
#define MODULE_TAG "rga_preprocess"
#endif

/* 函数职责：把格式化错误写入调用方缓冲，缓冲为空时安全忽略。 */
static void set_err(char *err, int err_len, const char *fmt, ...)
{
    if (!err || err_len <= 0) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, (size_t)err_len, fmt, ap);
    va_end(ap);
}

// 检测前处理: 等比 letterbox 到 target×target，BGR→RGB，packed 输出。
// src:    BGR888 packed, src_w×src_h, 行跨度 src_stride 字节 (0 表示紧凑 = src_w*3)
// dst:    调用方分配的 target*target*3 uint8 缓冲 (packed RGB, NHWC)
// 返回 1 成功并写 out_ratio/out_pad_w/out_pad_h; <0 失败。
int rga_letterbox_rgb_size(const uint8_t *src, int src_w, int src_h,
                           int src_stride, uint8_t *dst, int dst_size,
                           int dst_w, int dst_h,
                           float *out_ratio, int *out_pad_w, int *out_pad_h,
                           char *err, int err_len)
{
    if (!src || !dst || src_w <= 0 || src_h <= 0 ||
        dst_w <= 0 || dst_h <= 0) {
        set_err(err, err_len, "invalid args");
        return -1;
    }
    if (dst_size < dst_w * dst_h * 3) {
        set_err(err, err_len, "dst too small: need %d got %d",
                dst_w * dst_h * 3, dst_size);
        return -1;
    }

    float r = (float)dst_h / (float)src_h;
    float rw = (float)dst_w / (float)src_w;
    if (rw < r) {
        r = rw;
    }
    int new_w = (int)(src_w * r + 0.5f);
    int new_h = (int)(src_h * r + 0.5f);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;
    if (new_w > dst_w) new_w = dst_w;
    if (new_h > dst_h) new_h = dst_h;
    int pad_w = (dst_w - new_w) / 2;
    int pad_h = (dst_h - new_h) / 2;

    memset(dst, 114, (size_t)dst_w * dst_h * 3);

    int sstride = src_stride > 0 ? src_stride : src_w * 3;

    rga_buffer_t s;
    rga_buffer_t d;
    memset(&s, 0, sizeof(s));
    memset(&d, 0, sizeof(d));
    // 源: BGR888 packed, wstride 用字节跨度换算成像素 (RGB888 每像素3字节)。
    s = wrapbuffer_virtualaddr_t((void *)src, src_w, src_h,
                                 sstride / 3, src_h, RK_FORMAT_BGR_888);
    d = wrapbuffer_virtualaddr_t(dst, dst_w, dst_h, dst_w, dst_h,
                                 RK_FORMAT_RGB_888);

    im_rect src_rect;
    im_rect dst_rect;
    im_rect pat_rect;
    rga_buffer_t pat;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&pat_rect, 0, sizeof(pat_rect));
    memset(&pat, 0, sizeof(pat));

    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = src_w;
    src_rect.height = src_h;
    // 目标子矩形: 居中放缩放后的图，四周留 114 padding。
    dst_rect.x = pad_w;
    dst_rect.y = pad_h;
    dst_rect.width = new_w;
    dst_rect.height = new_h;

    int ck = imcheck(s, d, src_rect, dst_rect);
    if (ck != IM_STATUS_NOERROR) {
        set_err(err, err_len, "imcheck: %s", imStrError((IM_STATUS)ck));
        return -1;
    }
    // improcess 同时做缩放(src_rect→dst_rect 尺寸不同) + 格式转换(BGR→RGB)。
    IM_STATUS st = improcess(s, d, pat, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (st != IM_STATUS_SUCCESS) {
        set_err(err, err_len, "improcess: %s", imStrError(st));
        return -1;
    }

    if (out_ratio) *out_ratio = r;
    if (out_pad_w) *out_pad_w = pad_w;
    if (out_pad_h) *out_pad_h = pad_h;
    return 1;
}

/* 函数职责：把RGB输入按比例缩放到模型画布并填充letterbox背景。 */
int rga_letterbox_rgb(const uint8_t *src, int src_w, int src_h, int src_stride,
                      uint8_t *dst, int dst_size, int target,
                      float *out_ratio, int *out_pad_w, int *out_pad_h,
                      char *err, int err_len)
{
    return rga_letterbox_rgb_size(
        src, src_w, src_h, src_stride, dst, dst_size, target, target,
        out_ratio, out_pad_w, out_pad_h, err, err_len);
}

/* 函数职责：把NV12 DMA-BUF硬件缩放为任意尺寸模型输入并返回映射参数。 */
int rga_letterbox_nv12_fd_size(int src_fd, int src_w, int src_h,
                               int src_hstride, int src_vstride,
                               uint8_t *dst, int dst_size,
                               int dst_w, int dst_h,
                               float *out_ratio, int *out_pad_w,
                               int *out_pad_h, char *err, int err_len)
{
    if (src_fd < 0 || !dst || src_w <= 0 || src_h <= 0 ||
        src_hstride < src_w || src_vstride < src_h ||
        dst_w <= 0 || dst_h <= 0) {
        set_err(err, err_len, "invalid NV12 letterbox arguments");
        return -1;
    }
    if (dst_size < dst_w * dst_h * 3) {
        set_err(err, err_len, "NV12 letterbox dst too small");
        return -1;
    }
    float ratio = (float)dst_h / (float)src_h;
    float width_ratio = (float)dst_w / (float)src_w;
    if (width_ratio < ratio) {
        ratio = width_ratio;
    }
    int resized_w = (int)(src_w * ratio + 0.5f);
    int resized_h = (int)(src_h * ratio + 0.5f);
    if (resized_w < 1) resized_w = 1;
    if (resized_h < 1) resized_h = 1;
    if (resized_w > dst_w) resized_w = dst_w;
    if (resized_h > dst_h) resized_h = dst_h;
    int pad_w = (dst_w - resized_w) / 2;
    int pad_h = (dst_h - resized_h) / 2;
    memset(dst, 114, (size_t)dst_w * dst_h * 3u);

    rga_buffer_t source =
        wrapbuffer_fd_t(src_fd, src_w, src_h, src_hstride, src_vstride,
                        RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t destination =
        wrapbuffer_virtualaddr_t(dst, dst_w, dst_h, dst_w, dst_h,
                                 RK_FORMAT_RGB_888);
    rga_buffer_t pattern;
    im_rect source_rect = {0, 0, src_w, src_h};
    im_rect destination_rect = {pad_w, pad_h, resized_w, resized_h};
    im_rect pattern_rect;
    memset(&pattern, 0, sizeof(pattern));
    memset(&pattern_rect, 0, sizeof(pattern_rect));

    int checked = imcheck(source, destination, source_rect, destination_rect);
    if (checked != IM_STATUS_NOERROR) {
        set_err(err, err_len, "NV12 letterbox imcheck: %s",
                imStrError((IM_STATUS)checked));
        return -1;
    }
    IM_STATUS status = improcess(
        source, destination, pattern,
        source_rect, destination_rect, pattern_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        set_err(err, err_len, "NV12 letterbox improcess: %s",
                imStrError(status));
        return -1;
    }
    if (out_ratio) *out_ratio = ratio;
    if (out_pad_w) *out_pad_w = pad_w;
    if (out_pad_h) *out_pad_h = pad_h;
    return 1;
}

/* 函数职责：兼容接口：把NV12 DMA-BUF转换为原车牌320×320模型输入。 */
int rga_letterbox_nv12_fd(int src_fd, int src_w, int src_h,
                          int src_hstride, int src_vstride,
                          uint8_t *dst, int dst_size, int target,
                          float *out_ratio, int *out_pad_w, int *out_pad_h,
                          char *err, int err_len)
{
    return rga_letterbox_nv12_fd_size(
        src_fd, src_w, src_h, src_hstride, src_vstride,
        dst, dst_size, target, target,
        out_ratio, out_pad_w, out_pad_h, err, err_len);
}

/* 函数职责：从NV12 DMA-BUF提取偶数对齐ROI并转换到预分配BGR缓冲。 */
int rga_nv12_roi_bgr_fd(int src_fd, int src_w, int src_h,
                        int src_hstride, int src_vstride,
                        int requested_x, int requested_y,
                        int requested_w, int requested_h,
                        uint8_t *dst, int dst_size,
                        int dst_stride_pixels, int dst_vstride_pixels,
                        int *out_x, int *out_y, int *out_w, int *out_h,
                        char *err, int err_len)
{
    if (src_fd < 0 || !dst || src_w <= 0 || src_h <= 0 ||
        src_hstride < src_w || src_vstride < src_h ||
        requested_w <= 0 || requested_h <= 0) {
        set_err(err, err_len, "invalid NV12 ROI arguments");
        return -1;
    }
    int64_t requested_x2 = (int64_t)requested_x + requested_w;
    int64_t requested_y2 = (int64_t)requested_y + requested_h;
    int x1 = requested_x < 0 ? 0 :
             (requested_x > src_w ? src_w : requested_x);
    int y1 = requested_y < 0 ? 0 :
             (requested_y > src_h ? src_h : requested_y);
    int x2 = requested_x2 < 0 ? 0 :
             (requested_x2 > src_w ? src_w : (int)requested_x2);
    int y2 = requested_y2 < 0 ? 0 :
             (requested_y2 > src_h ? src_h : (int)requested_y2);
    x1 &= ~1;
    y1 &= ~1;
    x2 = (x2 + 1) & ~1;
    y2 = (y2 + 1) & ~1;
    if (x2 > src_w) x2 = src_w & ~1;
    if (y2 > src_h) y2 = src_h & ~1;
    int width = x2 - x1;
    int height = y2 - y1;
    int minimum_stride = (width + 3) & ~3;
    if (width <= 0 || height <= 0 ||
        dst_stride_pixels < minimum_stride ||
        dst_vstride_pixels < height ||
        (size_t)dst_size <
            (size_t)dst_stride_pixels * dst_vstride_pixels * 3u) {
        set_err(err, err_len,
                "invalid/aligned ROI %d,%d %dx%d or dst too small",
                x1, y1, width, height);
        return -1;
    }

    rga_buffer_t source =
        wrapbuffer_fd_t(src_fd, src_w, src_h, src_hstride, src_vstride,
                        RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t destination =
        wrapbuffer_virtualaddr_t(dst, width, height,
                                 dst_stride_pixels, dst_vstride_pixels,
                                 RK_FORMAT_BGR_888);
    rga_buffer_t pattern;
    im_rect source_rect = {x1, y1, width, height};
    im_rect destination_rect = {0, 0, width, height};
    im_rect pattern_rect;
    memset(&pattern, 0, sizeof(pattern));
    memset(&pattern_rect, 0, sizeof(pattern_rect));
    int checked = imcheck(source, destination, source_rect, destination_rect);
    if (checked != IM_STATUS_NOERROR) {
        set_err(err, err_len, "NV12 ROI imcheck: %s",
                imStrError((IM_STATUS)checked));
        return -1;
    }
    IM_STATUS status = improcess(
        source, destination, pattern,
        source_rect, destination_rect, pattern_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        set_err(err, err_len, "NV12 ROI improcess: %s", imStrError(status));
        return -1;
    }
    if (out_x) *out_x = x1;
    if (out_y) *out_y = y1;
    if (out_w) *out_w = width;
    if (out_h) *out_h = height;
    return 1;
}
