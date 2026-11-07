/*
 * RKNPU2持久化模型运行时。
 *
 * open阶段读取模型、查询I/O属性并只分配一次输入tensor memory；inference阶段
 * 复用输入和输出描述，避免逐帧malloc。输出数据的所有权仍属于RKNN runtime，
 * Python必须在下一次推理或close之前完成解析/复制。
 *
 * rknn_native_tensor_info和所有rknn_native_*导出函数是ctypes ABI。错误通过
 * 返回码和调用方提供的err缓冲传递，不从C层打印或终止进程。
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <rknn_api.h>
#include <rga/im2d.h>
#include <rga/rga.h>


#define RKNN_NATIVE_MAX_IO 8
#define RKNN_NATIVE_MAX_DIMS 16
#define RKNN_NATIVE_PLATE_CHANNELS 15
#define RKNN_NATIVE_PLATE_ANCHORS 3
#define RKNN_NATIVE_PLATE_HEADS 3
#define RKNN_NATIVE_PLATE_CLASSES 2
#define RKNN_NATIVE_POSE26_CHANNELS 18
#define RKNN_NATIVE_POSE26_HEADS 4
#define RKNN_NATIVE_POSE26_MAX_TOPK 300
#define RKNN_NATIVE_ROI_POSE_WIDTH 320
#define RKNN_NATIVE_ROI_POSE_HEIGHT 160
#define RKNN_NATIVE_ROI_TILE_WIDTH 160
#define RKNN_NATIVE_ROI_TILE_HEIGHT 80
#define RKNN_NATIVE_ROI_MAX 4
#define RKNN_NATIVE_OCR_WIDTH 168
#define RKNN_NATIVE_OCR_HEIGHT 48
#define RKNN_NATIVE_OCR_CHANNELS 3
#define RKNN_NATIVE_OCR_STEPS 21
#define RKNN_NATIVE_OCR_CLASSES 78
#define RKNN_NATIVE_OCR_COLORS 5

enum plate_head_layout {
    PLATE_HEAD_NCHW = 0,
    PLATE_HEAD_NHWC = 1,
    PLATE_HEAD_EXPLICIT_ANCHOR = 2,
    PLATE_HEAD_NC1HWC2 = 3,
};

struct pose26_head {
    uint32_t output_index;
    int grid_width;
    int grid_height;
    int row_stride;
    int channel_block;
    enum plate_head_layout layout;
};

static const int pose26_strides[RKNN_NATIVE_POSE26_HEADS] = {4, 8, 16, 32};

struct rknn_native_tensor_info {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[RKNN_NATIVE_MAX_DIMS];
    uint32_t n_elems;
    uint32_t size;
    int32_t type;
    int32_t format;
    int32_t qnt_type;
    int32_t zero_point;
    float scale;
};

struct rknn_native_plate_detection {
    float bbox[4];
    float landmarks[8];
    float landmark_visibility[4];
    float confidence;
    int32_t class_id;
};

struct rknn_native_roi_rect {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
};

struct rknn_native_roi_slot_info {
    int32_t roi_index;
    int32_t slot_left;
    int32_t slot_top;
    float ratio;
    float pad_x;
    float pad_y;
    struct rknn_native_roi_rect requested;
    struct rknn_native_roi_rect aligned;
};

struct rknn_native_roi_plate_detection {
    float bbox[4];
    float landmarks[8];
    float landmark_visibility[4];
    float confidence;
    int32_t class_id;
    int32_t roi_index;
};

struct rknn_native_detection_timing {
    float preprocess_ms;
    float npu_ms;
    float output_ms;
    float decode_ms;
    float nms_ms;
    float total_ms;
    uint32_t anchors_scanned;
    uint32_t candidates_decoded;
};

struct rknn_native_ocr_result {
    int32_t char_indices[RKNN_NATIVE_OCR_STEPS];
    uint32_t char_count;
    int32_t color_index;
    float text_confidence;
    float color_confidence;
    float resize_ms;
    float npu_ms;
    float decode_ms;
    float total_ms;
};

struct plate_candidate {
    float bbox[4];
    float landmarks[8];
    float landmark_visibility[4];
    float confidence;
    int32_t class_id;
};

struct pose26_anchor_candidate {
    int head_index;
    int x;
    int y;
    float scores[RKNN_NATIVE_PLATE_CLASSES];
    float maximum_score;
};

struct pose26_class_candidate {
    int anchor_index;
    int class_id;
    float score;
};

struct rknn_native {
    rknn_context context;
    void *model_data;
    size_t model_size;
    rknn_input_output_num io_count;
    rknn_tensor_attr input_attrs[RKNN_NATIVE_MAX_IO];
    rknn_tensor_attr output_attrs[RKNN_NATIVE_MAX_IO];
    rknn_tensor_attr native_output_attrs[RKNN_NATIVE_MAX_IO];
    rknn_tensor_mem *input_mems[2];
    rknn_tensor_mem *output_mems[RKNN_NATIVE_MAX_IO];
    int input_mem_bound;
    int output_mem_bound;
    int input_bound_index;
    int input_prepared[2];
    int input_width;
    int input_height;
    int input_channels;
    struct plate_candidate *plate_candidates;
    size_t plate_candidate_capacity;
    uint8_t *ocr_resized_u8;
    float *ocr_input_f32;
    float *ocr_text_output;
    float *ocr_color_output;
    int ocr_buffers_ready;
    float *roi_pose_outputs[RKNN_NATIVE_POSE26_HEADS];
    int roi_pose_outputs_ready;
};

/* 函数职责：把格式化错误写入调用方缓冲，缓冲为空时安全忽略。 */
static void set_err(char *err, size_t err_len, const char *text, int code)
{
    if (!err || err_len == 0) {
        return;
    }
    if (code) {
        snprintf(err, err_len, "%s: %d", text, code);
    } else {
        snprintf(err, err_len, "%s", text);
    }
}

static double monotonic_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static float clamp_float(float value, float minimum, float maximum)
{
    if (value < minimum) return minimum;
    if (value > maximum) return maximum;
    return value;
}

static float sigmoid_float(float value)
{
    value = clamp_float(value, -30.0f, 30.0f);
    return 1.0f / (1.0f + expf(-value));
}

/* 函数职责：把RKNN模型文件完整读入内存，所有权交给调用方释放。 */
static int load_file(const char *path, void **data, size_t *size,
                     char *err, size_t err_len)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        snprintf(err, err_len, "open %s failed: %s", path, strerror(errno));
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        set_err(err, err_len, "model seek failed", 0);
        return -1;
    }
    long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        set_err(err, err_len, "invalid model size", 0);
        return -1;
    }
    void *buffer = malloc((size_t)length);
    if (!buffer || fread(buffer, 1, (size_t)length, file) != (size_t)length) {
        free(buffer);
        fclose(file);
        set_err(err, err_len, "reading model failed", 0);
        return -1;
    }
    fclose(file);
    *data = buffer;
    *size = (size_t)length;
    return 0;
}

void rknn_native_close(void *handle);

/* 函数职责：创建RKNN上下文，查询输入输出属性并一次性分配持久内存。 */
void *rknn_native_open(const char *model_path, char *err, size_t err_len)
{
    if (!model_path || !model_path[0]) {
        set_err(err, err_len, "model path is empty", 0);
        return NULL;
    }
    struct rknn_native *native = calloc(1, sizeof(*native));
    if (!native) {
        set_err(err, err_len, "out of memory", 0);
        return NULL;
    }
    if (load_file(model_path, &native->model_data, &native->model_size,
                  err, err_len) != 0) {
        rknn_native_close(native);
        return NULL;
    }
    int result = rknn_init(&native->context, native->model_data,
                           (uint32_t)native->model_size, 0, NULL);
    if (result < 0) {
        set_err(err, err_len, "rknn_init failed", result);
        rknn_native_close(native);
        return NULL;
    }
    result = rknn_query(native->context, RKNN_QUERY_IN_OUT_NUM,
                        &native->io_count, sizeof(native->io_count));
    if (result < 0 || native->io_count.n_input != 1 ||
        native->io_count.n_output == 0 ||
        native->io_count.n_input > RKNN_NATIVE_MAX_IO ||
        native->io_count.n_output > RKNN_NATIVE_MAX_IO) {
        set_err(err, err_len, "unsupported RKNN input/output count", result);
        rknn_native_close(native);
        return NULL;
    }
    for (uint32_t i = 0; i < native->io_count.n_input; ++i) {
        native->input_attrs[i].index = i;
        result = rknn_query(native->context, RKNN_QUERY_INPUT_ATTR,
                            &native->input_attrs[i],
                            sizeof(native->input_attrs[i]));
        if (result < 0) {
            set_err(err, err_len, "RKNN_QUERY_INPUT_ATTR failed", result);
            rknn_native_close(native);
            return NULL;
        }
    }
    rknn_tensor_attr *input = &native->input_attrs[0];
    if (input->n_dims != 4) {
        set_err(err, err_len, "native RKNN input must have four dimensions", 0);
        rknn_native_close(native);
        return NULL;
    }
    if (input->fmt == RKNN_TENSOR_NHWC) {
        native->input_height = (int)input->dims[1];
        native->input_width = (int)input->dims[2];
        native->input_channels = (int)input->dims[3];
    } else if (input->fmt == RKNN_TENSOR_NCHW) {
        native->input_channels = (int)input->dims[1];
        native->input_height = (int)input->dims[2];
        native->input_width = (int)input->dims[3];
    } else {
        set_err(err, err_len, "unsupported RKNN input tensor format", 0);
        rknn_native_close(native);
        return NULL;
    }
    for (uint32_t i = 0; i < native->io_count.n_output; ++i) {
        native->output_attrs[i].index = i;
        result = rknn_query(native->context, RKNN_QUERY_OUTPUT_ATTR,
                            &native->output_attrs[i],
                            sizeof(native->output_attrs[i]));
        if (result < 0) {
            set_err(err, err_len, "RKNN_QUERY_OUTPUT_ATTR failed", result);
            rknn_native_close(native);
            return NULL;
        }
        native->native_output_attrs[i].index = i;
        result = rknn_query(native->context, RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                            &native->native_output_attrs[i],
                            sizeof(native->native_output_attrs[i]));
        if (result < 0) {
            set_err(err, err_len, "RKNN_QUERY_NATIVE_OUTPUT_ATTR failed", result);
            rknn_native_close(native);
            return NULL;
        }
    }
    set_err(err, err_len, "", 0);
    return native;
}

/* 函数职责：返回模型输出Tensor数量。 */
int rknn_native_output_count(void *handle)
{
    struct rknn_native *native = handle;
    return native ? (int)native->io_count.n_output : -1;
}

/* 函数职责：返回模型输入尺寸、布局、类型和量化参数。 */
int rknn_native_get_input_info(void *handle,
                               struct rknn_native_tensor_info *info,
                               char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || !info) {
        set_err(err, err_len, "invalid input info request", 0);
        return -1;
    }
    rknn_tensor_attr *attr = &native->input_attrs[0];
    memset(info, 0, sizeof(*info));
    info->index = attr->index;
    info->n_dims = attr->n_dims > RKNN_NATIVE_MAX_DIMS
                       ? RKNN_NATIVE_MAX_DIMS
                       : attr->n_dims;
    for (uint32_t i = 0; i < info->n_dims; ++i) {
        info->dims[i] = attr->dims[i];
    }
    info->n_elems = attr->n_elems;
    info->size = attr->size;
    info->type = attr->type;
    info->format = attr->fmt;
    info->qnt_type = attr->qnt_type;
    info->zero_point = attr->zp;
    info->scale = attr->scale;
    return 0;
}

/* 函数职责：返回指定输出Tensor的尺寸与量化信息。 */
int rknn_native_get_output_info(void *handle, int output_index,
                                struct rknn_native_tensor_info *info,
                                char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || !info || output_index < 0 ||
        output_index >= (int)native->io_count.n_output) {
        set_err(err, err_len, "invalid output info request", 0);
        return -1;
    }
    rknn_tensor_attr *attr = &native->output_attrs[output_index];
    memset(info, 0, sizeof(*info));
    info->index = attr->index;
    info->n_dims = attr->n_dims > RKNN_NATIVE_MAX_DIMS
                       ? RKNN_NATIVE_MAX_DIMS
                       : attr->n_dims;
    for (uint32_t i = 0; i < info->n_dims; ++i) {
        info->dims[i] = attr->dims[i];
    }
    info->n_elems = attr->n_elems;
    info->size = attr->size;
    info->type = attr->type;
    info->format = attr->fmt;
    info->qnt_type = attr->qnt_type;
    info->zero_point = attr->zp;
    info->scale = attr->scale;
    return 0;
}

/* 函数职责：把RKNN输出转换或反量化为调用方提供的float缓冲。 */
static int copy_float_outputs(struct rknn_native *native,
                              void **output_buffers,
                              const size_t *output_sizes,
                              char *err, size_t err_len)
{
    rknn_output outputs[RKNN_NATIVE_MAX_IO];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < native->io_count.n_output; ++i) {
        size_t required = (size_t)native->output_attrs[i].n_elems * sizeof(float);
        if (!output_buffers[i] || output_sizes[i] < required) {
            set_err(err, err_len, "preallocated output is too small", 0);
            return -1;
        }
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 1;
        outputs[i].buf = output_buffers[i];
        outputs[i].size = (uint32_t)output_sizes[i];
    }
    int result = rknn_outputs_get(native->context, native->io_count.n_output,
                                  outputs, NULL);
    if (result < 0) {
        set_err(err, err_len, "rknn_outputs_get failed", result);
        return -1;
    }
    rknn_outputs_release(native->context, native->io_count.n_output, outputs);
    return 0;
}

static int validate_ocr_contract(struct rknn_native *native,
                                 char *err, size_t err_len)
{
    rknn_tensor_attr *input = &native->input_attrs[0];
    if (native->input_mem_bound || input->fmt != RKNN_TENSOR_NHWC ||
        input->n_dims != 4 || input->dims[0] != 1 ||
        native->input_width != RKNN_NATIVE_OCR_WIDTH ||
        native->input_height != RKNN_NATIVE_OCR_HEIGHT ||
        native->input_channels != RKNN_NATIVE_OCR_CHANNELS) {
        set_err(err, err_len,
                "OCR input must be NHWC float-compatible [1,48,168,3]", 0);
        return -1;
    }
    if (native->io_count.n_output != 2) {
        set_err(err, err_len, "OCR model must have two outputs", 0);
        return -1;
    }
    rknn_tensor_attr *text = &native->output_attrs[0];
    rknn_tensor_attr *color = &native->output_attrs[1];
    if (text->n_dims != 3 || text->dims[0] != 1 ||
        text->dims[1] != RKNN_NATIVE_OCR_STEPS ||
        text->dims[2] != RKNN_NATIVE_OCR_CLASSES ||
        text->n_elems != RKNN_NATIVE_OCR_STEPS * RKNN_NATIVE_OCR_CLASSES) {
        set_err(err, err_len, "OCR text output must be [1,21,78]", 0);
        return -1;
    }
    if (color->n_dims != 2 || color->dims[0] != 1 ||
        color->dims[1] != RKNN_NATIVE_OCR_COLORS ||
        color->n_elems != RKNN_NATIVE_OCR_COLORS) {
        set_err(err, err_len, "OCR color output must be [1,5]", 0);
        return -1;
    }
    return 0;
}

static int ensure_ocr_buffers(struct rknn_native *native,
                              char *err, size_t err_len)
{
    if (native->ocr_buffers_ready) {
        return 0;
    }
    if (validate_ocr_contract(native, err, err_len) != 0) {
        return -1;
    }
    size_t pixels = (size_t)RKNN_NATIVE_OCR_WIDTH *
                    RKNN_NATIVE_OCR_HEIGHT * RKNN_NATIVE_OCR_CHANNELS;
    native->ocr_resized_u8 = malloc(pixels);
    native->ocr_input_f32 = malloc(pixels * sizeof(float));
    native->ocr_text_output = malloc(
        (size_t)RKNN_NATIVE_OCR_STEPS * RKNN_NATIVE_OCR_CLASSES *
        sizeof(float)
    );
    native->ocr_color_output = malloc(
        (size_t)RKNN_NATIVE_OCR_COLORS * sizeof(float)
    );
    if (!native->ocr_resized_u8 || !native->ocr_input_f32 ||
        !native->ocr_text_output || !native->ocr_color_output) {
        free(native->ocr_resized_u8);
        free(native->ocr_input_f32);
        free(native->ocr_text_output);
        free(native->ocr_color_output);
        native->ocr_resized_u8 = NULL;
        native->ocr_input_f32 = NULL;
        native->ocr_text_output = NULL;
        native->ocr_color_output = NULL;
        set_err(err, err_len, "allocating persistent OCR buffers failed", 0);
        return -1;
    }
    native->ocr_buffers_ready = 1;
    return 0;
}

static int stable_argmax_probability(const float *logits, int count,
                                     int *out_index, float *out_probability)
{
    float maximum = -INFINITY;
    int maximum_index = 0;
    for (int index = 0; index < count; ++index) {
        if (!isfinite(logits[index])) {
            return -1;
        }
        if (logits[index] > maximum) {
            maximum = logits[index];
            maximum_index = index;
        }
    }
    double total = 0.0;
    for (int index = 0; index < count; ++index) {
        total += exp((double)logits[index] - maximum);
    }
    if (!isfinite(total) || total <= 0.0) {
        return -1;
    }
    *out_index = maximum_index;
    *out_probability = (float)(1.0 / total);
    return 0;
}

static int decode_ocr_logits(const float *text_logits,
                             const float *color_logits,
                             struct rknn_native_ocr_result *ocr_result,
                             char *err, size_t err_len)
{
    ocr_result->char_count = 0;
    ocr_result->color_index = -1;
    ocr_result->text_confidence = 0.0f;
    ocr_result->color_confidence = 0.0f;
    int previous = 0;
    float confidence_sum = 0.0f;
    for (int step = 0; step < RKNN_NATIVE_OCR_STEPS; ++step) {
        int index = 0;
        float probability = 0.0f;
        const float *row = text_logits + step * RKNN_NATIVE_OCR_CLASSES;
        if (stable_argmax_probability(row, RKNN_NATIVE_OCR_CLASSES,
                                      &index, &probability) != 0) {
            set_err(err, err_len, "invalid OCR text logits", 0);
            return -1;
        }
        confidence_sum += probability;
        if (index != 0 && index != previous &&
            ocr_result->char_count < RKNN_NATIVE_OCR_STEPS) {
            ocr_result->char_indices[ocr_result->char_count++] = index;
        }
        previous = index;
    }
    ocr_result->text_confidence =
        confidence_sum / (float)RKNN_NATIVE_OCR_STEPS;
    if (stable_argmax_probability(
            color_logits, RKNN_NATIVE_OCR_COLORS,
            &ocr_result->color_index, &ocr_result->color_confidence) != 0) {
        set_err(err, err_len, "invalid OCR color logits", 0);
        return -1;
    }
    return 0;
}

/* 纯解码测试入口，也作为CTC和颜色逻辑的稳定公共契约。 */
int rknn_native_decode_ocr_logits(
    const float *text_logits, size_t text_count,
    const float *color_logits, size_t color_count,
    struct rknn_native_ocr_result *ocr_result,
    char *err, size_t err_len)
{
    if (!text_logits || !color_logits || !ocr_result ||
        text_count != RKNN_NATIVE_OCR_STEPS * RKNN_NATIVE_OCR_CLASSES ||
        color_count != RKNN_NATIVE_OCR_COLORS) {
        set_err(err, err_len, "invalid OCR decode buffers", 0);
        return -1;
    }
    memset(ocr_result, 0, sizeof(*ocr_result));
    double started = monotonic_ms();
    if (decode_ocr_logits(text_logits, color_logits, ocr_result,
                          err, err_len) != 0) {
        return -1;
    }
    ocr_result->decode_ms = (float)(monotonic_ms() - started);
    ocr_result->total_ms = ocr_result->decode_ms;
    return 0;
}

/*
 * OCR融合快路径：接收OpenCV已缩放的168x48 BGR，以NHWC float32交给RKNN；
 * softmax、21步CTC和颜色分类全部在持久C缓冲上完成。
 */
int rknn_native_recognize_bgr(void *handle,
                              const uint8_t *src, int src_width,
                              int src_height, int src_stride,
                              size_t src_size,
                              struct rknn_native_ocr_result *ocr_result,
                              char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || !src || !ocr_result) {
        set_err(err, err_len, "invalid native OCR request", 0);
        return -1;
    }
    memset(ocr_result, 0, sizeof(*ocr_result));
    ocr_result->color_index = -1;
    double total_started = monotonic_ms();
    if (ensure_ocr_buffers(native, err, err_len) != 0) {
        return -1;
    }

    size_t resized_elements = (size_t)RKNN_NATIVE_OCR_WIDTH *
                              RKNN_NATIVE_OCR_HEIGHT *
                              RKNN_NATIVE_OCR_CHANNELS;
    size_t row_bytes = (size_t)RKNN_NATIVE_OCR_WIDTH *
                       RKNN_NATIVE_OCR_CHANNELS;
    if (src_width != RKNN_NATIVE_OCR_WIDTH ||
        src_height != RKNN_NATIVE_OCR_HEIGHT ||
        src_stride < (int)row_bytes) {
        set_err(err, err_len,
                "native OCR input must be a complete 168x48 BGR image", 0);
        return -1;
    }
    size_t source_required =
        (size_t)(RKNN_NATIVE_OCR_HEIGHT - 1) * (size_t)src_stride + row_bytes;
    if (src_size < source_required) {
        set_err(err, err_len, "native OCR input buffer is truncated", 0);
        return -1;
    }
    for (int row = 0; row < RKNN_NATIVE_OCR_HEIGHT; ++row) {
        memcpy(native->ocr_resized_u8 + (size_t)row * row_bytes,
               src + (size_t)row * src_stride, row_bytes);
    }
    double npu_started = monotonic_ms();
    for (size_t index = 0; index < resized_elements; ++index) {
        native->ocr_input_f32[index] = (float)native->ocr_resized_u8[index];
    }

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = native->ocr_input_f32;
    input.size = (uint32_t)(resized_elements * sizeof(float));
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = RKNN_TENSOR_NHWC;
    input.pass_through = 0;
    int code = rknn_inputs_set(native->context, 1, &input);
    if (code < 0) {
        set_err(err, err_len, "OCR rknn_inputs_set failed", code);
        return -1;
    }
    code = rknn_run(native->context, NULL);
    if (code < 0) {
        set_err(err, err_len, "OCR rknn_run failed", code);
        return -1;
    }
    void *output_buffers[2] = {
        native->ocr_text_output, native->ocr_color_output
    };
    size_t output_sizes[2] = {
        (size_t)RKNN_NATIVE_OCR_STEPS * RKNN_NATIVE_OCR_CLASSES *
            sizeof(float),
        (size_t)RKNN_NATIVE_OCR_COLORS * sizeof(float),
    };
    if (copy_float_outputs(native, output_buffers, output_sizes,
                           err, err_len) != 0) {
        return -1;
    }
    ocr_result->npu_ms = (float)(monotonic_ms() - npu_started);

    double decode_started = monotonic_ms();
    if (decode_ocr_logits(native->ocr_text_output, native->ocr_color_output,
                          ocr_result, err, err_len) != 0) {
        return -1;
    }
    ocr_result->decode_ms = (float)(monotonic_ms() - decode_started);
    ocr_result->total_ms = (float)(monotonic_ms() - total_started);
    if (err && err_len > 0) {
        err[0] = '\0';
    }
    return 0;
}

/* 为同一RKNN上下文创建两块持久输入DMA-BUF，供RGA/NPU流水重叠。 */
static int bind_zero_copy_input(struct rknn_native *native,
                                char *err, size_t err_len)
{
    if (native->input_mem_bound) {
        return 0;
    }
    rknn_tensor_attr *attr = &native->input_attrs[0];
    attr->type = RKNN_TENSOR_UINT8;
    attr->fmt = RKNN_TENSOR_NHWC;
    uint32_t bytes = attr->size_with_stride ? attr->size_with_stride : attr->size;
    for (int slot = 0; slot < 2; ++slot) {
        native->input_mems[slot] = rknn_create_mem(native->context, bytes);
        if (!native->input_mems[slot] ||
            !native->input_mems[slot]->virt_addr ||
            native->input_mems[slot]->fd < 0) {
            set_err(err, err_len, "rknn_create_mem input DMA-BUF failed", 0);
            goto failure;
        }
    }
    int result = rknn_set_io_mem(
        native->context, native->input_mems[0], attr
    );
    if (result < 0) {
        set_err(err, err_len, "rknn_set_io_mem input failed", result);
        goto failure;
    }
    for (int slot = 0; slot < 2; ++slot) {
        fprintf(stderr,
                "RKNN zero-copy input[%d]: %dx%dx%d stride=%ux%u "
                "bytes=%u fd=%d\n",
                slot, native->input_width, native->input_height,
                native->input_channels, attr->w_stride, attr->h_stride,
                native->input_mems[slot]->size,
                native->input_mems[slot]->fd);
    }
    native->input_mem_bound = 1;
    native->input_bound_index = 0;
    return 0;

failure:
    for (int slot = 0; slot < 2; ++slot) {
        if (native->input_mems[slot]) {
            rknn_destroy_mem(native->context, native->input_mems[slot]);
            native->input_mems[slot] = NULL;
        }
    }
    return -1;
}

static int bind_input_slot(struct rknn_native *native, int slot,
                           char *err, size_t err_len)
{
    if (!native || slot < 0 || slot >= 2 || !native->input_mem_bound ||
        !native->input_mems[slot]) {
        set_err(err, err_len, "invalid RKNN input slot", 0);
        return -1;
    }
    if (native->input_bound_index == slot) {
        return 0;
    }
    int result = rknn_set_io_mem(
        native->context, native->input_mems[slot], &native->input_attrs[0]
    );
    if (result < 0) {
        set_err(err, err_len, "switching RKNN input slot failed", result);
        return -1;
    }
    native->input_bound_index = slot;
    return 0;
}

/* 为四路Pose26 INT8输出绑定持久内存；该入口不兼容旧三输出模型。 */
static int bind_pose26_output_memory(struct rknn_native *native,
                                     char *err, size_t err_len)
{
    if (native->output_mem_bound) {
        return 0;
    }
    if (native->io_count.n_output != RKNN_NATIVE_POSE26_HEADS) {
        set_err(err, err_len, "Pose26 detector must have four outputs", 0);
        return -1;
    }
    for (uint32_t i = 0; i < native->io_count.n_output; ++i) {
        rknn_tensor_attr *logical = &native->output_attrs[i];
        rknn_tensor_attr *physical = &native->native_output_attrs[i];
        if (logical->type != RKNN_TENSOR_INT8 ||
            logical->qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
            logical->n_elems % RKNN_NATIVE_POSE26_CHANNELS != 0 ||
            !isfinite(logical->scale) || logical->scale <= 0.0f) {
            set_err(err, err_len,
                    "Pose26 output must be affine INT8 with 18 channels", 0);
            goto failure;
        }
        uint32_t bytes = physical->size_with_stride
                             ? physical->size_with_stride : physical->size;
        native->output_mems[i] = rknn_create_mem(native->context, bytes);
        if (!native->output_mems[i] || !native->output_mems[i]->virt_addr) {
            set_err(err, err_len, "rknn_create_mem Pose26 output failed", 0);
            goto failure;
        }
        int result = rknn_set_io_mem(
            native->context, native->output_mems[i], physical
        );
        if (result < 0) {
            set_err(err, err_len, "rknn_set_io_mem Pose26 output failed", result);
            goto failure;
        }
        fprintf(stderr,
                "RKNN Pose26 output[%u]: dims=%u elems=%u stride=%ux%u bytes=%u fd=%d\n",
                i, physical->n_dims, physical->n_elems,
                physical->w_stride, physical->h_stride,
                native->output_mems[i]->size, native->output_mems[i]->fd);
    }
    native->output_mem_bound = 1;
    return 0;

failure:
    for (uint32_t i = 0; i < native->io_count.n_output; ++i) {
        if (native->output_mems[i]) {
            rknn_destroy_mem(native->context, native->output_mems[i]);
            native->output_mems[i] = NULL;
        }
    }
    return -1;
}

static float dequantize_plate_value(int8_t value,
                                    const rknn_tensor_attr *attr)
{
    return ((float)value - (float)attr->zp) * attr->scale;
}

/* 把letterbox坐标还原到原始NV12帧，并限制在有效图像范围内。 */
static void map_plate_detection_to_source(
    const struct plate_candidate *candidate,
    float ratio, int pad_x, int pad_y,
    int source_width, int source_height,
    struct rknn_native_plate_detection *detection)
{
    for (int point = 0; point < 2; ++point) {
        detection->bbox[point * 2] = clamp_float(
            (candidate->bbox[point * 2] - (float)pad_x) / ratio,
            0.0f, (float)source_width
        );
        detection->bbox[point * 2 + 1] = clamp_float(
            (candidate->bbox[point * 2 + 1] - (float)pad_y) / ratio,
            0.0f, (float)source_height
        );
    }
    for (int point = 0; point < 4; ++point) {
        detection->landmarks[point * 2] = clamp_float(
            (candidate->landmarks[point * 2] - (float)pad_x) / ratio,
            0.0f, (float)source_width
        );
        detection->landmarks[point * 2 + 1] = clamp_float(
            (candidate->landmarks[point * 2 + 1] - (float)pad_y) / ratio,
            0.0f, (float)source_height
        );
        detection->landmark_visibility[point] =
            candidate->landmark_visibility[point];
    }
    detection->confidence = candidate->confidence;
    detection->class_id = candidate->class_id;
}

static int describe_pose26_head(const rknn_tensor_attr *logical,
                                const rknn_tensor_attr *physical,
                                uint32_t output_index,
                                struct pose26_head *head,
                                char *err, size_t err_len)
{
    memset(head, 0, sizeof(*head));
    head->output_index = output_index;
    if (logical->n_dims == 4 &&
        logical->dims[1] == RKNN_NATIVE_POSE26_CHANNELS) {
        head->grid_height = (int)logical->dims[2];
        head->grid_width = (int)logical->dims[3];
    } else if (logical->n_dims == 4 &&
               logical->dims[3] == RKNN_NATIVE_POSE26_CHANNELS) {
        head->grid_height = (int)logical->dims[1];
        head->grid_width = (int)logical->dims[2];
    } else {
        set_err(err, err_len,
                "Pose26 output must be [1,18,H,W] or [1,H,W,18]", 0);
        return -1;
    }

    if (physical->fmt == RKNN_TENSOR_NC1HWC2 && physical->n_dims == 5 &&
        physical->dims[2] == (uint32_t)head->grid_height &&
        physical->dims[3] == (uint32_t)head->grid_width) {
        head->layout = PLATE_HEAD_NC1HWC2;
        head->channel_block = (int)physical->dims[4];
    } else if (physical->fmt == RKNN_TENSOR_NCHW && physical->n_dims == 4) {
        head->layout = PLATE_HEAD_NCHW;
        head->channel_block = 1;
    } else if (physical->fmt == RKNN_TENSOR_NHWC && physical->n_dims == 4) {
        head->layout = PLATE_HEAD_NHWC;
        head->channel_block = 1;
    } else {
        set_err(err, err_len, "unsupported Pose26 native output layout", 0);
        return -1;
    }
    head->row_stride = physical->w_stride
                           ? (int)physical->w_stride : head->grid_width;
    if (head->grid_width <= 0 || head->grid_height <= 0 ||
        head->row_stride < head->grid_width || head->channel_block <= 0) {
        set_err(err, err_len, "invalid Pose26 output grid", 0);
        return -1;
    }
    return 0;
}

static int prepare_pose26_heads(
    struct rknn_native *native,
    struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    char *err, size_t err_len)
{
    if (native->input_width != 512 || native->input_height != 288 ||
        native->input_channels != 3 ||
        native->io_count.n_output != RKNN_NATIVE_POSE26_HEADS) {
        set_err(err, err_len,
                "AI Pose26 model must be RGB 288x512 with four outputs", 0);
        return -1;
    }
    for (uint32_t i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        if (describe_pose26_head(&native->output_attrs[i],
                                 &native->native_output_attrs[i], i, &heads[i],
                                 err, err_len) != 0) {
            return -1;
        }
    }
    for (int i = 0; i < RKNN_NATIVE_POSE26_HEADS - 1; ++i) {
        for (int j = i + 1; j < RKNN_NATIVE_POSE26_HEADS; ++j) {
            int area_i = heads[i].grid_width * heads[i].grid_height;
            int area_j = heads[j].grid_width * heads[j].grid_height;
            if (area_j > area_i) {
                struct pose26_head temporary = heads[i];
                heads[i] = heads[j];
                heads[j] = temporary;
            }
        }
    }
    for (int i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        if (heads[i].grid_width != native->input_width / pose26_strides[i] ||
            heads[i].grid_height != native->input_height / pose26_strides[i]) {
            set_err(err, err_len,
                    "Pose26 grids must match strides 4/8/16/32", 0);
            return -1;
        }
    }
    return 0;
}

/* ROI模型输出为逻辑NCHW FP16；outputs_get(want_float=1)写入持久FP32缓冲。 */
static int prepare_roi_pose26_heads(
    struct rknn_native *native,
    struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    char *err, size_t err_len)
{
    if (native->input_width != RKNN_NATIVE_ROI_POSE_WIDTH ||
        native->input_height != RKNN_NATIVE_ROI_POSE_HEIGHT ||
        native->input_channels != 3 ||
        native->input_attrs[0].fmt != RKNN_TENSOR_NHWC ||
        native->io_count.n_output != RKNN_NATIVE_POSE26_HEADS) {
        set_err(err, err_len,
                "ROI Pose26 model must be RGB NHWC 160x320 with four outputs",
                0);
        return -1;
    }
    for (uint32_t i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        rknn_tensor_attr *attr = &native->output_attrs[i];
        if (attr->type != RKNN_TENSOR_FLOAT16 ||
            attr->n_elems % RKNN_NATIVE_POSE26_CHANNELS != 0 ||
            describe_pose26_head(attr, attr, i, &heads[i],
                                 err, err_len) != 0) {
            set_err(err, err_len,
                    "ROI Pose26 outputs must be FP16 with 18 channels", 0);
            return -1;
        }
    }
    for (int i = 0; i < RKNN_NATIVE_POSE26_HEADS - 1; ++i) {
        for (int j = i + 1; j < RKNN_NATIVE_POSE26_HEADS; ++j) {
            int area_i = heads[i].grid_width * heads[i].grid_height;
            int area_j = heads[j].grid_width * heads[j].grid_height;
            if (area_j > area_i) {
                struct pose26_head temporary = heads[i];
                heads[i] = heads[j];
                heads[j] = temporary;
            }
        }
    }
    for (int i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        if (heads[i].grid_width != native->input_width / pose26_strides[i] ||
            heads[i].grid_height != native->input_height / pose26_strides[i]) {
            set_err(err, err_len,
                    "ROI Pose26 grids must match strides 4/8/16/32", 0);
            return -1;
        }
    }
    return 0;
}

static int ensure_roi_pose_output_buffers(struct rknn_native *native,
                                           char *err, size_t err_len)
{
    if (native->roi_pose_outputs_ready) return 0;
    for (uint32_t i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        native->roi_pose_outputs[i] = malloc(
            (size_t)native->output_attrs[i].n_elems * sizeof(float)
        );
        if (!native->roi_pose_outputs[i]) {
            set_err(err, err_len, "allocating ROI Pose26 output failed", 0);
            for (uint32_t j = 0; j <= i; ++j) {
                free(native->roi_pose_outputs[j]);
                native->roi_pose_outputs[j] = NULL;
            }
            return -1;
        }
    }
    native->roi_pose_outputs_ready = 1;
    return 0;
}

static float pose26_float_value(struct rknn_native *native,
                                const struct pose26_head *head,
                                int y, int x, int channel)
{
    const float *data = native->roi_pose_outputs[head->output_index];
    size_t index;
    if (head->layout == PLATE_HEAD_NCHW) {
        index = ((size_t)channel * head->grid_height + (size_t)y) *
                    head->grid_width + (size_t)x;
    } else {
        index = ((size_t)y * head->grid_width + (size_t)x) *
                    RKNN_NATIVE_POSE26_CHANNELS + (size_t)channel;
    }
    return data[index];
}

static int8_t pose26_head_value(const int8_t *data,
                                const struct pose26_head *head,
                                int y, int x, int channel)
{
    size_t index;
    if (head->layout == PLATE_HEAD_NCHW) {
        index = ((size_t)channel * head->grid_height + (size_t)y) *
                    head->row_stride + (size_t)x;
    } else if (head->layout == PLATE_HEAD_NHWC) {
        index = ((size_t)y * head->row_stride + (size_t)x) *
                    RKNN_NATIVE_POSE26_CHANNELS + (size_t)channel;
    } else {
        int group = channel / head->channel_block;
        int lane = channel % head->channel_block;
        index = (((size_t)group * head->grid_height + (size_t)y) *
                     head->row_stride + (size_t)x) *
                    head->channel_block + (size_t)lane;
    }
    return data[index];
}

static void pose26_heap_swap(struct pose26_anchor_candidate *left,
                             struct pose26_anchor_candidate *right)
{
    struct pose26_anchor_candidate temporary = *left;
    *left = *right;
    *right = temporary;
}

static void pose26_heap_sift_up(struct pose26_anchor_candidate *heap, int index)
{
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (heap[parent].maximum_score <= heap[index].maximum_score) break;
        pose26_heap_swap(&heap[parent], &heap[index]);
        index = parent;
    }
}

static void pose26_heap_sift_down(struct pose26_anchor_candidate *heap,
                                  int size, int index)
{
    for (;;) {
        int smallest = index;
        int left = index * 2 + 1;
        int right = left + 1;
        if (left < size &&
            heap[left].maximum_score < heap[smallest].maximum_score) {
            smallest = left;
        }
        if (right < size &&
            heap[right].maximum_score < heap[smallest].maximum_score) {
            smallest = right;
        }
        if (smallest == index) break;
        pose26_heap_swap(&heap[index], &heap[smallest]);
        index = smallest;
    }
}

static void pose26_heap_offer(struct pose26_anchor_candidate *heap,
                              int *size, int capacity,
                              const struct pose26_anchor_candidate *candidate)
{
    if (*size < capacity) {
        heap[*size] = *candidate;
        pose26_heap_sift_up(heap, *size);
        ++(*size);
    } else if (candidate->maximum_score > heap[0].maximum_score) {
        heap[0] = *candidate;
        pose26_heap_sift_down(heap, *size, 0);
    }
}

static int compare_pose26_anchor_desc(const void *left, const void *right)
{
    const struct pose26_anchor_candidate *a = left;
    const struct pose26_anchor_candidate *b = right;
    if (a->maximum_score < b->maximum_score) return 1;
    if (a->maximum_score > b->maximum_score) return -1;
    return 0;
}

static int compare_pose26_class_desc(const void *left, const void *right)
{
    const struct pose26_class_candidate *a = left;
    const struct pose26_class_candidate *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return 0;
}

static void decode_pose26_geometry(
    struct rknn_native *native,
    const struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    const struct pose26_anchor_candidate *anchor,
    int class_id, float confidence,
    struct plate_candidate *candidate)
{
    const struct pose26_head *head = &heads[anchor->head_index];
    const rknn_tensor_attr *attr = &native->output_attrs[head->output_index];
    const int8_t *data =
        (const int8_t *)native->output_mems[head->output_index]->virt_addr;
    float stride = (float)pose26_strides[anchor->head_index];
    float anchor_x = (float)anchor->x + 0.5f;
    float anchor_y = (float)anchor->y + 0.5f;
    float distances[4];
    for (int component = 0; component < 4; ++component) {
        distances[component] = dequantize_plate_value(
            pose26_head_value(data, head, anchor->y, anchor->x, component), attr
        );
    }
    candidate->bbox[0] = (anchor_x - distances[0]) * stride;
    candidate->bbox[1] = (anchor_y - distances[1]) * stride;
    candidate->bbox[2] = (anchor_x + distances[2]) * stride;
    candidate->bbox[3] = (anchor_y + distances[3]) * stride;
    for (int point = 0; point < 4; ++point) {
        int channel = 6 + point * 3;
        float raw_x = dequantize_plate_value(
            pose26_head_value(data, head, anchor->y, anchor->x, channel), attr
        );
        float raw_y = dequantize_plate_value(
            pose26_head_value(data, head, anchor->y, anchor->x, channel + 1), attr
        );
        float raw_visibility = dequantize_plate_value(
            pose26_head_value(data, head, anchor->y, anchor->x, channel + 2), attr
        );
        candidate->landmarks[point * 2] = (raw_x + anchor_x) * stride;
        candidate->landmarks[point * 2 + 1] = (raw_y + anchor_y) * stride;
        candidate->landmark_visibility[point] = sigmoid_float(raw_visibility);
    }
    candidate->confidence = confidence;
    candidate->class_id = class_id;
}

static int decode_pose26_candidates(
    struct rknn_native *native,
    const struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    float confidence_threshold,
    struct plate_candidate *decoded, int decoded_capacity,
    uint32_t *anchors_scanned, uint32_t *candidates_decoded)
{
    struct pose26_anchor_candidate heap[RKNN_NATIVE_POSE26_MAX_TOPK];
    struct pose26_class_candidate classes[RKNN_NATIVE_POSE26_MAX_TOPK *
                                          RKNN_NATIVE_PLATE_CLASSES];
    int heap_size = 0;
    uint32_t scanned = 0;
    for (int head_index = 0; head_index < RKNN_NATIVE_POSE26_HEADS; ++head_index) {
        const struct pose26_head *head = &heads[head_index];
        const rknn_tensor_attr *attr = &native->output_attrs[head->output_index];
        const int8_t *data =
            (const int8_t *)native->output_mems[head->output_index]->virt_addr;
        float sigmoid_lut[256];
        for (int value = -128; value <= 127; ++value) {
            sigmoid_lut[value + 128] = sigmoid_float(
                dequantize_plate_value((int8_t)value, attr)
            );
        }
        for (int y = 0; y < head->grid_height; ++y) {
            for (int x = 0; x < head->grid_width; ++x) {
                struct pose26_anchor_candidate candidate;
                candidate.head_index = head_index;
                candidate.x = x;
                candidate.y = y;
                for (int class_id = 0; class_id < RKNN_NATIVE_PLATE_CLASSES;
                     ++class_id) {
                    int8_t raw = pose26_head_value(
                        data, head, y, x, 4 + class_id
                    );
                    candidate.scores[class_id] = sigmoid_lut[(int)raw + 128];
                }
                candidate.maximum_score =
                    candidate.scores[0] > candidate.scores[1]
                        ? candidate.scores[0] : candidate.scores[1];
                pose26_heap_offer(heap, &heap_size, decoded_capacity, &candidate);
                ++scanned;
            }
        }
    }
    qsort(heap, (size_t)heap_size, sizeof(heap[0]), compare_pose26_anchor_desc);
    int class_count = 0;
    for (int anchor_index = 0; anchor_index < heap_size; ++anchor_index) {
        for (int class_id = 0; class_id < RKNN_NATIVE_PLATE_CLASSES; ++class_id) {
            classes[class_count].anchor_index = anchor_index;
            classes[class_count].class_id = class_id;
            classes[class_count].score = heap[anchor_index].scores[class_id];
            ++class_count;
        }
    }
    qsort(classes, (size_t)class_count, sizeof(classes[0]),
          compare_pose26_class_desc);
    int selected = 0;
    for (int index = 0; index < class_count && selected < decoded_capacity;
         ++index) {
        if (classes[index].score <= confidence_threshold) break;
        decode_pose26_geometry(
            native, heads, &heap[classes[index].anchor_index],
            classes[index].class_id, classes[index].score, &decoded[selected]
        );
        ++selected;
    }
    *anchors_scanned = scanned;
    *candidates_decoded = (uint32_t)selected;
    return selected;
}

static void decode_roi_pose26_geometry(
    struct rknn_native *native,
    const struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    const struct pose26_anchor_candidate *anchor,
    int class_id, float confidence, struct plate_candidate *candidate)
{
    const struct pose26_head *head = &heads[anchor->head_index];
    float stride = (float)pose26_strides[anchor->head_index];
    float anchor_x = (float)anchor->x + 0.5f;
    float anchor_y = (float)anchor->y + 0.5f;
    float distances[4];
    for (int component = 0; component < 4; ++component) {
        distances[component] = pose26_float_value(
            native, head, anchor->y, anchor->x, component
        );
    }
    candidate->bbox[0] = (anchor_x - distances[0]) * stride;
    candidate->bbox[1] = (anchor_y - distances[1]) * stride;
    candidate->bbox[2] = (anchor_x + distances[2]) * stride;
    candidate->bbox[3] = (anchor_y + distances[3]) * stride;
    for (int point = 0; point < 4; ++point) {
        int channel = 6 + point * 3;
        float raw_x = pose26_float_value(
            native, head, anchor->y, anchor->x, channel
        );
        float raw_y = pose26_float_value(
            native, head, anchor->y, anchor->x, channel + 1
        );
        float visibility = pose26_float_value(
            native, head, anchor->y, anchor->x, channel + 2
        );
        candidate->landmarks[point * 2] = (raw_x + anchor_x) * stride;
        candidate->landmarks[point * 2 + 1] = (raw_y + anchor_y) * stride;
        candidate->landmark_visibility[point] = sigmoid_float(visibility);
    }
    candidate->confidence = confidence;
    candidate->class_id = class_id;
}

/* 严格执行4250 anchor -> Top300 anchor -> 600 class -> Top300。 */
static int decode_roi_pose26_candidates(
    struct rknn_native *native,
    const struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS],
    float confidence_threshold,
    struct plate_candidate decoded[RKNN_NATIVE_POSE26_MAX_TOPK],
    uint32_t *anchors_scanned, uint32_t *candidates_decoded)
{
    struct pose26_anchor_candidate heap[RKNN_NATIVE_POSE26_MAX_TOPK];
    struct pose26_class_candidate classes[RKNN_NATIVE_POSE26_MAX_TOPK *
                                          RKNN_NATIVE_PLATE_CLASSES];
    int heap_size = 0;
    uint32_t scanned = 0;
    for (int head_index = 0; head_index < RKNN_NATIVE_POSE26_HEADS;
         ++head_index) {
        const struct pose26_head *head = &heads[head_index];
        for (int y = 0; y < head->grid_height; ++y) {
            for (int x = 0; x < head->grid_width; ++x) {
                struct pose26_anchor_candidate candidate;
                candidate.head_index = head_index;
                candidate.x = x;
                candidate.y = y;
                for (int class_id = 0; class_id < RKNN_NATIVE_PLATE_CLASSES;
                     ++class_id) {
                    candidate.scores[class_id] = sigmoid_float(
                        pose26_float_value(native, head, y, x, 4 + class_id)
                    );
                }
                candidate.maximum_score =
                    candidate.scores[0] > candidate.scores[1]
                        ? candidate.scores[0] : candidate.scores[1];
                pose26_heap_offer(heap, &heap_size,
                                  RKNN_NATIVE_POSE26_MAX_TOPK, &candidate);
                ++scanned;
            }
        }
    }
    qsort(heap, (size_t)heap_size, sizeof(heap[0]), compare_pose26_anchor_desc);
    int class_count = 0;
    for (int anchor_index = 0; anchor_index < heap_size; ++anchor_index) {
        for (int class_id = 0; class_id < RKNN_NATIVE_PLATE_CLASSES;
             ++class_id) {
            classes[class_count].anchor_index = anchor_index;
            classes[class_count].class_id = class_id;
            classes[class_count].score = heap[anchor_index].scores[class_id];
            ++class_count;
        }
    }
    qsort(classes, (size_t)class_count, sizeof(classes[0]),
          compare_pose26_class_desc);
    int selected = 0;
    for (int index = 0;
         index < class_count && selected < RKNN_NATIVE_POSE26_MAX_TOPK;
         ++index) {
        if (classes[index].score <= confidence_threshold) break;
        decode_roi_pose26_geometry(
            native, heads, &heap[classes[index].anchor_index],
            classes[index].class_id, classes[index].score, &decoded[selected]
        );
        ++selected;
    }
    *anchors_scanned = scanned;
    *candidates_decoded = (uint32_t)selected;
    return selected;
}

static int letterbox_nv12_to_input(struct rknn_native *native, int slot,
                                   int src_fd, int src_width, int src_height,
                                   int src_hstride, int src_vstride,
                                   float *out_ratio, int *out_pad_x, int *out_pad_y,
                                   char *err, size_t err_len)
{
    if (src_fd < 0 || src_width <= 0 || src_height <= 0 ||
        src_hstride < src_width || src_vstride < src_height ||
        native->input_channels != 3 || slot < 0 || slot >= 2) {
        set_err(err, err_len, "invalid NV12 zero-copy input", 0);
        return -1;
    }
    if (bind_zero_copy_input(native, err, err_len) != 0) {
        return -1;
    }

    int target_w = native->input_width;
    int target_h = native->input_height;
    float ratio = (float)target_w / (float)src_width;
    float height_ratio = (float)target_h / (float)src_height;
    if (height_ratio < ratio) {
        ratio = height_ratio;
    }
    int resized_w = (int)(src_width * ratio + 0.5f);
    int resized_h = (int)(src_height * ratio + 0.5f);
    int pad_x = (target_w - resized_w) / 2;
    int pad_y = (target_h - resized_h) / 2;

    rknn_tensor_mem *input_mem = native->input_mems[slot];
    if (!input_mem || !input_mem->virt_addr || input_mem->fd < 0) {
        set_err(err, err_len, "RKNN input slot is unavailable", 0);
        return -1;
    }
    int dst_wstride = native->input_attrs[0].w_stride
                          ? (int)native->input_attrs[0].w_stride
                          : target_w;
    int dst_hstride = native->input_attrs[0].h_stride
                          ? (int)native->input_attrs[0].h_stride
                          : target_h;
    if (pad_x != 0 || pad_y != 0 ||
        resized_w != target_w || resized_h != target_h ||
        dst_wstride != target_w || dst_hstride != target_h) {
        memset(input_mem->virt_addr, 114, input_mem->size);
    }
    rga_buffer_t source =
        wrapbuffer_fd_t(src_fd, src_width, src_height,
                        src_hstride, src_vstride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t destination =
        wrapbuffer_fd_t(input_mem->fd, target_w, target_h,
                        dst_wstride, dst_hstride, RK_FORMAT_RGB_888);
    rga_buffer_t pattern;
    memset(&pattern, 0, sizeof(pattern));
    im_rect source_rect = {0, 0, src_width, src_height};
    im_rect destination_rect = {pad_x, pad_y, resized_w, resized_h};
    im_rect pattern_rect;
    memset(&pattern_rect, 0, sizeof(pattern_rect));
    int checked = imcheck(source, destination, source_rect, destination_rect);
    if (checked != IM_STATUS_NOERROR) {
        set_err(err, err_len, "RGA zero-copy input imcheck failed", checked);
        return -1;
    }
    IM_STATUS status = improcess(source, destination, pattern,
                                 source_rect, destination_rect,
                                 pattern_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        set_err(err, err_len, "RGA zero-copy input failed", status);
        return -1;
    }
    if (out_ratio) *out_ratio = ratio;
    if (out_pad_x) *out_pad_x = pad_x;
    if (out_pad_y) *out_pad_y = pad_y;
    return 0;
}

/* 函数职责：复制普通输入并执行一次RKNN推理。 */
int rknn_native_run(void *handle, const void *input_data, size_t input_size,
                    int input_kind, int input_format,
                    void **output_buffers, const size_t *output_sizes,
                    char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || !input_data || input_size == 0 ||
        !output_buffers || !output_sizes) {
        set_err(err, err_len, "invalid native inference request", 0);
        return -1;
    }
    if (native->input_mem_bound) {
        set_err(err, err_len,
                "generic inference cannot follow zero-copy input binding", 0);
        return -1;
    }
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = (void *)input_data;
    input.size = (uint32_t)input_size;
    input.type = input_kind == 0 ? RKNN_TENSOR_UINT8 : RKNN_TENSOR_FLOAT32;
    input.fmt = input_format == 0 ? RKNN_TENSOR_NHWC : RKNN_TENSOR_NCHW;
    input.pass_through = 0;
    int result = rknn_inputs_set(native->context, 1, &input);
    if (result < 0) {
        set_err(err, err_len, "rknn_inputs_set failed", result);
        return -1;
    }
    result = rknn_run(native->context, NULL);
    if (result < 0) {
        set_err(err, err_len, "rknn_run failed", result);
        return -1;
    }
    return copy_float_outputs(native, output_buffers, output_sizes, err, err_len);
}

/* 把NV12帧同步写入指定输入槽；不同槽可与另一槽的NPU运行并行。 */
int rknn_native_prepare_nv12_slot(void *handle, int slot,
                                  int src_fd, int src_width, int src_height,
                                  int src_hstride, int src_vstride,
                                  float *out_ratio,
                                  int *out_pad_x, int *out_pad_y,
                                  float *out_preprocess_ms,
                                  char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || slot < 0 || slot >= 2) {
        set_err(err, err_len, "invalid native NV12 prepare request", 0);
        return -1;
    }
    native->input_prepared[slot] = 0;
    double started = monotonic_ms();
    if (letterbox_nv12_to_input(native, slot, src_fd, src_width, src_height,
                                src_hstride, src_vstride,
                                out_ratio, out_pad_x, out_pad_y,
                                err, err_len) != 0) {
        return -1;
    }
    if (out_preprocess_ms) {
        *out_preprocess_ms = (float)(monotonic_ms() - started);
    }
    native->input_prepared[slot] = 1;
    return 0;
}

static int roi_slot_left(int index)
{
    static const int values[RKNN_NATIVE_ROI_MAX] = {160, 0, 0, 160};
    return values[index];
}

static int roi_slot_top(int index)
{
    static const int values[RKNN_NATIVE_ROI_MAX] = {0, 0, 80, 80};
    return values[index];
}

/* RGA把1～4个NV12 ROI letterbox到固定四槽RGB画布。 */
int rknn_native_prepare_roi_pose26_slot(
    void *handle, int slot,
    int src_fd, int src_width, int src_height,
    int src_hstride, int src_vstride,
    const struct rknn_native_roi_rect *rois, int roi_count,
    struct rknn_native_roi_slot_info *slot_infos, int slot_info_capacity,
    float *out_preprocess_ms, char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || slot < 0 || slot >= 2 || src_fd < 0 ||
        src_width <= 0 || src_height <= 0 ||
        src_hstride < src_width || src_vstride < src_height ||
        !rois || roi_count < 1 || roi_count > RKNN_NATIVE_ROI_MAX ||
        !slot_infos || slot_info_capacity < roi_count) {
        set_err(err, err_len, "invalid ROI Pose26 prepare request", 0);
        return -1;
    }
    struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS];
    if (prepare_roi_pose26_heads(native, heads, err, err_len) != 0 ||
        bind_zero_copy_input(native, err, err_len) != 0) {
        return -1;
    }
    native->input_prepared[slot] = 0;
    rknn_tensor_mem *input_mem = native->input_mems[slot];
    int dst_wstride = native->input_attrs[0].w_stride
                          ? (int)native->input_attrs[0].w_stride
                          : native->input_width;
    int dst_hstride = native->input_attrs[0].h_stride
                          ? (int)native->input_attrs[0].h_stride
                          : native->input_height;
    if (!input_mem || !input_mem->virt_addr || input_mem->fd < 0) {
        set_err(err, err_len, "ROI Pose26 input slot unavailable", 0);
        return -1;
    }
    double started = monotonic_ms();
    memset(input_mem->virt_addr, 114, input_mem->size);
    rga_buffer_t source = wrapbuffer_fd_t(
        src_fd, src_width, src_height, src_hstride, src_vstride,
        RK_FORMAT_YCbCr_420_SP
    );
    rga_buffer_t destination = wrapbuffer_fd_t(
        input_mem->fd, native->input_width, native->input_height,
        dst_wstride, dst_hstride, RK_FORMAT_RGB_888
    );
    rga_buffer_t pattern;
    memset(&pattern, 0, sizeof(pattern));
    im_rect pattern_rect;
    memset(&pattern_rect, 0, sizeof(pattern_rect));
    for (int index = 0; index < roi_count; ++index) {
        struct rknn_native_roi_rect requested = rois[index];
        if (requested.x1 < 0 || requested.y1 < 0 ||
            requested.x1 >= requested.x2 || requested.y1 >= requested.y2 ||
            requested.x2 > src_width || requested.y2 > src_height) {
            set_err(err, err_len, "ROI Pose26 geometry is out of bounds", 0);
            return -1;
        }
        struct rknn_native_roi_rect aligned = {
            requested.x1 & ~1, requested.y1 & ~1,
            (requested.x2 + 1) & ~1, (requested.y2 + 1) & ~1
        };
        if (aligned.x2 > src_width) aligned.x2 = src_width & ~1;
        if (aligned.y2 > src_height) aligned.y2 = src_height & ~1;
        int crop_width = aligned.x2 - aligned.x1;
        int crop_height = aligned.y2 - aligned.y1;
        if (crop_width <= 0 || crop_height <= 0) {
            set_err(err, err_len, "ROI Pose26 aligned crop is empty", 0);
            return -1;
        }
        float ratio = (float)RKNN_NATIVE_ROI_TILE_WIDTH / crop_width;
        float height_ratio = (float)RKNN_NATIVE_ROI_TILE_HEIGHT / crop_height;
        if (height_ratio < ratio) ratio = height_ratio;
        int resized_width = (int)floorf(crop_width * ratio + 0.5f);
        int resized_height = (int)floorf(crop_height * ratio + 0.5f);
        if (resized_width < 1) resized_width = 1;
        if (resized_height < 1) resized_height = 1;
        if (resized_width > RKNN_NATIVE_ROI_TILE_WIDTH)
            resized_width = RKNN_NATIVE_ROI_TILE_WIDTH;
        if (resized_height > RKNN_NATIVE_ROI_TILE_HEIGHT)
            resized_height = RKNN_NATIVE_ROI_TILE_HEIGHT;
        int pad_x = (RKNN_NATIVE_ROI_TILE_WIDTH - resized_width) / 2;
        int pad_y = (RKNN_NATIVE_ROI_TILE_HEIGHT - resized_height) / 2;
        /*
         * RK3568旧RGA驱动会拒绝起点为奇数的RGB888目标子矩形，返回
         * IM_STATUS_INVALID_PARAM(-3)。向左/上最多偏移1像素仍保持在
         * 80x160槽内，且slot_info保存实际pad，坐标反算不受影响。
         */
        pad_x &= ~1;
        pad_y &= ~1;
        int slot_left = roi_slot_left(index);
        int slot_top = roi_slot_top(index);
        im_rect source_rect = {
            aligned.x1, aligned.y1, crop_width, crop_height
        };
        im_rect destination_rect = {
            slot_left + pad_x, slot_top + pad_y,
            resized_width, resized_height
        };
        int checked = imcheck(
            source, destination, source_rect, destination_rect
        );
        if (checked != IM_STATUS_NOERROR) {
            char detail[256];
            snprintf(
                detail, sizeof(detail),
                "RGA ROI Pose26 imcheck failed roi=%d "
                "src=%d,%d,%d,%d dst=%d,%d,%d,%d",
                index,
                source_rect.x, source_rect.y,
                source_rect.width, source_rect.height,
                destination_rect.x, destination_rect.y,
                destination_rect.width, destination_rect.height
            );
            set_err(err, err_len, detail, checked);
            return -1;
        }
        IM_STATUS status = improcess(
            source, destination, pattern, source_rect, destination_rect,
            pattern_rect, IM_SYNC
        );
        if (status != IM_STATUS_SUCCESS) {
            set_err(err, err_len, "RGA ROI Pose26 pack failed", status);
            return -1;
        }
        slot_infos[index].roi_index = index;
        slot_infos[index].slot_left = slot_left;
        slot_infos[index].slot_top = slot_top;
        slot_infos[index].ratio = ratio;
        slot_infos[index].pad_x = (float)pad_x;
        slot_infos[index].pad_y = (float)pad_y;
        slot_infos[index].requested = requested;
        slot_infos[index].aligned = aligned;
    }
    native->input_prepared[slot] = 1;
    if (out_preprocess_ms) {
        *out_preprocess_ms = (float)(monotonic_ms() - started);
    }
    set_err(err, err_len, "", 0);
    return 0;
}

/* 旧单槽兼容接口。 */
int rknn_native_prepare_nv12(void *handle,
                             int src_fd, int src_width, int src_height,
                             int src_hstride, int src_vstride,
                             float *out_ratio, int *out_pad_x, int *out_pad_y,
                             char *err, size_t err_len)
{
    return rknn_native_prepare_nv12_slot(
        handle, 0, src_fd, src_width, src_height,
        src_hstride, src_vstride,
        out_ratio, out_pad_x, out_pad_y, NULL,
        err, err_len
    );
}

/* 函数职责：对已准备的持久输入内存执行推理并复制输出。 */
int rknn_native_run_prepared(void *handle,
                             void **output_buffers,
                             const size_t *output_sizes,
                             char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    if (!native || !output_buffers || !output_sizes ||
        !native->input_mem_bound || !native->input_prepared[0]) {
        set_err(err, err_len, "native RKNN input is not prepared", 0);
        return -1;
    }
    /*
     * prepare返回时RGA已同步把缩放图写入input_mem，源NV12 DMA-BUF不再
     * 被引用。调用rknn_run前先清除准备状态，避免失败后意外复用旧输入。
     */
    native->input_prepared[0] = 0;
    if (bind_input_slot(native, 0, err, err_len) != 0) {
        return -1;
    }
    int result = rknn_run(native->context, NULL);
    if (result < 0) {
        set_err(err, err_len, "rknn_run prepared input failed", result);
        return -1;
    }
    return copy_float_outputs(native, output_buffers, output_sizes, err, err_len);
}

/* 对已写入指定槽的Pose26输入运行NPU和Native解码。 */
int rknn_native_detect_pose26_prepared(
    void *handle, int slot,
    int src_width, int src_height,
    float ratio, int pad_x, int pad_y, float preprocess_ms,
    float confidence_threshold,
    struct rknn_native_plate_detection *detections,
    int detection_capacity, int *detection_count,
    struct rknn_native_detection_timing *timing,
    char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS];
    struct plate_candidate decoded[RKNN_NATIVE_POSE26_MAX_TOPK];
    double total_start = monotonic_ms();
    if (!native || slot < 0 || slot >= 2 ||
        !native->input_prepared[slot] ||
        !detections || detection_capacity <= 0 ||
        detection_capacity > RKNN_NATIVE_POSE26_MAX_TOPK ||
        !detection_count || !timing) {
        set_err(err, err_len, "invalid native Pose26 detection request", 0);
        return -1;
    }
    *detection_count = 0;
    memset(timing, 0, sizeof(*timing));
    timing->preprocess_ms = preprocess_ms < 0.0f ? 0.0f : preprocess_ms;
    confidence_threshold = clamp_float(confidence_threshold, 0.0f, 1.0f);
    if (prepare_pose26_heads(native, heads, err, err_len) != 0 ||
        bind_pose26_output_memory(native, err, err_len) != 0) {
        return -1;
    }
    if (bind_input_slot(native, slot, err, err_len) != 0) {
        return -1;
    }
    native->input_prepared[slot] = 0;

    double run_start = monotonic_ms();
    int result = rknn_run(native->context, NULL);
    double run_end = monotonic_ms();
    if (result < 0) {
        set_err(err, err_len, "rknn_run Pose26 detector failed", result);
        return -1;
    }
    double run_wall_ms = run_end - run_start;
    rknn_perf_run performance;
    memset(&performance, 0, sizeof(performance));
    result = rknn_query(native->context, RKNN_QUERY_PERF_RUN,
                        &performance, sizeof(performance));
    if (result >= 0 && performance.run_duration > 0) {
        timing->npu_ms = (float)performance.run_duration / 1000.0f;
        timing->output_ms = (float)(run_wall_ms - (double)timing->npu_ms);
        if (timing->output_ms < 0.0f) timing->output_ms = 0.0f;
    } else {
        timing->npu_ms = (float)run_wall_ms;
        timing->output_ms = 0.0f;
    }

    double decode_start = monotonic_ms();
    int selected_count = decode_pose26_candidates(
        native, heads, confidence_threshold, decoded, detection_capacity,
        &timing->anchors_scanned, &timing->candidates_decoded
    );
    for (int index = 0; index < selected_count; ++index) {
        map_plate_detection_to_source(
            &decoded[index], ratio, pad_x, pad_y, src_width, src_height,
            &detections[index]
        );
    }
    *detection_count = selected_count;
    timing->decode_ms = (float)(monotonic_ms() - decode_start);
    timing->nms_ms = 0.0f;
    timing->total_ms = timing->preprocess_ms +
                       (float)(monotonic_ms() - total_start);
    set_err(err, err_len, "", 0);
    return 0;
}

static void map_roi_pose_detection(
    const struct plate_candidate *candidate,
    const struct rknn_native_roi_slot_info *slot_info,
    struct rknn_native_roi_plate_detection *detection)
{
    const struct rknn_native_roi_rect *requested = &slot_info->requested;
    const struct rknn_native_roi_rect *aligned = &slot_info->aligned;
    float crop_width = (float)(aligned->x2 - aligned->x1);
    float crop_height = (float)(aligned->y2 - aligned->y1);
    float resized_width = floorf(crop_width * slot_info->ratio + 0.5f);
    float resized_height = floorf(crop_height * slot_info->ratio + 0.5f);
    resized_width = clamp_float(
        resized_width, 1.0f, (float)RKNN_NATIVE_ROI_TILE_WIDTH
    );
    resized_height = clamp_float(
        resized_height, 1.0f, (float)RKNN_NATIVE_ROI_TILE_HEIGHT
    );
    /* RGA uses the rounded integer destination size, so X/Y scales differ. */
    float scale_x = resized_width / crop_width;
    float scale_y = resized_height / crop_height;
    for (int point = 0; point < 2; ++point) {
        float canvas_x = candidate->bbox[point * 2];
        float canvas_y = candidate->bbox[point * 2 + 1];
        detection->bbox[point * 2] = clamp_float(
            (canvas_x - slot_info->slot_left - slot_info->pad_x) /
                scale_x + aligned->x1,
            (float)requested->x1, (float)requested->x2
        );
        detection->bbox[point * 2 + 1] = clamp_float(
            (canvas_y - slot_info->slot_top - slot_info->pad_y) /
                scale_y + aligned->y1,
            (float)requested->y1, (float)requested->y2
        );
    }
    for (int point = 0; point < 4; ++point) {
        float canvas_x = candidate->landmarks[point * 2];
        float canvas_y = candidate->landmarks[point * 2 + 1];
        detection->landmarks[point * 2] = clamp_float(
            (canvas_x - slot_info->slot_left - slot_info->pad_x) /
                scale_x + aligned->x1,
            (float)requested->x1, (float)requested->x2
        );
        detection->landmarks[point * 2 + 1] = clamp_float(
            (canvas_y - slot_info->slot_top - slot_info->pad_y) /
                scale_y + aligned->y1,
            (float)requested->y1, (float)requested->y2
        );
        detection->landmark_visibility[point] =
            candidate->landmark_visibility[point];
    }
    detection->confidence = candidate->confidence;
    detection->class_id = candidate->class_id;
    detection->roi_index = slot_info->roi_index;
}

/* 对已组好的ROI四槽画布运行FP16 Pose26模型，每个有效槽仅保留最高分。 */
int rknn_native_detect_roi_pose26_prepared(
    void *handle, int slot,
    const struct rknn_native_roi_slot_info *slot_infos, int roi_count,
    float preprocess_ms, float confidence_threshold,
    struct rknn_native_roi_plate_detection *detections,
    int detection_capacity, int *detection_count,
    struct rknn_native_detection_timing *timing,
    char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    struct pose26_head heads[RKNN_NATIVE_POSE26_HEADS];
    struct plate_candidate decoded[RKNN_NATIVE_POSE26_MAX_TOPK];
    struct rknn_native_roi_plate_detection per_roi[RKNN_NATIVE_ROI_MAX];
    int occupied[RKNN_NATIVE_ROI_MAX] = {0, 0, 0, 0};
    double total_started = monotonic_ms();
    if (!native || slot < 0 || slot >= 2 ||
        !native->input_prepared[slot] || !slot_infos ||
        roi_count < 1 || roi_count > RKNN_NATIVE_ROI_MAX ||
        !detections || detection_capacity < 1 ||
        !detection_count || !timing) {
        set_err(err, err_len, "invalid prepared ROI Pose26 request", 0);
        return -1;
    }
    *detection_count = 0;
    memset(timing, 0, sizeof(*timing));
    timing->preprocess_ms = preprocess_ms < 0.0f ? 0.0f : preprocess_ms;
    confidence_threshold = clamp_float(confidence_threshold, 0.0f, 1.0f);
    if (prepare_roi_pose26_heads(native, heads, err, err_len) != 0 ||
        ensure_roi_pose_output_buffers(native, err, err_len) != 0 ||
        bind_input_slot(native, slot, err, err_len) != 0) {
        return -1;
    }
    native->input_prepared[slot] = 0;
    double run_started = monotonic_ms();
    int result = rknn_run(native->context, NULL);
    double run_finished = monotonic_ms();
    if (result < 0) {
        set_err(err, err_len, "rknn_run ROI Pose26 failed", result);
        return -1;
    }
    rknn_perf_run performance;
    memset(&performance, 0, sizeof(performance));
    result = rknn_query(native->context, RKNN_QUERY_PERF_RUN,
                        &performance, sizeof(performance));
    timing->npu_ms = result >= 0 && performance.run_duration > 0
                         ? (float)performance.run_duration / 1000.0f
                         : (float)(run_finished - run_started);

    void *output_buffers[RKNN_NATIVE_POSE26_HEADS];
    size_t output_sizes[RKNN_NATIVE_POSE26_HEADS];
    for (uint32_t i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        output_buffers[i] = native->roi_pose_outputs[i];
        output_sizes[i] =
            (size_t)native->output_attrs[i].n_elems * sizeof(float);
    }
    double output_started = monotonic_ms();
    if (copy_float_outputs(native, output_buffers, output_sizes,
                           err, err_len) != 0) {
        return -1;
    }
    timing->output_ms = (float)(monotonic_ms() - output_started);

    double decode_started = monotonic_ms();
    int selected = decode_roi_pose26_candidates(
        native, heads, confidence_threshold, decoded,
        &timing->anchors_scanned, &timing->candidates_decoded
    );
    for (int index = 0; index < selected; ++index) {
        float center_x = (decoded[index].bbox[0] + decoded[index].bbox[2]) * 0.5f;
        float center_y = (decoded[index].bbox[1] + decoded[index].bbox[3]) * 0.5f;
        for (int roi_index = 0; roi_index < roi_count; ++roi_index) {
            const struct rknn_native_roi_slot_info *info = &slot_infos[roi_index];
            if (!occupied[roi_index] &&
                center_x >= info->slot_left &&
                center_x < info->slot_left + RKNN_NATIVE_ROI_TILE_WIDTH &&
                center_y >= info->slot_top &&
                center_y < info->slot_top + RKNN_NATIVE_ROI_TILE_HEIGHT) {
                map_roi_pose_detection(&decoded[index], info,
                                       &per_roi[roi_index]);
                occupied[roi_index] = 1;
                break;
            }
        }
    }
    int output_count = 0;
    for (int roi_index = 0;
         roi_index < roi_count && output_count < detection_capacity;
         ++roi_index) {
        if (occupied[roi_index]) {
            detections[output_count++] = per_roi[roi_index];
        }
    }
    *detection_count = output_count;
    timing->decode_ms = (float)(monotonic_ms() - decode_started);
    timing->nms_ms = 0.0f;
    timing->total_ms = timing->preprocess_ms +
                       (float)(monotonic_ms() - total_started);
    set_err(err, err_len, "", 0);
    return 0;
}

/* 函数职责：返回驱动统计的最近一次rknn_run纯NPU执行时间（微秒）。 */
int rknn_native_get_last_run_us(void *handle, int64_t *duration_us,
                                char *err, size_t err_len)
{
    struct rknn_native *native = handle;
    rknn_perf_run performance;
    if (!native || !duration_us) {
        set_err(err, err_len, "invalid RKNN performance query", 0);
        return -1;
    }
    memset(&performance, 0, sizeof(performance));
    int result = rknn_query(native->context, RKNN_QUERY_PERF_RUN,
                            &performance, sizeof(performance));
    if (result < 0) {
        set_err(err, err_len, "RKNN_QUERY_PERF_RUN failed", result);
        return -1;
    }
    *duration_us = performance.run_duration;
    return 0;
}

/* 函数职责：按输出、输入内存、RKNN上下文和模型缓冲顺序释放资源。 */
void rknn_native_close(void *handle)
{
    struct rknn_native *native = handle;
    if (!native) {
        return;
    }
    if (native->context) {
        for (uint32_t i = 0; i < RKNN_NATIVE_MAX_IO; ++i) {
            if (native->output_mems[i]) {
                rknn_destroy_mem(native->context, native->output_mems[i]);
                native->output_mems[i] = NULL;
            }
        }
    }
    if (native->context) {
        for (int slot = 0; slot < 2; ++slot) {
            if (native->input_mems[slot]) {
                rknn_destroy_mem(native->context, native->input_mems[slot]);
                native->input_mems[slot] = NULL;
            }
        }
    }
    if (native->context) {
        rknn_destroy(native->context);
    }
    free(native->plate_candidates);
    free(native->ocr_resized_u8);
    free(native->ocr_input_f32);
    free(native->ocr_text_output);
    free(native->ocr_color_output);
    for (uint32_t i = 0; i < RKNN_NATIVE_POSE26_HEADS; ++i) {
        free(native->roi_pose_outputs[i]);
    }
    free(native->model_data);
    free(native);
}
