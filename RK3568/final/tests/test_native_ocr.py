"""OpenCV BGR resize与融合Native OCR的契约和解码回归测试。"""

import ctypes
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from core.rknn_native import NativeOcrResult, NativeRknnModel
from plate_det import plate_recognizer


class _NativeMappingModel:
    def recognize_plate_bgr(self, roi):
        self.roi = roi
        return [1, 2, 999], 3, 0.91, 0.83


class _ToolkitModel:
    def inference(self, inputs, data_format):
        self.input = inputs[0]
        self.data_format = data_format
        text = np.zeros((1, 21, 78), dtype=np.float32)
        text[0, :, 0] = 8.0
        text[0, 0, 1] = 12.0
        color = np.asarray([[0.0, 0.0, 9.0, 0.0, 0.0]], dtype=np.float32)
        return [text, color]


class _FailingOcrLibrary:
    def rknn_native_recognize_bgr(self, *_args):
        _args[-2].value = b"forced native OCR failure"
        return -1


class NativeOcrContractTests(unittest.TestCase):
    def test_ctypes_ocr_abi_is_stable(self):
        self.assertEqual(
            [name for name, _ctype in NativeOcrResult._fields_],
            [
                "char_indices", "char_count", "color_index",
                "text_confidence", "color_confidence", "resize_ms",
                "npu_ms", "decode_ms", "total_ms",
            ],
        )
        self.assertEqual(ctypes.sizeof(NativeOcrResult), 116)
        self.assertEqual(NativeOcrResult.char_count.offset, 84)
        self.assertEqual(NativeOcrResult.color_index.offset, 88)

    def test_native_indices_are_mapped_in_python(self):
        model = _NativeMappingModel()
        roi = np.zeros((20, 60, 3), dtype=np.uint8)
        result = plate_recognizer.recognize_rknn(model, roi)
        expected = (
            plate_recognizer.plateName[1] + plate_recognizer.plateName[2]
        )
        self.assertEqual(result, (expected, "白色", 0.91, 0.83))
        self.assertIs(model.roi, roi)

    def test_python_toolkit_keeps_opencv_bgr_resize(self):
        resized = np.zeros((48, 168, 3), dtype=np.uint8)
        resized[..., 0] = 17
        resized[..., 1] = 29
        resized[..., 2] = 43
        model = _ToolkitModel()
        with mock.patch.object(
            plate_recognizer.cv2,
            "resize",
            return_value=resized,
        ) as resize:
            plate_recognizer.recognize_rknn(
                model, np.zeros((16, 40, 3), dtype=np.uint8)
            )
        resize.assert_called_once()
        self.assertEqual(model.data_format, "nchw")
        self.assertEqual(model.input.shape, (1, 3, 48, 168))
        self.assertEqual(tuple(model.input[0, :, 0, 0]), (17.0, 29.0, 43.0))

    def test_native_error_is_explicit_for_caller_recovery(self):
        model = NativeRknnModel.__new__(NativeRknnModel)
        model._handle = 1
        model._lib = _FailingOcrLibrary()
        model.last_ocr_timing = None
        with self.assertRaisesRegex(RuntimeError, "forced native OCR failure"):
            model.recognize_plate_bgr(
                np.zeros((24, 80, 3), dtype=np.uint8)
            )

class NativeOcrDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[1]
            / "core" / "native" / "librknn_runtime.so"
        )
        if not path.exists():
            raise unittest.SkipTest("requires compiled native RKNN helper")
        try:
            cls.lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.lib.rknn_native_decode_ocr_logits.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(NativeOcrResult), ctypes.c_char_p, ctypes.c_size_t,
        ]
        cls.lib.rknn_native_decode_ocr_logits.restype = ctypes.c_int

    def test_ctc_blank_repeat_and_internal_color_decode(self):
        text = np.full((21, 78), -8.0, dtype=np.float32)
        sequence = [0, 1, 1, 0, 1, 2, 2] + [0] * 14
        for step, index in enumerate(sequence):
            text[step, index] = 8.0
        color = np.asarray([-4.0, -3.0, -2.0, 9.0, -1.0], dtype=np.float32)
        result = NativeOcrResult()
        err = ctypes.create_string_buffer(256)
        rc = self.lib.rknn_native_decode_ocr_logits(
            text.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), text.size,
            color.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), color.size,
            ctypes.byref(result), err, len(err),
        )
        self.assertEqual(rc, 0, err.value.decode("utf-8", "replace"))
        self.assertEqual(
            list(result.char_indices[:result.char_count]), [1, 1, 2]
        )
        self.assertEqual(result.color_index, 3)
        self.assertGreater(result.text_confidence, 0.99)
        self.assertGreater(result.color_confidence, 0.99)


if __name__ == "__main__":
    unittest.main()
