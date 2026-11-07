"""Production Pose26 model contract tests."""

import unittest

from plate_det.model_contracts import validate_pose26_model


class _FakeNativeModel:
    def __init__(self, shape=(1, 288, 512, 3), tensor_format=1):
        self.input_info = {"shape": tuple(shape), "format": int(tensor_format)}
        self.output_info = [
            {
                "shape": (1, 18, 288 // stride, 512 // stride),
                "type": 2,
                "quantization": 2,
                "scale": 0.25,
            }
            for stride in (4, 8, 16, 32)
        ]


class Pose26ModelContractTests(unittest.TestCase):
    def test_strict_pose26_tensor_contract(self):
        self.assertEqual(validate_pose26_model(_FakeNativeModel()), (288, 512))

    def test_rejects_wrong_input_layout_or_shape(self):
        with self.assertRaisesRegex(ValueError, "NHWC"):
            validate_pose26_model(_FakeNativeModel(tensor_format=0))
        with self.assertRaisesRegex(ValueError, "288,512"):
            validate_pose26_model(_FakeNativeModel(shape=(1, 512, 288, 3)))

    def test_rejects_missing_or_invalid_outputs(self):
        model = _FakeNativeModel()
        model.output_info.pop()
        with self.assertRaisesRegex(ValueError, "outputs"):
            validate_pose26_model(model)
        model = _FakeNativeModel()
        model.output_info[0]["scale"] = 0.0
        with self.assertRaisesRegex(ValueError, "scale"):
            validate_pose26_model(model)


if __name__ == "__main__":
    unittest.main()
