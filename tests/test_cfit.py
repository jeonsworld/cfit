import unittest
import cfit


class TestCarbonFit(unittest.TestCase):
    def setUp(self):
        self.print_test_name()

    def print_test_name(self):
        print(f"\nRunning: {self._testMethodName}")

    def assert_and_print(self, result):
        self.assertIsNotNone(result)
        print(result)

    def test_from_hf_with_safetensors(self):
        result = cfit.from_hf("HuggingFaceH4/zephyr-7b-beta")
        self.assert_and_print(result)

    def test_from_hf_without_safetensors(self):
        result = cfit.from_hf("google/electra-base-discriminator")
        self.assert_and_print(result)

    def test_from_hf_with_quantized_models(self):
        models = [
            "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
            "hugging-quants/Meta-Llama-3.1-70B-BNB-NF4-BF16"
        ]
        for model in models:
            with self.subTest(model=model):
                result = cfit.from_hf(model)
                self.assert_and_print(result)

    def test_from_hf_nonexistent_model(self):
        with self.assertRaises(Exception):
            cfit.from_hf("FaceAdapter/FaceAdapter")

    def test_from_params(self):
        test_cases = [
            ("405B", str),
            (405_000_000_000, str)
        ]
        for input_value, expected_type in test_cases:
            with self.subTest(input=input_value):
                result = cfit.from_params(input_value)
                self.assert_and_print(result)
                self.assertIsInstance(result, expected_type)
                self.assertIn("Required GPU Memory", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
