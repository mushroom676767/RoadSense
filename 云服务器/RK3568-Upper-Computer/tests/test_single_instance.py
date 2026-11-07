import os
import unittest

from backend.single_instance import _instance_key


class SingleInstanceTests(unittest.TestCase):
    def test_instance_key_is_stable_for_equivalent_paths(self):
        base = os.path.abspath(os.path.join("somewhere", "..", "manager"))
        equivalent = os.path.join(base, ".")
        self.assertEqual(_instance_key(base), _instance_key(equivalent))

    def test_different_directories_use_different_guards(self):
        self.assertNotEqual(
            _instance_key(os.path.abspath("manager-a")),
            _instance_key(os.path.abspath("manager-b")),
        )


if __name__ == "__main__":
    unittest.main()
