"""CPU-only regression checks: python -m unittest src_memtrec.test_precompute_memory."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src_memtrec import precompute_memory as memory


class PrecomputeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.items = {str(i): {"title": f"Item {i}", "main_cat": "Music"} for i in range(1, 9)}

    def train_windows(self, ids, size=3):
        return memory.build_windows_for_user("u", 0, {"train": ids, "test": [999]}, self.items, size, 30)

    @staticmethod
    def ids(windows):
        return [[x["item_id"] for x in window] for window in windows]

    def test_overlapping_tail_and_observed_history_only(self):
        train = self.train_windows(list(range(1, 9)))
        test = memory.build_behavior_windows(list(range(1, 9)), self.items, 3, 30)
        expected = [["1", "2", "3"], ["4", "5", "6"], ["6", "7", "8"]]
        self.assertEqual(self.ids([x["interaction_sequence"] for x in train]), expected)
        self.assertEqual(self.ids(test), expected)
        self.assertEqual([x["window_index"] for x in train], [0, 1, 2])
        self.assertNotIn("999", train[-1]["prompt"])

    def test_missing_metadata_keeps_mode_specific_tail_policy(self):
        self.assertEqual(self.train_windows([1, 2, 999]), [])
        test = memory.build_behavior_windows([1, 2, 999], self.items, 3, 30)
        self.assertEqual(self.ids(test), [["1", "2"]])

    def test_empty_short_and_duplicate_windows(self):
        self.assertEqual(self.train_windows([]), [])
        self.assertEqual(len(self.train_windows([1, 2])), 1)
        self.assertEqual(len(self.train_windows([1, 1, 1, 1, 1, 1])), 2)
        self.assertEqual(len(memory.build_behavior_windows([1] * 6, self.items, 3, 30)), 1)

    def test_shared_prompt_with_optional_task_identity(self):
        interactions = [{"item_id": "1", "item": "Album", "category": "Music", "action": "purchase"}]
        train = memory.build_behavior_extraction_prompt(interactions)
        test = memory.build_single_behavior_prompt(memory.BehaviorTask('u"_W0', 0, interactions))
        train_shape = json.loads(train.split("Return exactly:\n", 1)[1])
        test_shape = json.loads(test.split("Return exactly:\n", 1)[1])
        self.assertEqual(test_shape.pop("task_id"), 'u"_W0')
        self.assertEqual(test_shape, train_shape)

    def test_shared_structural_guard_accepts_both_record_shapes(self):
        train = [{"item_id": str(i), "item_category": "Music"} for i in (1, 2)]
        test = [{"item_id": str(i), "category": "Music"} for i in (1, 2)]
        result = memory.enforce_structural_consistency("repetition", "same item", "repeat", train)
        self.assertEqual(result, memory.test_structural_consistency("repetition", "same_item", "repeat", test))
        self.assertEqual(result[:3], ("unknown", "mixed", "unknown"))
        self.assertTrue(result[4]["same_known_category"])
        repeated = memory.enforce_structural_consistency("persistence", "mixed", "stable", [train[0]] * 3)
        self.assertEqual(repeated[:3], ("repetition", "same item", "repeat"))

    def test_separate_output_and_resume_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            test = Path(tmp) / "test.jsonl"
            memory.append_memory_records(train, [{"schema_version": memory.TRAIN_SCHEMA_VERSION, "precompute_id": 2}])
            memory.append_jsonl(test, {"schema_version": memory.TEST_SCHEMA_VERSION, "precompute_ok": True, "user_id": "u"})
            self.assertEqual(memory.inspect_existing_output(train), ({2}, 1))
            self.assertEqual(memory.processed_users(test), {"u"})
            with self.assertRaises(RuntimeError):
                memory.inspect_existing_output(test)

    def test_cli_and_legacy_entry_points_without_gpu_dependencies(self):
        folder = Path(memory.__file__).parent
        for script, args in (
            ("precompute_memory.py", []),
            ("precompute_memory.py", ["train"]),
            ("precompute_memory.py", ["test"]),
            ("precompute_memory_create_dual_behavior.py", []),
            ("precompute_test_behaviors_gemma_dual.py", []),
        ):
            with self.subTest(script=script, args=args):
                result = subprocess.run([sys.executable, str(folder / script), *args, "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--help", result.stdout)


if __name__ == "__main__":
    unittest.main()
