"""Deterministic LIS contract; runnable directly against any solution file."""

import itertools
from pathlib import Path
import runpy
import sys
import unittest

CASES = [
    ([], 0), ([1], 1), ([1, 2, 3, 4], 4), ([4, 3, 2, 1], 1),
    ([10, 9, 2, 5, 3, 7, 101, 18], 4), ([2, 2, 2], 1),
    ([1, 2, 2, 3], 3), ([3, 1, 4, 2, 5], 3),
    ([-5, -2, -3, -1], 3), ([0, -1, 2, -3, 4], 3),
]


def expected_length(nums):
    """Independent exhaustive oracle on tiny inputs, not an agent solution."""
    return max((len(sub) for size in range(len(nums) + 1)
                for sub in itertools.combinations(nums, size)
                if all(a < b for a, b in zip(sub, sub[1:]))), default=0)


def check_solution(path):
    function = runpy.run_path(str(path))["length_of_lis"]
    cases = CASES + [(list(seq), expected_length(seq))
                     for size in range(5)
                     for seq in itertools.product((-1, 0, 1), repeat=size)]
    for nums, expected in cases:
        original = nums.copy()
        actual = function(nums)
        if type(actual) is not int or actual != expected or nums != original:
            raise AssertionError(
                f"Input {original}: expected integer {expected}, got {actual!r}; "
                f"input after call: {nums}"
            )
    return len(cases)


class GeneratedSolutionsTests(unittest.TestCase):
    def test_generated_solutions(self):
        # Separate processes keep a broken generated module out of unittest.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import ralph
        paths = sorted((ralph.WORKSPACE / "solutions").glob("*.py"))
        if not paths:
            self.skipTest("No generated solutions yet; run ralph.py first")
        for path in paths:
            with self.subTest(solution=path.name):
                passed, details = ralph.verify(path)
                self.assertTrue(passed, details)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        count = check_solution(Path(sys.argv[1]))
        print(f"PASS: {count} deterministic cases")
    else:
        unittest.main()
