# Longest Increasing Subsequence

Given a list of arbitrary integers, return the length of its longest strictly
increasing subsequence. Elements retain their original order but need not be
contiguous. Equal values do not extend a strictly increasing subsequence.

Example: [10, 9, 2, 5, 3, 7, 101, 18] -> 4.
An empty list returns 0. Handle duplicates and negative values.

Expose `def length_of_lis(nums: list[int]) -> int`.
Use Python's standard library only. Do not mutate the input.

Goal: discover a correct solution, then meaningfully improve its asymptotic
worst-case time complexity for general inputs, considering space tradeoffs.
Do not exploit the small verification inputs or special integer bounds.
