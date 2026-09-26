# approach_name: Patience Sorting with Binary Search
# time_complexity: O(n log n)
# space_complexity: O(n)

from bisect import bisect_left

def length_of_lis(nums: list[int]) -> int:
    tails = []
    for x in nums:
        idx = bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    return len(tails)

