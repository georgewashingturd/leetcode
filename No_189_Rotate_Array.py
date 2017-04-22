###############################################################################
# 189. Rotate Array
###############################################################################

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        if (k == 0):
            return
        
        # k can be bigger than the length of nums something I didn't expect
        
        ln_nums = len(nums)
        ln_r = k + ln_nums
        r = []
        while(ln_r > 0):
            r += nums
            ln_r -= ln_nums
        
        # first let's try a stupid way using python
        
        
        for i in range(ln_nums):
            # rotate left as I misunderstood it at first
            #nums[i] = r[i+k]
            # rotate right as requested by the problem
            nums[i] = r[i+ln_nums-k]
            