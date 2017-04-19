###############################################################################
# 1. Two Sum
###############################################################################

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        if (not nums):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = {}
        
        # add all numbers into the dictionary
        
        for i in nums:
            if (i in d):
                n = d[i] + 1
                d[i] = n
            else:
                d.setdefault(i, 1)
        
        # loop over only items in the dictionary to save time
        for i in d:
            n = target - i
            
            if (n == i):
                if (d[i] > 1):
                    j = nums.index(i)
                    return [nums.index(i), nums.index(n, j+1)]
            else:        
                if (n in d):
                    return [nums.index(i), nums.index(n)]
                
                
        return []