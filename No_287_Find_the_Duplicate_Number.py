###############################################################################
# 287. Find the Duplicate Number
###############################################################################

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # since each integer is between 1 and n (inclusive) we can try positioning them
        # in the correct index and if that index is already occupied by the correct number
        # then it is the duplicate element
        
        if (not nums):
            return None
            
        i = 0
        ln = len(nums)
        while(i < ln):
            if (nums[i] == i):
                i += 1
                continue
            
            n = nums[i]
            # see if nums[n] already have n as its element
            if (nums[n] == n):
                return n
            # otherwise put n in nums[n] and put nums[n] in nums[i]
            nums[i], nums[n] = nums[n], nums[i]
            
            # we don't need to inrement i here since we still need to process the new element at i
            
            
            