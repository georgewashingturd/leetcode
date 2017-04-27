###############################################################################
# 55. Jump Game
###############################################################################

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        if (not nums):
            return True
            
        if (len(nums) == 1):
            return True
            
        if (nums[0] <= 0):
            return False
        
        # unlike word break we don't need to create a True False array and then process every element of it
        # here we are just interested in the max reach so just update it
        
        # max is inclusive of the end point
        max_reach = nums[0]
        
        ln = len(nums)
        
        for i in range(1,ln):
            
            if (max_reach >= ln - 1):
                return True
            
            if (i > max_reach):
                return False

            n = i + nums[i]    
            if (n > max_reach):
                max_reach = n
            
        return True
        