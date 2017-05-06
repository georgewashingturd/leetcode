###############################################################################
# 16. 3Sum Closest
###############################################################################

class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        ln = len(nums)
        if (ln < 3):
            return 0
        
        # try using two pointers high and low but first we need to sort nums
        # we then keep track of the difference
        
        nums.sort()
        
        ms = float('inf')
        tn = 0
        
        i = 0
        while (i <= ln - 3):
            st = i + 1
            ed = ln - 1
            
            
            
            while (st < ed):
                n = nums[i] + nums[st] + nums[ed]
                d = abs(n - target)
                
                if (d == 0):
                    return n
                    
                if (d < ms):
                    ms = d
                    tn = n
                    
                if (n > target):
                    ed -= 1
                else:
                    st += 1
            
            i += 1
            
        return tn
            
        