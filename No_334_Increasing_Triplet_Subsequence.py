###############################################################################
# 334. Increasing Triplet Subsequence
###############################################################################

# we can generalize this approach for n elements other than just 3
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        ln = len(nums)
        
        if (ln < 3):
            return False
        
        # create an array of two elements
        a = [float('inf')] * 2
        
        # the cases are
        # 1. 5 2 3 4 then a[0] = 2 a[1] = 3 and we return True
        # 2. 10 11 2 12 then a[0] = 2 a[1] = 11 and we return True
        # 3. 10 2 11 12 then a[0] = 2 a[1] = 11 and we return True
        # 4. 3 2 1 then a[0] = 1 and we return False
        # 5. 10 11 2 then a[0] = 2 and a[1] = 11 and we return False
        
        i = 0
        while(i < ln):
            # need to include = sign otherwise it won't work for duplicate elements
            if (nums[i] <= a[0]):
                a[0] = nums[i]
            elif (nums[i] <= a[1]):
                a[1] = nums[i]
            else:
                return True
                
            i += 1
                
        return False    
        
        