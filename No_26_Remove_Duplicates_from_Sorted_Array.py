###############################################################################
# 26. Remove Duplicates from Sorted Array
###############################################################################

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if (not nums):
            return 0
        
        tot = 0
        count = 0
        st = 0
        prev = nums[0]
        for i in range(len(nums)):
            if (nums[i] == prev):
                count += 1
                
                if (count <= 1):
                    nums[st] = nums[i]
                    st += 1
                
            else:
                tot += min(count, 1)
                count = 1
                prev = nums[i]
                nums[st] = nums[i]
                st += 1
                
        tot += min(count, 1)
        
        return tot
                
        