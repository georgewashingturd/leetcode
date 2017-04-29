###############################################################################
# 442. Find All Duplicates in an Array
###############################################################################

# this is very similar to No 287. Find the Duplicate Number
# except that we need to report all the numbers not just one

from collections import deque

class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        ln = len(nums)
        
        if (ln <= 1):
            return []
        
        i = 0
        r = deque([])
        
        while(i < ln):
            # we set duplicate items to -1 to avoid double counting
            # we can get double counting from input [4,3,2,7,8,2,3,1] swapping will generate the following
            # [7, 3, 2, 4, 8, 2, 3, 1]
            # [3, 3, 2, 4, 8, 2, 7, 1]
            # [2, 3, 3, 4, 8, 2, 7, 1]
            # [3, 2, 3, 4, 8, 2, 7, 1]
            # [3, 2, 3, 4, 1, 2, 7, 8]
            # here 3 is processed a second time causing double count
            # [1, 2, 3, 4, 3, 2, 7, 8]
            
            if (nums[i] == i+1 or nums[i] == -1):
                i += 1
                continue
            
            n = nums[i]
            
            # we found a duplicate and this becomes a bit tricky
            # while swapping them to the correct location the same duplciate might
            # be swapped twice and we will double count so we need to set it to -1
            
            if (nums[n-1] == n):
                r.append(n)
                nums[i] = -1
                i += 1
                continue
                
            nums[i], nums[n-1] = nums[n-1], nums[i]
            
            #print(nums)
            
        return list(r)