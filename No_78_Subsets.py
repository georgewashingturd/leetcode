###############################################################################
# 78. Subsets
###############################################################################

# the main idea is to recurse
# get the subset of the set without this current element
# copy those subsets for one copy you do nothing
# on the other copy you append this current element for example

# [1,2] -> curr = [1] subset of [2] is [[], [2]] now make a copy
# [], [2] and another copy to append curr [1,2] ,[1]
# so the subsets are [], [2], [1,2], [1]

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (not nums):
            return [[]]
            
        if (len(nums) == 1):
            return [[nums[0]],[]]
            
        r = self.subsets(nums[1:])
        n = len(r)
        
        for i in range(n):
            t = r[i][:]
            t.append(nums[0])
            r.append(t)
            
        return r
    
    
        