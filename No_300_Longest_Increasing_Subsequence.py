###############################################################################
# 300. Longest Increasing Subsequence
###############################################################################

class Solution(object):

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # similar to No 334 increasng triplet the difference here is that the array a
        # is not fixed but we let it grow and the answer is the length of a once
        # we are done with all the nums
        
        ln = len(nums)
        
        if (ln <= 1):
            return ln
        
        a = []
        
        for i in nums:
            found = False
            for j in xrange(len(a)):
                if (i <= a[j]):
                    a[j] = i
                    found = True
                    break
                
            if (not found):
                a.append(i)
            
        return len(a)
        