###############################################################################
# 41. First Missing Positive
###############################################################################

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # see assumption below
        if (not nums):
            return 1
        
        # so I assume the positive integers always start from 1 what I mean is that we will
        # never get a list like this [10000, 10002, 10003] and the first missing positive
        # integer is 10001
        
        # if this is the case then my solution is
        
        d = {}
        
        # we just need to ignore all non-positive numbers
        # and while we're at it we count how many +ve numbers we have
        c = 0
        for i in nums:
            if (i > 0):
                d[i] = None
                c += 1
                
        # now scan starting from 1 to c (inclusive)
        for i in range(1,c+1):
            if (i not in d):
                return i
                
        # if we reach this point it means that all positive numbers from 1 to c (inclusive)
        # are in d and therefore are in the list so the first missing +ve number is c+1
        
        return c + 1
        