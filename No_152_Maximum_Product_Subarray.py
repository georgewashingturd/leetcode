###############################################################################
# 152. Maximum Product Subarray
###############################################################################

class Solution(object):
    def processSubArray(self, l, mc, tot):
    
        if (len(l) == 1):
            return l[0]
    
        if (mc % 2 == 0):
            return tot
            
        # so we have two choices here say we have [10, -1, 2, 3, -5, 6, 7, -8, 9] it's either
        # [10, -1, 2, 3, -5, 6, 7] or [ 2, 3, -5, 6, 7, -8, 9] because for product we need as many
        # elements as possible
        
        # first up is get rid of the right most -ve number
        nr = tot
        for i in range(1,len(l)+1):
            nr //= l[-i]
            if (l[-i] < 0):
                break
        
        nl = tot    
        for i in range(len(l)):
            nl //= l[i]
            if (l[i] < 0):
                break
        
        return max(nr, nl)
    
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if (len(nums) == 1):
            return nums[0]
        
        # let me try Kadane and see and Kadane doesn't work :)
        # but actually we just want to take care of -1 and 0
        # we don't want to involve 0 so first we partition the list into regions with no zero
        # like when nums = [1,2,3,0,5,6,7] we partition it into [1,2,3] and [5,6,7] and
        # then check the individual max product
        # for each partition is the number of -ve numbers are even we just multiply everything
        # if the number of -ve numbers are odd we need to only include an even number of them
        # so we scan which end we can discard to get max product for example [-17, 1,-2, 5,-9, 8] and we discard
        # the -ve number that will cause problems 
        
        # first partition into non zero regions
        r = []
        st = 0
        res = -float('inf')
        
        i = 0
        nz = 0
        while(i < len(nums)):
            # first find a non-zero element
            while (i < len(nums) and nums[i] == 0):
                i += 1
            
            st = i
            
            # now find the next zero
            while (i < len(nums) and nums[i] != 0):
                i += 1
            
            if (i > st):
                r.append(nums[st:i])
                # we also keep track of the number of non-zero elements
                # we will need this to reset the maximum product for example when we have [0,-2,0]
                # we need to increase res to at least 0
                nz += i - st
        
        # this means that there is at least one zero so adjust the max product to at least zero
        if (nz < len(nums)):
            res = 0
        
        # this means that every number is zero
        if (len(r) == 0):
            return 0
            
        
        for l in r:
            mc = 0
            tot = 1
            # while we count the number of -ve numbers we also get the total product to save time
            # so that we can backtrack it if we need to get rid some of the negative numbers
            for i in range(len(l)):
                tot = tot * l[i]
                if (l[i] < 0):
                    mc += 1

            t = self.processSubArray(l, mc, tot)
            if (t > res):
                res = t
        
        return res