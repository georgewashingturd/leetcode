###############################################################################
# 46. Permutations
###############################################################################

class Solution(object):
    def permuteHelper(self, nums):
        
        ln = len(nums)

        if (ln <= 1):
            return [nums]
            
        if (ln == 2):
            return [[nums[0], nums[1]],[nums[1], nums[0]]]
        
        n = nums[0]    
        r = self.permuteHelper(nums[1:])    
        
        lr = len(r)
        for x in range(lr):
            l = r.pop()
            lnl = len(l) + 1
            for i in range(lnl):
                a = l[:]
                a.insert(i, n)
                r.insert(0,a)
            
        return r
        
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        # the idea is to of course use recursion like this
        # say the permutation of 123 first we take 1 and then permute 23
        # so we have 23 and 32 then we re attach 1 to every possible location
        # 23 -> 123 213 231 and ditto with 32
        # 32 -> 132 312 321
        
        # this is the result list to be returned to caller
        self.r = []
        
        if (not nums):
            return None
            
        return self.permuteHelper(nums)