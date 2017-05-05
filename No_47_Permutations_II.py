###############################################################################
# 47. Permutations II
###############################################################################

class Solution(object):

    # n is how many more we want to choose
    def mchoosen(self, st, n):
    
        if (n == 0):
            self.mmc.append(self.mc[:])
            return
            
        # start with st 
        while (st <= self.m - n):
            self.mc.append(self.d[st])
            self.mchoosen(st+1, n-1)
            self.mc.pop()
            st += 1
            
    def getindices(self, m, n):
        self.d = list(range(m))
        self.m = m
        self.mmc = []
        self.mc = []
        # this is the wrong one it generates all permutation
        #self.mchoosen(d, m, n)
        self.mchoosen(0,n)
        return self.mmc
        
    # this is actually wrong, it generates every single permutation
    #def mchoosen(self, d, m, n):
    
    # the basic idea is for example we have (11)(22) we can do 2 choose 1 from each bracket and then
    # exchange those two, for example we choose the left one from left bracket and left 2 from right bracket
    # and then exchange them and so on
    
    def mergetwo(self, g1, g2):
        
        ln1 = len(g1)
        ln2 = len(g2)
        
        # original one is the first one
        res = [g1 + g2]
        
        n = min(ln1, ln2)
        
        for i in range(1, n+1):
            ind1 = self.getindices(ln1, i)
            ind2 = self.getindices(ln2, i)
            
            for lt1 in ind1:
                for lt2 in ind2:
                    t1 = g1[:]
                    t2 = g2[:]
                    # each entry in ind can be a list so we need to go through each element one at a time
                    for k in range(i):
                        t1[lt1[k]], t2[lt2[k]] = t2[lt2[k]], t1[lt1[k]]
                    res.append(t1 + t2)
                    
        #print res
        
        return res
        
    def prepnumbers(self, nums):
        
        if (not nums):
            return [[]]
            
        ln = len(nums)
        if (ln == 1):
            return [nums]
            
        # gather same numbers
        nums.sort()
        
        p = nums[0]
        
        t = []
        numgroup = []
        
        for i in nums:
            if (i == p):
                t.append(i)
            else:
                numgroup.append(t)
                t = [i]
            p = i
        # don't forget to add the last group
        numgroup.append(t)
        
        if (len(numgroup) <= 1):
            return numgroup
            
        # now merge the first two followed by merging the merged two and the third
        res = self.mergetwo(numgroup[0], numgroup[1])
        
        for i in range(2, len(numgroup)):
            # merge the first two
            tmp = []
            for g in res:
                tmp += self.mergetwo(g, numgroup[i])
            res = tmp
        
        res.sort()
        #print res
        
        return res

    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        # initial impression is to do normal permutation with a change
        # we don't generate all and then filter out the duplicates but in the return recursive call
        # we check if the previous entry is the same as the one we are inserting
        
        
        
        #return self.recurse(nums)
        #import time
        #start_time = time.time()
        #r = self.recursebrute(nums)
        r = self.prepnumbers(nums)
        #print("--- %s seconds ---" % (time.time() - start_time))
        return r