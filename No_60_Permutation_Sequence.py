###############################################################################
# 60. Permutation Sequence
###############################################################################

class Solution(object):
    def recurse(self, n, k):
        # calculate the number of combinations for n-1
        # using factorials of course
        
        if (n == 1):
            self.s += str(self.d[0])
            return
        
        tot = 1
        for i in range(1,n):
            tot *= i
        
        # we want to know what the first number is
        # but if there's residue we need to increment the first number by one
        # for example n=3 k=3 here tot=2 but 3//2 = 1
        f = (k // tot) 
        if (k % tot):
            f += 1
        
        self.s += str(self.d[f-1])    
        self.d.pop(f-1)
        
        self.recurse(n-1, k % tot)
            
            
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        self.d = list(range(1,n+1))
        
        self.s = ""
        
        self.recurse(n, k)
        
        return self.s
        

        