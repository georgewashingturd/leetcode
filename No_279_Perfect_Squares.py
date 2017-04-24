###############################################################################
# 279. Perfect Squares
###############################################################################

class Solution(object):
        
    def coinChangeWay(self, n):
        
        # this approach is similar to the coin change way
        # d[i] is the min number of squares to form i so if we have n we want to see if we can get
        # d[n] from d[i] plus some square
        
        d = list(range(n+1))
            
        for i in range(1, len(d)):
            y = 1
            while(y*y <= i):
                x = y*y
                if (d[i-x]+1 < d[i]): 
                    d[i] = d[i-x] + 1
                y += 1
                    
        return d[n]
        
    def palPartitionWay(self, n):    
        # this is actually just like word break except that now the wordDict contains only square numbers
        # and instead of the string it is just a number, actually this is more of the palindrome partitioning
        # but this approach is way too slow
        
        # first build the wordDict
        wd = {}
        i = 1
        while(i*i <= n):
            wd[i*i] = None
            i += 1
            
        if (n in wd):
            return 1
        
        d={0:0}
        
        for i in range(n+1):
            if (i in d or i == 0):
                for w in wd.keys():
                    if (i + w > n):
                        break
                    if (i + w in d):
                        if (d[i] + 1 < d[i + w]):
                            d[i + w] = d[i] + 1
                    else:
                        d[i + w] = d[i] + 1
                        
        return d[n]
                
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """    
        return self.coinChangeWay(n)
        
        