###############################################################################
# 62. Unique Paths
###############################################################################

class Solution(object):

    def trydp(self, m, n):
        # set the top row and left column to all 1 because there is only one way
        # to get to any of those cells
        
        d = [[0]*n for i in range(m)]
        
        # set first row to all one
        for i in range(n):
            d[0][i] = 1
            
        for i in range(m):
            d[i][0] = 1
            
        # now we build the other cells for example for cell i,j
        # if the cell above it has A ways to get to it and the cell left to it
        # has B ways to get to it then for cell i,j there are in total A + B ways
        
        for i in range(1,m):
            for j in range(1,n):
                d[i][j] = d[i-1][j] + d[i][j-1]
                
        return d[m-1][n-1]
        
    def factorial(self, m, n):
        
        if (m <= 1 or n <= 1):
            return 1
        
        a = m+n-2
        b = m-1
        c = a-b
        
        d = max(b,c)
        
        tot = 1
        for i in range(a, d, -1):
            tot *= i
            
        if (d == b):
            m = c
        else:
            m = b
            
        tt = 1
        for i in range(2,m+1):
            tt *= i
            
        return tot/tt
        
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        
        # we have an explicit formula for this, for example if the grid is 3x5 meaning there are 3 rows
        # and 5 columns a path might be like so
        # right -> right -> down -> down -> right -> right or
        # right -> down -> right -> right -> down -> right
        
        # so there are in total 6 moves (3 + 5 - 2) and there are only at max 3-1 = 2 downs
        # so in essence it is 6 choose 2
        # the explicit formula is (m+n-2)!/(m-1)!(n-3)!
        
        #return self.factorial(m, n)
        return self.trydp(m, n)
        