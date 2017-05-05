###############################################################################
# 63. Unique Paths II
###############################################################################

class Solution(object):
    
    # try straightforward recurse but it's too slow
    def recurse(self, m, n):
        
        if (m >= self.m-1 and n >= self.n-1):
            self.c += 1
            return
        
        # right
        stuck = True
        if (n < self.n-1 and self.grid[m][n+1] == 0):
            self.recurse(m,n+1)
            stuck = False
        if (m < self.m-1 and self.grid[m+1][n] == 0):
            self.recurse(m+1,n)
            stuck = False
            
        if (stuck):
            return
    
    def trydp(self, m, n):
        
        d = [[0]*n for i in range(m)]
        
        # now the difference is once we found an obstacle when setting the top row
        # and the left column everything else after that obstacle has the value 0
        
        for i in range(m):
            if (self.grid[i][0] == 1):
                break
            d[i][0] = 1
            
        for i in range(n):
            if (self.grid[0][i] == 1):
                break
            d[0][i] = 1
            
        for i in range(1, m):
            for j in range(1, n):
                #d[i][j] = (d[i-1][j] + d[i][j-1])*(1 - self.grid[i][j])
                if (not self.grid[i][j]):
                    d[i][j] = (d[i-1][j] + d[i][j-1])
                    
        return d[m-1][n-1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        
        self.c = 0
        
        if (not obstacleGrid):
            return 0
            
        self.m = len(obstacleGrid)
        self.n = len(obstacleGrid[0])
        self.grid = obstacleGrid
        
        if (self.grid[self.m-1][self.n-1] == 1 or self.grid[0][0] == 1):
            return 0
        
        
        
        #self.recurse(0, 0)
        
        #return self.c
        
        return self.trydp(self.m,self.n)
        