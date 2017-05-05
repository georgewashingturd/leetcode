###############################################################################
# 64. Minimum Path Sum
###############################################################################

class Solution(object):
    
    # actually you need to use dp very similar to unique paths
    # need to use Dijsktra but this will be too slow as Dijkstra is greedy algorithm

    def Dijkstra(self):
        # this is the distance priority queue
        d = []
        
        dist = [[float('inf')]*self.n for i in range(self.m)]
        dist[0][0] = self.grid[0][0]
        
        heapq.heappush(d, (0, (0,0)))
        
        while (d):
            t = heapq.heappop(d)
            m = t[1][0]
            n = t[1][1]
            
            # update the distance to its neighbors
            # then right
            if ((n < self.n-1) and (dist[m][n] + self.grid[m][n+1] < dist[m][n+1])):
                dist[m][n+1] = dist[m][n] + self.grid[m][n+1]
                heapq.heappush(d, (dist[m][n+1], (m , n+1)))
            # down
            if ((m < self.m-1) and (dist[m][n] + self.grid[m+1][n] < dist[m+1][n])):
                dist[m+1][n] = dist[m][n] + self.grid[m+1][n]
                heapq.heappush(d, (dist[m+1][n], (m+1 , n)))
          
        return dist[self.m-1][self.n-1]
        
    # there are only two ways to get to a cell so it's a pretty straightforward dp
    def trydp(self, m, n):
        
        d = [[0]*n for i in range(m)]
        
        d[0][0] = self.grid[0][0]
        
        for i in range(1,n):
            d[0][i] = d[0][i-1] + self.grid[0][i]
            
        for i in range(1,m):
            d[i][0] = d[i-1][0] + self.grid[i][0]
            
        for i in range(1,m):
            for j in range(1,n):
                d[i][j] = min(d[i-1][j], d[i][j-1]) + self.grid[i][j]
                
        return d[m-1][n-1]
                
        
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        # first idea is of course to do a BFS or DFS
        # might need to do a two pointer BFS or DFS
        
        self.d = {}
        self.m = len(grid)
        self.n = 0
        if (self.m):
            self.n = len(grid[0])
        self.grid = grid
        
        #return self.Dijkstra()
        return self.trydp(self.m, self.n)
        