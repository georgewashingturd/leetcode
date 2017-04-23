###############################################################################
# 200. Number of Islands
###############################################################################

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        # my first idea is of course to do a BFS and each new root we found we add the count
        # BFS starts from top left
        
        if (not grid):
            return 0
            
        if (not grid[0]):
            return 0
        
        rmax = len(grid)
        cmax = len(grid[0])
        
        # we cannot use a dictionary as the visited map since a list is mutable so we cannot use [r,c] as a key
        # so we need to use a matrix as well
        vm = [[0]*cmax for row in range(rmax)]
        ic = 0
        
        for r in range(rmax):
            for c in range(cmax):
                
                if (vm[r][c] == 0 and grid[r][c] == '1'):
                    # this is a new root
                    ic += 1
                    
                    q = [[r,c]]
                    vm[r][c] = 1
                    
                    while(q):
                        row, col = q.pop(0)
                        #print "%d %d" % (row, col)
                        
                        # no need to call function just check it straight away
                        if (row > 0 and grid[row-1][col] == '1' and vm[row-1][col] == 0):
                            vm[row-1][col] = 1
                            q.append([row-1,col])
                        if (row < rmax-1 and grid[row+1][col] == '1' and vm[row+1][col] == 0):
                            vm[row+1][col] = 1
                            q.append([row+1,col])
                        if (col > 0 and grid[row][col-1] == '1' and vm[row][col-1] == 0):
                            vm[row][col-1] = 1
                            q.append([row, col-1])
                        if (col < cmax-1 and grid[row][col+1] == '1' and vm[row][col+1] == 0):
                            vm[row][col+1] = 1
                            q.append([row, col+1])
        
        

                            
        return ic