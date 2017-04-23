###############################################################################
# 240. Search a 2D Matrix II
###############################################################################

class Solution(object):
    def extractColumn(self, matrix, col):
        return [matrix[row][col] for row in len(matrix)]
    
    def binarySearch(self, l, target):
        start = 0
        end = len(l)
        
        while (start < end):
            mid = (start + end)//2
            if (l[mid] == target):
                return True
            if (l[mid] > target):
                end = mid
            else:
                start = mid+1
                
        return False

    def binarySearchCol(self, m, col, target):
        start = 0
        # this is how many rows we have as we are searching within a column
        end = len(m)

        while (start < end):
            mid = (start + end)//2
            if (m[mid][col] == target):
                return True
            if (m[mid][col] > target):
                end = mid
            else:
                start = mid+1
                
        return False
        
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # my idea is to find rows and columns where the number might be in
        
        # first search rows that might contain the target
        # we search using binary search
        
        totrow = len(matrix)
        # check number of rows
        if (totrow <= 0):
            return False
            
        totcol = len(matrix[0])
        # check number of columns
        if (totcol <= 0):
            return False
        
        rc = 0
        for x in matrix:
            if (x[0] > target):
                break
            if (x[0] == target or x[-1] == target):
                return True
            if (x[0] < target and x[-1] > target):
                if (self.binarySearch(x, target)):
                    return True
            rc += 1

        # if we've checked all rows and did not find target
        if (rc >= totrow):
            return False

        # now we check each column
        for x in range(totcol):
            if(matrix[0][x] > target):
                break
            if (matrix[0][x] == target or matrix[-1][x] == target):
                return True
            if (matrix[0][x] < target and matrix[-1][x] > target):
                if (self.binarySearchCol(matrix, x, target)):
                    return True

        return False