###############################################################################
# 48. Rotate Image
###############################################################################

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        
        # we will try to rotate it in place
        # since this is a clockwise rotation we need to move it clockwise
        
        #very first thing to do is to check if the list is empty
        if (not matrix):
            return matrix
        # but also we need to check if matrix is [[]] in this case not matrix is False
        if (not matrix[0]):
            return matrix
        
        #self.myprint(matrix)
        
        
        # first we need to see if the matrix size is odd or even
        n = len(matrix)
        
        # so to move it in place we need to move
        # top left to top right
        # top right to bottom right
        # bottom right to bottom left
        # bottom left to top left
        
        # first we need to get the outer layer so the top left row and column are row = 0 column = 0
        
        for c in range(n//2):
            # starting row and starting column
            sr = c
            sc = c
            # initial size is the full size
            sz = n - (2*c)
            
            for i in range(sz-1):
                tl = matrix[sr][sc+i] # for this one the column moves to the right
                tr = matrix[sr+i][-(sc+1)] # for this one the row moves down
                br = matrix[-(sr+1)][-(sc+i+1)] # for this one the column moves backward
                bl = matrix[-(sr+i+1)][sc] # for this one the row moves upward
                
                # now move things around clockwise
                # top right = top left
                matrix[sr+i][-(sc+1)] = tl
                # bottom right = top right
                matrix[-(sr+1)][-(sc+i+1)] = tr
                # bottom left = bottom right
                matrix[-(sr+i+1)][sc] = br
                # top left = bottom left
                matrix[sr][sc+i] = bl
        
                
        #self.myprint(matrix)