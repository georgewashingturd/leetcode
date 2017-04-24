###############################################################################
# 73. Set Matrix Zeroes
###############################################################################

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        
        # the one thing we need to be careful is to not to set zero recursively like this
        # say that (1,1) is zero so we set row 1 to all zero and col 1 to all zero
        # but we should not use the newly introduced zero in row 1 or col 1 to set other rows and cols
        # to zero as well so we must be able to differentiate which zeros are the original ones
        
        # first gather original zeros and put them in a dict
        
        if (not matrix):
            return
        
        if (not matrix[0]):
            return
        
        # now gather all original zeros we want two separate the dicts for rows and cols
        # so that we don't set the same row or col to zero twice
        dr = {}
        dc = {}
        row = len(matrix)
        col = len(matrix[0])
        
        for r in range(row):
            for c in range(col):
                if (matrix[r][c] == 0):
                    dr.setdefault(r, None)
                    dc.setdefault(c, None)
                    
        for r in dr.keys():
            for c in range(col):
                matrix[r][c] = 0
                
        for c in dc.keys():        
            for r in range(row):
                matrix[r][c] = 0
        
        
        
        
        
        
        
        