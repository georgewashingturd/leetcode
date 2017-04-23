###############################################################################
# 54. Spiral Matrix
###############################################################################

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        if (not matrix):
            return []
            
        if (not matrix[0]):
            return []
            
        # my initial idea is to do it manually with a direction
        # d is the direction first we go left then down then left then up
        # 0 is right
        # 1 is down
        # 2 is left
        # 3 is up
        
        # start with right
        d = 0
        
        row = len(matrix)
        col = len(matrix[0])
        
        # note that left and top are inclusive while right and bot are not 
        leftcol = 0
        rightcol = col
        
        # shift toprow right away since we are already on top row right now
        toprow = 0 + 1
        botrow = row
        
        
        # now we track which row and column we are on
        # current row and current column cr and cc
        cr = 0
        cc = 0
        
        r = []
        
        for i in range(row*col):
            r.append(matrix[cr][cc])
            
            # there are in total 4 directions
            if (d % 4 == 0):
                cc += 1
                if (cc >= rightcol):
                    d += 1
                    cc = rightcol - 1
                    cr += 1
                    rightcol -= 1
            elif (d % 4 == 1):
                cr += 1
                if (cr >= botrow):
                    d += 1
                    cr = botrow - 1
                    cc -= 1
                    botrow -= 1
            elif (d % 4 == 2):
                cc -= 1
                # not that leftcol is inclusive so must use strict inequality
                if (cc < leftcol):
                    d += 1
                    cc = leftcol
                    cr -= 1
                    leftcol += 1
            elif (d % 4 == 3):
                cr -= 1
                # not that toprow is inclusive so must use strict inequality
                if (cr < toprow):
                    d += 1
                    cr = toprow
                    cc += 1
                    toprow += 1
                    
        return r      
