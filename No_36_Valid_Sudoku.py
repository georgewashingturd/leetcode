###############################################################################
# 36. Valid Sudoku
###############################################################################

class Solution(object):

    # a cleverer approach would be to encode each element and use only one dict
    # we can encode for example 'r09' which means row 0 already has 9
    # if we get another 9 on row 0 we return False
    
    def onedict(self, board):
        
        d = {}

        for row in range(9):
            for col in range(9):
                if (board[row][col] != "."):
                    n = board[row][col]
                    rn = "r" + str(row) + n
                    cn = "c" + str(col) + n
                    cbn = "cb" + str(3*(row//3) + col//3) + n
                    if (rn in d or cn in d or cbn in d):
                        return False
                    else:
                        d[rn] = None
                        d[cn] = None
                        d[cbn] = None
                        
        return True     

    def twentySevenDict(self, board):
        
        if (not board):
            return False
        
        # we can create 3x9 dict's
        
        # one for row
        # one for column
        # one for each little square
        drow = [{} for i in range(9)]
        dcol = [{} for i in range(9)]
        dcube = [{} for i in range(9)]
        
        # note [{}] * 9 will create 9 references to the same dict and once you change one dict everything else will also change
        
        for row in range(9):
            for col in range(9):
                if (board[row][col] != "."):
                    n = board[row][col]
                    if (n in drow[row] or n in dcol[col] or n in dcube[3*(row//3) + col//3]):
                        return False
                    else:
                        drow[row][n] = None
                        dcol[col][n] = None
                        dcube[3*(row//3) + col//3][n] = None
                        
        return True                
        
        
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        
        return self.onedict(board)