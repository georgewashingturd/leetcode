###############################################################################
# 17. Letter Combinations of a Phone Number
###############################################################################

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        
        # the main idea here is just generating "numbers" from 00000 to max but now each digit
        # might have different range
        
        if (not digits):
            return []
            
        n = len(digits)            
        
        # first build a dictionary
        d = {"2":[3,"abc"], "3":[3,"def"], "4":[3,"ghi"], "5":[3,"jkl"], "6":[3,"mno"], "7":[4,"pqrs"], "8":[3,"tuv"], "9":[4,"wxyz"], "0":[1," "]}
        
        # this will be the current alphabet combo
        a = [0]*n
        
        # construct max and let's find out the total number of combinations at the beginning
        tot = 1
        
        for i in range(1, n+1):
            tot *= d[digits[-i]][0]
        
        l = []

        while (tot > 0):
            l.append("".join([d[digits[i]][1][a[i]] for i in range(n)]))
            tot -= 1
            
            if (tot > 0):            
                i = 1
                c = 1
                while (i < (n + 1) and c > 0):
                    a[-i] += 1
                    # this is the carry
                    t = d[digits[-i]][0]
                    c = a[-i] // t
                    # this is the actual digit 
                    a[-i] %= t
                    i += 1
        
        return l