###############################################################################
# 20. Valid Parentheses
###############################################################################

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        # I will use a stack and a dictionary
        d = {"(":")","{":"}","[":"]"}
        
        stk = []
        
        for i in range(len(s)):
            if (s[i] in d):
                stk.append(s[i])
            else:
                if (len(stk) == 0):
                    return False
                ch = stk.pop()
                if (d[ch] != s[i]):
                    return False
                
        if (len(stk) > 0):
            return False
            
        return True