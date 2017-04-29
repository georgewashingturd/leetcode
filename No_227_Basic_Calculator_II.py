###############################################################################
# 227. Basic Calculator II
###############################################################################

class Solution(object):
    def findnextoperand(self, s, idx):
        st = idx
        while(idx < self.ln and (s[idx].isdigit() or s[idx].isspace())):
            idx += 1
        
        return int(s[st:idx]), idx
            
    def findnextoperator(self, s, idx):
        st = idx
        while(idx < self.ln and (s[idx].isdigit() or s[idx].isspace())):
            idx += 1
                
        if (idx >= self.ln):
            return None
            
        return s[idx], idx + 1

    def findnextsafeoperand(self, s, idx):
            
        op1, idx = self.findnextoperand(s, idx)
        if (idx < self.ln):
            opr, idx = self.findnextoperator(s, idx)
            
            while (opr in ("*", "/")):
                op2, idx = self.findnextoperand(s, idx)
                if (opr == "*"):
                    op1 *= op2
                else:
                    op1 //= op2
                if (idx < self.ln):
                    opr, idx = self.findnextoperator(s, idx)
                else:
                    return op1, idx
            
            return op1, idx - 1
        else:
            return op1, idx
        
        
    def mycalculator(self, s):
        
        # find first number
        # find operator attached to it
        
        # if operator is multiply or divide calculate immediately
        # if operator is add or subtract we need to process further to make sure it is not followed
        # by multiply or divide
        
        self.ln = len(s)
        
        res, i = self.findnextoperand(s, 0)
        
        # find first number
        while (i < self.ln):
            
            opr, i = self.findnextoperator(s, i)
            
            if (opr in ("*", "/")):
                op2, i = self.findnextoperand(s, i)
                
                if (opr == "*"):
                    res *= op2
                else:
                    res /= op2
            else:
                # it's add or subtract so we need to parse further until 
                
                op2 , i = self.findnextsafeoperand(s, i)
                
                if (opr == "+"):
                    res += op2
                else:
                    res -= op2
        
        return res
        
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        return self.mycalculator(s)
        
