###############################################################################
# 22. Generate Parentheses
###############################################################################

class Solution(object):
    
    # so the idea is like so say n=3
    # first we just get the left backets
    # ((( we now have some choices to put the right brackets
    # we can choose position 1 which means the first left bracket
    # ()(( now wqe have two choices for the other 2 right brackets either
    # ()()() or ()(())
    # next we choose position 2 for the first right bracket
    # (()( now we have two choices for the right brackets
    # (()()) and (())()
    # and lastly we choose position 3 for the first right bracket
    # ((() and so we only have one choice left
    # ((()))
    
    # this doesn't work either
    def above(self, st, ed, n):
        
        #print(st, ed, n)
        if (st >= ed):
            s = ""
            for i in range(self.sz):
                s += "("
                if (self.tmp[i] == 1):
                    s += ")"
            #print(n)        
            for i in range(n):
                s += ")"
                
            print(s)
            
            return
                
            
        # we first see how many spots we can put our right bracket on
        for i in range(st, ed):
            self.tmp[i] = 1
            self.above(i+1, ed, n-1)
            self.tmp[i] = 0
            
    # this one starts with adding a left bracket and then recurse
    # if we can still add left bracket we do so if not we add right bracket
    # as long as there are left brackets to match it on the return from the recurse call it tries adding a right bracket if it is possible
    def recurse2(self):
        
        if (self.lc < self.sz):
            self.res.append("(")
            self.lc += 1
            self.recurse2()
            self.res.pop()
            self.lc -= 1
            
            if (self.lc > self.rc):
                self.res.append(")")
                self.rc += 1
                self.recurse2()
                self.res.pop()
                self.rc -= 1
                
        elif (self.rc < self.sz):
            self.res.append(")")
            self.rc += 1
            self.recurse2()
            self.res.pop()
            self.rc -= 1
        else:
            #print("".join(self.res))
            #input("j")
            self.r.append("".join(self.res))
            
            
    # this doesn't work
    def recurseway(self, n):
        # to generate the combination for n=2 for example we first generate the combination for 1 and then choose places where we can safely put extra parentheses around it
        # it can be encapsulating it or next to it
        
        if (not n):
            return []
            
        if (n == 1):
            return ["()"]
            
        if (n == 2):
            return ["(())","()()"]
            
        r = self.recurseway(n-1)
        
        res = []
        for a in range(len(r)-1):
            res.append ("(" + r[a] + ")") 
            res.append ("()" + r[a])
            res.append (r[a] + "()")
            
        # we want to separate the last entry because appending on both sides results in the same thing
        res.append ("(" + r[-1] + ")")
        res.append ("()" + r[-1])
        
        return res
    
        
        
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        
        # this problem should be done using recursion
        
        self.tmp = [0] * n
        self.sz = n
        self.res = []
        self.r = []
        self.lc = self.rc = 0
        
        #return self.above(0,n,n)
        self.recurse2()
        return self.r