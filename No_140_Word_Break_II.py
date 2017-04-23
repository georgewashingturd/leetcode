###############################################################################
# 140. Word Break II
###############################################################################

from collections import deque

class Solution(object):
    def rabinkarp(self, s, f):
        d = {f:None}
        lf = len(f)
        for i in range(len(s)-lf+1):
            if (s[i:i+lf] in d):
                return i
                
        return -1
        
    # it will be faster if we start from the back since not all of the keys in d point to a solution
    def trydp(self, s, wordDict):
        F = [False] * len(s)
        d = {}
                    
        for n in range(len(s)):
            if (n == 0 or F[n-1]):
                for w in wordDict:
                    #i = self.rabinkarp(s[n:],w)
                    i = s[n:].find(w)
                    m = n + len(w)
                    if (i == 0):
                        F[m-1] = True
                        d.setdefault(m, []).append(n)
                        
        if (F[len(s)-1] == False):
            return None
            
        return d
            
    # else construct the sentences but it will be faster if we start from the back
    # since not all of the keys in d point to a solution    
    def constructSentences(self, d, s, n, l, r):
            
        for w in d[n]:
            l.appendleft(s[w:n])
            if (w == 0):
                r.append(" ".join(l))
            else:
                self.constructSentences(d, s, w, l, r)    
            l.popleft()
        
    def wc(self, s, wordDict):
        if (not s):
            return False
        
        if (s in wordDict):
            return True
            
        q = [0]
        d = {0:None}
        
        while(q):
            n = q.pop(0)
            
            if (s[n:] in wordDict):
                return True
            
            for w in wordDict:
                i = s[n:].find(w)
                m = n + len(w)
                if(i == 0):
                    if (m not in d):
                        q.append(m)
                        d[m] = None
                        
        return False
            
    def wordBreakHelper(self, s, wordDict, n, l, r):   
        if (not s):
            return []
        
        for w in wordDict:
            i = s[n:].find(w)
            if (i == 0):
                l.append(w)
                m = n + len(w)
                if (m >= len(s)):
                    r.append(" ".join(l))
                else:    
                    self.wordBreakHelper(s, wordDict, m, l, r)
                l.pop()
            
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        #if(not self.wc(s,wordDict)):
        d = self.trydp(s,wordDict)
        if (not d):
            return []
        
        l = deque([])
        r = []
        
        #self.wordBreakHelper(s, wordDict, 0, l, r)
        self.constructSentences(d, s, len(s), l, r)
        
        return r
      
                    