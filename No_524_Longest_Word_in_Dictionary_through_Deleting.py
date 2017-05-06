###############################################################################
# 524. Longest Word in Dictionary through Deleting
###############################################################################

class Solution(object):
    # need to be able to check if a word is a subsequence of another
    def issub(self, s, d):
        
        # rules for recursion but recursion is way too slow for this
        # 1. if d is empty then return True since an empty set is always a subset
        # 2. if s is empty return False
        # 3. check if s[0] == d[0] if so recurse with s[1:] and d[1:]
        #          else recurse with s[1:] and d
        
        #if (not d):
        #    return True
            
        #if (not s):
        #    return False
            
        #if (s[0] == d[0]):
        #    return self.issub(s[1:], d[1:])
        #else:
        #    return self.issub(s[1:], d)
            
        i = 0
        ls = len(s)
        ld = len(d)
        
        di = 0
        
        # d must be shorter or equal length than s
        while(i < ls):
            if (d[di] == s[i]):
                di += 1
                if (di >= ld):
                    return True
            i += 1
        
        return False
        
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        
        d.sort()
        
        ml = 0
        ms = ""
        
        for i in d:
            ln = len(i)
            if (ln > ml and self.issub(s, i)):
                #print i
                if (ln > ml):
                    ml = ln
                    ms = i
                    
        return ms
        
        
        