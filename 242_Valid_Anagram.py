###############################################################################
# 242. Valid Anagram
###############################################################################

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if (len(s) != len(t)):
            return False
        
        # again I will use hashmap here and see if it works :)
        
        # prepare hashmap
        ds = {}
        dt = {}
        
        for i in s:
            if (i in ds):
                n = ds[i] + 1
                ds[i] = n
            else:
                ds.setdefault(i,1)
                
        for i in t:
            if (i in dt):
                n = dt[i] + 1
                dt[i] = n
            else:
                dt.setdefault(i,1)
                
        return ds==dt
            
        
        
        