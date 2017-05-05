###############################################################################
# 451. Sort Characters By Frequency
###############################################################################

class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        
        if (not s):
            return ""
            
        
        d = dict.fromkeys(list(s), 0)
        
        for i in s:
            d[i] += 1
        
        res = []
        
        for n in d:
            res.append((-d[n], n))
            
        res.sort()
            
        return "".join([i[1]*d[i[1]] for i in res])