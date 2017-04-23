###############################################################################
# 91. Decoding Ways
###############################################################################

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        # s can actually start with 0 so we must be ready
        if (not s or s[0] == '0'):
            return 0
        
        # this is exactly like word break but the words in the dictionary are restricted to
        # "1", "2", "3", ... , "26"
        
        # we can use dp again but the difference this time is that we want to find out the number of cobinations
        # which can be big
        
        # so first contruct the wordDict
        
        wordDict = dict.fromkeys(list(map(str, list(range(1,27)))), None)
        
        ln = len(s)
        
        if (ln == 1):
            return 1
        
        # 0 means that there is no word ending at index i
        dl = [0]*len(s)

        for i in range(len(s)):
            
            # only process i if the we are starting at a valid location where a valid word ended at i-1
            # the only problematic words are 10 and 20 because 0 cannot be a valid word
            
            # we will only process a substring if it starts at another valid substring or at the beginning
            if (i == 0 or dl[i-1] > 0):
                # but here at most two characters are checked so we need not loop through the dictionary like Word Break
                for j in range(2):
                    if (i + j < ln and s[i:i+j+1] in wordDict and (i == 0 or dl[i-1] > 0)):
                        
                        m = i + j
                        
                        # if we reach here from other paths then add the number of paths to get to the previous
                        # point to this one e.g. if we reach index i-1 in 3 ways then the there is at least 3 ways
                        # to j which is after i-1 is  
                        if (i > 0):
                            dl[m] += dl[i-1]
                        # if we reach this word straight from the beginning then just increase the number
                        # of path to 1
                        else:
                            dl[m] += 1
        
        return dl[ln-1]          