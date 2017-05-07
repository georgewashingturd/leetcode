###############################################################################
# 395. Longest Substring with At Least K Repeating Characters
###############################################################################

class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        
        ln = len(s)
        if (ln == 0 or k > ln):
            return 0
        
            
        if (k <= 1):
            return len(s)
            
        # first tabulate the count for each character
        d = {}
        
        for i in range(ln):
            d.setdefault(s[i], 0)
            d[s[i]] += 1
        
        # first scan see the possible starting point
        i = 0
        mw = 0
        while (i < ln):
            if (d[s[i]] < k):
                i += 1
            else:
                # we have found a starting point need to know what the end point is
                td = {}
                # we also count the frequency of each character while finding the end point
                j = i
                while (j < ln):
                    if (d[s[j]] >= k):
                        td.setdefault(s[j], 0)
                        td[s[j]] += 1
                    else:
                        # this is the end point for this particular window
                        # see if this window is valid
                        break
                    j += 1
                
                # we get out here if we reach the end of the string of we found an invalid character
                # either way we need to check if the window is valid
                
                # within this window itself there might be multiple subwindows
                # for example s = "bbaaacbd" our first window will be "bbaaa"
                # it is an invalid window as a whole but "aaa" is still valid
                
                # but if our window is "babaa" it doesn't work
                
                # this window must at least be k long but if it's exactly k long then there must only be one character
                if (j-i == k and len(td.keys()) == 1):
                    mw = max(mw, k)
                # if the whole string is valid then the whole string it is
                elif (j-i >= ln):
                    mw = max(mw, j-i)
                elif (j-i > k):
                    mw = max(mw, self.longestSubstring(s[i:j], k))
                
                
                i = j + 1
            
        
        return mw
        
        