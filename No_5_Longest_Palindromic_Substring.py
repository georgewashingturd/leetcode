###############################################################################
# 5. Longest Palindromic Substring
###############################################################################

class Solution(object):
    
    def checkPal(self, x, a, b):
        # this is the shortest and most Pythonic way to check for a palindrome but it's quite slow
        #return all(x[a]==x[-a-1] for a in xrange(len(x)>>1))
        for i in range((b-a)//2):
            if (x[a+i] <> x[b-(i+1)]):
                return False
        return True

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if (not s):
            return ""
            
        if (self.checkPal(s, 0, len(s))):
            return s
            
        ls = len(s)
        # ml is longest palindrome length so far
        ml = 1
        # a and b are the starting and ending point of the longest palindromic substring
        a = 0
        b = 1
            
        # do not skip to s[1] even though it might look obvious because the even palindromes need to start from 0
        for i in range(len(s)):
            # split into odd palindrome and even palindrome
            
            # first odd ones
            # the strategy here is go to character at i
            # check if there could be a palindrome centered on i
            # now if the shortest possible palindrome if s[i-1:i+2] is not there then there could be no longer palindrome centered at i
            # and we just need to start searching from our current max since the longest one should be longer than the current max but to save time we can check if the shortest palindrom centered at i is even possible
            # if it's possible we immediately jump into the current max+1
            
            maxj = min(i,ls-i-1)
            if (maxj > 0 and self.checkPal(s,i-1,i+1+1) == True):
                for j in range(ml//2, maxj+1):
                    ln = 2*j+1
                    if (self.checkPal(s,i-j,i+j+1) == False):
                        break
                    elif (ln > ml):
                        a = i-j
                        b = i + j + 1
                        ml = ln
                    
                
            # next even ones
            maxj = min(i,ls-i-2)
            if (i+1+1 <= ls and self.checkPal(s,i,i+1+1) == True):
                for j in range(ml//2,maxj+1):
                    ln = 2*j+2
                    if (self.checkPal(s, i-j, i+j+2) == False):
                        break
                    if (self.checkPal(s, i-j, i+j+2) == True and ln > ml):
                        a = i - j
                        b = i + j + 2
                        ml = ln
                    
        return s[a:b] 