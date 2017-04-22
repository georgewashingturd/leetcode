###############################################################################
# 131. Palindrome Partitioning
###############################################################################
        
class Solution(object):
    
    # check for palindrome
    def isPalindrome(self, s):

        if (len(s) <= 1):
            return True
            
        for i in range(len(s)//2):
            if (s[i] <> s[-(i+1)]):
                return False

        return True

    # let's try using dp on this one
    def trydp(self, s):
        
        # to keep track the starting and end point of each palindromic substring
        # d[i] = j means there's a Palindromic substring ending at i and starting at j
        # we need this to build up the list of results
        d = {}
        
        # so we scan s and check whether the substring ending at i is a palindrome
        for i in range(len(s)):
            # so we are at i we want to check if we can find a substring ending at i that is a palindrome
            # there are two possibilities either this substring ending at i starts at another previous
            # palindromic substring or it starts from the beginning so we scan every substring starting
            # from i and going backward to the beginning
            
            # something to note here is that this new substring must start at another palindromic substring
            # or it's the first letter in the string so we add an if
            
            for j in range(i,-1,-1):
                if (self.isPalindrome(s[j:i+1])):
                    d.setdefault(i, []).append(j) 
                    
        return d
        
    def gatherResults(self, s, d, n, l, r):
        # we will build this recursively
        
        if (n not in d):
            return
        
        for i in d[n]:
            l.insert(0,s[i:n+1])
            if (i == 0):
                r.append(l[:])
            else:
                self.gatherResults(s, d, i-1, l, r)
            l.pop(0)
            
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        
        d = self.trydp(s)
        
        r = []
        l = []
        
        self.gatherResults(s, d, len(s)-1, l, r)
        
        return r
        
                        
        