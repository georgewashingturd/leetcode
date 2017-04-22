# 139. Word Break

class Solution(object):
        
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # the way I see this is as a DFS on a tree
        # so we start at the top of the tree (index 0 of string s)
        # then we want to know if we can get a word in dict to be the first word
        # starting at 0, if we can't immediately return 0
        
        # if we can get  a word from dict to be our first substring at s[0] then we check if we can get
        # another hit at the last location of that substring +1
        # but then we might have multiple choices and the first one might not lead to the correct solution
        # we need to search the next option so it is like parsing through a tree depth wise a DFS
        
        # but before we do DFS we can check for trivial cases
        if (not s):
            return False
            
        if (s in wordDict):
            return True
        
        # the things we keep as nodes are the indices of the string as starting points of the next substrings
        stk = [0]
            
        while(stk):
            n = stk.pop()
            
            if (s[n:] in wordDict):
                return True
            
            # now we check if this node has any descendants by looping through the dictionary
            for w in wordDict:
                # this means that substring w starts at s[n]
                i = s[n:].find(w)
                if (i == 0):
                    stk.append(n+len(w))
                    
        return False
        
        
s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
s2 = "aaaaaaaaaaaaaaaab"
s3 = "aaaaaaaaaaaaaaaa"
wd = ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]        
                
                
                
                
                