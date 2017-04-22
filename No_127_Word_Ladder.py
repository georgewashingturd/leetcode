###############################################################################
# 127. Word Ladder
###############################################################################

class Solution(object):
    # it assumes the lengths are the same
    def Connected(self, s1, s2):
        # d is the number of different characters between the two
        d = 0
        for i in range(len(s1)):
            if (s1[i] != s2[i]):
                if (d == 1):
                    return False
                else:
                    d += 1

        return True
        
    # this one apparently takes too long because we check every word in the wordlist everytime
    def BFS(self, beginWord, endWord, wordList):

        q = [beginWord]
        q.append("")
        # this disctionary is to find which word has been visited
        d = {}
        d[beginWord] = 1
        
        # c indicates the distance to endWord
        c = 0
        while(q):
            #print("===")
            n = q.pop(0)
            #print(n)
            
            # the len(q) > 0 is to prevent infinite loop
            if (n == "" and len(q) > 0):
                c += 1
                q.append("")
                continue
            
            for x in wordList:
                # process it as long as it's not visited yet
                if (x not in d):
                    if (self.Connected(n, x)):
                        #print(x)
                        if (x == endWord):
                            return c+2
                        q.append(x)
                        d[x] = 1
                        
            #input("Enter")
                        
        return 0
        
    # use another method to modify the word itself one alphabet at a time and then checking against the dictionary and apparently it works
    def BFS2(self, beginWord, endWord, wordList):

        al = list(range(ord("a"),ord("a") + 26))
        al = list(map(chr,al))
        wd = dict.fromkeys(wordList, 1)
        
        q = [beginWord]
        q.append("")
        # this disctionary is to find which word has been visited
        d = {}
        d[beginWord] = 1
        
        # c indicates the distance to endWord
        c = 0
        while(q):
            #print("===")
            n = q.pop(0)
            #print(n)
            
            # the len(q) > 0 is to prevent infinite loop
            if (n == "" and len(q) > 0):
                c += 1
                q.append("")
                continue
            
            # so instead of looping the entire word list we try to change it by changing the starting word itself
            #l = list(n)
            for i in range(len(n)):
                prev = n[i]
                for ch in al:
                    #l[i] = ch
                    x = n[:i] + ch + n[i+1:]
                    #print(x)
                    
                    if (x not in d and x in wd):
                        if (self.Connected(n, x)):
                            #print("-=-=-=-=-=")
                            #print(x)
                            if (x == endWord):
                                return c+2
                            q.append(x)
                            d[x] = 1
                #l[i] = prev
            
                        
            #input("Enter")
                        
        return 0
    
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        
        return self.BFS2(beginWord, endWord, wordList)