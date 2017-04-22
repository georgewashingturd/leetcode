###############################################################################
# 126. Word Ladder II
###############################################################################

from heapq import *

class Solution(object):
    # This is a function to gather the paths it's like DFS but unlike DFS we need to print all paths
    # from top to bottom everytime while DFS only prints every node once
    def PrintBranch(self, beginWord, endWord, pred, prefix, dl):
        if (not pred):
            return None
            
        if (endWord in pred[beginWord]):
            p = prefix[:]
            p.append(beginWord)
            p.append(endWord)
            dl.append(p)
            return p
        
        prefix.append(beginWord)
        
        for x in pred[beginWord]:
            self.PrintBranch(x, endWord, pred, prefix, dl)
        
        prefix.pop()
    
    # a better strategy is to start at the end I think so that when we print the lists we can just move forward
    def Dijkstra2(self, beginWord, endWord, wordList):
        wd = dict.fromkeys(wordList, None)
        
        if (endWord not in wd):
            return []
            
        # add beginWord into the dictionary list as well
        wd[beginWord] = None
        del wd[endWord]
            
        al = list(range(ord("a"),ord("a") + 26))
        al = list(map(chr,al))
            
        # if it's not in the dictionary it wasn't marked, i.e. it wasn't visited
        marked = {}
        
        # this is the distance array
        dist = dict.fromkeys(wordList, float('inf'))
        dist[beginWord] = float('inf')

        dist[endWord] = 0
        pred = {}
        
        pd = []
        heappush(pd,(dist[endWord], endWord))
        
        while(pd):
            # nd is the n distance, and n is the word itself, note that this is a tuple
            nd, n = heappop(pd)
            
            if (n in marked):
                continue
            
            # indicate that this word has been visited        
            marked[n] = None
            
            # now get all the neighbors of n
            for i in range(len(n)):
                for ch in al:
                    x = n[:i] + ch + n[i+1:]
                    
                    if (x in wd):
                        if (dist[n] + 1 <= dist[x]):
                            pred.setdefault(x, []).append(n)
                            dist[x] = dist[n] + 1
                            heappush(pd, (dist[x], x))
            
        
        dl = []
        self.PrintBranch(beginWord, endWord, pred, [], dl)
        
        #print(dl)
     
        return dl      

    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        
        return self.Dijkstra2(beginWord, endWord, wordList)
        