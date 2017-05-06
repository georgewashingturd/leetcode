###############################################################################
# 310. Minimum Height Trees
###############################################################################

class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        
        # heuristically speaking we should start with the node with the most connection but not sure if it will work for all cases
        # one case it won't work is like so
        # say we have two nodes a and b where each has 1000 leaves
        # and a and b connects to c so c is obviously the root with min height
        # but a and b have the most connections
        
        # another heuristic algo is to start with the leaves
        # find nodes with just one node and then move inward
        # this works because the logic is as follows
        # say you only have a straight line with nodes on it
        # if you start from the end points and then move inward with the same speed
        # you will meet in the middle which is the root with minimum height
        # the same logic applies to trees as well
        
        # the difference is that we want to keep removing leaves until we are left with one of two nodes
        # and those one or two nodes are our answer
        
        if (n == 0):
            return []
            
        if (n == 1):
            return [0]
            
        if (n == 2):
            return [0,1]
        
        children = {}
        
        for i in edges:
            if (i[0] not in children):
                children[i[0]] = {i[1]:None}
            else:
                children[i[0]][i[1]] = None
            if (i[1] not in children):
                children[i[1]] = {i[0]:None}
            else:
                children[i[1]][i[0]] = None

        
        # first scan for leaves
        q = []

        for i in children:
            if (len(children[i]) == 1):
                q.append(i)

        q.append(None)
        # this addition of None is to indicate that we have removed one layer of leaves

        while(q):
            node = q.pop(0)
            
            if (node is None):
                # if there are only 1 or two nodes left then we are done
                if (n <= 2):
                    return q
                q.append(None)
            else:
                # we onlt decrement the nodes if it is a real node and not None
                n -= 1
                for c in children[node]:
                    # you don't want to go back out and because of this we do not need a visited dict
                    del children[c][node]
                    
                    if (len(children[c]) == 1):
                        q.append(c)