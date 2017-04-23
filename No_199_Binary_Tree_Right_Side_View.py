###############################################################################
# 199. Binary Tree Right Side View
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # I initially misunderstood this problem
        # my take was just go to the right child as much as possible and if you can't just go to the left and continue
        # in this way for the following tree we will get only 1,3 although what we want is 1,3,5
        #   1            <---
        # /   \
        #2     3         <---
        # \     
        #  5             <---
        # that case we need to do BFS with level recognition
        
        if (not root):
            return []
            
        l = []
        q = [root, None]
        # None acts as a dummy to indicate the level we are in
        
        while(q):
            n = q.pop(0)
            
            # must check if this is the last dummy
            # if so do not add another dummy if it is not the last dummy the add another dummy to indicate the start of the next level
            if (n is None):
                if (len(q) > 0):
                    q.append(None)
            else:    
                # now add the children but first check if it is the right most element of the tree
                if (len(q) > 0 and q[0] is None):
                    l.append(n.val)
                    
                # now add the children
                if (n.left is not None):
                    q.append(n.left)

                if (n.right is not None):
                    q.append(n.right)
        
        return l