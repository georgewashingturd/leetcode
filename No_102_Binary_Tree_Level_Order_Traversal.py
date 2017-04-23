###############################################################################
# 102. Binary Tree Level Order Traversal
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

from collections import deque

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if (not root):
            return []
            
        q = deque([root, None])
        r = deque([])
        l = deque([])
        while(q):
            n = q.popleft()
            if (n):
                l.append(n.val)
            if (not n):
                if (q):
                    q.append(None)
                r.append(list(l))
                l = deque([])
            else:
                if (n.left):
                    q.append(n.left)
                if (n.right):
                    q.append(n.right)
                
        return list(r)
                    
            
        