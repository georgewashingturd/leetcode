###############################################################################
# 103. Binary Tree Zigzag Level Order Traversal
###############################################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        if (not root):
            return []
            
        
        # do BFS with a marker for levels
        q = [root, None]
        lr = True
        r = []
        tmp = deque([])
        
        while(q):
            n = q.pop(0)
            
            if(n is None):
                # make sure this is not the last None which means that we have finished the whole tree
                if(len(q) > 0):
                    q.append(None)

                r.append(list(tmp))
                lr = not lr
                
                tmp = deque([])
            else:
                if (lr):
                    tmp.append(n.val)
                else:
                    tmp.appendleft(n.val)
                
           
                if (n.left):
                    q.append(n.left)
                if(n.right):
                    q.append(n.right)
            
        
        return r    
            
            
            