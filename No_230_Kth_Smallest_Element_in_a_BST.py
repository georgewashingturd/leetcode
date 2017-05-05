###############################################################################
# 230. Kth Smallest Element in a BST
###############################################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def inorder(self, root, k):
        
        if (not root):
            return None
        
        if ((not root.left) and (not root.right)):
            self.count += 1
            if (self.count == k):
                return root.val
            return None
        
        n = self.inorder(root.left, k)
        if (n is None):
            self.count += 1
            if (self.count == k):
                return root.val
        else:
            return n
        return self.inorder(root.right, k)
    
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        
        # my first idea is to to in order traversal until we get k elements and we stop
        
        self.count = 0
        
        if (not root):
            return None
            
            
        return self.inorder(root, k)
            
        
        