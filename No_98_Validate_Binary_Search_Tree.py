###############################################################################
# 98. Validate Binary Search Tree
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def getMin(self, root):
        n = root
        while(n.left):
            n = n.left
        return n.val
        
    def getMax(self, root):
        n = root
        while(n.right):
            n = n.right
        return n.val
        
    # let's try in order traversal
    def inorder(self, root):
        
        if (not root):
            return True
        
        if ((not root.left) and (not root.right)):
            if (self.l is not None):
                if (root.val <= self.l):
                    return False
            self.l = root.val
            return True
                
        if (self.inorder(root.left)):
            if (self.l is not None):
                if (root.val <= self.l):
                    return False
            self.l = root.val
            
            return self.inorder(root.right)
            
        return False
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.l = None
        return self.inorder(root)