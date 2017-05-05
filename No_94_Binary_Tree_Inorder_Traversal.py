###############################################################################
# 94. Binary Tree Inorder Traversal
###############################################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    # will try to convert this recursive thing into an iterative solution
    def recurse(self, root):
        
        if (not root):
            return
        
        if (root.left is None and root.right is None):
            self.res.append(root.val)
            
        self.recurse(root.left)
        self.res.append(root.val)
        self.recurse(root.right)
        
    def iterative(self, root):
        
        if (not root):
            return
        
        # we definitely need a stack
        # keep going to the left until we reach the end
        # if so at this end's val to the list
        # go up one level and do the right side
        # if the right side is not None keep going to the left of this guy
        
        stk = []
        while(True):
            # keep going to the left
            while (root is not None):
                stk.append(root)
                root = root.left
                
            # we reached the end of left and root.left is now None
            # so either this is the last node on the left
            # or it has a right node
            # in either case we need to add this node to the result
            
            root = stk.pop()
            
            self.res.append(root.val)
            
            while (root.right is None and stk):
                root = stk.pop()
                self.res.append(root.val)
                
            if (not stk and root.right is None):
                return
            
            root = root.right
        

        
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        
        self.res = []
        
        self.iterative(root)
        
        return self.res
        