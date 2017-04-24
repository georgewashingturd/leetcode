###############################################################################
# 124. Binary Tree Maximum Path Sum
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def maxPathHelper(self, root):
        # for each node we want to know the maximum sum we can get that
        # runs through from its left subtree to itself and to its right subtree
        # however when returning to its caller it can only give back
        # one path from itself which is either the sum from its left subtree plus itself
        # or the sum from its right subtree plus itself
        # so we need an extra variable that can keep track the maximum sum we have gathered so far
        
        if (not root):
            return 0
            
        # we gather the maximum we can get from our left subtree if we chose not to use our left subtree
        # then we just get 0 from our left subtree and so we compare its result to zero
        left_max = max(0, self.maxPathHelper(root.left))
        
        # do the same with the right subtree
        right_max = max(0, self.maxPathHelper(root.right))
        
        # now we check if the path going from left subtree to this current node and to its right subtree
        # is bigger than our curr_max
        # but what if we don't want to use one of our sebtrees because say its max is negative?
        # well this case has been handled above by comparing the result of the subtrees to zero
        
        self.curr_max = max(self.curr_max, root.val + left_max + right_max)
        
        # now we return to our caller and we have to note that we can only return the result
        # from one path so we have to choose the bigger one from our left and right subtrees
        
        return root.val + max(left_max, right_max)
        
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        self.curr_max = -float('inf')
        self.maxPathHelper(root)
        return self.curr_max
        
        