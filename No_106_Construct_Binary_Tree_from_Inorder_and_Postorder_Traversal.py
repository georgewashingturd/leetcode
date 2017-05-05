###############################################################################
# 106. Construct Binary Tree from Inorder and Postorder Traversal
###############################################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def constructtree(self, postorder, inorder):
        
        if ((not postorder) or (not inorder)):
            return None
        
        root = TreeNode(postorder[-1])
        
        # now find root from inorder to see how big the right subtree is this way we can go tot he left sub tree's root from post order
        # from in order [(... left ...) root (... right ...)]
        # while post order [(... left ...) (... right ...) root] we need to know how far back we need to go from the end of the post order array to get the root of the left subtree this we can get from inorder this is why we need to know that the tree has no duplicates
        
        i = inorder.index(root.val)
        right_sub_tree_size = len(inorder) - i - 1
        
        left_root_index = -(1 + right_sub_tree_size)
        
        root.left = self.constructtree(postorder[:left_root_index], inorder[:i])
        root.right = self.constructtree(postorder[-right_sub_tree_size-1:-1], inorder[i+1:])
        
        return root
        
        
        
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        
        # we might be able to construct the tree from the post order alone note that for post order the last element is the root we can then backtrack from there

        if((not postorder) or (not inorder)):
            return None
        
        return self.constructtree(postorder, inorder)
            
        