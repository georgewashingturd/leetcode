###############################################################################
# 236. Lowest Common Ancestor of a Binary Tree
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
       
    # the strategy here is to get the path into p and then search if p is a parent of q
    # if it's not we backtrack to each parent of p and see if it's also a parent of q
    # meanwhile we check if one of the parents of p is q itself
    # the quite confusing thing here is that we use these functions to find the node itself
    # and to also check for common ancestor
    
    def gatherPathQ(self, root, q):
        if (not root):
            return False
        if (root is q):
            return True
            
        if ((not self.gatherPathQ(root.left, q)) and (not self.gatherPathQ(root.right, q))):
            return False
            
        return True

    def gatherPathP(self, root, p, q):

        if (not root):
            return None, False
        if (root is p):
            # we found p, now we want to see if we can find q
            return p, self.gatherPathQ(root, q)
        
        lp, lq = self.gatherPathP(root.left, p, q);

        # if we found both q and p then return their parent
        if (lp and lq):
            return lp, lq
        
        if (not lp):
            rp, rq = self.gatherPathP(root.right, p, q)
            
            # if we found both q and p then return their parent
            if (rp and rq):
                return rp, rq
                
            if (not rp):
                # this means that this current root is not a parent of p
                return None, False
            
        # if we are out of the if statement above then this current root is a parent of p
        # now we need to check if this root is also a parent of q
        
        # first check if this root itself is q
        if (root is q):
            return root, True
        
        # we found p on the left but q is not on the left, so search for q on the right
        if (lp):
            return root, self.gatherPathQ(root.right, q)

        # we found p on the right but q is not on the right, so search for q on the left
        if (rp):
            return root, self.gatherPathQ(root.left, q)
        
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        if (not root):
            return None
            
        if (root is p):
            return p
            
        if (root is q):
            return q
            
        np, nq = self.gatherPathP(root, p, q)
        
        return np