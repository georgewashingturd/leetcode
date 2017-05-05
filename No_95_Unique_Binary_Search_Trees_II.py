###############################################################################
# 95. Unique Binary Search Trees II
###############################################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    # would definitely try recurse
    def tryrecurse(self, st, ed):
        # for recursive function we need to decide a few things
        # first what will the return value be for this case it is a list
        # a list of what? a list of nodes that can be the children of the caller
        
        # what is the stopping condition?
        
        # the stopping condition is when st >= ed but there are cases
        # if st == ed we return a node
        # if st > ed we return None meaning the caller doesn't have children
        
        if (st == ed):
            return [TreeNode(st)]
        
        # we need this because if we return nothing the for loop will exit immediately
        # say we have no left children but plenty of right children the loop still won't start if we return nothing
        if (st > ed):
            return [None]
            
        # if it is not the end point what do we do?
        
        # we go through each element in the list and make it a root
        # then the elements left to it will be the potential left children
        # and the elements right to it will be its right children and then recurse
        i = st
        res = []
        while(i <= ed):
            left_children = self.tryrecurse(st, i-1)
            right_children = self.tryrecurse(i+1,ed)
            
            for left in left_children:
                for right in right_children:
                    t = TreeNode(i)
                    t.left = left
                    t.right = right
                    res.append(t)
                    
            i += 1
            
        return res
            
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        
        if (not n):
            return []
        
        return self.tryrecurse(1, n)

        