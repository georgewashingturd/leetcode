###############################################################################
# 297. Serialize and Deserialize Binary Tree
###############################################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    # l is a list
    # we can do a BFS and in this way we will have the usual configuration
    # i is parent 2*i + 1 is kid left and 2*i +2 is kid right
    # but the difference is that we do not reserve space if a node has no children
    def serializeHelper(self, root):
    
        if (not root):
            l=[]
            #print l
            return str(l)

        l = [root.val]
            
        q = [root]
        
        while(q):
            n = q.pop(0)
            if (n.left):
                l.append(n.left.val)
                q.append(n.left)
            else:
                l.append(None)

            if (n.right):
                l.append(n.right.val)
                q.append(n.right)
            else:
                l.append(None)
                
        return str(l)
            
    def deserializeHelper(self, data):
    
        #print data
        data = data.lstrip('[')
        data = data.rstrip(']')
        
        data = data.split(', ')
        #print data
        if (not data or data[0] == ''):
            return None
            
        t = data.pop(0)
        if (t == 'None'):
            return None
            
        root = TreeNode(int(t))
        # this is BFS in reverse
        q = [root]
        
        while(q):
            n = q.pop(0)
            
            t = data.pop(0)
            if (t <> 'None'):
                n.left = TreeNode(int(t))
                q.append(n.left)
                
            t = data.pop(0)
            if (t <> 'None'):
                n.right = TreeNode(int(t))
                q.append(n.right)
            
        return root
            
        
            
    
    
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        return self.serializeHelper(root)
        
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        return self.deserializeHelper(data)
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))     
     