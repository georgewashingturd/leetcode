###############################################################################
# 138. Copy List with Random Pointer
###############################################################################

# Definition for singly-linked list with a random pointer.
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """        
        
        # my idea here is hash map the pointers as we go through the linked list
        
        # first as always, check if the head is valid
        if (not head):
            return None
            
        # make a copy of the head, just in case
        n = head
        
        # the prefix p is for python
        phead = RandomListNode(head.label)
        pn = phead
        
        # set up the dictionary
        d = {head : phead}
                
        # first pass is to just gather all the nodes and build the next pointer dict
        while (n.next):
            pn.next = RandomListNode(n.next.label)
            # add it to the dictionary
            d[n.next] = pn.next
            
            # go to the next node
            n = n.next
            pn = pn.next
            
        # now we take care of the random pointers
        n = head
        pn = phead
        
        while(n):
            if (n.random):
                pn.random = d[n.random]
            n = n.next
            pn = pn.next
            
        return phead