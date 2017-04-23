###############################################################################
# 206. Reverse Linked List
###############################################################################

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def reverseList(self, head):
        n = head
        
        # this will be our new head
        s = None
        
        while(n):
            pnext = n.next
            # reversed linked list is empty then this node will be the last node
            if (not s):
                n.next = None
            else:
                n.next = s
                
            s = n
                
            n = pnext
   
        return s