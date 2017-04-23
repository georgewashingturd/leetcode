###############################################################################
# 21. Merge Two Sorted Lists
###############################################################################

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """    
        if (not l1):
            return l2
            
        if (not l2):
            return l1
            
        n1 = l1
        n2 = l2
        # c is a previous pointer so that we can link the next node of the previous result tot he current one
        c = None
        
        while (n1 and n2):
            if (n1.val < n2.val):
                v = n1
                n1 = n1.next
            else:
                v = n2
                n2 = n2.next
            if (not c):
                c = v
                # we set the head of the return pointer here
                r = c
            else:
                c.next = v
                c = c.next
                
        if (n1):
            c.next = n1
        if (n2):
            c.next = n2
            
        return r