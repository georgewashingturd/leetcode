###############################################################################
# 160. Intersection of Two Linked Lists
###############################################################################

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if (not headA or not headB):
            return None
        
        # running time must be O(n) and space must be O(1)
        # if space is not O(1) we can just use a dict but since space must be O(1) we cannot
        # so what we can do is to traverse the linked lists together
        # but if the lengths are not the same then we can't do it
        # so we traverse twice
        # first is to find the lengths and once we've found the length we fast forward to the same starting point on the longer one
        
        # first iteration to calculate the lengths
        
        n = headA
        l1 = 0
        while(n):
            l1 += 1
            n = n.next
        n = headA

        n = headB
        l2 = 0
        while(n):
            l2 += 1
            n = n.next
            
        p1 = headA
        p2 = headB
        while(l1 > l2):
            p1 = p1.next
            l1 -= 1
            
        while(l2 > l1):
            p2 = p2.next
            l2 -= 1
            
        # now we should have the same starting point for both lists
        
        while(p1):
            if (p1 is p2):
                return p1
            p1 = p1.next
            p2 = p2.next
            
        return None
