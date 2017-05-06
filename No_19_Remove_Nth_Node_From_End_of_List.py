###############################################################################
# 19. Remove Nth Node From End of List
###############################################################################

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        # the trick here is to use two pointers
        # one is n ahead of the other one
        
        if (not head):
            return None
        
        tmp = head
        tmp2 = head
        while (n and tmp2):
            tmp2 = tmp2.next
            n -= 1
            
        # not enough nodes
        if (n > 0):
            return head
            
        # this means we want to remove head
        if (not tmp2):
            return head.next
            
        while(tmp2.next):
            tmp = tmp.next
            tmp2 = tmp2.next
            
        tmp.next = tmp.next.next
        
        return head
        
        