###############################################################################
# 328. Odd Even Linked List
###############################################################################

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if (not head):
            return None
         
        oddh = odd = head
        evenh = even = head.next
        
        if (not even):
            return odd
        
        while(odd or even):
            # this means that there are even number of nodes
            if (even.next is None):
                odd.next = evenh
                return oddh
            else:
                odd.next = even.next
                odd = odd.next
                
            if (odd.next is None):
                even.next = None
                odd.next = evenh
                return oddh
            else:
                even.next = odd.next
                even = even.next
        
        
        
        