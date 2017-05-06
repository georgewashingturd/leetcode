###############################################################################
# 147. Insertion Sort List
###############################################################################

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if (not head):
            return None
        
        # there's only one element here
        if (not head.next):
            return head
            
        
        prev = head    
        tmp = head.next
        
        while (tmp):
            
            # now compare tmp starting from the start of the list so that we can insert it at the right position
            t = head
            pt = None
            while (t.val <= tmp.val and t is not tmp):
                pt = t
                t = t.next
            
            # meaning tmp is already at its right location
            if (t is tmp):
                prev = tmp
                tmp = tmp.next
                continue
                
                
            # we need to insert tmp at the head
            if (not pt):
                prev.next = tmp.next
                tmp.next = t
                head = tmp
            else:
                pt.next = tmp
                # fix the next for previous first before changing next for tmp
                prev.next = tmp.next
                tmp.next = t
            tmp = prev.next
            
        return head
        