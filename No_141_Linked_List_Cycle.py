###############################################################################
# 141. Linked List Cycle
###############################################################################

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    # first try with extra space
    # we will use a dictionary
    def useDict(self, head):
        if (not head):
            return False
        
        d = {}
        
        n = head
        while(n):
            if (n in d):
                return True
            else:
                d[n] = None
            n = n.next
            
        return False
     
    def useTwoPointers(self, head):
        
        if (not head):
            return False
            
        n1 = head
        n2 = head
        
        while(n1 and n2):
            n1 = n1.next
            n2 = n2.next
            if (not n2):
                return False
            else:
                n2 = n2.next
            if (n1 is n2):
                return True
                
        return False
        
    # based on the principle of it's easier to ask forgiveness than permission    
    def useTwoPointersFast(self, head):
        
        if (not head):
            return False
            
        n1 = head
        n2 = head
        
        while(n1 and n2):
            try:
                n1 = n1.next
                # it's faster to get the collision if we go 3 at a time
                n2 = n2.next.next.next
                if (n1 is n2):
                    return True
            except:
                return False

        return False        
        
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        #return self.useTwoPointers(head)
        #return self.useDict(head)
        return self.useTwoPointersFast(head)
        

        
