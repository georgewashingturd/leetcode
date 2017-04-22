###############################################################################
# 234. Palindrome Linked List
###############################################################################

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def ReverseLL(self, head):
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
            
            
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        # check if linked list is empty
        if (not head):
            return True
            
            
        # my first idea is to build a dictionary of the val and its index but then it won't be O(1) in space
        # to get O(1) in space  we can split the checking into sections but then we will need to go through the linked 
        # list in O(n^2) time
        
        # the key here is to note that reversing a linked list does not consume extra space, it's an O(1) operation in space
        
        # so first we go through the linked list once to get the length and then we traverse to the middle of the length
        # and reverse the second half
                
        
        # first count the number of elements
        
        n = head
        c = 0
        while(n):
            c += 1
            n = n.next
            
        if (c == 1):
            return True
            
        # now get to the middle of the list, if the length is odd we skip one more because the middle is useless
        
        m = c // 2 + (c % 2)
        n = head
        
        while(m > 0):
            n = n.next
            m = m - 1
            
        l = self.ReverseLL(n)
        
        # now we compare the two halves to see if they're palindromic
        
        k = head
        
        # now the count is the same for odd and even because for odd counts we ignore the middle element anyway
        m = c // 2
        while (m > 0):
            if (k.val != l.val):
                return False
            k = k.next
            l = l.next
            m = m - 1
            
        return True