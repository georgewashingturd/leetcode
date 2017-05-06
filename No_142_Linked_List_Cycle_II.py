###############################################################################
# 142. Linked List Cycle II
###############################################################################

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if(not head):
            return None
            
        tmp = head
        tmp2 = head
        
        while (tmp):
            try:
                tmp = tmp.next
                tmp2 = tmp2.next.next
                
                # say tmp travels A nodes before reaching the start of the cycle and along the cycle
                # n times and the cycle length is B and tmp meets tmp2 C nodes after the start of the cycle
                # then tmp travels a total of A + nB + C before tmp meets tmp2
                # in the same way tmp2 travels A + mB + C meaning tmp2 go around the cycle m times but since tmp2 is twice as fast as tmp
                # A + mB + C = 2(A + nB + C)
                # (m-2n)B = A + C
                # now A + C = 0 mod B
                # so if we travel from tmp which is C away from the start of the cycle and we travel A nodes
                # we will be at the beginning of the cycle as A + C = 0 mod B we might have looped the
                # cycle multiple times by then but it's ok
                
                if (tmp == tmp2):
                    tmp3 = head
                    while(tmp != tmp3):
                        tmp = tmp.next
                        tmp3 = tmp3.next
                    return tmp
                
            except:
                return None
        
        