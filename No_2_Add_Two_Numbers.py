# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # runner pointers
        n1 = l1
        n2 = l2
        
        # pointer to the previous resultant node so that we can link the result
        prev = None
        # cr means carry
        cr = 0
        
        while (n1 is not None or n2 is not None):
            # r holds the temporary addition result for each number
            r = ListNode(0)
            
            if (n1 is not None and n2 is not None):               
                r.val = n1.val + n2.val + cr
                n1 = n1.next
                n2 = n2.next
            elif (n1 is not None and n2 is None):
                # if one number is longer than the other and the shorter one is already covered 
                # just break no need to process just be careful to tackle the carry
                if (cr == 0):
                    prev.next = n1
                    break
                r.val = n1.val + cr
                n1 = n1.next
            else:
                # if one number is longer than the other and the shorter one is already covered 
                # just break no need to process just be careful to tackle the carry
                if (cr == 0):
                    prev.next = n2
                    break
                r.val = n2.val + cr
                n2 = n2.next

            # get the carry and actual result
            cr = r.val // 10
            r.val = r.val % 10
            
            # if this is the first number set the head for the return pointer
            if (prev is None):
                res = r
            else:
                prev.next = r
            prev = r
        
        # outside of the loop we need to see if we need to add another digit due to the carry e.g. 9+9 = 18 so we need to add one more digit
        if (cr > 0):
            r = ListNode(cr)
            prev.next = r
            
        return res
        