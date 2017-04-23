###############################################################################
# 23. Merge k Sorted Lists
###############################################################################

from heapq import *
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
                r = c
            else:
                c.next = v
                c = c.next
                
        if (n1):
            c.next = n1
        if (n2):
            c.next = n2
            
        return r



    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        # my first idea is to merge two of them and them merge the third so it's merging two at a time
        # but I got time limit exceeded so next idea is to use a priority queue
        
        # but first some simple checking for simple cases
        if (len(lists) == 0):
            return []

        if (len(lists) == 1):
            return lists[0]
            
        if (len(lists) == 2):
            return self.mergeTwoLists(lists[0], lists[1])
        
        # al is the number of alive lists, for now we assign everything to alive
        al = len(lists)
        
        # h is the priority queue used for our sorting purpose
        h = []
        
        # this is a dictionary used to take care of duplicates because heap queue doesn't play well with duplicates
        # to counter this we add another element in the tuple, first element is val, second is a counter incase of duplicates
        # and the last item in the tuple is the linked list pointer
        d = {}

        for x in range(len(lists)):
            if (lists[x]):
                # if the value was already in the heapq increase its count
                if (lists[x].val in d):
                    d[lists[x].val] += 1
                else:
                    d[lists[x].val] = 0
                    
                # push the 3-tuple into the heapq
                heappush(h, (lists[x].val, d[lists[x].val], lists[x]))

            else:
                # if the linked list is empty reduce the number of alive linked lists still at play
                al -= 1

        r = None
        
        # our stopping condition is when there's only one linked list lest
        # in this case we just attach it to the end of our resulting linked list
        while (al > 1):
            v, c, n = heappop(h)
            
            # if the result linked list is still empty we take note of it
            if (not r):
                r = n
                ph = r
            else:
                ph.next = n
                ph = ph.next
            
            # we then go to the next element of this particular linked list            
            n = n.next
            
            # if the next element is not empty we push its value to the heapq
            if (n):
                if(n.val in d):
                    d[n.val] += 1
                else:
                    d[n.val] = 0
                heappush(h, (n.val, d[n.val], n))
            # if this particular linked list is done we reduce the number of alive lists
            else:
                al -= 1
        
        # and lastly just attached the remaining one list to the tail of our resulting linked list   
        if (al == 1):
            if (not r):
                v, c, r = heappop(h)
            else:
                v, c, ph.next = heappop(h)
            
        return r