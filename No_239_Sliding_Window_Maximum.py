###############################################################################
# 239. Sliding Window Maximum
###############################################################################

###############################################################################
# Fastest solution
###############################################################################
        
from collections import deque

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if (not nums):
            return []

        # lm is list of maximum numbers
        lm = []
        # l is the deque
        l = []            
        
        # first check the dequeue
        # if the dequeue is empty then we need select a max from the window
        # and fill up the deque with the window
        for j in range(0,k):
            l.append([nums[j],j])
        l.sort(reverse=True)
        l = deque(l)
        lm.append(l[0][0])
        
        # the outer loop starts with k because each window has k elements and the first k elements have been processed above
        for i in range(k,len(nums)):
            # if the deque is not empty 
            # add the new element in the dequeue in the correct place
            # process the queue, remove any item less than the new element
            # remove any element too far away
            
            if (nums[i] > l[0][0]):
                l = deque([[nums[i],i]])
            else:
                while(l and l[-1][0] < nums[i]):
                    l.pop()
                l.append([nums[i],i])

            while(l and l[0][1] < i - k + 1):
                l.popleft()
            lm.append(l[0][0])
            
        return lm

###############################################################################
# Shortest solution
###############################################################################

from heapq import *

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if (not nums):
            return []

        # lm is list of maximum numbers
        lm = []
        # l is the deque
        l = []            
        
        # first check the dequeue
        # if the dequeue is empty then we need select a max from the window
        # and fill up the deque with the window
        for j in range(0,k):
            heappush(l, (-nums[j],j))
        
        lm.append(-l[0][0])
        
        # the outer loop starts with k because each window has k elements and the first k elements have been processed above
        for i in range(k,len(nums)):
            # if the deque is not empty 
            # add the new element in the dequeue in the correct place
            # process the queue, remove any item less than the new element
            # remove any element too far away
            heappush(l, (-nums[i],i))
            

            while(l and l[0][1] < i - k + 1):
                heappop(l)
            lm.append(-l[0][0])
            
        return lm        

###############################################################################
# Most straightforward solution
###############################################################################

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if (not nums):
            return []

        # lm is list of maximum numbers
        lm = []
        # l is the deque
        l = []            
        
        # the outer loop starts with k-1 because each window has k elements
        for i in range(k-1,len(nums)):
        
            # first check the dequeue
            # if the dequeue is empty then we need select a max from the window
            # and fill up the deque with the window
            if (not l):
                for j in range(i-k+1,i+1):
                    l.append([nums[j],j])
                l.sort(reverse=True)
                
                lm.append(l[0][0])
            else:
            # if the deque is not empty 
            # add the new element in the dequeue in the correct place
            # process the queue, remove any item less than the new element
            # remove any element too far away
                j = 0
                while (j < len(l) and l[j][0] > nums[i]):
                    j += 1

                l.insert(j,[nums[i],i])

                j += 1
                l[j:] = []
                    
                j = 0
                while (j < len(l)):
                    if (l[j][1] < i - k + 1):
                        l.pop(j)
                    else:
                        j += 1
                        
                lm.append(l[0][0])
            
        return lm   
        
        
