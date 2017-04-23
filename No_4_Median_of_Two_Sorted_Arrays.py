###############################################################################
# 4. Median of Two Sorted Arrays
###############################################################################

import heapq

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        n = len(nums1) + len(nums2)
        if (n % 2 == 1):
            m = n//2
            TwoElem = False
        else:
            m = n//2 - 1
            TwoElem = True
        h1 = nums1 + nums2
        
        h1.sort()
        
        if (TwoElem):
            return (h1[m] + h1[m+1])/2.0
            
        return h1[m]/1.0
        
        #heapq.heapify(h1)
        
        #for i in range(m):
        #    med = heapq.heappop(h1)
            
        #if (TwoElem):
        #    med += heapq.heappop(h1)
        #    med = med/2.0
            
        #return (med/1.0)
        
        
        
        