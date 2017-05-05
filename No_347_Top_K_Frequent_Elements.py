###############################################################################
# 347. Top K Frequent Elements
###############################################################################

import heapq

class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        
        # since we are using python just use dict
        
        d = dict.fromkeys(nums, 0)
        
        for i in nums:
            d[i] += 1
            
        res = []
        
        for n in d.keys():
            heapq.heappush(res, (d[n], n))
            
        res = heapq.nlargest(k, res)
        
        return [i[1] for i in res]
            
        
            
            
        
        
        
        