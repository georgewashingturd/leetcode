###############################################################################
# 121. Best Time to Buy and Sell Stock
###############################################################################

from collections import *

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if (len(prices) <= 1):
            return 0
        
        # we only take into account prices[1:] onward the first element should not be in this list
        d = deque(sorted(prices[1:], reverse=True))
        
        
        # the logic here is to sort descendingly the stock prices from second day onward
        # we then take the difference between the highest stock price and the current stock price
        # but once we use the current stock price we should remove it from d
        # because it means that it will no longer be in the future
        p = 0
        
        for i in range(len(prices)-1):
            cp = d[0] - prices[i]
            if (cp > p):
                p = cp
            # we only remove if i > 0 because the first element is not in this list as stated above
            if (i > 0):
                d.remove(prices[i])
        return p
        
# another very clever solution is to use Kadane's algorithm which is an algorithm to find the max subarray of an array with positive and negative numbers
# I also have an algorithm to do that although it's not as fancy as Kadane's