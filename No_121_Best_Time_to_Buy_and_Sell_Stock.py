###############################################################################
# 121. Best Time to Buy and Sell Stock
###############################################################################

from collections import *

class Solution(object):
    def straightForwardSolution(self, prices):
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
        
###############################################################################
# Cleverer Approach with Explanations
###############################################################################
        
    # another very clever solution is to use Kadane's algorithm which is an algorithm to 
    # find the max subarray of an array with positive and negative numbers
    # here I will also modify Kadane's algorithm to produce the starting and end points of the array
    # that give the maximum sub array
    
    def Kadane(self, A):

    # the main point of Kadane's algorithm is this:
    # 1. If we have the maximum subarray ending at position i (so array[i] is included in the maximumk sub array)
    #    then the maximum subarray ending at position i+1 either includes the max subarray ending at i or it doesn't
    
        #print "~~~~~", A
    
        # max_end_here has the TOTAL possible maximum of the sub array ending at each index i in the loop i.e. starting from array[i]
        # how many elements backward can I include to have the maximum total
        max_end_here = A[0]
        
        # max_so_far has the TOTAL of maximum sub array thus far which is the maximum subarray itself
        # the subarray itself doesn't have to end at i but the elements considered so far is array[0..i]
        max_so_far = A[0]
        
        # array element 0 is already used above and here I'm not iterating through the array fpr i in A[1:]
        # so that it will be clearer
        
        start_point = 0
        end_point = 0
        
        possible_start_update = False
        possible_start_point = 0
        
        for i in range(1,len(A)):
        
            # these two variable is to keep track the starting and end points of the maximum sub array
            prev_end_here = max_end_here
            prev_so_far = max_so_far
            
            # the starting point of the maximum sub array only changes if we decide to discard our previous subarray
            # but it must also be the case that the end point is updated and we are using a flag so as not to create extra "previous" variables
            # an example of the above is 0 7 8 - 10 -20 5 at 5 we are updating max_end_here but the maximum sub array itself doesn't move
            # it is still at [7, 8] so we don't update the starting point
            
            #print "~", i, A[i], max_end_here + A[i]
            if (A[i] > max_end_here + A[i]):
                possible_start_update = True
                possible_start_point = i
            
            # so even if A[i] is negative we will still include it because this marks the possible maximum total of the subarray ending at i
            # so element array[i] has to be included of course if A[i] is a very big number and the previous max_end_here is negative 
            # we'll just use A[i]
            max_end_here = max(A[i], max_end_here + A[i])
            
            # the end point of the maximum sub array only changes if we decide to extend our current result
            if (max_end_here > max_so_far):
                end_point = i
                if (possible_start_update is True):
                    start_point = possible_start_point
                    possible_start_update = False
            
            # here if A[i] is negative we don't include it which makes sense as this is the maximum sub array of array[0..i]
            max_so_far = max(max_end_here, max_so_far)
            
        return max_so_far, start_point, end_point

    
    # this is just a helper function that can be absorbed into the Kadane function above
    # but I want to separate Kadane's function because it is more generic than this problem alone
    def useKadane(self, prices):
        
        if (len(prices) <= 1):
            return 0
        # first we create the difference list and send it to Kadane
        n, s, e = self.Kadane([prices[i] - prices[i-1] for i in range(1,len(prices))])
        
        # print n, "  " ,s, e, "      " ,prices[s], prices[e]
        
        # the end point is off by one because i the difference list we take the next element minus current element
        assert n == (prices[e+1] - prices[s])
        
        if (n < 0):
            n = 0
        
        return n

###############################################################################
# Cleverer Approach without Explanations
###############################################################################

    def useFastKadane(self, prices):
        max_end_here = 0
        max_so_far = 0
        
        # we are comparing against zero and initializing with zeros because the return value is zero if there's
        # no good day to buy and sell stocks
        
        for i in range(1, len(prices)):
            max_end_here = max(0, max_end_here + prices[i] - prices[i-1])
            max_so_far = max(max_end_here, max_so_far)
            
        return max_so_far
         
###############################################################################
# Main Function
###############################################################################
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        # return self.useKadane(prices)
        return self.useFastKadane(prices)
    
    

