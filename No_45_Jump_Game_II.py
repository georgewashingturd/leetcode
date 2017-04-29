###############################################################################
# 45. Jump Game II
###############################################################################

class Solution(object):
    
    # will have to use dp on this one
    # it is actually similar to the coin change problem except that the types of coins
    # are not fixed
    # the main idea is like this
    # set d[i] as the min number of steps to get to i so d[0] is 0 since we start at 0
    # now say we at index j we want to see if we can get from previous d[i]'s to j
    # if we can do it then we add 1 step to d[i] and find the min of d[i] + i for d[j]
    # but what if we require two steps from i to j then in that case we can alternatively use
    # d [i' > i]
    
    # this one is fast enough the thing to note is that we do not need to mark or even
    # have the array d[i] at all for example [5,2,8,...]
    # when we process the first element 5 we know that indices 1 to 5 will have 1 step
    # as their answer, when we process the next element, i.e. 2 it doesn't help at all
    # since from 0 we can already go to 1 .. 5 in one step this is the case of 
    # if (lmn < lm):
    #    continue
    # so the d array for min steps if we have one will look like this at this point
    # [0,1,1,1,1,1]
    # when we process 8 we don't change d[3..5] because it's already at min value
    # we will only change d[6..10], points reachable from 2 and d[6..8] 
    # is only one more step than d[1..5] and so we don't need an array a single 
    # variable d is enough to keep track of min steps needed
    
    
    def trydp3(self, nums):
        
        ln = len(nums)
        
        if (ln == 1):
            return 0
        
        if (nums[0] and nums[0] >= ln - 1):
            return 1
            
        # get an element

        # find out the max point for this elem and set starting point for itetration
        mp = 1
        lm = nums[0] + 1
        # set current d
        d = 1
        
        while (lm <= ln - 1):
            # now iterate in this range from mp to lm
            # save lm because this will be the starting point for the next iteration
            n = lm
            for j in range(mp, lm):
                # get next possible max point
                #lmn = min(j + nums[j] + 1, ln)
                lmn = j + nums[j] + 1
                
                # we have reached the end
                if (lmn > ln - 1):
                    lm = lmn
                    break
                    
                # can't reach any new point    
                if (lmn < lm):
                    continue
                    
                # we have reached a new point but there might be a farther reaching point down the line
                lm = lmn
            # set starting point as the previous end point
            mp = n    
            
            # increase min step by one
            d = d + 1
        

        return d
    
    # this one is faster but it's still not fast enough because we are still updating
    # multiple elements of d[] we need not do that
    def trydp2(self, nums):
        
        d = [float('inf')] * len(nums)
        d[0] = 0
        
        ln = len(nums)
        
        if (ln == 1):
            return 0
        
        if (nums[0] and nums[0] >= ln - 1):
            return 1
        
        # keep track of the max point
        mp = 1
        
        for i in range(ln - 1):
            if (mp > ln - 1):
                break
            
            lm = min(i + nums[i] + 1, ln)
            if (lm < mp):
                continue
            
            #for j in range(mp, lm):
            #    d[j] = d[i] + 1
                
            d[mp:lm] = [d[i] + 1] * (lm - mp)
            mp = lm
            #print(d)
            #print(mp)
        return d[ln - 1]
            
    # this one is too slow because of the inner loop checking for all previous solutions
    def trydp(self, nums):
        
        d = [float('inf')] * len(nums)
        d[0] = 0
        
        ln = len(nums)
        
        for i in range(1,ln):
            for j in range(i):
                if (nums[j] + j >= i):
                    d[i] = min(d[i], d[j] + 1)
                    
        return d[ln - 1]
    
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        #return self.trydp(nums)
        return self.trydp3(nums)
        