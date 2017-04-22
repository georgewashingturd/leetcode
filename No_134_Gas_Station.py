###############################################################################
# 134. Gas Station
###############################################################################

class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        
        # the approach here is to note that this question asks the following
        # take the difference between gas and cost -> nums = gas - cost
        # what we want is a starting point i where the cumulative sum starting at i is always positive or zero
        # because we want cost <= gas to be able to make it around the loop
        ln = len(gas)
        nums = [(gas[i] - cost[i]) for i in range(ln)]

        # now we need a few variables to keep track of things
        ln = len(gas)
        st = 0 # starting point
        ed = 0 # end point
        vs = 1 # number of visited gas stations we end our search once this reaches ln because we've successfully visited all stations
        csum = 0 # current cumulative sum from st to ed
        
        # first find possible starting point which means nums[st] >= 0 any negative ones are impossible
        while(st < ln and nums[st] < 0):
            st += 1
        
        # cost is greater than gas for every station   
        if (st >= ln):
            return -1
        
        # we begin our search here
        csum = nums[st]
        ed = st
            
        while(vs < ln):
            # we increase ed which means we visit the next gas station
            ed += 1
            vs += 1
            csum += nums[ed % ln]
            
            # if after visiting a new gas station we can't move forward we need to change the starting point
            # if no starting point before ed can save us this means that we need to change our starting point to ed or beyond
            if (csum < 0):
                while (csum < 0 and st <= ed):
                    csum -= nums[st % ln]
                    st += 1
                    vs -= 1
  
                # we must now search for a new starting point
                while(st < ln and nums[st] < 0):
                    st += 1
                
                if (st >= ln):
                    return -1
                
                if (st > ed):
                    csum = nums[st]
                ed = st
            
        return st
        

        