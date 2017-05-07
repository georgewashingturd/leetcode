###############################################################################
# 84. Largest Rectangle in Histogram
###############################################################################

class Solution(object):
    
    # this is too slow
    def brute(self, heights):
                
        # ok so my idea is to scan one item at a time
        # to form a rectangle the bar next to your left or right has to be lower or equal to you
        # so scan left and right to find the end points
        
        # this approach is too slow so let's improve it a bit maybe we can keep track the min bar on the left
        # and the right so we can immediately go to that position
        
        if (not heights):
            return 0
            
        ln = len(heights)
        if (ln == 1):
            return heights[0]
            
        i = 0
        # m holds the max area
        m = 0

        while (i < ln):
            
            # set current heights to be heights[i]
            h = heights[i]
                
            # a is area
            a = h
            n = 1
            
            # scan right
            j = i + 1
            while (j < ln):
                if (heights[j] < h):
                    break
                n += 1
                j += 1
                
            # scan left
            j = i - 1
            while (j >= 0):
                if (heights[j] < h):
                    break
                n += 1
                j -= 1
                
            a = h * n
            if (a > m):
                m = a
                
            i += 1
        
        return m

    # this is still too slow
    def recurse(self, heights):
        
        # Divide and Conquer strategy
        # find the min value the result is either
        # 1. max area left to min value
        # 2. max area right to min value
        # 3. min value times len(heights)

        ln = len(heights)
        
        if (ln == 0):
            return 0
        
        if (ln == 1):
            return heights[0]
            
        i = heights.index(min(heights))
        
        return max(heights[i] * ln, self.recurse(heights[:i]), self.recurse(heights[i+1:]))
        
    # clever idea using stack
    def usestack(self, h):
                
        ln = len(h)
        
        if (ln == 0):
            return 0
        
        if (ln == 1):
            return h[0]
            
        stk = []
        
        m = 0
        i = 0
        while(i < ln):
            # both are accepted but the one with = is a lot slower as we will have too many items on the stack
            # if ((not stk) or h[i] >= h[stk[-1]]):
            if ((not stk) or h[i] > h[stk[-1]]):
                stk.append(i)
                i += 1
            else:
                th = stk.pop()
                
                if (not stk):
                    m = max(h[th] * (i), m)
                else:
                    m = max(h[th]*(i - stk[-1] - 1), m)
        
        i = stk[-1] + 1
        while(stk):
            th = stk.pop()
            if (not stk):
                m = max(h[th] * (i), m)
            else:
                m = max(h[th]*(i - stk[-1] - 1), m)

                
        return m
        
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        
        #return self.brute(heights)
        #return self.recurse(heights)
        return self.usestack(heights)

        
