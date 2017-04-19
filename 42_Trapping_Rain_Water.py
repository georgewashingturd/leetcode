###############################################################################
# 42. Trapping Rain Water
###############################################################################

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # first find a non zero element
        # once you find a non zero element see if the next number is smaller
        # if it is find the wall, i.e. a higher number than the first
        
        # first, find nonzero element
        i = 0
        
        # lh is the length of the height list
        lh = len(height)
        while (i < lh and height[i] == 0):
            i += 1
            
        # this means that all the elements are zero or only the last element is zero
        if (lh - i <= 1):
            return 0
            
        # now we check if the next element is lower than current
        tot = 0
        while (i < lh-1):
            # hi is height at i
            # hio is height at i plus one
            hi = height[i]
            hio = height[i+1]
            
            if (hio < hi):
                # now we need to find the wall
                # the wall can be as high or higher than height[i] in which case the water level is height[i]
                # or the wall can be actually lower than height[i] in which case the water level is lower than height[i]
                
                # we first check if we can find a higher/equal height wall and then iterate by lowering the water level
                # the minimum water level is 1 + height[i+1] of course
                
                h = hi
                
                while(h > hio):
                    j = i+1
                    tmp = 0
                    
                    while (j < lh and h > height[j]):
                        tmp += h - height[j]
                        j += 1
                        
                    # this means that we found the wall at water level h
                    if (j < lh):
                        tot += tmp
                        break
                    # otherwise we lower the water level and repeat the process
                    else:
                        h = h - 1
                        
                # if we cannot find any wall the increment i and get to the outer loop
                if (h <= hio):
                    i += 1
                else:
                    i = j
            else:
                i += 1                
                
        return tot