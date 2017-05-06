###############################################################################
# 11. Container With Most Water
###############################################################################

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        # so the logic is to start with the widest one i.e. the first and last line
        # and depending whether the left end or the right end is the shortest one we remove the shorter one and iterate
        
        ln = len(height)    
        width = ln - 1
        
        left = 0
        right = ln - 1
        
        ma = width * min(height[left], height[right])
        
        while (width > 1):
            width -= 1
            if (height[left] > height[right]):
                right -= 1
            else:
                left += 1
                
            ma = max(ma, min(height[left], height[right])* width)
            
        return ma
            