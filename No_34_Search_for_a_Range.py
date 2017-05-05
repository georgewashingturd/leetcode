###############################################################################
# 34. Search for a Range
###############################################################################

class Solution(object):
    
    def binSearchUp(self, nums, target, st):
        
        #st = 0
        ed = len(nums) - 1
        ln = len(nums) - 1
        
        while (st <= ed):
            mid = (st + ed)//2
            
            if (nums[mid] == target):
                if (mid + 1 > ln):
                    return mid
                if (nums[mid+1] > target):
                    return mid
                st = mid + 1
            elif (nums[mid] < target):
                st = mid + 1
            else:
                ed = mid - 1
                
        return -1
        
    def binSearchDown(self, nums, target):
        
        st = 0
        ed = len(nums) - 1
        ln = len(nums) - 1
        
        while (st <= ed):
            mid = (st + ed)//2
            
            if (nums[mid] == target):
                if (mid == 0):
                    return mid
                if (nums[mid-1] < target):
                    return mid
                ed = mid - 1
            elif (nums[mid] < target):
                st = mid + 1
            else:
                ed = mid - 1
                
        return -1
    
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        # since this is log n it must be binary search
        
        st = self.binSearchDown(nums, target)
        
        if (st == -1):
            return [-1,-1]
        else:
            return [st, self.binSearchUp(nums, target, st)]
        
        
        
        