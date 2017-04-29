###############################################################################
# 324. Wiggle Sort II
###############################################################################

class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        # my first idea was to do merge sort and then reverse the right sub answer
        # and then merge the two
        
        # my next idea is to sort the whole thing and then swap them every couple elements
        # the only thing is that we might have duplicate numbers
        
        # this can be overcome by sorting the whole thing split it into two
        # and then interleave them but we have to interleave them from the back
        # otherwise for [4,5,5,6] we will have left = [4,5] right = [5,6] and after interleaving
        # we have [4,5,5,6] again but if we interleave from the back we have
        # [5,6,4,5]
        
        ln = len(nums)
        
        if (ln <= 1):
            return
        
        nums.sort()
        
        mid = ln//2 + (ln % 2)
        
        left = nums[:mid]
        right = nums[mid:]
        
        #for i in range(len(right)):
        #    nums[2*i], nums[2*i+1] = left[i], right[i]
        
        j = 0
        if (ln % 2):
            for i in range(len(left)-1, 0, -1):
                nums[2*j], nums[2*j+1] = left[i], right[i-1]
                j += 1
                
            nums[-1] = left[0]
        else:    
            for i in range(len(left)-1, -1, -1):
                nums[2*j], nums[2*j+1] = left[i], right[i]
                j += 1
            
        
        