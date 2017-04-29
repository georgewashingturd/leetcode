###############################################################################
# 162. Find Peak Element
###############################################################################

class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # the point is that every element is different so either they are ascending
        # desceding or combination of the two but you will not see same elements on a row
        # the thing is that the edges of the array nums[-1] and nums[n] (in C notation)
        # nums[-1] means something different in python, the edges are minus infinity
        # so if the entire array is descending then the first element is the peak
        # and is the entire array is ascending the last element is the peak
        
        # we should use binary search to find a peak, the strategy here is that if
        # the nums[mid] is ascending we move to the right and if it is descending we move
        # to the left because if the entire list is ascending the peak is at the end and 
        # if the entire list is descending the peak is at the beginning
        
        # here start is inclusive but end is not just like the normal python convention
        
        st = 0
        ed = len(nums)
        ln = ed - 1
        
        # I made it harder than it needed to be because ed is not inclusive
        # the goal here is to have st as the index of the element we want
        # so we squeze the range until st = ed
        
        while(st < ed):
            mid = (st + ed)//2
            
            # we are at the end of the array so this means that the last element is the peak
            # we do this because we check for nums[mid+1] and we don't want to get an IndexError
            if (mid == ln):
                if (mid > 0):
                    if (nums[mid] > nums[mid-1]):
                        #return nums[mid]
                        return mid
                    else:
                        ed = mid
                        continue
                else:
                    #return nums[mid]
                    return mid
            
            # if it is Ascending then go to the right
            if (nums[mid] < nums[mid+1]):
                st = mid + 1
                
            # since every element is guaranteed to be unique we don't need to check for equality
            else:
                ed = mid
            
        
        #return nums[st]    
        return st
        
        
        