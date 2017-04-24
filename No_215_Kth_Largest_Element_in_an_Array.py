###############################################################################
# 215. Kth Largest Element in an Array
###############################################################################
     
# We should do something cleverer one of them is to use something called a quick select
# which is a modified wuick sort

# The basic idea is to use Quick Select algorithm to partition the array with pivot:

# Put numbers < pivot to pivot's left
# Put numbers > pivot to pivot's right
# Then

# if indexOfPivot == k, return A[k]
# else if indexOfPivot < k, keep checking left part to pivot
# else if indexOfPivot > k, keep checking right part to pivot
# Time complexity = O(n)

# Discard half each time: n+(n/2)+(n/4)..1 = n + (n-1) = O(2n-1) = O(n), because n/2+n/4+n/8+..1=n-1.

# Quick Select Solution Code:

class Solution(object):
    def quickSelect(self, nums, k, st, ed):
        
        pivot = st
        
        
        for i in range(st+1, ed):
            if (nums[i] > nums[pivot]):
                # put nums[i] at the left of pivot in two steps
                # swap nums[i] to just to the right of pivot
                # and then swap pivot to its immediate neighbor to the right
                if (i != pivot + 1):
                    nums[pivot+1], nums[i] = nums[i], nums[pivot+1]

                nums[pivot+1], nums[pivot] = nums[pivot], nums[pivot+1]
                
                # and don't forget to update the pivot itself as it has moved to the right
                pivot = pivot+1
                #print nums


        #print nums
        #input('+++')
                
        #print '===>', pivot, k 
        if (pivot == k):
            return nums[k]
            
        if (pivot > k):
            return self.quickSelect(nums,k,st,pivot)
        if (pivot < k):
            return self.quickSelect(nums,k,pivot+1,ed)
    
    # Somehow this super straightforward way is accepted by Leetcode and the performance is much better than QuickSelect above    
    def straightforwardSort(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        if (not nums):
            return -1
            
        #nums.sort(reverse=True)
        nums.sort()
        
        #return nums[k-1]
        return nums[-k]

        
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        # I want to try this using quict select method which is the application of quick sort to searching
        # note that sort algorithms can be adapted to search algorithm like binary search
        # in this case it's quick sort to quick select
        
        # the idea is to choose a pivot put > pivot to its left and < pivot to its right
        # and then see if the pivot is the kth element of the array if so return pivot
        # if not repeat the process to the left hald or right half depending on whether
        # the kth element is at the left or right of the pivot
        
        return self.quickSelect(nums, k-1, 0, len(nums))
        
