###############################################################################
# 75. Sort Colors
###############################################################################

class Solution(object):
    
    # this is a very clever trick related to Lomuto's way of doing quick sort
    def onePass(self, nums):
        
        two_ind = 0
        one_ind = 0
        zero_ind = 0
        for i in nums:
            nums[two_ind] = 2
            two_ind += 1
            
            if (i < 2):
                nums[one_ind] = 1
                one_ind += 1
                
            # note that you don't want elif here because you want each if to always be executed
            # this is to ensure that the ind are correct
            if (i == 0):
                nums[zero_ind] = 0
                zero_ind += 1
    
    def twoPass(self, nums):
        
        d = {0:0, 1:0, 2:0}
        
        for i in nums:
            d[i] += 1
            
        nums[:d[0]] = [0]*d[0]
        nums[d[0]:d[0] + d[1]] = [1] * d[1]
        nums[d[0] + d[1]:d[0] + d[1] + d[2]] = [2] * d[2]
    
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        #self.twoPass(nums)
        self.onePass(nums)