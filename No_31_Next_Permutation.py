###############################################################################
# 31. Next Permutation
###############################################################################

class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        # my first idea is to swap them from the back because you want the smallest possible next permutation so you want the first number to stay the same
        # but this gives the wrong result
        # for example [1,3,2] -> [3,1,2] where it should have been [2,1,3]
        
        # next method is to start from nums[-2] and see if we can find the smallest number that is bigger than nums[-2] that is on the right of nums[-2] if so swap that with nums[-2]
        # if not we move to nums[-3] and see if we can find the smallest number that is bigger than nums[-3]
        # but we don't swap with nums[-3] we find the biggest number that is smaller than this number that is between this number and nums[-3]
        
        for i in range(-2, -len(nums)-1, -1):
            maxj = float('inf')
            jdx = len(nums)
            #print(i)
            #print(list(range(i+1, 0)))
            for j in range(i+1, 0):
                #print(j, nums[j], nums[i])
                if (nums[j] > nums[i]):
                    if (nums[j] < maxj):
                        maxj = nums[j]
                        jdx = j
            # if we find such number
            if (jdx != len(nums)):
                    nums[jdx], nums[i] = nums[i], nums[jdx]

                    tmp = nums[i+1:]
                    tmp.sort()
                    nums[i+1:] = tmp
                    #for m in range(jdx, kdx, -1):
                    #    nums[m], nums[m-1] = nums[m-1], nums[m]
                    return
                        
            
        nums.sort()