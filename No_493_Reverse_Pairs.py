###############################################################################
# 493. Reverse Pairs
###############################################################################

# this is similar to count number smaller than self problem No 315 however
# when we do the merge we want to separate the counting of smaller elements
# from the right sub array and the actualy merging of the arrays themselves
# because of the multiplicative factor of 2 if you merge and count at the same time
# it becomes very confusing very fast not to mention that you'll get the wrong answer

# this is because you want to count by multiplying the right sub list by 2 but you
# still want to merge the usual way without the factor of 2 why? because think about it
# say you have two sub arrays you want to merge, you want each sub array to be sorted
# if it's sorted by involving the factor of 2 you'll get something weird for example

# [2] and [1,3] you'll get [2 ,1, 3] as a merged result because you multiply the right sub list by 2
# during the merge comparison and when you go up in the recursion stacks this will cause all sorts of
# weird problems

# and it turns out that in python instead of merging it manually calling the sort function is a lot faster
# around 38% faster

class Solution(object):
    def merge(self, left, right):

        count  = 0
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        # first we count and then we merge
        while (ls < le and rs < re):
            if (left[ls] <= 2*right[rs]):
                self.d += count
                ls += 1
            else:
                count += 1
                rs += 1
                
        self.d += (le - ls)*count
        
        # it's much faster to just vcall the sorted function
        
        # now we merge
        # ls is left start and le is left end
        #ls = 0
        #le = len(left)
        
        # rd is right start and re is right end
        #rs = 0
        #re = len(right)
        #t = []
        
        #while (ls < le and rs < re):
        #    if (left[ls] <= right[rs]):
        #        t.append(left[ls])
        #        ls += 1
        #    else:
        #        t.append(right[rs])
        #        rs += 1

        #t += left[ls:] + right[rs:]
        
        t = sorted(left + right)

        return t
        
    def mergeSort(self, nums, st, ed):
        
        if (ed - st <= 1):
            return nums[st:ed]
            
        mid = (st + ed) >> 1
            
        left = self.mergeSort(nums, st, mid)
        right = self.mergeSort(nums, mid, ed)
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # this is how you time python execution, it is only accurate to roughly 10 ms
        #import time
        #start_time = time.time()
        
        # we don't need a dictionary here because we a re not keeping track the number smaller for each
        # number we are just interested in the total count
        self.d = 0
        
        self.mergeSort(nums, 0, len(nums))

        #print("--- %s seconds ---" % (time.time() - start_time))   
        
        return self.d      