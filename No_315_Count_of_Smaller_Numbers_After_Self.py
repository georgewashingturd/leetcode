###############################################################################
# 315. Count of Smaller Numbers After Self
###############################################################################

# the great thing is that sort algorithms can be used for other things to
# they are usually adept for searches and in this case we are searching for
# smaller elements on the right side of the array so merge sort is applicable to this

class Solution(object):
    def merge(self, left, right):
        
        count  = 0
        
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        t = []
        while (ls < le and rs < re):
            if (left[ls][0] <= right[rs][0]):
                t.append(left[ls])
                self.d[left[ls]] += count
                ls += 1
            else:
                t.append(right[rs])
                count +=1
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            self.d[left[ls]] += count
            ls += 1
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        # we might have duplicate entries so we need to keep track of the indices of these elements
        self.d = {(nums[i], i): 0 for i in range(len(nums))}
        
        numi = [(nums[i], i) for i in range(len(nums))]
        
        self.mergeSort(numi)
        
        t = []
        for i in range(len(nums)):
            t.append(self.d[(nums[i],i)])
            
        return t