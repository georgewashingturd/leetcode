###############################################################################
# 128. Longest Consecutive Sequence
###############################################################################

class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        
        # the idea here is to hash map every number and then scan if this number has consecutive neighbors but we do this from the hashmap
        
        d = dict.fromkeys(nums, None)
        
        allmax = 0
        
        # we need to delete the key in the middle of the loop so we must use d.keys()
        for n in d.keys():
        
            # key might no longer exists since it's been deleted from previous iteration
            if (n in d):
                del d[n]
            else:
                continue
                
            # now scan if this number has consecutive neighbors so the length of this continuous subarray is 1 which is just n
            currmax = 1
            
            # first scan upward
            i = n+1
            while(i in d):
                del d[i]
                currmax += 1
                i += 1
                
                
            # then scan backwards
            i = n-1
            while(i in d):
                del d[i]
                currmax += 1
                i -= 1
            
            # compare curr max with global max
            if (currmax > allmax):
                allmax = currmax
                
        return allmax
            