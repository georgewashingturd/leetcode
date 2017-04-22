###############################################################################
# 15. 3Sum
###############################################################################

# My approach here might not be optimum but it does the work but it's conceptually easy
# to get 3Sum just isolate one number and then the 2Sum for the remaining
# and I use dict to speed things up

# Note that there are two distribuitons of execution time on Leetcode I believe this isdue to the fact
# that they added a new test case that caused TLE (Time Limit Exceeded) for a lot of people

class Solution(object):

    # a slightly more efficient approach by not re creating the dict everytime this function is called
    # but there is no noticable difference on Leetcode
    def allTwoSumFaster(self, dn, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # print nums
        if (not dn):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = dn
        
        ans = []
        od = []
        for i in d.keys():
            n = target - i
            
            # since it's only the sum of two numbers once a number is used it cannot be reused bby another number
            # e.g. say the target is 5, and we found 3 has a partner that sums up to 5, but since it's only a sum of two numbers
            # the partner of 3 is unique, which is 2 and the partner of 2 is also unique, so we can immediately pop these two from the dictionary
            if (n in d):
                if (i <> n):
                    ans.append([i,n])
                    od.append([i,d[i]])
                    od.append([n,d[n]])
                    del d[i]
                    del d[n]
                else:
                    # in the case of n == i, d[i] must have length more than 1 otherwise it doesn't have a solution
                    if (d[i] >= 2):
                        ans.append([n,n])
                    od.append([n,d[n]])
                    del d[n]
        
        # this is to avoid copying the dict everytime
        for m in od:
            d[m[0]] = m[1]
                
        #print ans        
        return ans

    ###############################################################################
        
    # need to find a way to find all unique two numbers that add up to target
    def allTwoSum(self, dn, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # print nums
        if (not dn):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = dict(dn)
        
        ans = []
        for i in d.keys():
            n = target - i
            
            # since it's only the sum of two numbers once a number is used it cannot be reused bby another number
            # e.g. say the target is 5, and we found 3 has a partner that sums up to 5, but since it's only a sum of two numbers
            # the partner of 3 is unique, which is 2 and the partner of 2 is also unique, so we can immediately pop these two from the dictionary
            if (n in d):
                if (i <> n):
                    ans.append([i,n])
                    del d[i]
                    del d[n]
                else:
                    # in the case of n == i, d[i] must have length more than 1 otherwise it doesn't have a solution
                    if (d[i] >= 2):
                        ans.append([n,n])
                    del d[n]
                    
                
        #print ans        
        return ans

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (len(nums) < 3):
            return []
            
        if (len(nums) == 3): 
            if ((nums[0]+nums[1]+nums[2]) <> 0):
                return []
            else:
                return [nums]
            
        
        d = {}
        
        # add all numbers into the dictionary and because we iterate through nums the indices in tha values are sorted in ascending order
        for i in xrange(len(nums)):
            if (nums[i] in d):
                n = d[nums[i]]
                d[nums[i]] = n+1
            else:
                d[nums[i]] = 1
            
        r = []
        
        for n in d.keys():
            target = -n
            # print target
            # print d
            gone = True
            if (d[n] > 1):
                m = d[n]
                m -= 1
                d[n] = m
                gone = False
            else:
                del d[n]
                
            l = self.allTwoSum(d, target)
            # print l
            for j in l:
                r.append([-target,j[0],j[1]])
                
            if (gone == False):
                del d[n]
                
        return r