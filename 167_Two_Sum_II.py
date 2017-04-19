###############################################################################
# 167. Two Sum II
###############################################################################

class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """

        if (not numbers):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = {}
        
        # add all numbers into the dictionary
        
        for i in numbers:
            if (i in d):
                n = d[i] + 1
                d[i] = n
            else:
                d.setdefault(i, 1)
        
        for i in d:
            n = target - i
            
            if (n == i):
                if (d[i] > 1):
                    j = numbers.index(i)
                    m = numbers.index(n, j+1)
                    if (j < m):
                        return [j+1,m+1]
                    else:
                        return [m+1,j+1]
            else:        
                if (n in d):
                    j = numbers.index(i)
                    m = numbers.index(n)
                    if (j < m):
                        return [j+1,m+1]
                    else:
                        return [m+1,j+1]
                
                
        return []