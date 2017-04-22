###############################################################################
# 238. Product of Array Except Self
###############################################################################

class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        if (not nums):
            return []
        
        # since I cannot use division I will use repeated additions :)
        
        tot = 1
        
        # this is to track zeros
        zc = 0
        
        for i in nums:
            if (i == 0):
                zc += 1
            else:
                tot = tot * i
            
        if (zc > 1):
            return [0] * len(nums)
            
        l = []
        # now do the repeated sums as an alternative to division :)
        for i in nums:
            if (zc > 0):
                if (i == 0):
                    l.append(tot)
                else:
                    l.append(0)
            else:
                j = abs(i)
                ntot = abs(tot)
                
                # first we need to find out the overall sign of the final product
                if ((i < 0 and tot > 0) or (i > 0 and tot < 0)):
                    s = -1
                else:
                    s = 1
                    
                if (j == 1):
                    l.append(ntot*s)
                else:
                    
                    # if we just use normal addition if the number is large like millions it will take too long
                    # so we use the logarithmic summation approach we don't add one but add ii^n
                    ii = abs(i)
                    p = 1
                    while(j < ntot):
                        pp = p
                        p = j
                        j *= ii

                    # the difference of the multiplier is searched again using 2^n until we get the final result
                    if (j > ntot):
                        k = pp
                        m = 2
                        w = 1
                        
                        while (k*ii <> ntot):
                            m = 2 ** (w-1)
                            k += m
                            m = 1
                            
                            w = 0

                            while ((k+m)*ii < ntot):
                                m *= 2
                                w += 1
                            if ((k+m)*ii == ntot):
                                k += m
                            
                        l.append(k*s)
                    else:
                        l.append(p*s)
                
        return l
                