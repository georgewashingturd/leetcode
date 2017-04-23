###############################################################################
# 50. Pow(x, n)
###############################################################################

# we can sort of do binary search by keep squaring the number
# x =x^2 followed by x = x^2 = (x^2)^2 and so on until we ovvershoot
# and then backtrack slowly

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        
        return x**n