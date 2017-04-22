###############################################################################
# 89. Gray Code
###############################################################################

# The trick here is to notice that everytime you add a bit from n -> n+1 you just go through the Gray Code for
# the n in reverse order so you're just adding 1 >> n+1 to them

class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if (n == 0):
            return [0]
            
        tot = 1 << n
        l = [0,1]
        
        if (n == 1):
            return l
        
        for i in xrange(2,tot):
            if (i & (i-1) == 0):
                j = -1
                n = i
            l.append(l[i+j] + n)
            j -= 2
        
        
        return l