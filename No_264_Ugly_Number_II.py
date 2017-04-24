###############################################################################
# 264. Ugly Number II
###############################################################################

class Solution(object):
    
    # this is the brute force way and it apparently took too long
    def isUgly(self, n):
        if (n % 2 > 0 and n % 3 > 0 and n % 5 > 0):
            return False
            
        while (n & 1 == 0):
            n >>= 1
        
        while (n % 3 == 0):
            n //= 3
            
        while (n % 5 == 0):
            n //= 5
            
        if (n > 1):
            return False
            
        return True
    
    def bruteWay(self, n):
        
        if (not n):
            return 0
            
        if (n == 1):
            return 1
            
        c = 1
        k = 2
        while (c < n):
            if (self.isUgly(k)):
                c += 1
                if (c == n):
                    return k
            k += 1
    
    def dpishWay(self, n):
        # the idea here is to see the pattern like so
        # it starts with 1 as the first ugly number the next can be 2*1 3*1 or 5*1
        # m2, m3, m5 are the pointers to the list as to where we can multiply 2,3, and 5 respectively
        # we choose the minimum of 2*1 3*1 and 5*1 and in this case it is 2*1 we add this to the list of ugly numbers
        # the minimum multiplier for 2 is then incremented to the next element in the list which in this case is 2 itself
        # while the multiplier for 3 and 5 are still the same i.e. 1
        
        # the complication is when u[m2]*2 u[m3]*3 u[m5]*5 are all the same in this case we need to increment each of them
        # until we find the next number otherwise after each iteration we get the same ugly number
        
        
        if (not n):
            return 0
            
        if (n == 1):
            return 1
        
        m2 = 0
        m3 = 0
        m5 = 0
        
        # I was experimenting to see whether preparing the list in advance speeds up things compared to calling append
        u = [1]*n
        
        c = 1
        while (c < n):

            curr = min(u[m2]*2, u[m3]*3, u[m5]*5)
            if (curr == u[m2]*2):
                m2 += 1
                while(u[m3]*3 <= curr):
                    m3 += 1
                while(u[m5]*5 <= curr):
                    m5 += 1
            elif (curr == u[m3]*3):
                m3 += 1
                while(u[m2]*2 <= curr):
                    m2 += 1
                while(u[m5]*5 <= curr):
                    m5 += 1
            elif (curr == u[m5]*5):
                m5 += 1
                while(u[m2]*2 <= curr):
                    m2 += 1
                while(u[m3]*3 <= curr):
                    m3 += 1
                    
            u[c] = curr
            
            c += 1

        return u[n-1]
    
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        #print n    
        # can we use Chinese remainder theorem? since 2,3,5 are all primes
        
        return self.dpishWay(n)