###############################################################################
# 204. Count Primes
###############################################################################

###############################################################################
# Fastest solution
###############################################################################   
     
class Solution(object):
    
    # this function to determine if a number is prime is actually not used
    def isPrime(self,n):
        i = 2
        while((i * i) < n):
            if (n % i == 0):
                return False
            else:
                i += 1
        return True
    
    # the solution uses the sieve method
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 3:
            return 0
        primes = [True] * n
        primes[0] = primes[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if primes[i]:
                primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
        print len(primes)
        return sum(primes)
        
###############################################################################
# Shortest and most Pythonic solution
###############################################################################  

class Solution(object):
    
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 3:
            return 0
        primes = [1] * n
        primes[0] = primes[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if primes[i]:
                primes[i * i: n: i] = [0] * len(primes[i * i: n: i])
        print len(primes)
        return sum(primes)  
        