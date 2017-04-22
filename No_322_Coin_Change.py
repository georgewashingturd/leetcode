###############################################################################
# 322. Coin Change
###############################################################################

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        
        # this is definitely a DP problem so first let's create an array for the possible solutions
        # d[i] where i indicates the total amount and d[i] indicates the min number of coins needed to
        # form i, d[i]=-1 if we can't find a way to form i using coins
        
        d = [-1] * (amount+1)
        d[0] = 0 # it takes zero coins to form zero
        
        # to find out how many coins we need for amount we scan d[i]
        for i in range(1,amount+1):
            
            for c in coins:
                # if I decide to use the solution for d[i-c] then the solution for d[i] will be
                # d[i-c] + 1 since we can certainly form amount i by just adding coin c
                # but we must be careful to see if the coin is not too big and that there's a valid
                # solution for d[i-c]
                if (i >= c and d[i-c] > -1):
                    if (d[i] == -1):
                        d[i] = d[i-c] + 1
                    elif (d[i-c]+1 < d[i]):
                        d[i] = d[i-c] + 1
                        
        
        return d[amount]
                    
            