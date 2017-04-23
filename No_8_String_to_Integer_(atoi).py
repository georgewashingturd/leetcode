###############################################################################
# 8. String to Integer (atoi)
###############################################################################

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        if (not str):
            return 0
        
        # remove leading white space
        str = str.lstrip()
        
        # sign dictionary
        sd = {"1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "0":0}
        
        # I was initially overthinking this I thought the following are valid
        # "+++++ 123" -> not valid there must be no space between the plus minus sign and the actual numbers
        # "++123" -> not valid
        # "--123" -> not valid
        # only one plus or one minus sign is allowed
        # "a 123" -> not valid
        # also it returns 0 if the number is not valid
        # need to check for max int size as well
        
        # first scan for first valid character or plus minus sign
        i = 0
        
        # n is a sign multiplier
        n = 1
        if(str[0] == "-"):
            n = -1
            i += 1
        if(str[0] == "+"):
            n = 1
            i += 1
        
        # sp is starting character of the digits
        sp = i
        
        while (i < len(str) and str[i] in sd):
            i += 1
        
        nums = str[sp:i]
        
        
        if (len(nums) <= 0):
            return 0
            
        INT_MAX = 2147483647
        INT_MIN = -2147483648
        
        # now we need to discard leading zeros
        nums = nums.lstrip("0")
        tot = len(nums)
        
        # if the number is too big
        if (tot > 10):
            if (n < 0):
                return INT_MIN
            else:
                return INT_MAX
                
        i = 1
        # power of ten
        e = 1
        # the final number
        m = 0
        
        while(i <= tot):
            m += sd[nums[-i]]*e
            e *= 10
            i += 1
            
        if (n > 0):
            return min(INT_MAX, m)
        else:
            return max(INT_MIN, n*m)