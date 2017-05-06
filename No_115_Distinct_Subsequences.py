###############################################################################
# 115. Distinct Subsequences
###############################################################################

class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        
        return self.trydp(s, t)
        
    def tryrecurse(self, s, t):
        # here are the rules of solving this problem
        
        # 1. if t is None then return 1 as an empty set is always a subset
        # 2. if s is None then return 0 because an empty set doesn't contain anything
        # 3. if s[0] != t[0] then recurse with s[1:] and t
        # 4. if s[0] == t[0] then recurse with
        #    4.1 s[1:] and t this is because the first character can be repeated e.g. s = "AAB" t = "AB"
        #    4.2 s[1:] and t[1:]
        # 5. finally we return the sum of 4.1 and 4.2
        
        if (not t):
            return 1
            
        if (not s):
            return 0
            
        if (s[0] == t[0]):
            a = self.tryrecurse(s[1:], t)
            b = self.tryrecurse(s[1:], t[1:])
            
            return a + b
            
        return self.tryrecurse(s[1:], t)
        
    def trydp(self, s, t):
        # the rules for dp are similar to that of the recursive one and let's translate the
        # recursive rules to a dp, it is clearer to translate the rules from recurse to dp
        # rather than trying to come up with the dp rules straightaway
        
        # let d[i][j] be the number of dictinct subsequences of s[:i+1] and t[:j+1]
        # 1. if t is None then return 1 as an empty set is always a subset
        #    this means that d[i][0] = 1 for any i
        # 2. if s is None then return 0 because an empty set doesn't contain anything
        #    this means that d[0][j] = 0 for any j because s is empty
        # 3. if s[0] != t[0] then recurse with s[1:] and t
        #    here it is easier to compare the tail rather than the head
        #    if s[-1] != t[-1] then d[i][j] = d[i-1][j]
        # 4. if s[-1] == t[-1] then recurse with
        #    4.1 s[:-1] and t this is because the first character can be repeated e.g. s = "AAB" t = "AB"
        #    4.2 s[:-1] and t[:-1]
        #    this means that if s[-1] == t[-1] then
        #    d[i][j] = d[i-1][j] + d[i-1][j-1]
        # 5. finally we return the sum of 4.1 and 4.2
        #    we return d[len(s)-1][len(t)-1]
        
        # one complication here is that if you notice above there's no rule to build
        # d[i][j] from d[i][j-1], in fact this is impossible since for example we have s="BAA" and t="AB"
        # up to s="BAA" and t="A" we have 2 subsequences but once we increase t to "AB" the number
        # of subsequences actually goes down to zero
        
        # coming up with my own rule I get the following

        # then d[i+1][j] is given by
        #     if the additional character at s[i] == t[j-1] is the last character of t[:j]
        #     then d[i+1][j] = d[i][j] * 2 if d[i][j] != 0 and len(t) > 1
        #          d[i+1][j] = d[i][j] + 1 if d[i][j] == 0 or len(t) == 1
        #          the first case is because if we already have "AABC" so the count is 2 and we add another "C"
        #          each of the 2 count will have another choice for "C" and so the count doubles
        #          the difference is when t only has one character and in that case the count only increases by 1
        #          but this still misses the point where for example we have "AAB" and "AB" because
        #          "AA" doesn't contain "AB" and so d[i][j] is zero but once we add "B" to "AAB" it suddenly jumps from
        #          0 to 2 so we need d[i-1][j-1] as well
        #     else if the additional character is not the same then d[i+1][j] = d[i][j]
        # we cannot build d[i][j+1] based on d[i][j] as mentioned above
        # which still misses some cases    
        
        
        # because we are also considering strings of length 0 so we have d[0][0]
        # and for the actual length of the strings we have d[len(s)][len(t)] 
        # so we need to add 1
        ls = len(s)+1
        lt = len(t)+1
        
        d = [[0]*lt for i in range(ls)]

        # make sure d[0][0] is one so the range starts with 1
        for j in range(1,lt):
            d[0][j] = 0

        for i in range(ls):
            d[i][0] = 1
        
        for j in range(1, lt):        
            # t can at most be as long as s so start the s loop from j to save time
            for i in range(j, ls):
                if (s[i-1] <> t[j-1]):
                    d[i][j] = d[i-1][j]
                else:
                    d[i][j] = d[i-1][j] + d[i-1][j-1]
        
        #for i in d:
        #    print i
        
        return d[ls-1][lt-1]