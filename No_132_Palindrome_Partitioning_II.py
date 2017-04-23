###############################################################################
# 132. Palindrome Partitioning II
###############################################################################

class Solution(object):

    # let's try using dp on this one
    # the main difficulty is not in getting the solution but in getting it fast enough
    # we can use the same approach to PalinDrom Partitioning part I but then it will be to slow
    # my initial approach was to build the tree of palindromes and then doing DFS to see which
    # branch is the shortest but one of the test cases, see below, has 2^33 branches LOL
    # the main idea is as we build the tree we keep track which level we are in
    
    def trydp(self, s):
        
        # dl is a list to indicates the level of a node in the tree
        # dl[i] is the level of a node ending at index i i.e. the substring ending at i
        # is a palindrome
        
        # we start at -1 at index 0 just to avoid doing a minus 1 at the final result
        # the code is actually counting the number of palindromes and the number of cuts
        # is just the number of palindromes minus one for example is s = "a" dl[0] = -1
        # but we are actually returning dl[1]
        dl = list(range(-1,len(s)))
        
        # hopefully a faster way to check for palindromes
        pl = [[False]*(len(s)+1) for i in range(len(s)+1)]
        
        # The idea is to see how we can build a tree but the thing is that we want to indicate which level each node is
        # the trick here is that to avoid checking for palindromes manually we have to structure the loops in such a way
        # that the next loop cycles can utilize previous ones
        # I initially scan with
        # for i in range(len(s)):
        #     for j in range((i+1, len(s)+1)):
        # in this case we choose a point and scan till the end but in this way we can't utilize previous results on which
        # substrings are palindromes it is very subtle indeed
        for j in range(1,len(s)+1):
            for i in range(j-1,-1,-1):
                # the first condition j - i <= 1 is for when the substring only has 1 character in it
                # s[i] == s[j-1] and pl[i+1][j-1] == True is a shortcut to see if the substring without the endpoints
                # is already a palindrome
                # the last one j-1-i-1 <= 1 is to cover when the substring is even in length
                if (j - i <= 1 or (s[i] == s[j-1] and (pl[i+1][j-1] == True or j-1-i-1 <= 1))):
                    
                    pl[i][j] = True

                    if (dl[i] + 1 < dl[j]):
                        dl[j] = dl[i] + 1
        
        return dl[len(s)]

    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """

        return self.trydp(s)
 

###############################################################################
# difficult test cases
###############################################################################

# this first one has 2^33 branches in the Palindrome tree
#tc1 = "apjesgpsxoeiokmqmfgvjslcjukbqxpsobyhjpbgdfruqdkeiszrlmtwgfxyfostpqczidfljwfbbrflkgdvtytbgqalguewnhvvmcgxboycffopmtmhtfizxkmeftcucxpobxmelmjtuzigsxnncxpaibgpuijwhankxbplpyejxmrrjgeoevqozwdtgospohznkoyzocjlracchjqnggbfeebmuvbicbvmpuleywrpzwsihivnrwtxcukwplgtobhgxukwrdlszfaiqxwjvrgxnsveedxseeyeykarqnjrtlaliyudpacctzizcftjlunlgnfwcqqxcqikocqffsjyurzwysfjmswvhbrmshjuzsgpwyubtfbnwajuvrfhlccvfwhxfqthkcwhatktymgxostjlztwdxritygbrbibdgkezvzajizxasjnrcjwzdfvdnwwqeyumkamhzoqhnqjfzwzbixclcxqrtniznemxeahfozp"        

# this one has one two palindromes but building the tree takes an enormouse amount of time due to its size
# the main time taken is due to cheking whether a substring is a palindrome using a loop
#tc2 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"        
                          
             
             
             

        
        
             
             