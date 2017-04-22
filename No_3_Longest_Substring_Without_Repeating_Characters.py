###############################################################################
# 3. Longest Substring Without Repeating Characters
###############################################################################

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        # try with running pointers and increase the tail, if we hit a repeating character we move the head forward
        # to speed things up we use a dict to see which characters are already in teh substring
        if (not s):
            return 0
        
        # start pointer end pointer and the length of the current substring
        st = 0
        ln = 1
        
        d={s[0]:0}
        
        # current max len of substring
        cm = 0
        
        for ed in range(1,len (s)):
            # check if substring already has this character
            if (s[ed] not in d):
                ln += 1
                d[s[ed]] = ed
            # reset the substring to include s[ed] but we must get rid of the previous copy
            else:
                
                # record the length of previous substring but this might not be right sine we are not deleting keys anymore
                if (ln > cm):
                    cm = ln

                # reset the dictionary, but take note of where the other copy of s[ed] is
                #p = d[s[ed]] + 1
                #for i in range(st,d[s[ed]]+1):
                    #del d[s[i]]
                # reset the starting point  
                #st = p
                
                # a cleverer trick is to not do deletions on the dict but just updating the starting point
                # is current starting point is less than the index of the duplicate character then move starting point forward
                # but since we are not deleting stuffs we might have numerous duplicate notofications but it's ok
                # as long as those duplicates are before st so that's why we are only taking the max
                st = max(st,d[s[ed]]+1)
                
                # now update the dictionary entry
                d[s[ed]] = ed
                
                # and reset the length as well
                ln = ed-st+1

        # return the maximum length
        return max(cm, ln)          