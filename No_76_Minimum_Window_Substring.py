###############################################################################
# 76. Minimum Window Substring
###############################################################################

class Solution(object):
    
    # This is a brute force method and it is very slow
    def slowBrute(self, s, t):
        ln = len(s)
        
        # first build a dict with locations for each char of t but the characters 
        # might repeat and we have to keep track of it
        dt = {}
        for i in t:
            if (i in dt):
                dt[i] += 1
            else:
                dt[i] = 1

        ds = {}
        for i in range(ln):
            if (s[i] in dt):
                ds.setdefault(s[i], []).append(i)
                
        # now we scan from the right until we cannot scan anymore
        # i.e. we can no longer find chars of t in s
        
        if (len(ds.keys()) < len(dt.keys())):
            return ""
            
        
        for n in ds.keys():
            if (len(ds[n]) < dt[n]):
                return ""

        # this is how you deep copy a dictionary
        #dtmp = {key: value[:] for key, value in ds.items()}
        
        cmin = float('inf')
        ts = (0,0)
        
        # create another dict for min index to speed up search
        dy = dict.fromkeys(list(t), 0)

        for i in range(ln):
            if (s[i] in ds):
                #print i
                # we see the min window needed if we start from here
                cni = dt[s[i]]
                
                
                # but we need to know that we have enough char s[i] to complete t
                
                # to save time we don't use index function
                #ind_i = ds[s[i]].index(i)
                if (dy[s[i]] <> -1):
                    ind_i = dy[s[i]]
                            
                    lsi = len(ds[s[i]])
                    while (ind_i < lsi and ds[s[i]][ind_i] < i):
                        ind_i += 1
                        
                    #assert ds[s[i]][ind_i] == i
                    
                    dy[s[i]] = ind_i     
                else:
                    ind_i = len(ds[s[i]])

                if (ind_i + cni > len(ds[s[i]])):
                    return s[ts[0]:ts[1]]
                
                # so start must be from i
                st = i
                
                # if we have enough char left
                ed = ds[s[i]][ind_i + cni - 1] # this is a potential end point
                
                # now we check other char
                for n in dt.keys():
                    if (dy[n] == -1):
                        ed = -1
                        break
                    if (n <> s[i]):
                        # count the number of char needed for this char
                        cn = dt[n]
                        
                        # find a location that makes sense i.e. beyond start
                        y = dy[n]
                        
                        lsn = len(ds[n])
                        while (y < lsn and ds[n][y] < st):
                            y += 1
                        
                        dy[n] = y
                        
                        # either we don't have enough char or all the positions are not good
                        if (y + cn - 1 >= lsn):
                            # one of the characters doesn't work so we must say that starting from location i is a bust
                            dy[n] = -1
                            ed = -1
                            break

                        # if the position is good update ed                        
                        ed = max (ed, ds[n][y + cn - 1])
                        
                if (ed <> -1):
                    clen = ed + 1 - st
                    if (clen < cmin):
                        cmin = clen
                        ts = (st, ed+1)
                    
        return s[ts[0]:ts[1]]

    
    # This is a much faster way of sliding window and finding minimum the code is a tad ugly but who cares, I'm really tired
    # but the logic is very clear first from the beginning of s we find a window that contains t
    # we then start moving the beginning of the window if by moving its starting point
    # we lose a character of t we move the end to recoup, while we are doing this we take note of which
    # window is shortest if moving the beginning doesn't make us lose any character of t we keep moving it
    
    def uglyButFast(self, s, t):
        
        # I will try another method by sliding a window
        # so first find a character that's in t and then find a window starting at that location
        # that contains t
        
        # first build a dict with locations for each char of t but the characters 
        # might repeat and we have to keep track of it
        dt = {}
        for i in t:
            if (i in dt):
                dt[i] += 1
            else:
                dt[i] = 1
        
             
        ls = len(s)
        lt = len(t)
        
        # tuple for the start and end of window
        ts = (0,0)
        
        # the current min window length
        cmin = float('inf')
        
        # start of window
        st = ls
        
        # find the beginning of window
        i = 0
        while (i < ls and s[i] not in dt):
            i += 1
            
        # not a single character of t is found in s
        if (i >= ls):
            return ""
            
        # found the beginning of window
        st = i
        
        # now find the end of the window
        # but while finding the window we might encounter some extra characters from t
        # we keep them for when we want to shorten the window
        de = {key: 0 for key in dt.keys()}
        
        j = i
        while (j < ls and lt > 0):
            if (s[j] in dt):                       
                if (dt[s[j]] > 0):
                    dt[s[j]] -= 1
                    lt -= 1
                else:
                    de[s[j]] += 1
            j += 1
        
        # j is one more than the inclusive end so we need to subtract 1 from it    
        # if lt is zero it means that we've found all characters from t               
        if (lt == 0):
            ed = j - 1
            
            clen = ed + 1 - st
            if (clen < cmin):
                cmin = clen
                ts = (st, ed + 1)
        else:
            return ""

            
        # now we start moving st and see how much ed much move and see if we can find a shorter window
        
        while (st < ls - lt):   
            # mc stands for missing character
            mc = s[st]
            
            # reduce the number of extra for mc
            de[s[st]] -= 1
            
            # we are really missing s[st] from our window now so we must extend the end of the window
            if (de[s[st]] < 0):
                ed += 1
                
                while (ed < ls and s[ed] != mc):
                    # we might gather other members of t so keep track of them
                    if (s[ed] in dt):
                        de[s[ed]] += 1
                    ed += 1
                
                if (ed >= ls):
                    return s[ts[0]:ts[1]]
                
                # s[ed] is mc and we've added to our window so make sure its extra count is updated
                de[s[ed]] += 1
                
                st += 1
                # make sure the beginning is part of t
                while (s[st] not in dt):
                    st += 1
                    
                clen = ed + 1 - st
                if (clen < cmin):
                    cmin = clen
                    ts = (st, ed+1)

            else:    
                # we only removed an extra so we can go on removing other characters and shorten the window
                # by moving st forward
                st += 1
                while (st < ls and s[st] not in dt):
                    st += 1
                
                # we might find a shorter window here
                if (st < ls):
                    clen = ed + 1 - st
                    if (clen < cmin):
                        cmin = clen
                        ts = (st, ed+1)
                
        return s[ts[0]:ts[1]]
        
        
        
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        
        #return self.slowBrute(s,t)
        return self.uglyButFast(s,t)