###############################################################################
# 179. Largest Number
###############################################################################

class Solution:
    def findTrouble(self, clm, d):

        for i in range(len(clm)-1):
            if (len(clm[i]) > len(clm[i+1]) and clm[i][:len(clm[i+1])] == clm[i+1]):
                if ((clm[i+1] + clm[i]) > (clm[i] + clm[i+1])):
                    return i
                
        return -1
    
    def largestNumber(self, nums):
        
        # I think we need to group by same starting digit
        
        # process digit by digit starting with the biggest one 9
        r = ""
        
        strnum = map(str, nums)
        nm = ['9', '8' ,'7' ,'6' ,'5' ,'4' ,'3', '2', '1', '0']
        
        for d in nm:
            clm = [w for w in strnum if (w[0] == d)]
            
            # for '0' we just need to cat them all
            if (d > '0'):
                # Python's internal sort already almost got it right the only thing is that when there is something like
                # 391 and 39 they will be sorted as 391, 39 and if you just concatenate them you will get 39139 < 39391
                clm.sort(reverse=True)
                    
                # the thing is that we might have something like 51 51 5 in this case we need to search for 51 5 twice
                # because even after we shuffle 51 5 to 5 51 the overall status is still 51 5 51 which is still not correct
                # so we need a helper function to scan the list again
                j = self.findTrouble(clm, d)
                
                while (j > -1):
                    # we need this loop in case there are multiple items with the same value lie
                    # 51 5 5 5 so we need to shift 51 down multiple times
                    for i in range(j, len(clm)-1):
                    
                        # I initially tried different methods to decide whether to swap or not and each had a hole in the logic
                        # but the simplest actually works just try each combo and see which one works
                        if (len(clm[i]) > len(clm[i+1]) and clm[i][:len(clm[i+1])] == clm[i+1]):
                            if ((clm[i+1] + clm[i]) > (clm[i] + clm[i+1])):
                                tmp = clm[i]
                                clm[i] = clm[i+1]
                                clm[i+1] = tmp
                            else:
                                break
                                
                    # scan again to see if there's still some problem
                    j = self.findTrouble(clm, d)
            
            
            r += "".join(clm)

        # this is to check we return something like "000000" which is not accepted
        if(int(r) == 0):
            return "0"
        
        return r       