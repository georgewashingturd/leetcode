###############################################################################
# 413. Arithmetic Slices
###############################################################################

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        
        # first thought was using a sliding window next maybe dp but dp might not work unless
        # you can tell which one is of the same form for example if we have  1, 2, 3, 5, 7
        # we have 2 slices if we add 9 to the end of the array we will have 2 extra slices from
        # 3, 5, 7 as 3, 5, 7, 9 has a total of three but 1, 2, 3 will not have any additional slice
        
        # while if we have 1,2,3,4 adding 5 will include all previous slices with an addition of 3,4,5
        # as a new slice
        
        # so maybe we can do something like this
        # scan the array to calculate the difference between two numbers
        # if the difference changes then we have a new slice
        # for the subsequence with a constant difference the number of slices is
        # n - (3 - 1) + n - (4 - 1) + n - (5 - 1) + ... + n - (n - 1)
        # = n - 2 + n - 3 + n - 4 + n - 5 + ... + n - (n - 1)
        # = n(n-2) - ((n-1)(n)/2 - 1) since there are (n-2) number of n's and the other term is gauss sum
        # = (n^2 - 3n + 2)/2
        # n-(3-1) is the number of slices with length 3 and so on
        
        ln = len(A)
        if (ln < 3):
            return 0

        # first order of business is to find the first slice
        st = -1
        for i in range(2, ln):
            if (A[i] - A[i-1] == A[i-1] - A[i-2]):
                # found a new sequence
                st = i-2
                ed = i
                prev = A[i] - A[i-1]
                break
        
        if (st == -1):
            return 0
            
        # n is curr len
        n = 3
        count  = 0
        i = ed+1
        while (i < ln):
            if (A[i] - A[ed] == prev):
                n += 1
                ed = i
                i += 1
            # end of this particular sequence need to find a new one
            else:
                count += (n**2 - 3*n + 2)//2
                
                # find next slice
                st = -1
                for j in range(ed+2, ln):
                    if (A[j] - A[j-1] == A[j-1] - A[j-2]):
                        st = j - 2
                        ed = j
                        n = 3
                        prev = A[j] - A[j-1]
                        break
                # no more slices available
                if (st == -1):
                    return count
                i = ed + 1
        
        # don't forget to add the last one
        count += (n**2 - 3*n + 2)//2
                
        return count
        
        
        