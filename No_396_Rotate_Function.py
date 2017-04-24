###############################################################################
# 396. Rotate Function
###############################################################################

class Solution(object):

    def brute(self, A):
        ln = len(A)
        
        F_0 = sum([A[i]*i for i in range(len(A))])
        print(F_0)
        
        for i in range(1,ln):
            R = A[-i:] + A[:-i]
            F_i = sum([R[l]*l for l in range(len(R))])
            print(F_i)
        
        print("===")
    

    def maxRotateFunction(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        
        # so my first idea is to find the formula for F but it doesn't seem obvious
        # the next best thing I could come up with is to get F(1) from F(0) and I think it can be generated quite easily
        # Like for the example above from F(0) to F(1) we add
        # 4 + 3 + 2 + (-3)*6
        # and from F(1) to F(2) we just add
        # 4 + 3 + (-3)*2 + 6
        # and from F(2) to F(3) we just add water LOL
        # 4 + (-3)*3 + 2 + 6
        
        # so basically we are just adding the total of the array except for one element and 
        # that element starts at the end and ends at the second element and it is also multiplied
        # by - (len(A)-1)
        
        # first build F(0) python style
        
        F_0 = sum([A[i]*i for i in range(len(A))])
        curr_max = F_0
        
        # we first initialize delta to the sum of all elements
        delta = sum(A)
        ln = len(A)

        for i in range(ln-1, 0, -1):
            d = (delta) - (ln)*A[i]
            curr_max = max(curr_max, F_0 + d)
            # don't forget to update F_0 itself as what we are doing is just adding the delta
            # at each step
            F_0 += d
            
        return curr_max