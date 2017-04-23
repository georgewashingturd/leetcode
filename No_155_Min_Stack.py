###############################################################################
# 155. Min Stack
###############################################################################

# the idea here is that everytime an element is pushed into the stack we keep track of the current min
# since the elements are pushed one at a time and popped one at a time the current min at each instant of time is enough
# to ensure we know the current min even if an element is popped we don't need to keep track of previous min

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stk = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if (not self.stk):
            self.stk.append((x,x))
            return
        
        if (x < self.stk[-1][1]):
            self.stk.append((x,x))
        else:
            self.stk.append((x,self.stk[-1][1]))
        

    def pop(self):
        """
        :rtype: void
        """
        # we need to update min
        n, m = self.stk.pop()
        return n
        

    def top(self):
        """
        :rtype: int
        """
        return self.stk[-1][0]


    def getMin(self):
        """
        :rtype: int
        """
        return self.stk[-1][1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()