###############################################################################
# 225. Implement Stack using Queues
###############################################################################

class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.count = 0
        self.q1 = []
        self.q2 = []
        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        
        self.q1.append(x)


    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        tmp = []
        while (len(self.q1) > 1):
            tmp.append(self.q1.pop(0))
        n = self.q1.pop(0)
        self.q1 = tmp
        return n


    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        tmp = []
        while (len(self.q1) > 1):
            tmp.append(self.q1.pop(0))
        n = self.q1.pop(0)
        self.q1 = tmp
        self.q1.append(n)
        return n
        
        
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        
        if (not self.q1):
            return True
            
        return False
        


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()