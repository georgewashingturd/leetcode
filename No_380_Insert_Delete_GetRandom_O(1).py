###############################################################################
# 380. Insert Delete GetRandom O(1)
###############################################################################

from random import randint

class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.d = {}
        

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        
        if (val in self.d):
            return False
            
        self.d[val] = None
        return True
        

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        
        if (val not in self.d):
            return False
            
        del self.d[val]
        return True
        

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        
        v = self.d.keys()
        ln = len(self.d.keys())
        
        n = randint(0, ln-1)
        
        return v[n]
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()