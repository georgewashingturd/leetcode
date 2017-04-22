###############################################################################
# 146. LRU Cache
###############################################################################

class Node(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None
        
class DoublyLL(object):
    def __init__(self):
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.count = 0
        
    def InsertFirst(self, node):
        if (node is None):
            return
        node.next = self.head.next
        node.prev = self.head
        self.head.next = node
        node.next.prev = node
        self.count += 1
        
    def RemoveTail(self):
        if (self.tail.prev is not self.head):
            node = self.tail.prev
            node.prev.next = self.tail
            self.tail.prev = node.prev
            self.count -= 1
            return node
    
    def Remove(self, node):
        if (node is None):
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.count -= 1
    

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.l = DoublyLL() # to maintain which key is recently used, it contains only key
        self.d = {} # to speed up look up contains key value pairs
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        try:
            n = self.d[key]
        except KeyError:
            return -1
        # at this point key is guaranteed to exist
        self.l.Remove(n)
        self.l.InsertFirst(n)
        
        return n.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        
        # first check if the key already exists
        
        if (key in self.d):
            node = self.d[key]
            self.l.Remove(node)
            self.l.InsertFirst(node)
            node.value = value
        else:
            if (self.l.count >= self.capacity):
                node = self.l.RemoveTail()
                del self.d[node.key]
                
            node=Node(key,value)
            self.l.InsertFirst(node)
            self.d[key]=node
        
        

        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)      
        
        
        