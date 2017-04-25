###############################################################################
# 295. Find Median from Data Stream
###############################################################################

class MedianFinder(object):
    
    import heapq
    
    # so the idea here is to use two heaps one min heap and one max heap
    # we want to store the bigger half in a min heap so the the top is the lowest of the bigger half
    # and the lower half to be a max heap so that the top is the biggest of the lower half
    # and we want to maintain the size to be equal \pm 1
    # also note that python heapq is only a minheap so to make it a maxheap we need to multiply by -1

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.maxheap = []
        self.maxsize = 0
        
        self.minheap = []
        self.minsize = 0
        
        self.size = 0
        
        # for the very first element just put it in the max heap for now because during the size balancing
        # between the two heaps we might still move things around anyway

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        
        if (not self.size):
            heapq.heappush(self.maxheap, (-num, self.maxsize))
            self.maxsize = 1
            self.size = 1
            return
        
        # note that since the min heap is done by mutiplying -1 we need abs to compare
        if (num <= -self.maxheap[0][0]):
            heapq.heappush(self.maxheap, (-num, self.maxsize))
            self.maxsize += 1
        # minheap might be empty since we always initialize with maxheap
        else:
            heapq.heappush(self.minheap, (num, self.minsize))
            self.minsize += 1
            
        self.size += 1

        # now we check for balance what we want is that maxheap is at most one more than min heap
        # this is our convention
        while (self.maxsize > self.minsize + 1):
            # move one item from maxheap to minheap
            heapq.heappush(self.minheap, ((-1)*heapq.heappop(self.maxheap)[0], self.minsize))
            self.maxsize -= 1
            self.minsize += 1
            
        while (self.minsize > self.maxsize):
            # move one item from minheap to maxheap
            heapq.heappush(self.maxheap, ((-1)*heapq.heappop(self.minheap)[0], self.maxsize))
            self.minsize -= 1
            self.maxsize += 1


    def findMedian(self):
        """
        :rtype: float
        """
        
        if (not self.size):
            return 0.0
        
        if (self.size & 1):
            return -self.maxheap[0][0]/1.0
            
        return (-self.maxheap[0][0] + self.minheap[0][0])/2.0
        
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()