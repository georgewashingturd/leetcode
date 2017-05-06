###############################################################################
# 207. Course Schedule
###############################################################################

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        
        # looks like the idea is very similar to minimum height tree
        # we want to remove the leaves and keep removing leaves until we have
        # no more leaves if so then we are successful
        
        # the only way we cannot finish all courses is if there are cycles
        
        if (numCourses <= 1):
            return True
            
        # build the dict, keys are the courses and values are the prereq
        d = {}
        neighbor = {}
        for i in prerequisites:
            d.setdefault(i[0], []). append(i[1])
            neighbor.setdefault(i[1], []).append(i[0])
            
        # gather leaves, courses with no prereq
        q = []
        for i in range(numCourses):
            if (i not in d):
                q.append(i)
            
        while(q):
            n = q.pop(0)
            numCourses -= 1
            
            # see if n is a prereq to some other course
            if (n in neighbor):
                for c in neighbor[n]:
                    # we remove this prereq
                    d[c].remove(n)
                    # if c now becomes a leaf add it to the queue
                    if (not d[c]):
                        q.append(c)
                        del d[c]
            
        if (numCourses == 0):
            return True
            
        return False
        