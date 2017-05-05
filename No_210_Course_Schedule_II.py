###############################################################################
# 210. Course Schedule II
###############################################################################

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        
        # the only way there's no way to get to every course is if there's a cycle
        # this is the only way the path is impossible
        
        # we can start with any course that has no prerequisite
        # so first is to scan courses with no prerequisites
        
        # build the prereq dict from prerequisites
        # key means this course has prerequisites
        
        d = {}
        # this is a directed graph so we have to be careful whose neighbor is whose
        neighbor = {}
        
        for i in prerequisites:
            if (i[0] not in d):
                d[i[0]] = {i[1]:None}
            else:
                d[i[0]][i[1]] = None
                
            neighbor.setdefault(i[1], []).append(i[0])
            #d.setdefault(i[0], []).append(i[1])
        
        # now do a BFS on those who have not prerequisites
        
        q = []
        for i in range(numCourses):
            if (i not in d):
                q.append(i)
             
        res = []   
        
        while(q):
            n = q.pop(0)
            
            res.append(n)
            numCourses -= 1
            
            # we take course n so we can eliminate n from the prereq dict
            # this dict search is too slow let's try changing it from a list into a dict
            # but it's still too slow so I guess we need to build a neighbor dict or list
            
            # if n is a prereq
            if (n in neighbor):
                for c in neighbor[n]:
                #for c in d.keys():
                    if (n in d[c]):
                        del d[c][n]
                        # if because of this c no longer has prerequisite add c to the queue
                        if (not d[c]):
                            q.append(c)
                            del d[c]
        
        # need to check if there is a path to the last course
        if (numCourses == 0):
            return res
        return []
        
        