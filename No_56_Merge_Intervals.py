###############################################################################
# 56. Merge Intervals
###############################################################################

# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        
        if (not intervals):
            return []
            
        ln = len(intervals)
        
        if (ln == 1):
            return intervals
        
        # first thing to do is sort the intervals
        
        intervals.sort(key=lambda interval: interval.start) 
        
        # now go through the intervals one by one
        
        st = 0
        i = 1
        res = []
        while (i < ln):
            # we need to merge
            if (intervals[i].start <= intervals[st].end):
                intervals[st].end = max(intervals[i].end, intervals[st].end)
                i += 1
            # we found a separate intervals
            else:
                res.append(intervals[st])
                st = i
                i = i + 1
                
        res.append(intervals[st])
        
        
        return res
        