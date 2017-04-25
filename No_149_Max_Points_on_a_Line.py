###############################################################################
# 149. Max Points on a Line
###############################################################################

# Definition for a point.
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

# note that float alone is not enough precision to keep track of the slopes
# I've tried Fraction and Decimal but they are way too slow so I use gcd instead

class Solution(object):
    # some debugging functions to convert a list of numbers into a list of points
    def toPoints(self, nums):
        
        p = []
        for i in nums:
            p.append(Point(i[0], i[1]))
            
        return p
        

    # the trick for a simpler brute is to reset the dictionary for every outer loop so that
    # you don't need to manage double counting manually
    def simplerBrute(self, points):
        import fractions
        #from fractions import Fraction
        #from decimal import *
        
        ln = len(points)
        
        if (ln <= 2):
            return ln
            
        curr_max = 0
            
        for i in range(ln):
            
            # we reset the hashmap for every point
            d = {}
            
            # counter to check if point[i] is duplicated elsewhere in the following loop
            # note that we only care for point[i] this is the beauty of resetting the hashmap
            # we need not care about other duplicates except for the one we are currently handling
            csame = 0
            
            for j in range(ln):
                if (i <> j):
                    sameFlag = False
                    # first check if the this point is the same as point[i], we only care if point[i] is duplicated
                    if (points[i].x == points[j].x and points[i].y == points[j].y):
                        csame += 1
                        sameFlag = True
                        
                    # there is a potential problem here since same point and same x coordinate will trigger simultenously
                    # below we add d['inf'] = 2 or += 1 if we find a point where the x coordinate is the same as point[i]
                    # and at the end of the j loop we will add c same to the total number of points in this case we will be
                    # double counting and so we need to take a note of it
                        
                    # now calculate the slope and this is another beauty of resetting the dict for every point
                    # only the slope matters we don't need to calculate the constant factor why?
                    # because all these lines are with respect to point[i] only so if we have two line segments
                    # AB and BC (and the two share the same point B) have the same slope and since they share the
                    # same point B they must be on the same line so we just need to only care about the slope
                    # and on top of that we can just use float instead of fractions
                    
                    dy = points[i].y - points[j].y
                    dx = points[i].x - points[j].x
                    
                    if (dx == 0):
                        slope = float('inf')
                    else:
                        g = fractions.gcd(dy,dx)
                        #slope = Decimal(dy, dx)
                        slope = (dy/g,dx/g)
                        
                    if (slope in d and sameFlag == False):
                        d[slope] += 1
                    else:
                        # there must be at least 2 points to start a line except that in the case of same points
                        if (sameFlag == False):
                            d[slope] = 2
                        else:
                            d[slope] = 1
                    
                    #print d
            # now we can extract the max number of points for the line emanating from point[i]
            # that has the max number of points
            # and we also take this opportunity to update curr_max
            
            dv = d.values()
            #print dv, " ~~~ ", csame
            
            # note that max will return an exception if the list is empty
            curr_max = max(curr_max, max(dv) + csame)
        
        return curr_max
        
    # note that I initially use Fraction to keep track of the slopes but it is way too slow and it keeps giving me
    # TLE (Time Limit Exceeded) and only until I changed it into manually computing the gcd and manually keeping
    # the numerator and denominator that it was fast enough to compete
    def fasterBrute(self, points):
        import fractions
        
        # we assume there is no rounding error LOL as I don't know whether the points all have integer coordinates or not
        # we gather the slopes between any two points and then tabulate them using a dict but two lines
        # can have the same slope even though they are different lines so we can't just rely on the slopes
        # and we have to take care of duplicate points as well where in this case we just increase everyone's count by one since we are calculating every line between two points
        #y1 = (y1-y2)/(x1-x2) x1 + c
        #c = (y1(x1-x2) - (y1 - y2)x1)/(x1-x2)  
        
        ln = len(points)
        
        if (ln <= 2):
            return ln

        # first gather all the points we want to know if some of them are duplicates
        dp = {}
        for p in points:
            tp = (p.x, p.y)
            if (tp in dp):
                dp[tp] += 1
            else:
                dp[tp] = 1

        # now we gather the points within each line
        # note that if we just use float or double there was a rounding error causing a wrong result
        # the purpose of the valus being a dictionary is we want to keep track which point has been added 
        # to which line as we don't want to double count the same point
        # double counting happens in the subsequent loops not in the current one for example if points are
        # [[1,1],[1,1],[2,2],[2,2]] when we are looping over the lines for the second [1,1] we count the second
        # [2,2] again but they are all on the same line
        
        d = {}
        lp = dp.keys()
        lnp = len(lp)

        for i in range(lnp-1):
            for j in range(i+1, lnp):
                dy = lp[j][1]-lp[i][1]
                dx = lp[j][0]-lp[i][0]
                if (dx != 0):
                    g = fractions.gcd(dy,dx)
                    slope = (dy/g,dx/g)
                    
                    cn = lp[i][1]*dx - lp[i][0]*dy
                    cd = dx
                    g = fractions.gcd(cn,cd)
                    
                    c = (cn/g,cd/g)

                    key = (slope, c, 0)
                else:
                    slope = lp[i][0]
                    c = 0
                    key = (slope, c, 1)

                if (key not in d):
                    # need at least two points to be on a line so start with two
                    d[key] = {lp[i]:None, lp[j]:None}
                else:
                    d[key].setdefault(lp[j], None)
                    
        curr_max = 0
        
        dv = d.values()
        
        # this means there's only one point but there might be multiple of them
        if (not dv):
            return dp.values()[0]
        
        for dl in d.values():
            t = 0
            for m in dl.keys():
                t += dp[m]
            if (t > curr_max):
                curr_max = t
                
        return curr_max
        
        
    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        
        #return self.simplerBrute(points)
        return self.fasterBrute(points)
        
        
                
# difficult test cases

# this will break if you just use float
#tc1 = [[0,0],[94911151,94911150],[94911152,94911151]]

# these two will exceed time limit if you use Fraction or Decimal
#tc2 = [[29,87],[145,227],[400,84],[800,179],[60,950],[560,122],[-6,5],[-87,-53],[-64,-118],[-204,-388],[720,160],[-232,-228],[-72,-135],[-102,-163],[-68,-88],[-116,-95],[-34,-13],[170,437],[40,103],[0,-38],[-10,-7],[-36,-114],[238,587],[-340,-140],[-7,2],[36,586],[60,950],[-42,-597],[-4,-6],[0,18],[36,586],[18,0],[-720,-182],[240,46],[5,-6],[261,367],[-203,-193],[240,46],[400,84],[72,114],[0,62],[-42,-597],[-170,-76],[-174,-158],[68,212],[-480,-125],[5,-6],[0,-38],[174,262],[34,137],[-232,-187],[-232,-228],[232,332],[-64,-118],[-240,-68],[272,662],[-40,-67],[203,158],[-203,-164],[272,662],[56,137],[4,-1],[-18,-233],[240,46],[-3,2],[640,141],[-480,-125],[-29,17],[-64,-118],[800,179],[-56,-101],[36,586],[-64,-118],[-87,-53],[-29,17],[320,65],[7,5],[40,103],[136,362],[-320,-87],[-5,5],[-340,-688],[-232,-228],[9,1],[-27,-95],[7,-5],[58,122],[48,120],[8,35],[-272,-538],[34,137],[-800,-201],[-68,-88],[29,87],[160,27],[72,171],[261,367],[-56,-101],[-9,-2],[0,52],[-6,-7],[170,437],[-261,-210],[-48,-84],[-63,-171],[-24,-33],[-68,-88],[-204,-388],[40,103],[34,137],[-204,-388],[-400,-106]]
#tc3 = [[560,248],[0,16],[30,250],[950,187],[630,277],[950,187],[-212,-268],[-287,-222],[53,37],[-280,-100],[-1,-14],[-5,4],[-35,-387],[-95,11],[-70,-13],[-700,-274],[-95,11],[-2,-33],[3,62],[-4,-47],[106,98],[-7,-65],[-8,-71],[-8,-147],[5,5],[-5,-90],[-420,-158],[-420,-158],[-350,-129],[-475,-53],[-4,-47],[-380,-37],[0,-24],[35,299],[-8,-71],[-2,-6],[8,25],[6,13],[-106,-146],[53,37],[-7,-128],[-5,-1],[-318,-390],[-15,-191],[-665,-85],[318,342],[7,138],[-570,-69],[-9,-4],[0,-9],[1,-7],[-51,23],[4,1],[-7,5],[-280,-100],[700,306],[0,-23],[-7,-4],[-246,-184],[350,161],[-424,-512],[35,299],[0,-24],[-140,-42],[-760,-101],[-9,-9],[140,74],[-285,-21],[-350,-129],[-6,9],[-630,-245],[700,306],[1,-17],[0,16],[-70,-13],[1,24],[-328,-260],[-34,26],[7,-5],[-371,-451],[-570,-69],[0,27],[-7,-65],[-9,-166],[-475,-53],[-68,20],[210,103],[700,306],[7,-6],[-3,-52],[-106,-146],[560,248],[10,6],[6,119],[0,2],[-41,6],[7,19],[30,250]]
