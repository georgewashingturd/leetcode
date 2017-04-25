def fm(a):
    s = 0
    e = len(a)-1
    while (s <= e):
        if (a[s] <= a[e]):
            print "found min %d a[%d] = %d" % (s, s, a[s])
            return s
        else:
            mid = (s+e)//2
            # print "mid %d %d %d %d" % (mid, a[s], a[mid], a[e])
            if (a[s] > a[mid]):
                e = mid
            elif (a[mid] > a[e]):
                s = mid + 1
                
                

def roro(a,n):
    return a[n % len(a):] + a[:n % len(a)]
    
def rere(a,n):
    return a[-n % len(a):] + a[:-n % len(a)]
    
    
def shsh(s):

    d = {}
    a = []
    for i in s:
        d.setdefault(i,1)
        a.append(i)
        
    lm = []
    for i in a:
        ll = 1
        j = i + 1
        while(d.has_key(j) == True):
            ll += 1
            del d[j]
            a.remove(j)
            j += 1
        j = i - 1
        while(d.has_key(j) == True):
            ll += 1
            del d[j]
            a.remove(j)
            j -= 1
        lm.append(ll)
    print "ma %d" % max(lm)
    
    
# binary tree implemented as array

# first task print levels

# so we do a breadth first search with a dummy node to indicate that one level is already processed

def BFSM(a):
    # first check if the tree is empty
    if (len(a) == 0):
        return
    v = [] # first create an empty list to be our queue
    # next put the head in that queue
    v.append(a[0])
    
    #next append the dummy node to indicate that we have finished one level
    v.append('d')
    print v
    while (len(v) > 0):
        # we pop first item in v
        n = v.pop(0)

        # if we see a dummy node we need to add a dummy node to indicate a level completion
        if (n == 'd'):
            if (len(v) > 0):
                v.append('d')
                print "====="
        else:
        # here we add all the children of n into the queue since this is a binary tree we only need to add 2 of them
        # but we need to be careful and check if we hit the end of the tree
        # first we print each level
            print n
            li = (2*a.index(n) + 1)
            ri = (2*a.index(n) + 2)
            if (li < len(a) and a[li] <> -1):
                v.append(a[li])
            if (ri < len(a) and a[ri] <> -1):
                v.append(a[ri])

        #input("press enter")

def DFS(a):
    # first check if the tree is empty
    if (len(a) == 0):
        return
    # if it is not empty create a stack
    v = []
    # add the root to the stack
    v.append(a[0])
    
    while (len(v) > 0):
        # first, pop one item and display it
        n = v.pop()
        print n
        
        # now insert its children to the stack, we add the right children first because we want to print left to right and this is a stack
        li = (2*a.index(n) + 1)
        ri = (2*a.index(n) + 2)
        if (ri < len(a) and a[ri] <> -1):
            v.append(a[ri])
        if (li < len(a) and a[li] <> -1):
            v.append(a[li])
        

# second task, find max sum in one branch

class Node:
    def __init__(self, v):
        self.left = None
        self.right = None
        self.value = v

    
# convert array to a tree so that we can do recursive modified DFS
def catt(a):
    # here we do not need to do any BFS or DFS we just need a loop, a loop through the list
    # but first we need to create a list of Nodes to make it easier
    
    # first check if a is empty
    if (len(a) == 0):
        return None
    
    # first prepare an empty list
    nl = []
    
    for i in a:
        # first create the node object, then add it to the list
        # but check if it is a valid node
        if (i <> -1):
            n = Node(i)
            nl.append(n)
        
    for i in range(len(nl)):
        li = 2*i + 1
        ri = 2*i + 2
        if (li < len(nl)):
            nl[i].left = nl[li]
        if (ri < len(nl)):
            nl[i].right = nl[ri]
            
    return nl[0]
        

# to display if we ge the conversion right        
def BSMT(a):
    # here a is the root of the tree
    # first check if it is empty
    
    if (a is None):
        return
        
    # next create an empty queue and append the root
    v = []
    v.append(a)
    
    while (len(v) > 0):
        # pop the head, since this is a queue
        n = v.pop(0)
        print n.value
        
        # now add the children of n
        if (n.left is not None):
            v.append(n.left)
        if (n.right is not None):
            v.append(n.right)

# find max sum in one branch
def DFSM(a):
    # first check if we have met the stopping condition
    if (a is None):
        return 0
    if (a.left is None and a.right is None):
        return a.value
        
    # this is the recursive part
    psl = 0
    psr = 0
    if (a.left is not None):
        psl = a.value + DFSM(a.left)
    if (a.right is not None):
        psr = a.value + DFSM(a.right)
        
    return max(psl, psr)
    
    
# next task, fidn the longest consecutive characters

st = "fffasjkdfhakshaaaaaaaaaooji2wj3rhjkfkngfppppppppppppppppppp84jkjksgh"
    
# we loop through the string one character at a time and check if the next character is the same as the previous one

def gl(a):
    # first check if it is an empty string
    if (len(a) == 0):
        return None
        
    if (len(a) == 1):
        l = []
        l.append(a[0])
        l.append(1)
        return l
        
        
    prev = a[0]
    # set the current max char
    cm = a[0]
    lm = 0
    l = 1
    
    for i in range(1,len(a)):
        if (a[i] == prev):
            l += 1
        else:
            if (l > lm):
                lm = l
                cm = a[i-1]
            l = 1
            prev = a[i]
    
    if (l > lm):
        lm = l
        cm = a[i-1]
    
    li=[]
    li.append(cm)
    li.append(lm)
    
    return li
    
    

# task, find is an element is in a tree

def iat(a, n):

    if (a is None):
        return False
    # an empty node is a member of everything
    if (n is None):
        return True
    
    if (a.value == n):
        return True
    
    # first search left subtree
    if (a.left is not None):
        if (iat(a.left, n) == True):
            return True
    # if it's not on the left subtree we check the right subtree
    if (a.right is not None):
        if (iat(a.left, n) == True):
            return True
    
    # this only means that it's not on the right nor on the left subtree
    return False


# next task, find the first common ancestor

def ffca(a,n,m):
    # first check if a is empty
    if (a is None):
        return None
        
    # but this doesn't work if n is the ancestor of m or vice versa
    if (iat(a.left, n) == True and iat(a.right, m) == True):
        return a
        
    if (a.value == n and (iat(a.left, m) == True or iat(a.right, m) == True)):
        return a

    if (a.value == m and (iat(a.left, n) == True or iat(a.right, n) == True)):
        return a

    if (a.left is not None):
        al = ffca(a.left,n,m)
        if (al is not None): 
            return al

    if (a.right is not None):
        ar = ffca(a.right,n,m)
        if (ar is not None): 
            return ar
        
    return None
        
        
# today's task, find the longest connected 1's in a matrix, a neighbor can be left right up down or diagonal
# my first thought is to do a BFS since the neighbors of an element

# first create a matrix
# n is the size of the matrix, i.e. nr x nc
def cm(nr, nc):
    return [[0]*nc for i in range(nr)]

# next make a function to get neighbors of an element
def gn(m,n, mr, mc):
    if (m < 0 or n < 0):
        return None
        
    l = []
    for x in range(n-1,n+1+1):
        for y in range(m-1,m+1+1):
            if ((x >= 0 and y >= 0) and (x < mc and y < mr) and (y <> m or x <> n)):
                l.append([y,x])
                
    return l
    
    
# now we are ready to do the BFS
def flp(a, mr, mc):
    # first create a queue for the BFS
    
    # next create the visited matrix
    vm = [[0]*mc for i in range(mr)]
    lmax = 0
    
    # we need to loop through the whole matrix here
    # separate 1's mean that they are separate trees
    # so we need to do BFS on each one of them
    for r in range(mr):
        for c in range(mc):
            # need to check visited matrix to see if we have visited this tree
            if (a[r][c] <> 0 and vm[r][c] == 0):
                # we found a new tree so start BFS for this tree
                # set visited to 1
                vm[r][c] = 1
                
                # add this tree/graph into the queue but we also have to count its length
                v = []
                v.append([r,c])
                
                ln = 0
                
                while(len(v) > 0):
                    n = v.pop(0)
                    ln += 1
                    nb = gn(n[0],n[1], mr, mc)
                    
                    for g in nb:
                        if (a[g[0]][g[1]] == 1 and vm[g[0]][g[1]] == 0):
                            v.append(g)
                            vm[g[0]][g[1]] = 1
                
                # done with this tree/graph check if we have the longest path
                
                if (ln > lmax):
                    lmax = ln
    return lmax
                    
# somehow pprint doesn't work so
def dt(a):
    for i in a:
        print i
    print " "
        
        
# next task, place 8 queens in a chess board so that none of them are on the same column or the same row


def qc(ql, qm = 3, l = []):
    # for any recursive function we need a stopping condition
    if (ql == 0):
        m = cm(qm, qm)
        for o in l:
            m[o[0]][o[1]] = 1
            
        dt(m)
        return

    # if ql (queens left) is not zero then we try to place it somewhere
    # we figure out the row for it based on ql, ql == 8 means row 0, so row is 8 - ql
    # then we choose the column, to check which row, column is avaiable we check the list l
    # for row 0 l is always reset
    for col in range(qm):
        row = qm - ql
        ll = [row, col]
        # we need to check if it is in l, if it is not put it in l
        # and recursively call for the remaining queens
        occ = False
        for o in l:
            if (o[1] == col):
                occ = True
        if (occ == False):
            l.append(ll)
            qc(ql-1, qm, l)            
            l.remove(ll)
            
            
# coins' assortments, the coins are 2, 3, 5 all primes

## itas, is there a solution
#def itas(d, n, c):
#    for i in range(n,0,-1):
#        if (d[i] <> [-1,-1,-1]):
#            # this means that we have a solution for i-1
#            # so we just need to know if we can make a solution for n
#            # if d[i] has no solution this means that we cannot build on top of it this is the improtant fact we need
#            # if d[i] has a solution, meaning i has a solution we need to see if we can build on top of it
#            # and we don't need to take care of the cases of multiple coins, for example if we consider the case of 8
#            # and we want to know if we can use 2 we only need to consider the case for 6 only, we don't need to consider the case of 4
#            # because if we use 4 we need to add 2 2's, but 6 already includes that but this is not clear from the outset
            
    

#def cca(m):
#    # we will try to use dynamic programming for this
#    # we will need a helper function to see if we can find a solution 
#    # 0 total always has a solution so we can always use this
    
#    # first set solution 0 as good
#    d = []
#    d.append([0,0,0])
    
#    ns = [-1,-1,-1]
#    c = [2,3,5]
    
#    for n in range(1,m+1):
#        itas()
     




# Distinct Subsequence

def ds(s, t, si, ti):
    if (not s or not t):
        return 0
    
    if (len(t) > len(s)):
        return 0
    
    if (ti > len(t)-1):
        return 0
    
    c = 0
    for i in range(si,len(s) - (len(t) - ti - 1)):
        
        if (s[i] == t[ti]):
            if (ti == len(t)-1):
                return 1
            else:
                c += ds(s, t, i+1, ti+1)
            
        
    return c
        
    
# how to check if binary tree is correct

def cibtic(bt):
    if (not bt):
        return True
        
    if (bt.left <> None):
        if (get_max(bt.left) > bt.value):
            return False
        return cibtic(bt.left)

    elif (bt.right <> None):
        if (get_min(bt.right) < bt.value):
            return False
        return cibtic(bt.right)
    else:
        return True
        
        
        
        
        
        
        
        
        
        
        
        
        
#################################################################################
# LRU Cache
#################################################################################

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
        
############################################################################
#
############################################################################

# class Solution(object):
    # def maxSlidingWindow(self, nums, k):
        # """
        # :type nums: List[int]
        # :type k: int
        # :rtype: List[int]
        # """
        # if (not nums):
            # return []
            
        # l = []
        # for i in range(len(nums)-k+1):
            # l.append(max(nums[i:i+k]))
            
        # return l        
        
class MaxSlidingWindowSolution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if (not nums):
            return []

        lm = []
        l = []            
        
        # the outer loop starts with k-1
        for i in range(k-1,len(nums)):
        
            # first check the dequeue
            # if the dequeue is empty then we need select a max from the window
            # and fill up the deque with the window
            if (not l):
                for j in range(i-k+1,i+1):
                    l.append([nums[j],j])
                l.sort()
                l.reverse()
                
                lm.append(l[0][0])
            else:
            # if the deque is not empty 
            # add the new element in the dequeue in the correct place
            # process the queue, remove any item less than the new element
            # remove any element too far away
                j = 0
                while (j < len(l) and l[j][0] > nums[i]):
                    j += 1

                l.insert(j,[nums[i],i])

                j += 1
                l[j:] = []
                    
                j = 0
                while (j < len(l)):
                    if (l[j][1] < i - k + 1):
                        l.pop(j)
                    else:
                        j += 1
                        
                lm.append(l[0][0])
            
        return lm
        
        
class TwoSumSolution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        if (not nums):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = {}
        
        # add all numbers into the dictionary
        
        for i in nums:
            if (i in d):
                n = d[i] + 1
                d[i] = n
            else:
                d.setdefault(i, 1)
        
        for i in d:
            n = target - i
            
            if (n == i):
                if (d[i] > 1):
                    j = nums.index(i)
                    return [nums.index(i), nums.index(n, j+1)]
            else:        
                if (n in d):
                    return [nums.index(i), nums.index(n)]
                
                
        return []
        
class ValidAnagramSolution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if (len(s) != len(t)):
            return False
        
        # again I will use hashmap here and see if it works :)
        
        # prepare hashmap
        ds = {}
        dt = {}
        
        for i in s:
            if (i in ds):
                n = ds[i] + 1
                ds[i] = n
            else:
                ds.setdefault(i,1)
                
        for i in t:
            if (i in dt):
                n = dt[i] + 1
                dt[i] = n
            else:
                dt.setdefault(i,1)
                
        return ds==dt        

        
class ProdArraySolution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        if (not nums):
            return []
        
        # since I cannot use division I will use repeated additions :)
        
        tot = 1
        
        # this is to track zeros
        zc = 0
        
        for i in nums:
            if (i == 0):
                zc += 1
            else:
                tot = tot * i
            
        if (zc > 1):
            return [0] * len(nums)
            
        l = []
        # now do the repeated sums as an alternative to division :)
        for i in nums:
            if (zc > 0):
                if (i == 0):
                    l.append(tot)
                else:
                    l.append(0)
            else:
                j = abs(i)
                ntot = abs(tot)
                if ((i < 0 and tot > 0) or (i > 0 and tot < 0)):
                    s = -1
                else:
                    s = 1
                    
                if (j == 1):
                    l.append(ntot*s)
                else:
                    
                    ii = abs(i)
                    p = 1
                    while(j < ntot):
                        pp = p
                        p = j
                        j *= ii
                        
                    # print "%d %d %d    %d" % (pp, j, ntot, pp*j)

                    if (j > ntot):
                        k = pp
                        m = 2
                        w = 1
                        
                        while (k*ii <> ntot):
                            m = 2 ** (w-1)
                            k += m
                            m = 1
                            
                            w = 0
                            # print "\nstart %d" % k
                            while ((k+m)*ii < ntot):
                                # print m
                                m *= 2
                                w += 1
                            if ((k+m)*ii == ntot):
                                k += m
                            
                            # print "after %d %d %d    %d" % (m, k, ntot, (k+m)*ii)
                            
                        # print (k*s)
                        l.append(k*s)
                    else:
                        # print (p*s)
                        l.append(p*s)
                
        return l

        
class LongestPalSolution(object):
    
    def checkPal(self, a):
#        la = len(a)//2
#        l = list(a[:la])
#        r = list(a[la + (len(a)%2):])
#        r.reverse()
        
#        return l == r
        
        for i in range(len(a)//2):
            if (a[i] <> a[-(i+1)]):
                return False
        return True

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if (not s):
            return ""
            
        ls = len(s)
        ml = 1
        mp = s[0]
            
        for i in range(len(s)):
            # split into odd palindrome and even palindrome
            
            # first odd ones
            maxj = min(i,ls-i-1)
            for j in range(maxj,0,-1):
                sb = s[i-j: i + j + 1]
                ln = 2*j+1
                if (self.checkPal(sb) == True and ln > ml):
                    mp = sb
                    ml = ln
                    break
                
            # next even ones
            maxj = min(i,len(s)-i-2)
            for j in range(maxj, -1, -1):
                sb = s[i-j: i + j + 2]
                ln = 2*j+2
                if (self.checkPal(sb) == True and ln > ml):
                    mp = sb
                    ml = ln
                    break
                    
        return mp
        
# This is actually similar to that of maximum sliding window we need to maintain the maximum within a window but in this case the window is shrinking everytime        
class BuySellStockSolution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        if (not prices or len(prices) <= 1):
            return 0
            
        l = prices[1:]
        l.sort(reverse=True)
        
        # this means that it is monotonically decreasing
        if (l == prices[1:] and prices[0] >= l[0] and len(l) > 1):
            return 0
        
        # current maximum profit
        p = 0
        for i in range(len(prices)):
            cp = l[0] - prices[i]
            if (cp > p):
                p = cp
            if (i > 0):
                l.pop(l.index(prices[i]))
            # print l
            
        return p

        
        
        
        
        
        
        
        
        
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        if (not nums):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = {}
        
        # add all numbers into the dictionary
        
        for i in nums:
            if (i in d):
                n = d[i] + 1
                d[i] = n
            else:
                d.setdefault(i, 1)
        
        for i in d:
            n = target - i
            
            if (n == i):
                if (d[i] > 1):
                    j = nums.index(i)
                    return [nums.index(i), nums.index(n, j+1)]
            else:        
                if (n in d):
                    return [nums.index(i), nums.index(n)]
                
                
        return []        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class ThreeSumSolution(object):
    # need to find a way to find all unique two numbers that add up to target
    def allTwoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # print nums
        if (not nums):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = {}
        
        # add all numbers into the dictionary and because we iterate through nums the indices in tha values are sorted in ascending order
        for i in xrange(len(nums)):
            d.setdefault(nums[i],[]).append(i)

        ans = []
        for i in d.keys():
            n = target - i
            
            # since it's only the sum of two numbers once a number is used it cannot be reused bby another number
            # e.g. say the target is 5, and we found 3 has a partner that sums up to 5, but since it's only a sum of two numbers
            # the partner of 3 is unique, which is 2 and the partner of 2 is also unique, so we can immediately pop these two from the dictionary
            if (n in d):
                if (i <> n):
                    ans.append([i,n])
                    del d[i]
                    del d[n]
                else:
                    # in the case of n == i, d[i] must have length more than 1 otherwise it doesn't have a solution
                    if (len(d[i]) >= 2):
                        ans.append([n,n])
                    del d[n]
                    
                
        #print ans        
        return ans

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (len(nums) < 3):
            return []
            
        if (len(nums) == 3): 
            if ((nums[0]+nums[1]+nums[2]) <> 0):
                return []
            else:
                return [nums]
            
        # this is to remove duplicates
        nums.sort()
        
        r = []
        i = 0
        ln = len(nums)-2
        ll = len(nums)-1
        while (i < ln):
            target = -nums[i]
            l = self.allTwoSum(nums[i+1:], target)
            for j in l:
                r.append([nums[i],j[0],j[1]])

            # now ignore duplicates
            prev = nums[i]
            while (i < ll and nums[i+1] == prev):
                i += 1
            # we need to get to the next different element
            i += 1
                
        return r
            
            
            
# second try to avoid Time Limit Exceeded

class ThreeSumSolutionTake2(object):
    # need to find a way to find all unique two numbers that add up to target
    def allTwoSum(self, dn, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # print nums
        if (not dn):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = dict(dn)
        
        ans = []
        for i in d.keys():
            n = target - i
            
            # since it's only the sum of two numbers once a number is used it cannot be reused bby another number
            # e.g. say the target is 5, and we found 3 has a partner that sums up to 5, but since it's only a sum of two numbers
            # the partner of 3 is unique, which is 2 and the partner of 2 is also unique, so we can immediately pop these two from the dictionary
            if (n in d):
                if (i <> n):
                    ans.append([i,n])
                    del d[i]
                    del d[n]
                else:
                    # in the case of n == i, d[i] must have length more than 1 otherwise it doesn't have a solution
                    if (d[i] >= 2):
                        ans.append([n,n])
                    del d[n]
                    
                
        #print ans        
        return ans

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (len(nums) < 3):
            return []
            
        if (len(nums) == 3): 
            if ((nums[0]+nums[1]+nums[2]) <> 0):
                return []
            else:
                return [nums]
            
        
        d = {}
        
        # add all numbers into the dictionary and because we iterate through nums the indices in tha values are sorted in ascending order
        for i in xrange(len(nums)):
            if (nums[i] in d):
                n = d[nums[i]]
                d[nums[i]] = n+1
            else:
                d[nums[i]] = 1
            
        r = []
        
        for n in d.keys():
            target = -n
            # print target
            # print d
            gone = True
            if (d[n] > 1):
                m = d[n]
                m -= 1
                d[n] = m
                gone = False
            else:
                del d[n]
                
            l = self.allTwoSum(d, target)
            # print l
            for j in l:
                r.append([-target,j[0],j[1]])
                
            if (gone == False):
                del d[n]
                
        return r
            
       









# third try to avoid Time Limit Exceeded

class ThreeSumSolutionTake3(object):
    # need to find a way to find all unique two numbers that add up to target
    def allTwoSum(self, dn, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # print nums
        if (not dn):
            return []
        
        # solve it using a hashmap or dictionary in Python
        d = dn
        
        ans = []
        od = []
        for i in d.keys():
            n = target - i
            
            # since it's only the sum of two numbers once a number is used it cannot be reused bby another number
            # e.g. say the target is 5, and we found 3 has a partner that sums up to 5, but since it's only a sum of two numbers
            # the partner of 3 is unique, which is 2 and the partner of 2 is also unique, so we can immediately pop these two from the dictionary
            if (n in d):
                if (i <> n):
                    ans.append([i,n])
                    od.append([i,d[i]])
                    od.append([n,d[n]])
                    del d[i]
                    del d[n]
                else:
                    # in the case of n == i, d[i] must have length more than 1 otherwise it doesn't have a solution
                    if (d[i] >= 2):
                        ans.append([n,n])
                    od.append([n,d[n]])
                    del d[n]
        
        # this is to avoid copying the dict everytime
        for m in od:
            d[m[0]] = m[1]
                
        #print ans        
        return ans

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (len(nums) < 3):
            return []
            
        if (len(nums) == 3): 
            if ((nums[0]+nums[1]+nums[2]) <> 0):
                return []
            else:
                return [nums]
            
        
        d = {}
        
        # add all numbers into the dictionary and because we iterate through nums the indices in tha values are sorted in ascending order
        for i in xrange(len(nums)):
            if (nums[i] in d):
                n = d[nums[i]]
                d[nums[i]] = n+1
            else:
                d[nums[i]] = 1
            
        r = []
        
        for n in d.keys():
            target = -n
            # print target
            # print d
            gone = True
            if (d[n] > 1):
                m = d[n]
                m -= 1
                d[n] = m
                gone = False
            else:
                del d[n]
                
            l = self.allTwoSum(d, target)
            # print l
            for j in l:
                r.append([-target,j[0],j[1]])
                
            if (gone == False):
                del d[n]
                
        return r
            
              

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# note that all keys now point to the same object, we are not replicating the object value here            
dict.fromkeys([1, 2, 3], 1)            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
def OthreeSum(nums):
    res = []
    nums.sort()
    for i in xrange(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res        
        
        
        
#lt = [82597,-9243,62390,83030,-97960,-26521,-61011,83390,-38677,12333,75987,46091,83794,19355,-71037,-6242,-28801,324,1202,-90885,-2989,-95597,-34333,35528,5680,89093,-90606,50360,-29393,-27012,53313,65213,99818,-82405,-41661,-3333,-51952,72135,-1523,26377,74685,96992,92263,15929,5467,-99555,-43348,-41689,-60383,-3990,32165,65265,-72973,-58372,12741,-48568,-46596,72419,-1859,34153,62937,81310,-61823,-96770,-54944,8845,-91184,24208,-29078,31495,65258,14198,85395,70506,-40908,56740,-12228,-40072,32429,93001,68445,-73927,25731,-91859,-24150,10093,-60271,-81683,-18126,51055,48189,-6468,25057,81194,-58628,74042,66158,-14452,-49851,-43667,11092,39189,-17025,-79173,13606,83172,92647,-59741,19343,-26644,-57607,82908,-20655,1637,80060,98994,39331,-31274,-61523,91225,-72953,13211,-75116,-98421,-41571,-69074,99587,39345,42151,-2460,98236,15690,-52507,-95803,-48935,-46492,-45606,-79254,-99851,52533,73486,39948,-7240,71815,-585,-96252,90990,-93815,93340,-71848,58733,-14859,-83082,-75794,-82082,-24871,-15206,91207,-56469,-93618,67131,-8682,75719,87429,-98757,-7535,-24890,-94160,85003,33928,75538,97456,-66424,-60074,-8527,-28697,-22308,2246,-70134,-82319,-10184,87081,-34949,-28645,-47352,-83966,-60418,-15293,-53067,-25921,55172,75064,95859,48049,34311,-86931,-38586,33686,-36714,96922,76713,-22165,-80585,-34503,-44516,39217,-28457,47227,-94036,43457,24626,-87359,26898,-70819,30528,-32397,-69486,84912,-1187,-98986,-32958,4280,-79129,-65604,9344,58964,50584,71128,-55480,24986,15086,-62360,-42977,-49482,-77256,-36895,-74818,20,3063,-49426,28152,-97329,6086,86035,-88743,35241,44249,19927,-10660,89404,24179,-26621,-6511,57745,-28750,96340,-97160,-97822,-49979,52307,79462,94273,-24808,77104,9255,-83057,77655,21361,55956,-9096,48599,-40490,-55107,2689,29608,20497,66834,-34678,23553,-81400,-66630,-96321,-34499,-12957,-20564,25610,-4322,-58462,20801,53700,71527,24669,-54534,57879,-3221,33636,3900,97832,-27688,-98715,5992,24520,-55401,-57613,-69926,57377,-77610,20123,52174,860,60429,-91994,-62403,-6218,-90610,-37263,-15052,62069,-96465,44254,89892,-3406,19121,-41842,-87783,-64125,-56120,73904,-22797,-58118,-4866,5356,75318,46119,21276,-19246,-9241,-97425,57333,-15802,93149,25689,-5532,95716,39209,-87672,-29470,-16324,-15331,27632,-39454,56530,-16000,29853,46475,78242,-46602,83192,-73440,-15816,50964,-36601,89758,38375,-40007,-36675,-94030,67576,46811,-64919,45595,76530,40398,35845,41791,67697,-30439,-82944,63115,33447,-36046,-50122,-34789,43003,-78947,-38763,-89210,32756,-20389,-31358,-90526,-81607,88741,86643,98422,47389,-75189,13091,95993,-15501,94260,-25584,-1483,-67261,-70753,25160,89614,-90620,-48542,83889,-12388,-9642,-37043,-67663,28794,-8801,13621,12241,55379,84290,21692,-95906,-85617,-17341,-63767,80183,-4942,-51478,30997,-13658,8838,17452,-82869,-39897,68449,31964,98158,-49489,62283,-62209,-92792,-59342,55146,-38533,20496,62667,62593,36095,-12470,5453,-50451,74716,-17902,3302,-16760,-71642,-34819,96459,-72860,21638,47342,-69897,-40180,44466,76496,84659,13848,-91600,-90887,-63742,-2156,-84981,-99280,94326,-33854,92029,-50811,98711,-36459,-75555,79110,-88164,-97397,-84217,97457,64387,30513,-53190,-83215,252,2344,-27177,-92945,-89010,82662,-11670,86069,53417,42702,97082,3695,-14530,-46334,17910,77999,28009,-12374,15498,-46941,97088,-35030,95040,92095,-59469,-24761,46491,67357,-66658,37446,-65130,-50416,99197,30925,27308,54122,-44719,12582,-99525,-38446,-69050,-22352,94757,-56062,33684,-40199,-46399,96842,-50881,-22380,-65021,40582,53623,-76034,77018,-97074,-84838,-22953,-74205,79715,-33920,-35794,-91369,73421,-82492,63680,-14915,-33295,37145,76852,-69442,60125,-74166,74308,-1900,-30195,-16267,-60781,-27760,5852,38917,25742,-3765,49097,-63541,98612,-92865,-30248,9612,-8798,53262,95781,-42278,-36529,7252,-27394,-5021,59178,80934,-48480,-75131,-54439,-19145,-48140,98457,-6601,-51616,-89730,78028,32083,-48904,16822,-81153,-8832,48720,-80728,-45133,-86647,-4259,-40453,2590,28613,50523,-4105,-27790,-74579,-17223,63721,33489,-47921,97628,-97691,-14782,-65644,18008,-93651,-71266,80990,-76732,-47104,35368,28632,59818,-86269,-89753,34557,-92230,-5933,-3487,-73557,-13174,-43981,-43630,-55171,30254,-83710,-99583,-13500,71787,5017,-25117,-78586,86941,-3251,-23867,-36315,75973,86272,-45575,77462,-98836,-10859,70168,-32971,-38739,-12761,93410,14014,-30706,-77356,-85965,-62316,63918,-59914,-64088,1591,-10957,38004,15129,-83602,-51791,34381,-89382,-26056,8942,5465,71458,-73805,-87445,-19921,-80784,69150,-34168,28301,-68955,18041,6059,82342,9947,39795,44047,-57313,48569,81936,-2863,-80932,32976,-86454,-84207,33033,32867,9104,-16580,-25727,80157,-70169,53741,86522,84651,68480,84018,61932,7332,-61322,-69663,76370,41206,12326,-34689,17016,82975,-23386,39417,72793,44774,-96259,3213,79952,29265,-61492,-49337,14162,65886,3342,-41622,-62659,-90402,-24751,88511,54739,-21383,-40161,-96610,-24944,-602,-76842,-21856,69964,43994,-15121,-85530,12718,13170,-13547,69222,62417,-75305,-81446,-38786,-52075,-23110,97681,-82800,-53178,11474,35857,94197,-58148,-23689,32506,92154,-64536,-73930,-77138,97446,-83459,70963,22452,68472,-3728,-25059,-49405,95129,-6167,12808,99918,30113,-12641,-26665,86362,-33505,50661,26714,33701,89012,-91540,40517,-12716,-57185,-87230,29914,-59560,13200,-72723,58272,23913,-45586,-96593,-26265,-2141,31087,81399,92511,-34049,20577,2803,26003,8940,42117,40887,-82715,38269,40969,-50022,72088,21291,-67280,-16523,90535,18669,94342,-39568,-88080,-99486,-20716,23108,-28037,63342,36863,-29420,-44016,75135,73415,16059,-4899,86893,43136,-7041,33483,-67612,25327,40830,6184,61805,4247,81119,-22854,-26104,-63466,63093,-63685,60369,51023,51644,-16350,74438,-83514,99083,10079,-58451,-79621,48471,67131,-86940,99093,11855,-22272,-67683,-44371,9541,18123,37766,-70922,80385,-57513,-76021,-47890,36154,72935,84387,-92681,-88303,-7810,59902,-90,-64704,-28396,-66403,8860,13343,33882,85680,7228,28160,-14003,54369,-58893,92606,-63492,-10101,64714,58486,29948,-44679,-22763,10151,-56695,4031,-18242,-36232,86168,-14263,9883,47124,47271,92761,-24958,-73263,-79661,-69147,-18874,29546,-92588,-85771,26451,-86650,-43306,-59094,-47492,-34821,-91763,-47670,33537,22843,67417,-759,92159,63075,94065,-26988,55276,65903,30414,-67129,-99508,-83092,-91493,-50426,14349,-83216,-76090,32742,-5306,-93310,-60750,-60620,-45484,-21108,-58341,-28048,-52803,69735,78906,81649,32565,-86804,-83202,-65688,-1760,89707,93322,-72750,84134,71900,-37720,19450,-78018,22001,-23604,26276,-21498,65892,-72117,-89834,-23867,55817,-77963,42518,93123,-83916,63260,-2243,-97108,85442,-36775,17984,-58810,99664,-19082,93075,-69329,87061,79713,16296,70996,13483,-74582,49900,-27669,-40562,1209,-20572,34660,83193,75579,7344,64925,88361,60969,3114,44611,-27445,53049,-16085,-92851,-53306,13859,-33532,86622,-75666,-18159,-98256,51875,-42251,-27977,-18080,23772,38160,41779,9147,94175,99905,-85755,62535,-88412,-52038,-68171,93255,-44684,-11242,-104,31796,62346,-54931,-55790,-70032,46221,56541,-91947,90592,93503,4071,20646,4856,-63598,15396,-50708,32138,-85164,38528,-89959,53852,57915,-42421,-88916,-75072,67030,-29066,49542,-71591,61708,-53985,-43051,28483,46991,-83216,80991,-46254,-48716,39356,-8270,-47763,-34410,874,-1186,-7049,28846,11276,21960,-13304,-11433,-4913,55754,79616,70423,-27523,64803,49277,14906,-97401,-92390,91075,70736,21971,-3303,55333,-93996,76538,54603,-75899,98801,46887,35041,48302,-52318,55439,24574,14079,-24889,83440,14961,34312,-89260,-22293,-81271,-2586,-71059,-10640,-93095,-5453,-70041,66543,74012,-11662,-52477,-37597,-70919,92971,-17452,-67306,-80418,7225,-89296,24296,86547,37154,-10696,74436,-63959,58860,33590,-88925,-97814,-83664,85484,-8385,-50879,57729,-74728,-87852,-15524,-91120,22062,28134,80917,32026,49707,-54252,-44319,-35139,13777,44660,85274,25043,58781,-89035,-76274,6364,-63625,72855,43242,-35033,12820,-27460,77372,-47578,-61162,-70758,-1343,-4159,64935,56024,-2151,43770,19758,-30186,-86040,24666,-62332,-67542,73180,-25821,-27826,-45504,-36858,-12041,20017,-24066,-56625,-52097,-47239,-90694,8959,7712,-14258,-5860,55349,61808,-4423,-93703,64681,-98641,-25222,46999,-83831,-54714,19997,-68477,66073,51801,-66491,52061,-52866,79907,-39736,-68331,68937,91464,98892,910,93501,31295,-85873,27036,-57340,50412,21,-2445,29471,71317,82093,-94823,-54458,-97410,39560,-7628,66452,39701,54029,37906,46773,58296,60370,-61090,85501,-86874,71443,-72702,-72047,14848,34102,77975,-66294,-36576,31349,52493,-70833,-80287,94435,39745,-98291,84524,-18942,10236,93448,50846,94023,-6939,47999,14740,30165,81048,84935,-19177,-13594,32289,62628,-90612,-542,-66627,64255,71199,-83841,-82943,-73885,8623,-67214,-9474,-35249,62254,-14087,-90969,21515,-83303,94377,-91619,19956,-98810,96727,-91939,29119,-85473,-82153,-69008,44850,74299,-76459,-86464,8315,-49912,-28665,59052,-69708,76024,-92738,50098,18683,-91438,18096,-19335,35659,91826,15779,-73070,67873,-12458,-71440,-46721,54856,97212,-81875,35805,36952,68498,81627,-34231,81712,27100,-9741,-82612,18766,-36392,2759,41728,69743,26825,48355,-17790,17165,56558,3295,-24375,55669,-16109,24079,73414,48990,-11931,-78214,90745,19878,35673,-15317,-89086,94675,-92513,88410,-93248,-19475,-74041,-19165,32329,-26266,-46828,-18747,45328,8990,-78219,-25874,-74801,-44956,-54577,-29756,-99822,-35731,-18348,-68915,-83518,-53451,95471,-2954,-13706,-8763,-21642,-37210,16814,-60070,-42743,27697,-36333,-42362,11576,85742,-82536,68767,-56103,-63012,71396,-78464,-68101,-15917,-11113,-3596,77626,-60191,-30585,-73584,6214,-84303,18403,23618,-15619,-89755,-59515,-59103,-74308,-63725,-29364,-52376,-96130,70894,-12609,50845,-2314,42264,-70825,64481,55752,4460,-68603,-88701,4713,-50441,-51333,-77907,97412,-66616,-49430,60489,-85262,-97621,-18980,44727,-69321,-57730,66287,-92566,-64427,-14270,11515,-92612,-87645,61557,24197,-81923,-39831,-10301,-23640,-76219,-68025,92761,-76493,68554,-77734,-95620,-11753,-51700,98234,-68544,-61838,29467,46603,-18221,-35441,74537,40327,-58293,75755,-57301,-7532,-94163,18179,-14388,-22258,-46417,-48285,18242,-77551,82620,250,-20060,-79568,-77259,82052,-98897,-75464,48773,-79040,-11293,45941,-67876,-69204,-46477,-46107,792,60546,-34573,-12879,-94562,20356,-48004,-62429,96242,40594,2099,99494,25724,-39394,-2388,-18563,-56510,-83570,-29214,3015,74454,74197,76678,-46597,60630,-76093,37578,-82045,-24077,62082,-87787,-74936,58687,12200,-98952,70155,-77370,21710,-84625,-60556,-84128,925,65474,-15741,-94619,88377,89334,44749,22002,-45750,-93081,-14600,-83447,46691,85040,-66447,-80085,56308,44310,24979,-29694,57991,4675,-71273,-44508,13615,-54710,23552,-78253,-34637,50497,68706,81543,-88408,-21405,6001,-33834,-21570,-46692,-25344,20310,71258,-97680,11721,59977,59247,-48949,98955,-50276,-80844,-27935,-76102,55858,-33492,40680,66691,-33188,8284,64893,-7528,6019,-85523,8434,-64366,-56663,26862,30008,-7611,-12179,-70076,21426,-11261,-36864,-61937,-59677,929,-21052,3848,-20888,-16065,98995,-32293,-86121,-54564,77831,68602,74977,31658,40699,29755,98424,80358,-69337,26339,13213,-46016,-18331,64713,-46883,-58451,-70024,-92393,-4088,70628,-51185,71164,-75791,-1636,-29102,-16929,-87650,-84589,-24229,-42137,-15653,94825,13042,88499,-47100,-90358,-7180,29754,-65727,-42659,-85560,-9037,-52459,20997,-47425,17318,21122,20472,-23037,65216,-63625,-7877,-91907,24100,-72516,22903,-85247,-8938,73878,54953,87480,-31466,-99524,35369,-78376,89984,-15982,94045,-7269,23319,-80456,-37653,-76756,2909,81936,54958,-12393,60560,-84664,-82413,66941,-26573,-97532,64460,18593,-85789,-38820,-92575,-43663,-89435,83272,-50585,13616,-71541,-53156,727,-27644,16538,34049,57745,34348,35009,16634,-18791,23271,-63844,95817,21781,16590,59669,15966,-6864,48050,-36143,97427,-59390,96931,78939,-1958,50777,43338,-51149,39235,-27054,-43492,67457,-83616,37179,10390,85818,2391,73635,87579,-49127,-81264,-79023,-81590,53554,-74972,-83940,-13726,-39095,29174,78072,76104,47778,25797,-29515,-6493,-92793,22481,-36197,-65560,42342,15750,97556,99634,-56048,-35688,13501,63969,-74291,50911,39225,93702,-3490,-59461,-30105,-46761,-80113,92906,-68487,50742,36152,-90240,-83631,24597,-50566,-15477,18470,77038,40223,-80364,-98676,70957,-63647,99537,13041,31679,86631,37633,-16866,13686,-71565,21652,-46053,-80578,-61382,68487,-6417,4656,20811,67013,-30868,-11219,46,74944,14627,56965,42275,-52480,52162,-84883,-52579,-90331,92792,42184,-73422,-58440,65308,-25069,5475,-57996,59557,-17561,2826,-56939,14996,-94855,-53707,99159,43645,-67719,-1331,21412,41704,31612,32622,1919,-69333,-69828,22422,-78842,57896,-17363,27979,-76897,35008,46482,-75289,65799,20057,7170,41326,-76069,90840,-81253,-50749,3649,-42315,45238,-33924,62101,96906,58884,-7617,-28689,-66578,62458,50876,-57553,6739,41014,-64040,-34916,37940,13048,-97478,-11318,-89440,-31933,-40357,-59737,-76718,-14104,-31774,28001,4103,41702,-25120,-31654,63085,-3642,84870,-83896,-76422,-61520,12900,88678,85547,33132,-88627,52820,63915,-27472,78867,-51439,33005,-23447,-3271,-39308,39726,-74260,-31874,-36893,93656,910,-98362,60450,-88048,99308,13947,83996,-90415,-35117,70858,-55332,-31721,97528,82982,-86218,6822,25227,36946,97077,-4257,-41526,56795,89870,75860,-70802,21779,14184,-16511,-89156,-31422,71470,69600,-78498,74079,-19410,40311,28501,26397,-67574,-32518,68510,38615,19355,-6088,-97159,-29255,-92523,3023,-42536,-88681,64255,41206,44119,52208,39522,-52108,91276,-70514,83436,63289,-79741,9623,99559,12642,85950,83735,-21156,-67208,98088,-7341,-27763,-30048,-44099,-14866,-45504,-91704,19369,13700,10481,-49344,-85686,33994,19672,36028,60842,66564,-24919,33950,-93616,-47430,-35391,-28279,56806,74690,39284,-96683,-7642,-75232,37657,-14531,-86870,-9274,-26173,98640,88652,64257,46457,37814,-19370,9337,-22556,-41525,39105,-28719,51611,-93252,98044,-90996,21710,-47605,-64259,-32727,53611,-31918,-3555,33316,-66472,21274,-37731,-2919,15016,48779,-88868,1897,41728,46344,-89667,37848,68092,-44011,85354,-43776,38739,-31423,-66330,65167,-22016,59405,34328,-60042,87660,-67698,-59174,-1408,-46809,-43485,-88807,-60489,13974,22319,55836,-62995,-37375,-4185,32687,-36551,-75237,58280,26942,-73756,71756,78775,-40573,14367,-71622,-77338,24112,23414,-7679,-51721,87492,85066,-21612,57045,10673,-96836,52461,-62218,-9310,65862,-22748,89906,-96987,-98698,26956,-43428,46141,47456,28095,55952,67323,-36455,-60202,-43302,-82932,42020,77036,10142,60406,70331,63836,58850,-66752,52109,21395,-10238,-98647,-41962,27778,69060,98535,-28680,-52263,-56679,66103,-42426,27203,80021,10153,58678,36398,63112,34911,20515,62082,-15659,-40785,27054,43767,-20289,65838,-6954,-60228,-72226,52236,-35464,25209,-15462,-79617,-41668,-84083,62404,-69062,18913,46545,20757,13805,24717,-18461,-47009,-25779,68834,64824,34473,39576,31570,14861,-15114,-41233,95509,68232,67846,84902,-83060,17642,-18422,73688,77671,-26930,64484,-99637,73875,6428,21034,-73471,19664,-68031,15922,-27028,48137,54955,-82793,-41144,-10218,-24921,-28299,-2288,68518,-54452,15686,-41814,66165,-72207,-61986,80020,50544,-99500,16244,78998,40989,14525,-56061,-24692,-94790,21111,37296,-90794,72100,70550,-31757,17708,-74290,61910,78039,-78629,-25033,73172,-91953,10052,64502,99585,-1741,90324,-73723,68942,28149,30218,24422,16659,10710,-62594,94249,96588,46192,34251,73500,-65995,-81168,41412,-98724,-63710,-54696,-52407,19746,45869,27821,-94866,-76705,-13417,-61995,-71560,43450,67384,-8838,-80293,-28937,23330,-89694,-40586,46918,80429,-5475,78013,25309,-34162,37236,-77577,86744,26281,-29033,-91813,35347,13033,-13631,-24459,3325,-71078,-75359,81311,19700,47678,-74680,-84113,45192,35502,37675,19553,76522,-51098,-18211,89717,4508,-82946,27749,85995,89912,-53678,-64727,-14778,32075,-63412,-40524,86440,-2707,-36821,63850,-30883,67294,-99468,-23708,34932,34386,98899,29239,-23385,5897,54882,98660,49098,70275,17718,88533,52161,63340,50061,-89457,19491,-99156,24873,-17008,64610,-55543,50495,17056,-10400,-56678,-29073,-42960,-76418,98562,-88104,-96255,10159,-90724,54011,12052,45871,-90933,-69420,67039,37202,78051,-52197,-40278,-58425,65414,-23394,-1415,6912,-53447,7352,17307,-78147,63727,98905,55412,-57658,-32884,-44878,22755,39730,3638,35111,39777,74193,38736,-11829,-61188,-92757,55946,-71232,-63032,-83947,39147,-96684,-99233,25131,-32197,24406,-55428,-61941,25874,-69453,64483,-19644,-68441,12783,87338,-48676,66451,-447,-61590,50932,-11270,29035,65698,-63544,10029,80499,-9461,86368,91365,-81810,-71914,-52056,-13782,44240,-30093,-2437,24007,67581,-17365,-69164,-8420,-69289,-29370,48010,90439,13141,69243,50668,39328,61731,78266,-81313,17921,-38196,55261,9948,-24970,75712,-72106,28696,7461,31621,61047,51476,56512,11839,-96916,-82739,28924,-99927,58449,37280,69357,11219,-32119,-62050,-48745,-83486,-52376,42668,82659,68882,38773,46269,-96005,97630,25009,-2951,-67811,99801,81587,-79793,-18547,-83086,69512,33127,-92145,-88497,47703,59527,1909,88785,-88882,69188,-46131,-5589,-15086,36255,-53238,-33009,82664,53901,35939,-42946,-25571,33298,69291,53199,74746,-40127,-39050,91033,51717,-98048,87240,36172,65453,-94425,-63694,-30027,59004,88660,3649,-20267,-52565,-67321,34037,4320,91515,-56753,60115,27134,68617,-61395,-26503,-98929,-8849,-63318,10709,-16151,61905,-95785,5262,23670,-25277,90206,-19391,45735,37208,-31992,-92450,18516,-90452,-58870,-58602,93383,14333,17994,82411,-54126,-32576,35440,-60526,-78764,-25069,-9022,-394,92186,-38057,55328,-61569,67780,77169,19546,-92664,-94948,44484,-13439,83529,27518,-48333,72998,38342,-90553,-98578,-76906,81515,-16464,78439,92529,35225,-39968,-10130,-7845,-32245,-74955,-74996,67731,-13897,-82493,33407,93619,59560,-24404,-57553,19486,-45341,34098,-24978,-33612,79058,71847,76713,-95422,6421,-96075,-59130,-28976,-16922,-62203,69970,68331,21874,40551,89650,51908,58181,66480,-68177,34323,-3046,-49656,-59758,43564,-10960,-30796,15473,-20216,46085,-85355,41515,-30669,-87498,57711,56067,63199,-83805,62042,91213,-14606,4394,-562,74913,10406,96810,-61595,32564,31640,-9732,42058,98052,-7908,-72330,1558,-80301,34878,32900,3939,-8824,88316,20937,21566,-3218,-66080,-31620,86859,54289,90476,-42889,-15016,-18838,75456,30159,-67101,42328,-92703,85850,-5475,23470,-80806,68206,17764,88235,46421,-41578,74005,-81142,80545,20868,-1560,64017,83784,68863,-97516,-13016,-72223,79630,-55692,82255,88467,28007,-34686,-69049,-41677,88535,-8217,68060,-51280,28971,49088,49235,26905,-81117,-44888,40623,74337,-24662,97476,79542,-72082,-35093,98175,-61761,-68169,59697,-62542,-72965,59883,-64026,-37656,-92392,-12113,-73495,98258,68379,-21545,64607,-70957,-92254,-97460,-63436,-8853,-19357,-51965,-76582,12687,-49712,45413,-60043,33496,31539,-57347,41837,67280,-68813,52088,-13155,-86430,-15239,-45030,96041,18749,-23992,46048,35243,-79450,85425,-58524,88781,-39454,53073,-48864,-82289,39086,82540,-11555,25014,-5431,-39585,-89526,2705,31953,-81611,36985,-56022,68684,-27101,11422,64655,-26965,-63081,-13840,-91003,-78147,-8966,41488,1988,99021,-61575,-47060,65260,-23844,-21781,-91865,-19607,44808,2890,63692,-88663,-58272,15970,-65195,-45416,-48444,-78226,-65332,-24568,42833,-1806,-71595,80002,-52250,30952,48452,-90106,31015,-22073,62339,63318,78391,28699,77900,-4026,-76870,-45943,33665,9174,-84360,-22684,-16832,-67949,-38077,-38987,-32847,51443,-53580,-13505,9344,-92337,26585,70458,-52764,-67471,-68411,-1119,-2072,-93476,67981,40887,-89304,-12235,41488,1454,5355,-34855,-72080,24514,-58305,3340,34331,8731,77451,-64983,-57876,82874,62481,-32754,-39902,22451,-79095,-23904,78409,-7418,77916]        

########################################################################################
# Gray Code
########################################################################################


        
class GrayCodeSolution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if (n == 0):
            return [0]
            
        tot = 1 << n
        l = [0,1]
        
        if (n == 1):
            return l
        
        # the pattern is that of reflection, it's a lot easier to show visually, everytime you go to the next higher power of two
        # you repeat the previous pattern backwards at least this is what you see in terms of bits
        # in terms of integers you just add the higer power of two backwards to the previous pattern
        
        for i in xrange(2,tot):
            if (i & (i-1) == 0):
                j = -1
                n = i
            l.append(l[i+j] + n)
            j -= 2
        
        
        return l 
        
        
########################################################################################
# Rotate Image
########################################################################################        
        
class RotateImageSolution(object):

    def myprint(self, matrix):
        for y in matrix:
            for x in y:
                print "%2d" % x,
            print ""
            
        print ""

    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        
        # we will try to rotate it in place
        # since this is a clockwise rotation we need to move it clockwise
        
        #very first thing to do is to check if the list is empty
        if (not matrix):
            return matrix
        # but also we need to check if matrix is [[]] in this case not matrix is False
        if (not matrix[0]):
            return matrix
        
        #self.myprint(matrix)
        
        
        # first we need to see if the matrix size is odd or even
        n = len(matrix)
        
        # so to move it in place we need to move
        # top left to top right
        # top right to bottom right
        # bottom right to bottom left
        # bottom left to top left
        
        # first we need to get the outer layer so the top left row and column are row = 0 column = 0
        
        for c in range(n//2):
            # starting row and starting column
            sr = c
            sc = c
            # initial size is the full size
            sz = n - (2*c)
            
            for i in range(sz-1):
                tl = matrix[sr][sc+i] # for this one the column moves to the right
                tr = matrix[sr+i][-(sc+1)] # for this one the row moves down
                br = matrix[-(sr+1)][-(sc+i+1)] # for this one the column moves backward
                bl = matrix[-(sr+i+1)][sc] # for this one the row moves upward
                
                # now move things around clockwise
                # top right = top left
                matrix[sr+i][-(sc+1)] = tl
                # bottom right = top right
                matrix[-(sr+1)][-(sc+i+1)] = tr
                # bottom left = bottom right
                matrix[-(sr+i+1)][sc] = br
                # top left = bottom left
                matrix[sr][sc+i] = bl
        
                
        #self.myprint(matrix)
        
        
#####################################################################################
# Number of Islands
#####################################################################################       

class NumIslandsSolution(object):
    
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        # my first idea is of course to do a BFS and each new root we found we add the count
        # BFS starts from top left
        
        if (not grid):
            return 0
            
        if (not grid[0]):
            return 0
        
        rmax = len(grid)
        cmax = len(grid[0])
        
        # we cannot use a dictionary as the visited map since a list is mutable so we cannot use [r,c] as a key
        # so we need to use a matrix as well
        vm = [[0]*cmax for row in range(rmax)]
        ic = 0
        
        for r in range(rmax):
            for c in range(cmax):
                
                if (vm[r][c] == 0 and grid[r][c] == '1'):
                    # this is a new root
                    ic += 1
                    
                    q = [(r,c)]
                    vm[r][c] = 1
                    
                    while(q):
                        row, col = q.pop(0)
                        #print "%d %d" % (row, col)
                        
                        # no need to call function just check it straight away
                        if (row > 0 and grid[row-1][col] == '1' and vm[row-1][col] == 0):
                            vm[row-1][col] = 1
                            q.append((row-1,col))
                        if (row < rmax-1 and grid[row+1][col] == '1' and vm[row+1][col] == 0):
                            vm[row+1][col] = 1
                            q.append((row+1,col))
                        if (col > 0 and grid[row][col-1] == '1' and vm[row][col-1] == 0):
                            vm[row][col-1] = 1
                            q.append((row, col-1))
                        if (col < cmax-1 and grid[row][col+1] == '1' and vm[row][col+1] == 0):
                            vm[row][col+1] = 1
                            q.append((row, col+1))

        return ic
        
        
############################################################################################
# Binary Tree Right Side View
############################################################################################        

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class BinaryTreeRightSideViewSolution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # I initially misunderstood this problem
        # my take was just go to the right child as much as possible and if you can't just go to the left and continue
        # in this way for the following tree we will get only 1,3 although what we want is 1,3,5
        #   1            <---
        # /   \
        #2     3         <---
        # \     
        #  5             <---
        # that case we need to do BFS with level recognition
        
        if (not root):
            return []
            
        l = []
        q = [root, None]
        # None acts as a dummy to indicate the level we are in
        
        while(q):
            n = q.pop(0)
            
            # must check if this is the last dummy
            # if so do not add another dummy if it is not the last dummy the add another dummy to indicate the start of the next level
            if (n is None):
                if (len(q) > 0):
                    q.append(None)
            else:    
                # now add the children but first check if it is the right most element of the tree
                if (len(q) > 0 and q[0] is None):
                    l.append(n.val)
                    
                # now add the children
                if (n.left is not None):
                    q.append(n.left)

                if (n.right is not None):
                    q.append(n.right)
        
        return l
        
###############################################################################################
# Kth Largest Element in an Array
###############################################################################################        

class KthLargestSolution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        if (not nums):
            return -1
            
        # not sure if I can just do this but it works
        nums.sort()
        
        return nums[-k]
        
############################################################################################
# Group Anagrams
############################################################################################

class GroupAnagramsSolution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        
        if (not strs):
            return []
            
        # my strategy would be to sort the string and then use a dict
        
        d = {}
        for x in strs:
            s = list(x)
            s.sort()
            d.setdefault("".join(s), []).append(x)
            
        l = []
        for x in d.values():
            l.append(x)
            
        return l
        
        
########################################################################################
# Letter Combinations Phone Number
########################################################################################

class LetterComboPhoneSolution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        
        if (not digits):
            return []
            
        n = len(digits)            
        
        # first build a dictionary
        d = {"2":[3,"abc"], "3":[3,"def"], "4":[3,"ghi"], "5":[3,"jkl"], "6":[3,"mno"], "7":[4,"pqrs"], "8":[3,"tuv"], "9":[4,"wxyz"], "0":[1," "]}
        
        # this will be the current alphabet combo
        a = [0]*n
        
        # construct max and let's find out the total number of combinations at the beginning
        tot = 1
        
        for i in range(1, n+1):
            tot *= d[digits[-i]][0]
        
        l = []

        while (tot > 0):
            l.append("".join([d[digits[i]][1][a[i]] for i in range(n)]))
            tot -= 1
            
            if (tot > 0):            
                i = 1
                c = 1
                while (i < (n + 1) and c > 0):
                    a[-i] += 1
                    # this is the carry
                    t = d[digits[-i]][0]
                    c = a[-i] // t
                    # this is the actual digit 
                    a[-i] %= t
                    i += 1
        
        return l

        
        
####################################################################################
# Intersection of Two Linked Lists
####################################################################################        

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class IntersectionLLSolution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if (not headA or not headB):
            return None
        
        # running time must be O(n) and space must be O(1)
        # if space is not O(1) we can just use a dict but since space must be O(1) we cannot
        # so what we can do is to traverse the linked lists together
        # but if the lengths are not the same then we can't do it
        # so we traverse twice
        # first is to find the lengths and once we've found the length we fast forward to the same starting point on the longer one
        
        # first iteration to calculate the lengths
        
        n = headA
        l1 = 0
        while(n):
            l1 += 1
            n = n.next
        n = headA

        n = headB
        l2 = 0
        while(n):
            l2 += 1
            n = n.next
            
        p1 = headA
        p2 = headB
        while(l1 > l2):
            p1 = p1.next
            l1 -= 1
            
        while(l2 > l1):
            p2 = p2.next
            l2 -= 1
            
        # now we should have the same starting point for both lists
        
        while(p1):
            if (p1 is p2):
                return p1
            p1 = p1.next
            p2 = p2.next
            
        return None

##################################################################
# Subsets
##################################################################

class SubsetsSolution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if (not nums):
            return [[]]
            
        if (len(nums) == 1):
            return [[nums[0]],[]]
            
        r = self.subsets(nums[1:])
        n = len(r)
        print r
        
        for i in range(n):
            t = r[i][:]
            print t
            t.append(nums[0])
            r.append(t)
            
        return r

###############################################################
# Lowest Common Ancestor Binary Tree
###############################################################

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
       
    # the strategy here is to get the path into p and then search if p is a parent of q
    # if it's not we backtrack to each parent of p and see if it's also a parent of q
    # meanwhile we check if one of the parents of p is q itself
    # the quite confusing thing here is that we use these functions to find the node itself
    # and to also check for common ancestor
    
    def gatherPathQ(self, root, q):
        if (not root):
            return False
        if (root is q):
            return True
            
        if ((not self.gatherPathQ(root.left, q)) and (not self.gatherPathQ(root.right, q))):
            return False
            
        return True

    def gatherPathP(self, root, p, q):

        if (not root):
            return None, False
        if (root is p):
            # we found p, now we want to see if we can find q
            return p, (self.gatherPathQ(root.left, q) or self.gatherPathQ(root.right, q))
        
        lp, lq = self.gatherPathP(root.left, p, q);

        # if we found both q and p then return their parent
        if (lp and lq):
            return lp, lq
        
        if (not lp):
            rp, rq = self.gatherPathP(root.right, p, q)
            
            # if we found both q and p then return their parent
            if (rp and rq):
                return rp, rq
                
            if (not rp):
                # this means that this current root is not a parent of p
                return None, False
            
        # if we are out of the if statement above then this current root is a parent of p
        # now we need to check if this root is also a parent of q
        
        # first check if this root itself is q
        if (root is q):
            return root, True
        
        # we found p on the left but q is not on the left, so search for q on the right
        if (lp):
            return root, self.gatherPathQ(root.right, q)

        # we found p on the right but q is not on the right, so search for q on the left
        if (rp):
            return root, self.gatherPathQ(root.left, q)
        
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        if (not root):
            return None
            
        if (root is p):
            return p
            
        if (root is q):
            return q
            
        np, nq = self.gatherPathP(root, p, q)
        
        return np
     
     
     
####################################################################################
# Serialize Deserialize Binary Tree
####################################################################################     
     
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    # l is a list
    # we can do a BFS and in this way we will have the usual configuration
    # i is parent 2*i + 1 is kid left and 2*i +2 is kid right
    def serializeHelper(self, root):
    
        if (not root):
            l=[]
            #print l
            return str(l)

        l = [root.val]
            
        q = [root]
        
        while(q):
            n = q.pop(0)
            if (n.left):
                l.append(n.left.val)
                q.append(n.left)
            else:
                l.append(None)

            if (n.right):
                l.append(n.right.val)
                q.append(n.right)
            else:
                l.append(None)
                
        return str(l)
            
    def deserializeHelper(self, data):
    
        #print data
        data = data.lstrip('[')
        data = data.rstrip(']')
        
        data = data.split(', ')
        #print data
        if (not data or data[0] == ''):
            return None
            
        t = data.pop(0)
        if (t == 'None'):
            return None
            
        root = TreeNode(int(t))
        # this is BFS in reverse
        q = [root]
        
        while(q):
            n = q.pop(0)
            
            t = data.pop(0)
            if (t <> 'None'):
                n.left = TreeNode(int(t))
                q.append(n.left)
                
            t = data.pop(0)
            if (t <> 'None'):
                n.right = TreeNode(int(t))
                q.append(n.right)
            
        return root
            
        
            
    
    
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        return self.serializeHelper(root)
        
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        return self.deserializeHelper(data)
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))     
     
     
     
     
     
###########################################################
# Search 2D matrix
###########################################################     
     
class Solution(object):
    def extractColumn(self, matrix, col):
        return [matrix[row][col] for row in len(matrix)]
    
    def binarySearch(self, l, target):
        start = 0
        end = len(l)
        
        while (start < end):
            mid = (start + end)//2
            if (l[mid] == target):
                return True
            if (l[mid] > target):
                end = mid
            else:
                start = mid+1
                
        return False

    def binarySearchCol(self, m, col, target):
        start = 0
        # this is how many rows we have as we are searching within a column
        end = len(m)

        while (start < end):
            mid = (start + end)//2
            if (m[mid][col] == target):
                return True
            if (m[mid][col] > target):
                end = mid
            else:
                start = mid+1
                
        return False
        
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # my idea is to find rows and columns where the number might be in
        
        # first search rows that might contain the target
        # we search using binary search
        
        totrow = len(matrix)
        # check number of rows
        if (totrow <= 0):
            return False
            
        totcol = len(matrix[0])
        # check number of columns
        if (totcol <= 0):
            return False
        
        rc = 0
        for x in matrix:
            if (x[0] == target or x[-1] == target):
                return True
            if (x[0] < target and x[-1] > target):
                if (self.binarySearch(x, target)):
                    return True
            elif (x[0] > target):
                break
            rc += 1

        # if we've checked all rows and did not find target
        if (rc >= totrow):
            return False

        # now we check each column
        for x in range(totcol):
            if (matrix[0][x] == target or matrix[-1][x] == target):
                return True
            if (matrix[0][x] < target and matrix[-1][x] > target):
                if (self.binarySearchCol(matrix, x, target)):
                    return True
            elif(matrix[0][x] > target):
                break
            
        return False
        
###############################################################
# Validate BST
###############################################################        

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def getMin(self, root):
        n = root
        while(n.left):
            n = n.left
        return n.val
        
    def getMax(self, root):
        n = root
        while(n.right):
            n = n.right
        return n.val
    # sadly this approach is too slow
    # need to use in order traversal to speed it up
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        if (not root):
            return True
        if ((not root.left) and (not root.right)):
            return True
            
        
        if (root.left):
            if (self.isValidBST(root.left) == False):
                return False
            if (self.getMax(root.left) >= root.val):
                return False
            if (root.left.val >= root.val):
                return False
                
        
        
        if (root.right):
            if (self.isValidBST(root.right) == False):
                return False
            if (self.getMin(root.right) <= root.val):
                return False
            if (root.right.val <= root.val):
                return False

        return True

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
        
    # let's try in order traversal
    def inorder(self, root, l):
        if (not root):
            return True
        if ((not root.left) and (not root.right)):
            if (l):
                if (root.val <= l[-1]):
                    return False
            l.append(root.val)
            return True
                
        if (self.inorder(root.left, l)):
            if (l):
                if (root.val <= l[-1]):
                    return False
            l.append(root.val)
            return self.inorder(root.right, l)
            
        return False
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        l = []
        return self.inorder(root,l)    




#################################################
# Binary Tree Level Order Traversal
#################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if (not root):
            return []
            
        q = deque([root, None])
        r = deque([])
        l = deque([])
        while(q):
            n = q.popleft()
            if (n):
                l.append(n.val)
            if (not n):
                if (q):
                    q.append(None)
                r.append(list(l))
                l = deque([])
            else:
                if (n.left):
                    q.append(n.left)
                if (n.right):
                    q.append(n.right)
                
        return list(r)
                    
            
        

















# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def getMin(self, root):
        n = root
        while(n.left):
            n = n.left
        return n.val
        
    def getMax(self, root):
        n = root
        while(n.right):
            n = n.right
        return n.val
        
    # let's try in order traversal
    def inorder(self, root):
        
        if (not root):
            return True
        
        if ((not root.left) and (not root.right)):
            if (self.l is not None):
                if (root.val <= self.l):
                    return False
            self.l = root.val
            return True
                
        if (self.inorder(root.left)):
            if (self.l is not None):
                if (root.val <= self.l):
                    return False
            self.l = root.val
            
            return self.inorder(root.right)
            
        return False
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.l = None
        return self.inorder(root)
        












tt = [9051,5526,2264,5041,1630,5906,6787,8382,4662,4532,6804,4710,4542,2116,7219,8701,8308,957,8775,4822,396,8995,8597,2304,8902,830,8591,5828,9642,7100,3976,5565,5490,1613,5731,8052,8985,2623,6325,3723,5224,8274,4787,6310,3393,78,3288,7584,7440,5752,351,4555,7265,9959,3866,9854,2709,5817,7272,43,1014,7527,3946,4289,1272,5213,710,1603,2436,8823,5228,2581,771,3700,2109,5638,3402,3910,871,5441,6861,9556,1089,4088,2788,9632,6822,6145,5137,236,683,2869,9525,8161,8374,2439,6028,7813,6406,7519]




class Solution:
    # @param {integer[]} nums
    # @return {string}
    
    def findTrouble(self, clm, d):

        for i in range(len(clm)-1):
            if (len(clm[i]) > len(clm[i+1]) and clm[i][:len(clm[i+1])] == clm[i+1]):
                if ((clm[i+1] + clm[i]) > (clm[i] + clm[i+1])):
                    return i
                
        return -1
    
    def largestNumber(self, nums):
        
        # I think we need to group by same starting digit
        
        # process digit by digit starting with the biggest one 9
        r = ""
        
        strnum = map(str, nums)
        nm = ['9', '8' ,'7' ,'6' ,'5' ,'4' ,'3', '2', '1', '0']
        
        for d in nm:
            clm = [w for w in strnum if (w[0] == d)]
            
            # for '0' we just need to cat them all
            if (d > '0'):
                # Python's internal sort already almost got it right the only thing is that when there is something like
                # 391 and 39 they will be sorted as 391 39 and if you just concatenate them you will get 39139 < 39391
                clm.sort(reverse=True)
                    
                # the thing is that we might have something like 51 51 5 in this case we need to search for 51 5 twice
                # because even after we shuffle 51 5 to 5 51 the overall status is still 51 5 51 which is still not correct
                # so we need a helper function to scan the list again
                j = self.findTrouble(clm, d)
                
                while (j > -1):
                    # we need this loop in case there are multiple items with the same value lie
                    # 51 5 5 5 so we need to shift 51 down multiple times
                    for i in range(j, len(clm)-1):
                    
                        # I initially tried different methods to decide whether to swap or not and each had a hole in the logic
                        # but the simplest actually works just try each combo and see which one works
                        if (len(clm[i]) > len(clm[i+1]) and clm[i][:len(clm[i+1])] == clm[i+1]):
                            if ((clm[i+1] + clm[i]) > (clm[i] + clm[i+1])):
                                tmp = clm[i]
                                clm[i] = clm[i+1]
                                clm[i+1] = tmp
                            else:
                                break
                                
                    # scan again to see if there's still some problem
                    j = self.findTrouble(clm, d)
            
            
            r += "".join(clm)

        # this is to check we return something like "000000" which is not accepted
        if(int(r) == 0):
            return "0"
        
        return r       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class Solution(object):
    
    # check for palindrome
    def isPalindrome(self, s):

        if (len(s) <= 1):
            return True
            
        for i in range(len(s)//2):
            if (s[i] <> s[-(i+1)]):
                return False

        return True

    # let's try using dp on this one
    def trydp(self, s):
        
        # to keep track the starting and end point of each palindromic substring
        # d[i] = j means there's a Palindromic substring ending at i and starting at j
        # we need this to build up the list of results
        d = {}
        
        # so we scan s and check whether the substring ending at i is a palindrome
        for i in range(len(s)):
            # so we are at i we want to check if we can find a substring ending at i that is a palindrome
            # there are two possibilities either this substring ending at i starts at another previous
            # palindromic substring or it starts from the beginning so we scan every substring starting
            # from i and going backward to the beginning
            
            # something to note here is that this new substring must start at another palindromic substring
            # or it's the first letter in the string so we add an if
            
            for j in range(i,-1,-1):
                if (self.isPalindrome(s[j:i+1])):
                    d.setdefault(i, []).append(j) 
                    
        return d
        
    def gatherResults(self, s, d, n, l, r):
        # we will build this recursively
        
        if (n not in d):
            return
        
        for i in d[n]:
            l.insert(0,s[i:n+1])
            if (i == 0):
                r.append(l[:])
            else:
                self.gatherResults(s, d, i-1, l, r)
            l.pop(0)
            
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        
        d = self.trydp(s)
        
        r = []
        l = []
        
        self.gatherResults(s, d, len(s)-1, l, r)
        
        return r
        
           














class Solution(object):
    # check for palindrome
    def isPalindrome(self, s):

        if (len(s) <= 1):
            return True
            
        for i in range(len(s)//2):
            if (s[i] <> s[-(i+1)]):
                return False

        return True

    # let's try using dp on this one
    def trydp(self, s):
        
        # to keep track the starting and end point of each palindromic substring
        # d[i] = j means there's a Palindromic substring ending at i and starting at j
        # we need this to build up the list of results
        d = {}
        
        # so we scan s and check whether the substring ending at i is a palindrome
        for i in range(len(s)):
            # so we are at i we want to check if we can find a substring ending at i that is a palindrome
            # there are two possibilities either this substring ending at i starts at another previous
            # palindromic substring or it starts from the beginning so we scan every substring starting
            # from i and going backward to the beginning
            
            # something to note here is that this new substring must start at another palindromic substring
            # or it's the first letter in the string so we add an if
            
            for j in range(i,-1,-1):
                if (self.isPalindrome(s[j:i+1])):
                    d.setdefault(i, []).append(j) 
                    
        return d
        
    def DFSWithDepth(self, d, n):
        
        stk = [(n,0)]
        
        lm = float('inf')
        
        while(stk):
            t, l = stk.pop()
            
            for i in d[t]:
                if (i == 0):
                    # print l
                    if (l < lm):
                        lm = l
                else:
                    stk.append((i-1, l+1))
                    
        return lm
        
        
    def findShortestCut(self, s, d, n, l, r):
        if (n not in d):
            return -1
        
        for i in d[n]:
            l += 1
            if (i == 0):
                if (l < r):
                    return l
            else:
                r = self.findShortestCut(s, d, i-1, l, r)
            l -= 1
            
        return r
        
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        if (len(s) <= 1):
            return 0
            
        if (self.isPalindrome(s)):
            return 0
        
        d = self.trydp(s)
        
        #self.DFSWithDepth(d, len(s) - 1)
        
        return d
        #r = self.findShortestCut(s,d,len(s)-1,0,float('inf'))
        
        #return r - 1
        
lst = "apjesgpsxoeiokmqmfgvjslcjukbqxpsobyhjpbgdfruqdkeiszrlmtwgfxyfostpqczidfljwfbbrflkgdvtytbgqalguewnhvvmcgxboycffopmtmhtfizxkmeftcucxpobxmelmjtuzigsxnncxpaibgpuijwhankxbplpyejxmrrjgeoevqozwdtgospohznkoyzocjlracchjqnggbfeebmuvbicbvmpuleywrpzwsihivnrwtxcukwplgtobhgxukwrdlszfaiqxwjvrgxnsveedxseeyeykarqnjrtlaliyudpacctzizcftjlunlgnfwcqqxcqikocqffsjyurzwysfjmswvhbrmshjuzsgpwyubtfbnwajuvrfhlccvfwhxfqthkcwhatktymgxostjlztwdxritygbrbibdgkezvzajizxasjnrcjwzdfvdnwwqeyumkamhzoqhnqjfzwzbixclcxqrtniznemxeahfozp"        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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





















class Solution(object):

    def aNewHope(self, s, t):
        ln = len(s)
        
        # first build a dict with locations for each char of t but the characters 
        # might repeat and we have to keep track of it
        dt = {}
        for i in t:
            if (i in dt):
                dt[i] += 1
            else:
                dt[i] = 1

        ds = {}
        for i in range(ln):
            if (s[i] in dt):
                ds.setdefault(s[i], []).append(i)
                
        # now we scan from the right until we cannot scan anymore
        # i.e. we can no longer find chars of t in s
        
        if (len(ds.keys()) < len(dt.keys())):
            return ""
            
        
        for n in ds.keys():
            if (len(ds[n]) < dt[n]):
                return ""

        # this is how you deep copy a dictionary
        #dtmp = {key: value[:] for key, value in ds.items()}
        
        cmin = float('inf')
        ts = (0,0)
        
        # create another dict for min index to speed up search
        dy = dict.fromkeys(list(t), 0)

        for i in range(ln):
            if (s[i] in ds):
                #print i
                # we see the min window needed if we start from here
                cni = dt[s[i]]
                
                
                # but we need to know that we have enough char s[i] to complete t
                
                # to save time we don't use index function
                #ind_i = ds[s[i]].index(i)
                if (dy[s[i]] <> -1):
                    ind_i = dy[s[i]]
                            
                    lsi = len(ds[s[i]])
                    while (ind_i < lsi and ds[s[i]][ind_i] < i):
                        ind_i += 1
                        
                    #assert ds[s[i]][ind_i] == i
                    
                    dy[s[i]] = ind_i     
                else:
                    ind_i = len(ds[s[i]])

                if (ind_i + cni > len(ds[s[i]])):
                    return s[ts[0]:ts[1]]
                
                # so start must be from i
                st = i
                
                # if we have enough char left
                ed = ds[s[i]][ind_i + cni - 1] # this is a potential end point
                
                # now we check other char
                for n in dt.keys():
                    if (dy[n] == -1):
                        ed = -1
                        break
                    if (n <> s[i]):
                        # count the number of char needed for this char
                        cn = dt[n]
                        
                        # find a location that makes sense i.e. beyond start
                        y = dy[n]
                        
                        lsn = len(ds[n])
                        while (y < lsn and ds[n][y] < st):
                            y += 1
                        
                        dy[n] = y
                        
                        # either we don't have enough char or all the positions are not good
                        if (y + cn - 1 >= lsn):
                            # one of the characters doesn't work so we must say that starting from location i is a bust
                            dy[n] = -1
                            ed = -1
                            break

                        # if the position is good update ed                        
                        ed = max (ed, ds[n][y + cn - 1])
                        
                if (ed <> -1):
                    clen = ed + 1 - st
                    if (clen < cmin):
                        cmin = clen
                        ts = (st, ed+1)
                    
        return s[ts[0]:ts[1]]
        
        
    def slidingWindow()
        
                
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        
        # my gut feeling tells me to first scan from the left and then scan from the right
        # making sure we still have the substring t
        
        ln = len(s)
        
        # first build a dict with locations for each char of t but the characters 
        # might repeat and we have to keep track of it
        dt = {}
        for i in t:
            if (i in dt):
                dt[i] += 1
            else:
                dt[i] = 1

        ds = {}
        for i in range(ln):
            if (s[i] in dt):
                ds.setdefault(s[i], []).append(i)
                
        # now we scan from the right until we cannot scan anymore
        # i.e. we can no longer find chars of t in s
        
        if (len(ds.keys()) < len(dt.keys())):
            return ""
            
        print ds
        
        for n in ds.keys():
            if (len(ds[n]) < dt[n]):
                return ""
        
        st = 0
        ed = ln -1
        
        # make a temporary copy of the dict since we are going to modify it
        #dtmp = dict(ds)
        dtmp = {key: value[:] for key, value in ds.items()}
        print " ~~~~~~~~~~~~~~~~~~~ ", dtmp
        for i in range(ln):
            if (s[i] in ds):
                # stop here since if we move beyond i we won't have s[i] no more
                if (len(ds[s[i]]) == dt[s[i]]):
                    st = i
                    break
                ds[s[i]].remove(i)
        
        print st
        print " ~~~~~~~~~~~~~~~~~~~ ", dtmp
        # now scan from the end
        #ds = dict(dtmp)
        ds = {key: value[:] for key, value in dtmp.items()}
        print " ~~~~~~~~~~~~~~~~~~~ ", ds
        for i in range(ln-1, -1, -1):
            if (s[i] in ds):
                print ds
                if (len(ds[s[i]]) == dt[s[i]]):
                    ed = i
                    break
                ds[s[i]].remove(i)
        
        print ed
        
        return s[st:ed+1]
                
s1 = "kgfidhktkjhlkbgjkylgdracfzjduycghkomrbfbkoowqwgaurizliesjnveoxmvjdjaepdqftmvsuyoogobrutahogxnvuxyezevfuaaiyufwjtezuxtpycfgasburzytdvazwakuxpsiiyhewctwgycgsgdkhdfnzfmvhwrellmvjvzfzsdgqgolorxvxciwjxtqvmxhxlcijeqiytqrzfcpyzlvbvrksmcoybxxpbgyfwgepzvrezgcytabptnjgpxgtweiykgfiolxniqthzwfswihpvtxlseepkopwuueiidyquratphnnqxflqcyiiezssoomlsxtyxlsolngtctjzywrbvajbzeuqsiblhwlehfvtubmwuxyvvpwsrhutlojgwktegekpjfidgwzdvxyrpwjgfdzttizquswcwgshockuzlzulznavzgdegwyovqlpmnluhsikeflpghagvcbujeapcyfxosmcizzpthbzompvurbrwenflnwnmdncwbfebevwnzwclnzhgcycglhtbfjnjwrfxwlacixqhvuvivcoxdrfqazrgigrgywdwjgztfrbanwiiayhdrmuunlcxstdsrjoapntugwutuedvemyyzusogumanpueyigpybjeyfasjfpqsqotkgjqaxspnmvnxbfvcobcudxflmvfcjanrjfthaiwofllgqglhkndpmiazgfdrfsjracyanwqsjcbakmjubmmowmpeuuwznfspjsryohtyjuawglsjxezvroallymafhpozgpqpiqzcsxkdptcutxnjzawxmwctltvtiljsbkuthgwwbyswxfgzfewubbpowkigvtywdupmankbndyligkqkiknjzchkmnfflekfvyhlijynjlwrxodgyrrxvzjhoroavahsapdiacwjpucnifviyohtprceksefunzucdfchbnwxplhxgpvxwrmpvqzowgimgdolirslgqkycrvkgshejuuhmvvlcdxkinvqgpdnhnljeiwmadtmzntokqzmtyycltuukahsnuducziedbscqlsbbtpxrobfhxzuximncrjgrrkwvdalqtoumergsulbrmvrwjeydpguiqqdvsrmlfgylzedtrhkfebbohbrwhnhxfmvxdhjlpjwopchgjtnnvodepwdylkxqwsqczznqklezplhafuqcitizslzdvwwupmwqnlhxwlwozdogxekhasisehxbdtvuhrlucurbhppgsdoriyykricxpbyvxupencbqwsreiimclbuvbufudjrslsnkofobhptgkmmuuywizqddllxowpijhytvdkymzsulegfzfcjguojhzhxyyghhgbcllazmuuyzafahjjqgxznzinxgvgnbhrmuuljohjpkqpraahgajvzriyydengofskzgtppefzvwrvxadxjaydjydocqvsxpdyxyondvmyrfvqiaptanwllbaquxirmlqkmgzpbnputmldmcwoqvadwavqxeilraxdiwulmlffxsilvgcnbcsyeoqdsaolcorkmlxyzfdyznkuwmjxqcxusoxmqlxtzofocdmbiqzhflebzpbprajjqivhuvcvlhjnkwquosevfkzfzcwtcietqcamxcikltawrsshkydsiexkgvdidjbuldgkfqvrkxpdpjlakqsuurecmjkstomgrutzlqsxnjacuneedyzzrfbgpoykcmsvglwtdoqqztvugzakazlrhnxwdxifjccsozlrpckpxfldglpgnbauqzstxcaiecaudmotqyknfvsliiuvlurbvjwulwdsadmerazjyjydgrrobnmmjdpeplzcjcujhhpbhqmizlnhcgwftkrcnghctifcmbnvifwsvjcxwpeyycdrmwucedexnlbznquxvtpretoaluajxfajdwnhbugofjpuzmuxflolfenqynzxubjxawgbqmsyvhuwvotaajnfpaxqnwnjzwgzvmdnaxlxeiucwpcyzqfoqcegaspcqybnmgbndomkwgmvyqvxgblzfshimykeynslutaxjmjgvvdtmysubfvjxcrjddobsgombomkdvsatvvfwnzxsvgszzbccrgxzulclzpqvhsqfnvbcwywrfotgsxlldilatnntcfqmxgrkdsozsktzbogtlrerzrtuhiplnfxknqwsikudwncxdiqozxhaoavximjvuihjzdcjpwmmlploxeezbmzrmwrxlauficojhqtxohlzwwpwcuvfgwzuvqrgqmlaozmxshuiohingzjitgobcnwzdpfvdsxrujroqlwhvgubgdlzjzdnozptqwqurqnlzezssvznctokybljdoyrppngmdcdvpqvuppmmqbqlrajsmuvcupskcawhcbdrrangrbuhcnstndobzjgtyydcabkccpvtpjbgmyprljkamaelkkgkkmnknzosojnfupnhncyalkazlemxoshaewkuvymjkzqeqdlfflfsygrjmdidypdcmoyjoktykldapqiwenpcviniovnmkqqygpivbdvloaoftwcxltzhbmrrhedtuuudleamjvaxwqfrohozcpidbzxkfafcwbfjffwocyoaotrccfdeumjxngjthrvfsapyhnojlcmbxddzlidhwnhktqdcjykcazcjoqszveaskfsvnxckkjwgczknzupbvtkjmeihlkzvptqfrurdgnjkouxpqpqmicvugebrqdmgyenpbethrerhaqwrfodoqaiyemqxredpjqhkwctpgmwjcsaiifyyfiwmuojntmdeemqaolpwxnfbffjpmjnssleocncxbhbhttjjeyfdllessqjfzwxtjdilsmivvlcqglzmlepyrwskmbrnzqhivrwnfgiasmsaxrnkxeipaaboduccklmfckuhrcjlqblnuaxrfhihjlwofyqrleynpswiwhvmigbejavojgvsrtgztysefrrulpekwzwghakqaigkocehxnirlbvqspmfgqpdrolzowrqgycuchdzumqhnmyhdmojfeowsaxiypyenbapidoerhletlnyychdgwbayziwoicbjcsthixzplfkwtiwvsbdodfocpksxmvhqnczvaylnexjxgguyhzomecotoiqcdzuqctoesbrwyavgiresquewyvrmdhqhjkzleppwqgupirxtkcncytyxqpjuyadhmeuqulomtidcbbxlfmndfnawcmsdoxkadhtzshmmsrotsnfxzudgifcmtwpjtamzhfybmkneedawqhwrbzyjgawaznjunvtwcsypenvirvhhcdbgezrkbnmadyvsvopyippnckxviedmjgsnfkaizmjckgviwmghdvwhhtdpaicjowxvgzdglokyufgtroawjwesrhirrmbfiacrzfzmujmqpujiilftjlmdswulkxquzslyzkltuzmluxtcjawulkxfguqqrikrcwreiezeelpyjlaulyqziogqizgbhtsmrmqzqreudvsogviuvyveobuyadakwuwngomxfsrkomywhiqejlixnfwpiwzesxrznfwvapfobekkmdpxqzvdettopusjsliftgatwijzmvmaydypcvujopxfksocfxjydmrbuabiwpkwsklwfihtxhahzpleoxftjwvfzynxnzthkhfppnloulyvajbqigktdhyefnbunwdxfiowxzltljonuqgfwqybxzhemkybjdyolnnjmaczjtpfjvmhlllkkuoyhoaqrageptnetsdelumwpyfclsoxpewwsymlasqqofuhzxomucijaqookclzhjxbhjniefaukudkrrwlthxwznklpvnyfkaowpyauyugsxsmrrzmayiblsmdqzdxmfniuoiqoezjdtvukwhmxrnkkcncbtzpyoxvchnrlmarixnuvwhsowfayuccndrmrpjuwwnklnbtmqnhnwcbthbrbfvpkndbemxaikmvzpqlgettcxwvezpfgmwqzzrfnriybutdmkosqjnsglqkkhsqtogvqzadsibudvzphnjxxyrfjhsvokniyvdowfupovlrskizxtwwroecxwzmgmwghbxdgruumfnfpxensdlltpnqloeraayjdxpmojiapwhgvotorhbmypckdjdgjdrpagbosjrhhyndojthsutqlwcrfizqlqhozieruvabwpgmabusynljlncsvbljusztddkxbkyzbhlcifugthsexuxsykdsccnfixcljdkkkvmudjbwstcppbkplnhqsuvctanedxppudjxomvhhywzbgonwydquasoahsfejuangybsvkctjbxppknnpfirxyugszmcwnpcnswifycobbfrltgcaovaopghptjngartunbzofppgvitqpxqftquixbzqmmwmdrituijaxaiujckabbfwrbfqeggqveeaxntsxcuwfcrqgbqiexgybnofuxypdepbrltqpnnurgkyjcioaoobcciybgnflegerzvhokdfqofzsnpsedvgieejmtoxzuervzajldrbtwcmmsgqvcyfiepuzduayyrvztfkxylquneihlyfpykxczrddledlxykrfwofjgcznkgyllnjwkovdrarxfcvepmllatvivuvfcvsoickushylirjntetkqhwsatcvpyctvvheztardaenrncnrwxjfvbhqechegbzdifcegvhiauapwnhukqbiuiyamgethfwrnbvvyedyadozsxvfpxnlhllutrpaxnumorhnyknyqdziavpsucdbqpxjimmhaitqzadxltybwxlzsbrofwqxlnjwcvdcfxsexyektcnpbbqucjkjgahtqqpsntwssoyrocchgrzispewzoghrpajfqbulxmdnmuerylxqmhkenhfcpmvemelehfretwtftxwrlwjxfwdtdivuwsalwlpdsjjlfcqyapsnnbmsxqlcaiyxayiylzbdimoophorygelkqdhirmjzmgcloaynecyqsofbcyzjemtvscfjqokdumqknoarsyjnoroqpqbucwrwbwhtlswhgouvfuoxuykipbagreavatudqbxdvvmekgzpaqowobgfchlvlosnhotxsqcnnptxvtowlduzebgiirfvfzkpofmgxpvlgpibkxzvcuwivcqfvxcbwoqkueqrvmcbdnfnmeioaewxiocdlgehvwurdkkyypcdchqonaeoealmqqqbwwktvemyrxyrocvqlngzokpsmahcszfrvrmsyzsryrkmvfehvkgjwxdixkmsjtjhmvbkwubwnmiitopoaxxwudgunumznxiasjmrfqnscybxmsonqnlmquigowfetpeoasfgiuymsbmhuawmphagbjmsftwbkcpkuusdqrrjudqqdmetvfbzqprvkpwurnenjxsaqkjmnbdtphomorlegecqtammoqazpuzhekuunzpcidpxwcdcjhueigryytxqnzzujtqxufbdkscgxfkgpgkxmdxmwwemxegjzwgudjxncbvzifhxlrwzvtntssfpwnyxlgrosqduryvadcxqvupspdmjxtbvhbssimjacowwntysvvjsraljfxscqvxzxsuhedjirfegyczvakntkycqxxuqsuprmlysqhakofqojrjjbzdozhgxgapbwgskstciafsrkjfvapheqmsptaoccddzkxjeqomttfkfqpcsgjywvqopsctviwuuvfymvkhortvhiycrhapftdwlipgqlcmikkufwdwowtxhrbybcpgvvcidrpvethmdtjlpmhfjadqugqxifffchofafcgylueefpwuybdagvunntvuydxhrehwhpwukazrrvlpyqsflmsaipvguuolyxjhniczkdqcyetyuldiaxiipfojzexacghpqlqboidomwnhispkqzshfiqgnngpwqgmbwnqesgtrtabmrleqdlxldeatmrrcxfgvvycneveaxhoossgxfglimlbydudcajavhcilzpnwwbmrtuoaazjmlmlqhqshzseiwotxdjvckhteeheejprueemlwguvbydzmyxshswscpygyemhwfdajpnhyhczhhytivnpqjjsyjazqmgmsmoddblbipcpxbkhyawqjiiktkjjzrxrflwjmjmwbpnysahqafhjnrvjegsdaswfdsqsedofuefmemegrnrfhnolviovakdgetaiyonuusgyeneyawdjltugdkegwhobcojdezxztgzatgyvcdhpwbxbobhkixlnlxqqypprvotquroyuvpsynumodzzbmmmlecjvtdtiwjeozdiusdvhxhwcgxdvlsgpwqmqvfarrehqjsnevilurjwagcvrwbistviockitprkyjxcghqayzzygdtzzvirqfcfhmpbdgnesmgatydrycqgflheipxzwbggovjtdwxxigydedwefommausilphirpohmxasvypfepiksepzvblvvdfhvnrmehvrgvjbvagbsqmmwdmrezmcfslaheonljergpseqafkstwowiibkwfpoqrxwfnhqryyjsczukjmdfcaqkdchirxakxwpkfhbffkxkltuwfxehxwscybkpymzvkqfpzjuevtqjmkfrilbhhvkfwwwwjxutpzlokfviblrnwyhgkinrfzzbwxzhvtcmvnbhvwpwjilfhsntadmhclkyjkfgdksaxviutxqdgckpuyixbugfiretblzgvthvpppioilwmyliwvhzsoeafktgwumtnvqckinbqyxcwlkugstygaankttinhedfuhrcusstswrdjojbjkjjkyugtcvcgyhdgzfaravlwpohdaimktwwtscmwypdywigvnjppeaotrvyrglbbjzvbchxcwcctkjqashpykdubzssfdvgowbpalnchrhccsvekctkozepazjhegntdridcxilrpovjlzvnufctmttlfcpnqiqjtwnsgajxqegbdbrygvtuopfvrvjjfbsyxhrdkaaahickjtksoemetuakpjwwmqvkyopiqskxamkkhuexyqctkegbpcybsvnsjdcbvnzvbjhjfligekzoqhshlqjenwywbbxwqyurjbcpnlvrxuhqezxprgvefucgxcfnazgkalbpwwivqslwtlmlthrydwhaampcnyopjpfhlpcehqabdmowwhxzdecdsrihrwwambjxrmbaecbhqpfmxcmcioichqjbmgbyjlyczzdfbeoswvgvysziihlszwtocwomjmqkmtrpafjwdqtbksjvcwpdtkrxiglsuceivwyvdjtgbmjohvljrammmgkumvogztpvrpswaodeaosjjdsdfhprnblbzajyvavmpqksenwgcoqntkqytirglehketlbtiplyadepzntpdhkpxkjhbptfzfmsspnbfybfbiwcvqtyxdpwpyeqqzyrgklzbycgxdnankfiayizeyvtybmoakfjwsixrmgqptsffxnfywgcwcxudjjgvqrrjzralxscskhyixfitmkpqjjttubonsiwtbygaqlscuskrysmmedcopxjjbzytjupksxkkifnfxzyxuljyqloflzrmzpfikwveuhremejmfijzvtalgotrqkdpblznhwsdelisrtewdowyjwkpmdjcpmtqzvehrymxjwqaqwytmuuvsgnhlwjakfskayttwfrhejjufipsejpxrcecypeluxvwvqquxhdbqnxxxnpbyfyqjcukszvowsltibpfktjcggzdvrgwwfofjipdjmshefbmisuslfutlvyjmfhkkvstfhxpwrwawzpeslydquxdpvmeatomqgiwqflmyjmwjdadoaieufkldwpfseuimcgejtqhdfdoiftzfbfbjpmmmctginqwpxbysthxymljreiyinzrdmdrqyzampydozejvngtaueraosicrzhfcaxdzzrqyuhzaquoyqoswhtlzbprbyyfywizvbrvwyvyqrmpjajgicrjegaheboexzduauqyvnxngjcqmmwwqpfwvidbjufitkctgusbjrpqatiohekytsuqaatjuytpkvkqsesdvjvwedmmjxmepgupinaenvtxseqkiiogtlexpqlynxdeezvopvteqoejuuvfellxxpwofimvfrzlrvaisvmllswgmtlqfypenmsuugrbyrnropbgkptvipdujonqocudooykurmuibnwqofceyzoqdiztvpiylloblzxdnsnohmewtgcrfqhkaieyzowmbmnplluafhomvsioamiiersfaydboboqnbfbobbtiqgekqsldcthunouorvcsjuinbjbcmvlcdrptlayoviikweyqzdklxsabdbdqyqubhjxvlxjiftvexvuyyejyzlzcncuntprcaoxmniwtbrzynvxnbilaumdjrxykopopfodtwboyvpyikxxlilmcxrnoqlahdrwifbdberbzgahsxhefssjrjygbkiipzgxehdimujldjvxjebowtaneyvgkgqzfxzmgcusydgdpdbyjcohyowcpskzabbfyatecnzcthgimhgvlucllyqasazsdcadctkgjljcurmgaudnqhzbhpdarduxfwakqmqbxbfrurzuncidljhtykvrproxorhgdzhbdouxhusuycxkleflpccgttdjkkppmyqpmnmthgnhintvtygkclrweufqhxftvyiklwfudtdlixjbxpmaafygzfyicebaejmpirllmoyunhylzknbrlissgbxmcxqojhwcsjpjhjwphjotwpkjzfcigbwkynyjuwlpfaanmweviupelbqnguvovrzvgxedwrubuuqiqwdyqufcwxrermtofujotprpintchjkziqaykhcietnxhilpcudvlwntjgysrkrbaralyeteyibhsmwuibsvxhpippaizskalknkqiqqrsyjeugpwakvhbeyqggqyqskcfwtnmlggjqlgyceymzhqfhgmnwmgqykvufljrrpajcghgkhvmqltxrhlqutgsdepjpaairasujbcvjxhzidxckksedazgeopvqrehsybbbgykdimmllovxicaidiprakhrqqjqumsaledtsrgfhdcpfcailpbnbkeudokxcexplaifsqlbunrbstmdipslcwffmscpyzejbppvfcxbpqdhjrmtgeeibknnepecqjosafphhmjyeognfuglznwczlrddwfthvnjadqjrbiwpolwiskjhgngrhjudqqmmdvoinycpgagwldgfnqmtfsmmbjruzdgeawnqurfarlwcodidvuwcplpquimtosxormeldlouzkxnphrmdppsaglwoxpcsptjbzufnkbmvabmhublmsfoyqlooatiportpnacybovesjnmkaycgncwyrmakwsannpdnsdqiyuchipuipyqebpesdheazgpovsbpvtavqcfatzkjxbpaquutdyveelscdtrboryeudenepyesoeikzdpzztgmlkpgbsuvrzrwfgdufhmvnwhocwqdpypaigilzcgsdxqjetflcuiqspgulwxcuoeevmquowouwedcakncevuzkbtxztbwhfacrqyhnoblvgnvrazxxwwjxsgynhuwoxbfnzocqqnnyakrcjpzfniehqfuzbrnyssmthqlzeyxgipjugkbttgblnxloqynkrbgarrzxqyganpuvbwpuesgrrtfiruukvakpslmkmskbjhxlfohjgpczijjaeembexqykdiwxvyjevujjdurtwsdhcdliqnopfnbixyisyrytnheqbxjzpnhtsubkjqbpddzcbjxbavundaegxpgoepizljjvbnmgqljtvguxbqlgqrnpitdnknmvxxqakehcqqvnwpmzfxgxqtmzosoafcvebofosmqkemzcdmbllhiccxdsrtswvodweimwudjaricuudfgnqirpnkigcedshbyrovnreniopejtmpiezztolduqhzkrajkoeyhklrybwjwiaembfhprkljtktmncxdivluievjaehkhlyithymrjvnwqbjtymdvzyeevpgzxrikprdzprqaofyfhhahwffvdlbaaicxbkksbprshktvprcybcixinunyyoagqajckbeztwgdreulgvmldltshisfyunquwteyzgtvowpusabpomsbhmirgqqbdixcbeaaktyarnvlpwbdkvgtkqmqgetmnrooxqrhjrjceepvcqaooghywwqdamrnffecvxlgoukuzzrdsrsgxbgvxyaykrnrytttsebgblccntffmxnlzvhwthwgbmxzhtvaxwiaklcxgietfregonfhxdpyppzziwzhthifukcdsgyazegnslwrpmrgbqflpgskoapkntpfznhauopoblfszwzcoendlipbienvxfdyukaoapccbjuchvhwcubncrqnxfkvknvfawtalyeojbtrwapywqnbjfohlyexozcovyqjyvzhysywsnvpgkqpseydecasefibdzmdumkuelqxanmgyeyskyvodxjherkhqxmmjxgpkxmkkarkercqpzfszqzdhmnajzmuyvjiuytgymrfdvvsxclwsmbcaocjqqfolrzhpsopehwjcsbbwozpbbtbpxnhnuwblvicpwsvdqeiiflhwlwxmradoplbmjezencvlwqroubxbmexxcwjvzpjqamcjrepeikrgaiuwjrzwxxvbvhwuwflwuphltwrdgijiregcxfaveyyafxubehzyzgjueaymlhwelcgjhjgoheombkpgsqgtwqncslodvkhgmqrvzzjgezuhklpebwxxombgapkvztdnjxiiwuqctmkdxbhyzgxntywvngqobvblsmtpgmrbydfwfowgxuwsusniadwbaamtmikuubvufcsmtpuqlqifkxnkcmmcoavakfwgjrbqwtepkyhvrrbboqmqeasscqsnxotgiwwescvcbcyuvbvjrjzwjtoojjbhzwpjtopnqkptopnjqlskutigpyuyhxhjtxuonkbdwtzliowbzrlwktczrwabtjmoigfvsibpbacmqynspaocvqdoodrjndxvmetgnuvqwzcmyrgprurokvdaaujpahnmnguacstyrxmpptfinyxataawwklwtykggfjixegobtgsondblcdfeedaqoxlphrvocelimckhkevxhzilcppisqahplwyftnjxasmeetoiplsqudujtdelflarjywenheozqsdjhoaugqojnaeqepvrpocqmgdukrvcmzovpopvheguglmmcjdsyhimnlgafecrfsmuhbpqhxmpkabnjghjnrybcedjihanaojjsabbyptkpuxabemoxkrcqwlbeoqgeapwasaahtlwpiyspkjmuaqnzcodselwecvhuvhbqszfdzaskjggjxtkbhntoapscmzwjjzzbaheahykqhsbgmmjvbcjcteegfdpocrqdawdhxpzehamvovtqxeusiaaodseijwnjtqsqhrwqtimkdhcclwwuyxkcrqmanzlgrfyywworgcbljfpxbltakfebvqdiroqitogrmwlszodkushvgcxqhdudvxlmmcuhscbhzmzyafkuzusfwryexshspuhdybnreqadczdegpbgjvnionmlfvgxpncqqrhuhhzjieqrutxfomneabqhwyeoljebanyiwztgejxaewhujvzlyrsvpzlairdkgbscbokhzijegsyrnxcmqvjfwwhgnlqvmlzpzjyytjeacdrrcwfvrqapmivbtnwhavzbhzcalgcnlugqqmljclarvxuomrynozovyqwjzaykufcmtnxvbzowwjnjcnyqmmomfkyemyqhdtvwivujukitvxexkcqyqemvqkbcjulophnxejjavtvrqzocbeasimpyzrbknmacvnbbymeufzlxzbzagbuqqkrpduzqibsgpthpgnjnnwlmykogojyetrtwlumwvgcmzlehznvmmgamsosixsmqkhoxdctbixnkjrrkhoyvuyhpmbxktmcfttdxvyyewnofhhpqwnlsnshzvhlovowohznexlivbyaigrycgjutdevyngoiwkyflmklzugavjibcifjtbeyhzwwineexfklrwysqgtlpdosovdhxhggjedrrjxrohehdcawitjayyumvzusicbrgekanirervnvxnthicjjcfvyersnvtgczrcnzvtqqgnpwiygnippplpueihadnedgisfdlyvtnkrykylunakaqehanhdalihvaoynotjefacpynkopuqhaxjnepprxymanmndfwjkznczcmnsiebiblvickbzjqzycpdilnumcwmfkzzsfaajnvtpgurrkqpnhsxvhrhmqtravstautlirezzzqqljhyqbxllrvxhoojjiemcaxkrcmsdekqnrrguotffocodsaoaecdgapxedrjibzvuqdphxuonzdstfhwjjbqlqpruhaekcbfufubeqqrpvkjarhjcibqathwakammcrghlincaetvoevhlgmticwblsdvtphxjqgwataaxuxoysvmyjhindlzwqurieoruwnwpowmkvcknaynjgyxljwjcsakwqbsmqodgmsshudwubkejkdvtzevrfhqxkkmbzpjnjcxehtyifeuphpliticasuacemfddptntqtalrktyekhvxqpguraorcsfiyjztyslruykgrkncsibkzjgrjukmoqwobvhzpsslrerkpgohrqtqzzjjvuxjktalohmfceqvihfzqughmbzncjyxfvjrojeesqjuwbrglxgbtokhqjuuutszqdshowlgoxdkyurltzmonwvfisacluedxwklvwtjvwwvtphoovdduajcslgffmjcjtpgrirohnkcetsuqvykyjoquvyzhjdscuawcsklhwporiiifiudrjwngumpdrkhlmdqgqbqotegdoqixkoqvkedgqvlifsvtylaqpqeiwmlkcvhnjaiveobwjmgqhcjhnjxcbbmxozksvtfgtqcxefupucfbeoisahwbkjbailtfeyoyqsxwxtltwquaheuhlrkwclymyrsfsidiacfzwstujpuqurxhijfkxeyvvsanafyckfcgxxagmwyinxsxhxjedibbacqbjoftthbtgdbtaadpxvpgvykyimzjqqmzgrvcbwvhawccygtwdicajpzrdactsoubipdloasqyxsxfnviyzjhqkytmbofrjgbalmonheleykjohtmhctzmttzwhgosortbejolqbrqoaevtseylemfznditrbjjwkphacxetqsvwpqpwoaadhbqljmemvpkieskirobhiyeypvufxwifzbsinyxkohuuzhdrvgagfggwbcdzyogzpajeygqoonlpuqirwjxdrdbtffufvaekcoqaugrktcanskfqvewnixinecvzezlipbimibwdfytzjyqecotmlbcsfxtjwfrmgcaqfwwlxpkgncmrqgeejgwpdpabwupdwpvdorolgbtvdhnuntyzbwoenohkizpgomkeeapmdikhqxdfsdetzuzojgytfpcwcoagqsuucebudgcvjiqkdpyoyzjfoldqbgysyvmczpxdvzaghtqmiqaipkogxrwzxxtxsfarwzwryzzhupuchwnzibfgudhuaatuhtodsmwslvafmwktsxsdaxjudsqfskazfoeaaasovuvhsfcnevqgubdxttdnffoltsltsfjpafyumchrxxxuyattwygbdumymzsgfdxhixbvajoziltjopcknjntfcrublaipapxxruzcizkwstkhywxjlsonjipxlxmnplmgimkqlumfqypqwmziepxpaomlmaadmmjuyvmjpbphbgxyswiyofnouczicblonwkorzxiaoqbupboojmcrcqnervgsixhgjvxivhkjzmgpnwdzlfqpxftoabikapqmlpwgwhrwvzlkyqjjxbyugtkiwsszjklwzhewoyslxfwxvhisjgorbyaasuzbfkaetecicuvcbrzwziqkqebueduwatahfalyeqijmenoxmkwwdgphklmpfkpakwhkkhraqcmwpatdnscqyrzkelajwpliouvybmarqmpfpjkcbmubftojhouffhnvbitdwvwmimtxxgbasvdaqrjxhgennskrceuzzsnjgjifpjfjgljzcvykddzqvjhxpdyryocgzlmowtzfelejicvlfudcfncxscoqqszpdnfcifhnsnyaptnxpqbwddtygjoycpohcvjlcsaufawxtjukcvghbafjjwhnlxvqtgbvbdsgdofyabiczolaqrjcnqpyqmojvrgwuyezpkxrlfvkwgmmxvqkogleubpptvlpwspncmepuzaqfmefkvradcexevnzsaoosfbwshmmgoeoaitmdmmgpsgvnvwgvmwqsaokhayoocermzdqlsyckezazvixigpagdmogkewokpajtsunrzxxyzzuhwconewqqmycqvqakqlfjischsbftabbyfrllnebccdszvvmoirzqmdzgehzficdqtrbjxdlzifbgcnulgvduydackahscnkygmjdrwebdfhgudtbywvwzwzjyiecxocfclitjnvnuetpimsikpfkngvlqbqosstugeoptlcprxeblykworgbxdpnffdxzzrffhxyuznmxupkuzgismmpsbfxikujlnpqvmornefmyvxswddpspekslhcydljqcanqqfeumsuhppevutnlzzdnihlluudxwrnnytjkmeudvayptnaumfhumuhgplfnevaeokqcqkgnkgcfueagjjdxekfvxnidilhyvybnkjavpeclkestoxsujzphmzkukuthswwchwzycckmgehwbbqkbhtnhgiradbubxwkwyiyasniyuhzyxmzhfmolwhnbihegeocexcgrpoqmvpkvdlcjxswwzkzkreqxsubrwhjoyzavqmsuxufwmbsonuyfyqeskikirvwpwwnokfkhcpeqtyegsensuslgaxprunjqmwewdpefgkwzicgdhtvdnimnhdhbqstcqaztpfbxpxxxbfciyyomicbsfktxcpaupboggrdxoawpfzagzquhsvzzivwmkyhbbxhhokeeldvscxybnskhqmiajhmvfvwhsdqgnxkaagtedtftorcgdlgmsuhzqfikizryydyalterplkdcmliztutnetflbqucjscmscamirbtbgyprgrkckzidfuhxojgiouaqumblpuovsgmyxybhyjnffuctfibtecmzqnkgjzbehqeeohlotatyuvscfvxkzjjqyvyiwbodrshiavwtxqrsnlwvhtfifqadnynkabptwzbwuaptsilhujcddjlizmnpmbzoeqiaplylnpffysnucrxhbkpcwerlhszmjhoyvpkierqcjdwlwvsxgjceotpvopjxyaxtpjetauykzwxvsqvazxepswlgnwflwdhlonhphhbidydxoazqzehscuyprecpjjdxwkofrpnwotwcvvuvbcndwnuptxfcgicfuqmngcluxhysfvngzcmxqgnjduomestyifnqhymeinxnimvhphghotqtpgftyytjeibnjourbrglfbuuladbwwulcahdacoglpuufonihttownlhqoimkfzpfishliowzisfyfnhajvyyggqvqchvewcmqkzyyxyipfiwuryfmxetfuqxzxtfuxrkjrljoltgqbeksdshawchssetrzynxsaijlszylhopmajpsqrqsajmeegedvdvvngifhgtpwidzturmlkgnvtrzbxewczuqhcxrlqihaliwfcismofhjwikwnjaodqyfqpsixcjikhpmphadohnmszihijvlbvtrklajnltasimhrnmfwbsqbcxlnvxpqsiddimtvgvnqjwiylpxmhnpmlbzgaoyszbxhosfwnumzrwxemusviihvlhdnxbfwfegtzlrofdnyalemhxhrfrmsyrfxtmuasctrpmiwpvvribdsynjfewxenebiqxilbdeqpmnhikyslekkrurxsrdhvesoeczfidwxqlgavfuglhscdkbxbeeykymobnwjrsijdplawfghmblrnooqstoctdtuqpdjjosofdoblwkzzxrmlhefcvhwycqqtwackuspagslfcmebftmnxrpsalsfejyajbmfvdfcvsjzfnckiozghpahctmgenipqwulzanwsyxzmzkunjcuadefyslefwtvsnhspwqkvojjntfbfixhndmnyardwmbqzkkribbdpkoegwfnefwtkjiijbmhnpzozkfhnincvbgbxoqdvmyfviogjcpytpnsbubodoqysybrorxjdxrbghjrlzqeqfyrzzfeqxekxhbmyaxeyqfgzqxkppqhuoyuqfagchqaotonuntueaadqzqgpgpjxofntgvynnadgoqcbjwhlyncydkplzaingxjouhdhhgqxzsakrkwrxyrcigpjhazpziduribaotxnozafjwyiuqmeycnhemydwbxnlbpcuopiorpznqijgwngadqeroioshddktfhhxpxohxexhnfddnegdixusnmcrrmmztvyltosssrzgwejjirptqjnexrxoelnzgnmbcgvuxwcvblhlewgvhwzenystjivrvhvxoqunnwjqksbookjvgkzvktpvdmnztkyjasxprihbalpqvbndnoyyhptnbimxsmgvhfmypubrfshatwfjkdhuvchynvvfhecqjpobjrsokcqdyvmlavfvaounjrusvzlhozkztcpdefgdavqasvysluhwleqjdbstdcswghzlmhqocgqxdorokzmyfcjiedbhhvdnccyqjqgoywnbnerxfdvkggukjxrquweyyeojxwzvtmxitqhyuflclhmvhflbbrabwqmtnmqiapqtgxuceheqqslxxyzaqbdorpjgdzqepizxitzmcfctxnswotyeubuoqdwuenavvzsdvtqxyusvkltyerebsypldlhhajfwixbfrtdroxnyiiypacqucfhfsqotztrktspmoudcsscabcrbehxiohphxdczztjsnagbvjfncjpdlhgrqfbfmqmkwqlvjupywvcrgjgivynihvcxddcpuqbqgmnriesfbwyhffscfhlmfnjisoxmebenptgyxyfyufickcerjrdoepwzdwnjzswonfonftnczvelxkdjcixixwhauvktpihepwrvrfxsadeanjjrriapejragbtvmdcbbtwuavndytdmeofvwfmdredxhzvvexxfsvbttowrjzfnplllsxojduhlvcizbhgtfhmyyirhjxvykgmcfaojwvwrzesaoattkiriskrchmctuoycrmlnjjhipbkdcfymnrwgcnklcmfwdfyurljcwskmuwrybqkrhizjvlxxzcwchwyaiicgcoswmmwnciglqhvmsujswfwfcvhmflshznglzcabnjodqlplfbfbibimyradctlbitohxrkkayoskfdpiitrmdfkvgoxbbbjwpqtgcgicwbmbeempzfbeknzsbzteaccruwaweizalnbqtphuukzhxazbzthbxkvsgvqkfrmrhpjafsvzcugzuicwbkzyuuscrlozcaqjpbmwthyxdgyzobvrlvqhrkjlvzkazclqfxnyelxhrvjiwezjbrqvcbgzcsbbunuzkjzpwwyprhxqoxbrososvuuymvbiixhtggkeekuyutokqpbhjqbhmvhbrijbgrwozgtgtpeniuxblyivqlefhlgavpddiaskgnqzeuomolnzmxwcjcjsnpujfxmpxgzgvphzjwozhbbvbzamcjgpzaagxpvxvvdtkmswigxskynlanzcxftwhucelxdgsizfvubdaavmbjzbyydvfytmsvtfvwmphnucjxfmadyboycmyhiefbqazlfuitlpjitshyehddirdmebtohzrybqjgmtayklgddnekxhnfbsshvnvdmmbnfmwriyrsjzwmpcwmlltmfltbzqenfhdmtbdbzxwkwuwlwvxhirwqerrfkxvreydzzgdafzkcvinrflaqeygiqzqcwltcjwbkegmkfymcgbtomweswmdontqlejnphqbmxnyelmhtnchcynuxbxloqezwpmlxfolcbjgoxnkkqtmqhhnkgbzzfavupjtuxgwbpermsbzivlaesqqbrpawsvsheobeuzfmdwavazfxlidfytgpjgndehkkvloqmvcpsenuesrpauwyndpylebpmnahaqzmmdcgipamjtdmnzmcecdqggfuuhcuydhkhlqrcsfllizajmcoqfewojrirveralbjytaclfqyppyzbanrirqwajgtgkmgynqwiszvyptngrmiuzbcsravcgqiqdpoqhnbzowzdqpljbtddxhvqjstpjdimuyeblfdzecewrqfvllaiuxzequvzxflservpxyrddxzungrbrrjopkshgpgevtyzktqjuhmxfntuulvnehbvttcajkeedefqxvbtdoskbmgxzcldlmbegderokxtocjoxlhcbaoeovcnzpskbtojzvanfxiubsizfrdirsmfnqumfnwjimshhtwnsfjwztxgkaoesclhvgtavrlfgennzjctqlppgmxrnlvhqgploknlczridexepcxinnzubxnqiiyaqplhzzsdajnfpnjhpgpevoaxnnnwxnwqekxglrlldzsuobbdhshfjkcwxruvhtnldkrqbddgevbuwexfxgkiqoeqgblnuyyzkspnxsudpxxnbwsyefgvtnlboyokypjkjazgutjjsiwuequivbfrqssgshqsybchdcnxxyiqgemmwgushajzsdfnkxfkairdmgcjekbmqnynuvhdvsfpvmcgkiqmskqeqspxftjvgkicfriajnrfzwnledyypqcgktdpmonvyxxdphfyuxdwvqhbwhzlxdtkwyzlpmgnmftbpyzjxdhuvpvryufwzagyzhjowrdiyylnzkzwpgtlxoewllskmainjtdhkiplhaygnzecgxpdmwzxjtjxvsmhnpoaaglhijpvltartrwfpyezrpypjlwxckxcqilztawewyxpquwhsailrhabuwywjvbbfmcanzfjxeypywxswhepogqkdoholipswutjtauseukfqjxoqbehiwrnladyiuthomqjmnesjoxczvsrywyivsvgsagdvzkjnctzrxubnrfmmxfnwevrytzqtfvohfyuktwbsqefhjtwjylfegwczxvveqjlyjgkeelwusbddwbmvmtztreuozptpbgpmozrsnnofrsknvziiskdmozcnrveseuestxlilbncvaprzabtkehyfklwhpqllxehurphufjqfbhgbizlohjvtwuankmzjsigjxndezvjacrbmtocaxyvvsviehjobtwkrjqvinhkqynerbhtahovbmgfvrvxlrjedbbnpiyhjllcnuypdolearkwkhtvbczegbrvlrcudnzskzhnnusqneoynjtontstwmohdqeggfaqeidrcxjsbxqpgjfueynckcknuqwavyqbgwultedxdvokgtglttpfxbvurruopnryjmleqqbtlrfrnleghumgvjapaxfrkxrvjezbwdwhwcbjphsybefznhqpeviqsclbgnedsbmofpywxqvawcbllrcqnmifowjxlpcdqqrxgrufjrhhvjugawyplhxbfgyqvctotmnbgjirapzvyvebscqfjpgqvhgtjbjeowhwipnkfoifkmieobjaqrqoyzjkasqqqymynclustcefkivntveloslnbvnvmenksozwzwpsatxkesaxqekeyytbtgwhmjthxtdyruqropdukdtajgghezixhgfwsydrtgqmdgfrwjngogiamqhqpsqyyvuqvxirgadjgjceurmkxukmkucbmsampwfnwpeepjnvurcokobhllvxhyztsbikznxvedztmoqhhsxrjybumwrtuspptxztmyqqbqjpebwmuudduezgdqcxjroeqpzcpqsqgdimeeaaagfvmedpxfpgwdmovoqkekdmnlaupbushantgqjdyylwdylymbjsoogbzgvqhskosxiyqtcqrpmunxtbbevpubsviolpgygavqanjedauqwjprnsxihrmamuqalyytavjajsfoubgrkyixqgiazucmqsodsmzujiwudbjxerqppboghygiuiwkckqwypsizoecsncwhsnwtceccmlxwhjauzwuqocsdjuvwwtfrhcmlipnvckrlwsfxqaohegmpcjobctsjxewgvkdxwulsosqhsgqzocoqncrpylbvsxbpjepgbaarsqanyqomucfhhyzfkacezetlkitnktfdmhqkyezuiecrcbyodbuehqpraihlcooqvgcvmbquhummvmcanxewpuqpexailqyiydekewfoqtanhcnvbrrqnhgsgictbdcbgimketywmajtfwqgpkxdtfxcpzvscxuihivnbvjpyskdqbjgnijipikhjgbmfxwnftagrmvpegsvjdbyomgdurmdxorclqrrluzcgxtmdnhorkmhyumulzaiptncjetxavzgsttxiclzpzgnttifuipjbelnlhklclswgntppgqefxuaschomzxpdkcqtocxawtijoxauofbnzchdjvrkugogleazizbgtkphjscxxfmyhtdtqudmattlatnisqpncgxmyblxgyxqtajkuowxirkimxshqjusqtequsobqlrlocxiqwhynsiarbjhkfaczgttajwqupeitbtkzaknutpmwatifvleegaepyxlqdwjepvgrgvmbmaxmthrrsyxyabpyabodxakovqhblvexvblrlckdchecktdlwtqvtajpstsdoxckanblslruuxxescrpvikpiwgbpygfxjpaysnqfenbbmguoxljhxyqvetmfbscmjelesdlwqnacvwqujacgyuefnqkokodutoalrljkajmxhfsqfuxfobmdtwtenwaimavgneqqqenxjhooblwroiwhhqnwonebpzjddzvvditnshdtubnkmjttbhrswvvmwerbdmxiwqxoxdktgxdsiwnmdegabboxhfzmtwupmuglzqefphyzflnjsvltybrfgqrzooqaiyljorofymhfawossblqcewkuplidekncxwtolyozhubektwohmslkdiaosfudvxcrqreqhgydheqbaekowvbzddqwikbdjdxzwbsgjqjrwzuydemcbpdhsvfdiggrlftijlosdzjzzfxellindmuvhzyphhjumcevpuqsuztvhsbsdkfiybefcexjnggckkxzfnrzpkwwexqjjwshzwnccmnsstucetbjjukyggwkfmtpklwkjlmoxzgrvavzvykesweacakjpgybrxldkzrkbzbwbxxwxjbqffqjidszjtacsofcpymqxqoypnxmwyedfvmpahqjcrluhtywrnprdfpsvoetojxugllgucrwvescxrarerijzxbuohncsmwyykgumzrhejlbclsidcreauyqcnsuapvabdundnuhzfkbsfmxhfbjabzepsyrjvrkznxebtlaaxducpsmvvvxxxbmpbfazwkyjjcmepmgmqarhudesybpvedmgimfgyfrhjenyqvmiwdqsiumjydecitkshrygwsphwdhoyuxwzprilfrljxorsakaskgqcxyaafpbbmzbdsloyeajdqcwqmwkzebdbbeegthihilkdyadccmfuobldzxcdyhfblligssnafzfdciheumzjqmopjstkzallnvrfphhugnzivgpstodjcraljvxkxlvkgprwwbseqspnyeftuiewbfjmajxyusicjayhdtmtlbglhofdwjbaqkmrvjrpezspbsxeljzymgkvurysczstlhkhsbywcsmgnlmywmziujdocdiecdxnkjwcmvbrppxccjdyxcgrnlhpoqtsvthfeejrbmiqdhvfsydscljrwrgmjbeewolbswstjclolgufgxxklhbhahwllcphqcikycfjgryrzszgrwqcfsmfbiqomhwdlgpkjvtrmrjllktfgmjmgvtjfkystwxfcmrnhmfveqjraewgqguydoklyiftecaiqtwagbxmtgdtlcbdkwepkoakwffcjgmazichlztggxztbdylzcalqbvoicssifskpwtdvnjdklwnovapxaaqdxemzxekeywtupwrcgvortbhqbwqeygjvmxwldgpbclxuhrwponihqpyunqylwhikhfwklnmqylmieixsxiaozainqiaorjayowjbmxtmkfupquhncjdblnqbvrrpcsnqqytcvnrgulvkfeajodnztddqwwfyotqhrklpzcvtqsgvyihmnuogxtxupcugxlnpvagwfbrvvxsthzivtybtrypidjoaudrrexnjahgxgytoydjtbkmuldcqnrsjwkdcjqnmxtleseuqhdjkpyecbjsqclcyxfbixouxigdamxswadirbnufyoxazbyyizhgqddyliayfswhveyztsgrjfrkjsjrrsxhvchhkhhpbdyzhxabriuanmlwewqxophlsqueabwbeucotcrlklnazayfwodmgmayeueewaicxksqnjdzvrburccqqheytgwxbdezfmdcwwvlfyncpozjqbbjihhdogcpjgqosttrpocxopzggapwbshlbpopxtgizwyusrnhhrewjgcpgtnknvxomoglhmtilrcdlrmsjtosbshjlutyrrnfhtqohlwudbzosmibhztirarfxldqesylnobxcxrfyrvtfdumeouwqlbpygiebgrowslysffsycldigpmwliowlgxsodtwvjspxthcycabnfjvwvthgkftqcleggadsuspnctogvrvacrwzmffwkzopmsxvuanrueeyxueulawntunhzuglvuaipxflzucwyjuxdbrrdpeyjtfkolvjiwtxtyivsnrqwlpboisrxxpiqxseioqqkjpfiacanttjhkkvtdpscedudvbmlrexawdiwhljbgsiqefljuglbjieejssazpavulrcycdgwnyaskmpszqbhuhqpzsopvjjiclgeqmadkcywonoqdrxzbqoaierdndlrqtxhfzmscmqszfieyggzywagbtmqwzinthwahiujvldwsmdqujsshuainobkojzjsvlzfgicroblcebwptqwnshdhinxtogljrlqzvtsludnahlpabnqwwtfgiczyknjdazfxsxjobwoiifklvchtvmafqzfcooioonmikvxntnvbvgnqiwjgllndltgbhqqxdegpzfdusywnglfrhrzucoypjgrtwjqznwqodtiwuglcrrtbkuvolteuzxnxggyvepjiiteearhrkcsuxqlilpenoukuajpgaaoituptmwnyyhvxxgfjvmwjggfirworwdcpzrllnymnfhlfnsoiektksyskwjdsmfjlwwzlwlzcsmjfexpjjkvdkqpgztynajaxgzkcmlggxpncoenxokbyasgblgomqplyqvcfjjlgazirfxouiruikbmxhcjxrwxkuapdlcjnduejtvdbozmaayyiajhugfgchxfszqfdmmmghrjvqvytvwbyhkspparwxyqwpybdkplrgvozqmsokvmsfgjvgbtdzmbqkdwwshpbcdcvsulepxwwdgrljnqkbtdlyzjtiyundykvuvajsyfxgzkpoccpvfcuglnoljbohlmbgomamtzvujkklobjtddzqsqbdrskfennsbqgwpbtbodocfxwkcxbunexhwjmcyzuzrbyherksuyvbwrmaopvevuluexwcyteqwezavjtrjothhhixwdbtxmndlmimfmedsddnoygxobhvipvgswarlpvybfkitolltpahpgheiobsdwzxtbqgqnesspalfmwzdwgufpypdahiwihrlyzclewgvwrakqzpoiktxkkdmuljctcqtesgiomfmefzuhqirceonotzkfkpapfttxzdnmibspcowtynvcektxkdewsglzbeybhlprpjhrxfmvviazutyeyfdtlosnvrqcehwtqdnpizkdjmeeffkgknbvevizmydtfzoxedmndzsolzstefzoualyvuzkuffcevzjzoyenssyrnrvsslmuxnstmeprjrygtjfhjjubeutrmqjgcxbbkvuzzjskdqqwbydkjzonthehzpobnrezmyynkrxnjpllnqxxpnmkurswiixvbcvncvaomzdyooleekdvypguiaqrkqolulawhcnvdqvrhgxlrajpqrjatctdsiqmvqkeuxcxoxqveoivotmgkxhdzagrkydioominohdmyasmlpaebjpysuaudrfbxnqnbaaskggigyurpwyqfnqxjqumkzohoguriwjscpclpwcevawcpncbfwnrxdynjueutqovhrixplbknypfnrgupvoozobijheextbfkkczxfmkfbruuxrltfzutaeejjihckomusnrxbdfevzauqlpvmjrmovyxefbqvpkncdmggrkdtlajwphjbyxrvxqzcrpzgbmkgzqpsdslouxmkgxngxebuwdznsyvwlhcmargdmurgtfsbdhhcliwhnxwiwrejbixbcdeozxwscpthwclknktahsidloomxlnzrtpqqllvceebjqnwaiwhafnmtdysrorihgaaulvyszzbrmzrvvbxcqjwtkomyiuhqxhmkcleekhvzgpwcqtgidcdptqpdmwvjgiuktaofimeobastwzdzdgqywfbjdxujfquarkfzwknzmiuihxjmezlcgonklitsynspaaqwmeyyosjxsujwuuacjomwygryvlovleuxksyamhxedhcxddcotoltuersphnbohaabybohyulhxklcmzxbmqxshpirflfmlfqmggthagoztbbfdyirqikaxsrcdwrqdhcmpytpgjpzlshotpfjzplxjcboaufgdjssjnkqwzpjtyzmqfypwlmioqwqkdwopqiydsoctglnglbwsbmqnaydqxvdautpkbqqwgupofevsirmjddyeddwazbdtuufykxurrlhzqcryugjxlidolcrmdwhaqnadkwwchdbzguzccjbntfegjxtmtaolpoockkeuqhtydvaggbvzizhrwcgrfudulrwvecrwlnuriuzovupewyxsbdkiapclgbimfaitncmxwlnufcabcxwdbildxqoftyuaycvnhhkdszzabzdexbjajxdiptoirikqmgftnsziryivalbxnkzyjchyvximhtevzmpoeqwvqgsqstnhgmhpqqqnnyawrgsssjjwzfupmmggivpokwfcnxqcwqzpytkrctpdylumacuialykkfmxdebbucqvvamztdeupzipdfzdqxpaeezhibifdbabbocxrjtikbzngyrxxgzckzlvcwafxaiaonzhwsqgwptcqmbnvdrhcdcebclaeucujccfwunslgvdvyrqbixdnajqevqjrtfcvjhrdumrchuxhnpjyponpaevvugonmqkvzebvqsxzoxgcorcdgpugtqdqwtdklqgkwjfobeprdvvspxvvpqdiiiygberowzyuifiljxcpuahjzegowlzetznwdgkbfkxppfcbegqbgzmgdvwagvgcvtaabncizzxphzsngkqojdumyrogovkjqlkuaoczpcqybiojseabhwqhpeqebanulzolpqnfsdfwmthgyhkhearvghzkhpevosiekvlfzqdnkktiudrafivopxyxmcsawagiytnxkqkwddgzjxzqgwextcwfgoicrqlpkaowryswdyvmxwdfsjdmmztsqdxjukaekocdouzruwgrslzluczgdubvyaeskgyuucyiwxcsuilotkbnoawuxzpvzwpjcpdxrdhpdlhxtunofpcvrapozflabajdrocksmzvxiutdhhkwlhqlxemeqmfsxkkopwgdydcgxxlgfxfsfveczsirhpkgzwuqsavqfyzcvmlckqvrnbvovdgcssjtqfkiihbnfcsdqdhcvctifowbredlyofourzcyapgehjumqxhvsrahqbauhmtvtjrkivikloouzljskqpckrlatowccrzxbrhjoruzikwjxhbhzsmjkzlfpkhtbiqezxqormpzwniwumdwgrfqrrxibecenslosbejnwillshignengldzcbifhkmwexnwarfppnabiklozpbfafhfipmrbaofxbcxuhmmxppfbatcrvjrcqymhndintqnwskrzzdyqsdtuzvlugeyzdkgsprfjhsfzfyagdzyxertdoezwjxvojcgbtnmaitbnsdvxetxjmqoavqwgnuwzwecrrotzhwobzlzeeypqofslvrrllpkjtmancvojrdlqqmwvklnyrimobfzovlxozzdbuslnxwdnkakunaugigvlmwzhiyxhxbjttwgqulfenqydqsjabglnfmwnktwjwhcrdkjzvpdjhdmxwxzxgmmlyiyidwtcevtjhfjcumndqzcozbnfnzqnjqxmmqqsnkcuplylzmqrnsfmvutkvjxebutqmjibvmvmztzzfpqthyhcubmqmlwqhrgvbqwyfrlenmfucwnlwxqqyeimqfczwkvzdraijnmyzssbqipoffgfjprbwubugwwjpwrajgyiarmszbgggolrdctbywyfsuohsikbqilqzziaaqgierckwxnutlbnhswbcgcspqiqpsrskxcqrecnbqjudljnfgohmtzbjjsadlunkxfbdiqfaabdnlqsrhzdayhljbnmmibyfgaioqposrufemxxtwcvwjgfmwtaeuzldzttrlkkcepispfonhjmtstfongujffmgygzlqlpwdxosccfmorfinmyausjbxyjxgjjyuyuxwkpapxofqvndilfcfrgcppbvykttlpsjzklqhavzviffvkkabqesfgpgsibfztobdcrmcnozxthtfrvhjabqirwabxgbtwcyytlcibcgulzzeuipwkfebefcbnfwljvkdgmnhqtdduehrpwnbcsegzijoogzxdlclyquqjxugdbxsakteexkfgvvwdvtggwrgolzzktfowciwpeavbmdjkyfnvbwexpmndlmnesdbwmznsaltoofyokoklkqkzcxcoksuexxcjxsdzdxntcojocvcolmpqmugymaejygvxayyfxnabbksfhuttillwfruttyerhbqjaopzepdfbvnjrieztpwaistbtyvczllbccdgugnvdagulkvebtfjmalfagjatudvxidtototvxnghlllesuquhxvpgxjdcgclkvbflvaiihsvudaoxsihqfwycfdfofqactfoofwzmouituwtnhafmwfpejwsomsrhzhwedvturbnmagbhiitfnklwgfluzecylyeqvxmmbcadkdktlfazqlvwsirrotpzttsqtgcwifztklmwqmvgzsfrbwkrrzrmulcprgnanmvyzhuqvrywjdpmbzeuaxgwkkmdrzfeoacnjyfepgcbjrpnsvjsuqpgcdkvepauqekzfpgwvkzhepwwmfrjxipanzvrpxmheyijnohlbhufyqbkvuttcjvpkqhbqviutrfihmvitpnrfreblzwokrdvhmwxjhteyyphnkqiafchwgiiheysxdpfslebsaisaxanjsylxatwibgpighhjwmbktfevpzvoblxtjjsjcqrfwipcsxoqlragyhbngzbhkzcvkugmnztjhjozieqtvsauzdjhcloomslwprkcqfnaqbbyhrsdmuzlzivcvlsjeehtxrxqzkqporovliaxczkbkffftapxewsdqptrzjgnbuloellinvmxdrewtvvxunbmcwddgjtjgexkriteffdiykgsfbrbounsrqwzcrikxfuoppvuwafxknpspitdfhyyzwxdnfnmmkcdzxzyagvkumcjwyobfozgkykzfzlrjowlnwjpceaysehoeyslmrsxseyxdkpnanapjjfophmhnwxswpdijxhbbiyhspwudwlofsewztdprchnsmxbkhenpcujuqdxoqntjspkluzcxirrvouhvyukcptwhytwpjrybjiofksesjvnzpuvnrhqidmpbinsrhsusbixlccflztmppjegruoujgxrvqkckgulquysyefxzrmqbrxunnqwtnprfbtqhqdxmerkkwjloybrraleobdjquayywqfovfazymlvvwlvacmaptoswaksciqyyymwfmvdajywflrfpggezwyvyjpbrgzsgoolclodupzcasjqyruxovuoempvurpahfljtbmpqnrtibjgsfgiaczeqqckjtkqzxauzojrcdkkgtsabajbfkivakikfscgscattmkvpvhvqbvtcgvjqfetrofwhhdbmfufrecgbjdumbnohkxapevguafbjiexnyehdipgttcguqudcufsaaaucfyopcnfdsmiadowwrcsjyylsdfugirkppyftmmwgaeidvecogwzfukzaswgcnqreryzfmwlmcvszcuniqmplzrltntvcjogcpfhbduqiqihscvcujuhilubanyczpibepjhvdxdvhkplhsgronbzidzxdbwslyycjixofckpnbawvgpjwrigwjdzmauzauclcbzkelztnzpkifugyemuopvcrrctmgeqhgalrbegdurlbntzrftfwqkoimhwsomzuplnqrwtlngoazntgizdbjrjxahpqtqkidybvwwijfxlemenxfqyqhjpjsganxmwamzxivgtafehtlwqsmqlvbaethghgtfgfggodmjetqnmbjdvxvgsxjyssfbyaigfomvsyfyrvneyyjvvgenfnlyhsbfsklayyxsyeqsbdyzbhxypbnvqztxtwemgpohplzqqqirvgtzpqxetlzlmukfrotxfhevvgnlwvetdzssrsykdyxruhylvslbbrywxljvioplnhfhzdredpxyywluoyxrqxqpkzlspcffjkmlnfrrwprvlcrbboutnkixkvrbxwgjgswvanowefrpkksnaninxqmjodlnbrwjmuhokvkodegpoycpnkelcrwswhgybejpzqdjmpxrtfgkekbrwfeyydrvmzvpdeeevjnjznausgeysfsonymlbfiqgdfmimqdvgmwvorpwxooficsiqfmyocerhqzvjerpsnmigbeersjzwdiniulwyrohkpdtwukqjwzsxdwpyhqukahsuurumhcwyfrubbnmxmeurclpjckvctozqftcoxamcthfuusriynzbbrkddghwwgmdcqrzsdelmlhidaqeosfrjrwozdmxmbaqttrmxxzljhuohmsxexoprkdqqsoctqsmkjyjhaushqgqdzohyezppeghwwjpmdeyoehxeepnuwexvuhvyilpmtwlrhcfkgezllwrrmfmkllhfscdcpvpircbzehlllmyyjpqxwejhpqmkssmkndugfwhkdtxofvmqmepjiyqaodkjvytkcpqhlytorqcylrpaxejifzitesebszlupceaaxrtovrbmgbmnmuumcpswuuovflmhppjmgccbugqcjcrwwlkabcyxwmuezhnowxdczhfvnvazztvxfhzijxhimbmlizcgyfkamqvcnxbpryoyfogqlpazqmlbonhosuogkmkzjnqanfauvglgvlgoatpqgyeastiefvgqiwtwdtqycdhzrpnriejqdnqxiacpvkgggbcmuukyaryrgnkqutgzgywpzagwkmctngfkdpkdjiupyeisxorfnpwoqdmarpthxytaqdfzozqmfygenzkcoisgnejfveotfhvvkbwacmmknuocvaemjwswcddyirtnaqiltglwsaghwwjsxneglyeqckokbjzczjptpcrjexrbtatsqzadxftzrjluskwstnniseswbjxwlspbtpfrwasnkgicqgwmwjwonqkqbknneeqyvxruyljirtmkcjwysuoilyymknzptqppngllruvloivqttqnnevyewlpjmgdexusfffwzdjpnefgzxvnbrngpaxypfdtcxnhevfgypvnvdvxaqkejzkycjzrockscyzgtgotxkgtndqptctfdfdfocwehjwlhxtceixdsfyhiqwbssgyxmdmqcsxlpiumcwsktweawlinjjfbsfbglpwnhuaweulisfundynalvnvguyaazglofayzcufwhqqnkfslsswphtxcuyevgrmlrkmteurokjbuotfioezvxwxtwbpfsbqllfgrbzflkghycpisthnsenmcqdhleayphfgigmvrucryraqmjghpvvsnybycbztkebsmnnfkqxkjzdyilhagcstctugxtzhzcalsptchqotrrrbcabictegpymotnvspxfhnylwqaphtgjwgkgkgksvkejdtgxledstdgpuifbbnmxwxgidsbadhjibrimvngqqmikwnoycaxyzdjcwkebtpluhtiegrvzfdjcszauaxvzdeezqicjynghjqlrhgqnlcpddpadovfrblstvucvikctbokiwdvihmlkbnosivjlicedwjzgtdzwrmforcyypliszykhvuvdyzhuiideleafsftjkrhbcqtdqelmagpyhyuqixuwcwbuewgonnprfivwoikoqbozajpkilgkihhodncsctmidjgstqijtzitiagqnmdxumlzbpsdignmpvginqthpdczfistvvsrjeqkvkneqdshyaroskvkqxjvgbqdyjuhpmtkzjqrmkzmmqafdxptvaqdchicqemhqfrpokjcuirharodrdymdprhwzmofyvvfonwywyvgregjsmaxkoabhkziynnwkkdzzzzxaudramcsyepkusmdenlcbaalldmmfokhinalbddzjvmgpajtnxvqxxckaqxhejopwrayufzfcnlrnvkbuhtrexehodqumhjkylkurmztznnnkxqehjxnjbwxgnruwdfuorohdpdfeqsopkswvqmetidcxxwtadpqvdyavbfzrxqgjifharnwbryzouuqykgvpevkfbmyachkmmtnfglgzjhnfcddmkydquoyvcwvgfhstchyjshibqyhsdmgpnkgjaggecdkklcdnkeyeljidhpwkdecqqizqohuslgeyjilvmmzzvhceyaqehswxayoqfklqrpoczczloamlirpitfoaylyajotoaeftvtvvxuilvxyfonphvekvpivawqoywvsspnpokzahwaomofwljggaqdaourmqmekbuzqgzsnxsdkiiitsobugfvuzmgrylcchduneoccifotnwohmqcsvzmwnupnhrpovldzbfdkkrfapthdjlxgkjexgulnncekqhktnpfzhkbyejcpubyhlcgtyrgmwkwouihjynihxmfdvjtrlrayvhruazgwmekgnspnhzuelbiindcywcyrjrlkqtieavkppbcwdwcozmzmpdmlfkfrbsduolaogndxoftrdwickjramdntoukrnpnbdqpjdujtbcneaxilgbnufcnvxaompensueexanqbrfhscrfitseywstvuhhllwehakeazxdqckkhcprvjwtiqodkxlvjhbqklxmkwqdaanhcfzxqvsyngexepupywhijgvwclzfnomsvnrmkkmxsquyhrpgnxqwfnskatpokdgfrxtbjgbeyxcgosobrgawuuiwoxmadsfuszxdhahfymnmedqxkqvgmeccdpcgjicdjcjqcqvcelpkohhazfiljlqocsxncbtlfcbiuzmzmbjakrdjwlxhynxafuqiyjtdlxlfyaekdqmohpxfbuobhlepiuxgdsjqrxxpxezaumqbdoiekyiujmjtgiblcldukaljeljyzcbcqjjchrwtffolpqysdbonbzyimjgnlisbtbeducyzkmmzekthnhwojuyxeflppdtkgzpabnblcnnizgqmqoczlbyrijgzobjuazweoxpsmeblltzbhccufalpmuoavzvyatcptthfyitlbsolwqfdsrsengfigcggxkxqghwerrrvlfgivnnpggutobeufmdcxctdjrjbkawvheferttabptpsxmxmpcjtxodqzeabcvigbmxwfzxxtmpsyneprpjofoegsehubrcedbalbxiwmxommvefmnvubzavfkrjbtnhuosxnctzbkmqhqvvfrpissfaeyjhtyopoinckgdmqrezjzvosmynefpkdaswygmnpbrbhupamtbvvwmpjmcpstavtsuuyihpjyrsugugawexuqmdcpvukvsenoynyhdhhubbcykcuozqgouifpfpmztzptonarwjnznqxxutnwktkgdrxkgjlckavapysnhhnuzktwgiuqbgyhxxtdiffshduxbomkwzhxmcvslpkmvkvuhpapuhjdkdcnspyqohncjjuuagjyqesxtuopcxvmbxcvmaldhyfqjcnkiwkzaewwzlnbkipngriuttxcufvdhippeboggncfasimaxairidmecryhmdqwvgcwwiclnjnfrahpaqiktwovnmfbqvxldbjakmlzxzgpbngbdolwkauambbeyzrcabcmppnmqtdrhjukzgjmifmwgnbeougrxkfoctdsmgmvqgipordwwptefwkvqtooduchpfcdoozewhvlybhwsccxymvrfjfldjssruyfixlfeisejzcccebcswtyiajlmdgdyekidxpjhhprjrvnxyppvlpflasxeaceeomjlesvnhowajlzjbpgmglzhexnebrlqfudlpyzwmxzxjyhkpnywzzfdwshaluqspteesfcecdelclxryxpzxucvomvbilwxhcdzhiflnjxpwwjvuautxzonnmpvxlvepafwknjywintfyqbzmuynklcyaawnvccxkaixkgvsvqzwicxxyvkjgphjpfymkbjwdrtpzqkvjlrroqomlbxrqhafobraecwhwrzvilsrwlkkbzzmcyesfgzwjswnutvqkwmuhgndsyqtzrgbpmvxuuteoprikobvknrtvfznfphginjxbeuvanboftypjzmwoepkloxsfeypthnwnjsnzvdeqbtadiwtondbedtljrbdaoznztcypfcfiyhngfhutkvznqkzhlcjbdwbzybqfexpcfdkmkgfwdemtrlceplgujjtjkloqjfuqiecsmkrpqdkhoomdbuxkftvrejippchzjmuvwlnooxwcokhpggalzveghmvyrfzcrtxnembumzyalfprmrelpjryxpyouxqltqaxeduqfssqykkkdoxuxruvislcsepnwrkawetbznuxxejdsixszunhoalgectavgdsmyvllpklttoocngdhvnwvaecdaeagjsuhqitfsgboursjxvruiprlttotaobjjytzgogauhoqyizxesrohlzbgpwggyspwlqebwevgvngdmgjfzymkzprpmzvmrmiecohaseiprptgxenwcbhcratfzortczzenhghlpzfguldyyzbzifskyxmoqfqpgwencobrngzhtchreismpnxpmjddqvuwrbmwrppgxpnqrujvfdkwqhmfsrikneigfwcwlscijraaojyvnirfrfklzsvmxdueczdntfyczyenkckxkdcbjnmihjwgpslrrogmvgouftjpvvoizqlifyspjovyiipayymwedslwectzqdyjpiqpafyqdxznobompmrseeqkhmkzpkievbtvisolrpbfsrshdlajrsuyegawskcnlwluqbhjqkoibipgvrsnmonqwnmoypclnphfqrogbeqfnhnkzfzltndpcsgcjkoyzxfpotvqznxoukzvqyilnodqgjsrtiruiwjwrgoqtlfpwgloetbsjgkpreimnzgvxptovihqtogruthaojuypenhjytyzcvxrqnpyuhsinlegtfvczttgicriyriamfrcuidpqxzfieobljfmebsodlxbchdqovtifyxvvzdddrhocoagpvwvgvwjcazsxubeuacpzfveweuysvgkscfaabjedlxchahealtcqqgmubmnysxswesyhedlpapyayccjygzonnjqzyqtxisbaqietvvvlwxlaavagwpvkanladhznzaqrkwnguitqgesibtoykmimzvofcnudpkvzhihdsctyhngkixiulsikowrslowaytahbjktmfuvvcfdzqykiixyfqtqeprfktmlezgqgclbftfgjfrtpquugnnnegsobgbffjwomjptcwzndqjtidnpsccwiwkjcnpkwnykyndbkdqpvbieeyvyazulvgwxtnhsafczvgchluwgwfmcfhbfgxsdepbzcktdsweneiurcqzaxmdcvgkedchjrkjuxezpcfcgpyzrqzuonxsifcoifysyidufjvumcyurwulntfdievsjyhrfflfpqavaxrmasgroccdxhgfncywcfqjmdobuqywoqkqkshavmesvmyjoyhsfxbrbssxirvyjfovzyvukzphmpqcnyxsxyjasdxdwubhzsczqyvcxkjlzcooclgblrcgvbjrwmzxebywzuazgmkiqawundvzwxhrzqlxemujpvhrpiqxwcfmdcgqhbrdbaxcqizvwvglqpchzshvxmjaqpsxtmdwmayfkzcguitnxwwzqdunqepivkiqwlmknkpzlthibavbwtanxwcpcsxycfoasdkfncgbvnjzogcmzcbyaendyndauranumukobnmqwlecntdrucrarwygzabtcqppmbfporgkemotfumcygepcfycfbxitmxfoevcwbpcpfyvbkzzqdrvipzqqwyjixqybldzoucebabdmccdhtyskzicvgnfdeeerjeidimekpjwrmzyhzqsyauvgiblmhsfqynvibbsqlojzqkgrcyqdtlldozrvtwnrkkznmtgzhpyzbergfjjauwdlxuiqqyhciiqwdwvnnvmrcdxhmihjekwcgeswccblmthsrffearxcxuljcsuheqlkvfjzjkkyfuqwbvcdmagjutvxmrgsitdfymeiyrgdnybotdrhfztcvozhhhzpcewasdkqusvjqyzlcstjumlpamwhtxzxcvywyknufkfxjejnxcwroigquezqbanfwncwartfdycmdlbcpytdizvefvwivmotbssblniolldqyesqentoopttmrchdsdfmvixirnvesoexiudwqsmhngtekckcfroabnksciyvwfsezxcrrpyevrlhuxjbozcpookqwklspfzpdbgwqhdnyouundwdeassoelnpwfecgvhiedylimlygglmcosejnsnyfeunlytihdmnezvqluythmvwfzfstuqbbmqmllwupjhtkflvbjoiwledmlviiljjzzxmmthbeqwpxhmytbhrolubzplhcuqbvpqhixlpphenpqlktsbkgvlfmwoosgrpvhyfgjuwrajitwsoptyxqlccjssbyjsttrazavxtxibqfvfwqdtipxkypccnouijlikofihdslvxiavzgqhxnuxeyztmwdvjmgxtqfwissiaudbaujxjfxodyjsunnebvdulirwaledfheibphnuebadmszzhdcdjxqtqmwbsziadqdtaqpetpjghjhrokuxjmrzbnctmmvrfzmfrclntvtdgcuuoquxclydacxopufwtubnyoarlvurxqomgiljvmovonlfiqddmgqkvqcglyalpozvymgtyvopizwqrauggmqddumqtmdkcchapotpljetkdmddijlucpzwbsrpvdtaltpbogmznhlqvqvdagaexbcaturrzkrmfintobublhtgyomyxoewpgxlcoccbvogdecrhwvseobcfnccbiysoxdrohvpclxhrzbzywtcmugrjktmsufitibhsyyukwcdvdctbyxnufferaqlwxuqixxawywiualrdqngokuksboldhxagtqoqpqonnwjnkdtaqzqcvauyrgcafcgyuvqlnfbadkqpnjrfkhiikegdkidrmjpggisunvwsfmivxzakjksvpaybjbtvkketsyphtlaaakxzlalukfrzgpdpqwmdgswzsnilaioculcgnpcyvuzbjcsaagfhzditqqhqbvvvixdwogwjavmlaenalsjmvtfnmmgnjbksaoqowmayjmjjrxmervsnrxcdmksvlepnsfeqteofqyojvcpmyhbrhgutqmqeqgrlnvbjzqosqednhvzhcdyjluebgvfydzlsupbwhpqxwvkdqlmvqmgbglfoimonlafirhgyywvutfzkqnfhznkfypvickvpmbxdkdbibwfvdehvsoerbpzcewkigiyulxzvbadcyixbpgxhudawsbyzykjgshddfcaifemdomhfsgonjrjeshdsrmbmwgpdiqnnwztmlcseiirtzaxsjcmcrponvgsgyftueoisqcwalpukpteawtcnitcgzumxbhthwdzogvptkldbekdadgnnzrtalhzmpsunvvbdtswfdchimmkrsagtrcvdjuryonnmjppchohnqqvedhvxhdtzfoepjvpxldjusbybphohylldtiaaltbrifitdxqoackudiwupmbpamsvwtxhozaygirjgkxgojdrrbnjnoypctkymifcjibpqsbweyfskklblymaonqqgcoumwwvexnwsovyoarykecxfoxtbnkhsadeyacczpztfusvwluipyfumeqquczmcdsbfufelgyqcsmydiiugpruuhmojoanrsoypwcacctfwhfbrqwzfptzyiphwrwnknlbjpdluwsqfaswqqvxfrxdzomvgiksxxtgowuivobcotbglyesrdxqrlnxtzprykznjtxeiyiytjfgijybmvsdghdcjsvjfwyoqysawcasmfawqzxcwrifbljmfgvkcuausswqxxkjenhvogzkahrbucpckkongpjajgtgffrjsyeghethnmikpzmzjaosryubgpjlvbupbbwihrpudzeuyswkvmemmypuozflccaffyomrfgvhnrhfptacxnqyelmdszhuznbadnrkuorjiboxupbuessifgoxfvsgktgnjyyfmrouucuprvlxpdgujusqpmwahdpqdpxrhlmdoiuhfffvknrvhllyxiahnqzqfxhokvuydoijupgeezallnlkvmtictyjsiacdudhknsecalgzntkaexyqqiujhgdpiguejbowvvaviivvdhxhgvaxgwinememsysdchzxebvmjsjmzgvgevfkunzyzfntffwjovjsweehwptvwrdcmioiynhxuwrbbplslukjyrtugorefxgshwcfljfffwhjlubyonkoedknrdrxaarkwtpkfupmsyvguygimsyqxifpmzoozhyxcqhjmzwhztnuuvegqdynneezduqebobgglrtmpanlphfmolcrjqenugspwwsmaatnsmzjooudwzpkksdwkjlerevncqceqxrkjsuzhqcfqtrcnuyxfmexsbmurobilsuwnoxufzbxvszkoaitooqgpfchtdrhylflcdhjekgbchmqjwjxkcdzhiicnzlllrypwhyareevlqlvmxuwzaxiigkvhwjupjjlthnknribjtwvssttstbaboyimjdbsoolwdyjvgmgbicudxpvqrwrhwatffwztklrnldkrptuavqdugylzwcgsebzlxzhjmyyxblqtektbbyvhyiivnoprrpfqomzckzzokrhobcrecnnaxvvklneccjssvtuhtvvfmewpinumcjnafkksxoxzzrtunnegkblwihmsxifzrsbycmtacqbcjvywkeevgqtozhgwplyjfzjinckqulugiuctzhqhzjvardzirxtwyfgmhqeeeakmuohvxzcyjzzcdamuzgsxlzjuppatpmimdrharmbcnaeqcsmbyfylekivzkrpxzdcsmcjydlmfrejktkkyjyevizusqlrrotkdyvbdjplawukrfxrqvvldwhxptpgjksmioqoxhzzftmwajnsgkdjhniwixetixisnygmhhnpbzndibwonbzuoisyhxsrwamlaszjamydomiypencioxyyrxiopzmrlxrgyaglblzntxeamgkdfidxhlnylcpyziealmnmzmbvlfxalhftriszwfzhhrhjzmjkywwykzvxeszszglspftlmtuukbflpqwoouoaujzadzucfhuwmojzjfvvjagyseoizeivnjxqksuszromhalqfgloirpcwdkxiuuxdrqojkglbwxiykagslucfjedpqaxmuurnkfzxnkymjjmvlsnzpreaxhpnjmzfhqstytqndrbocojngymvpqwmznbiacnjvicvbvihmmxoahbihqgzgenirpqttgcoohjpcuyqgweifhqmxwngabouqgnjmsaczjsddsvlobttkkiiivqlwgfoouabmadnlxsrqnegoyczkkjuhbfgjmwxhwyvyejmbdftrjchheitqtoxwvfficdradmbjxnlbugudibnomgefweesjjclotoshpbheoiyxedhhqwtzabxefpqqtdfmrzwicdmtnmqtqzaivwjwxhceyswswbgpccyfhmawzqpppqznmjllsnrqinigrrvtvvadiabhsrykjzzxikkczbthqdphundsifxsiybcrzocbnblxloxohgfmyojlrbvzyohypjqrausnwdlhrkyasdpqrydlzzkzptxgepxekrsixfrqrprlsgemghxfijgxedeivdnhxhpumgkfwdqqlucrebymkemokszhvmosvfojlzojcgaqdehhggvsmhjybifzwvrraiaohqxlwzbzelgmteeflpenxlmvhfmjnoxwexkdwlccwlorpvxcmdrmqfhxlwjamgzrdjuawyevjvmyslbeosshaaglypnkwobzzxbwyithdfixrldhfwofwvsukgeiwikwhnufvnpasxxlidtnzjtvglrttxbshnopbiwyvlzpkydudwichdxqqqjmomexauobelbbycqldtfgrpjbsirlzcxpodmoxbrdoteovwzyshflwompwfaxhzmwcbmbceesxkisgkrarwmnmaeiiggpfunjvamzestdccvlmoldmzsyaxjidrzllpwjplivdcyievtskgxygpzioisubrhmgdvxoguuiucoqxicdfalwztljlhkfozntwnkchemciasfkwipyuoapzjbgwiwvvptizwqdiunlosxkdnkzbkocwrkszgpvvifxgxcrrqefibrlqcktmzxkshcsptpneukmsmdlcjtllkydolxdkunceswtnlgtiqgcflnhvguittljeimqfcujwtejeldmdloiweqrmmbhhnrlxfdehandtupxfsmkdctbilvvzwchltslzypiyfotzdsgxeqcjldryamyxrfimzvnzlwzsmsrrlzorpsadxudyidvzkfmmbzloxlgjdzaqyqipiisgeyenbbmmgayjalzxwfjcelseccuknosrmweqztcitcrtnttkyeofpilrlfeawwrivwyupceieyesrzerdsysckmppqplohwlfwiuzmefkkxbjbvovtjydvpvnfidmeprdxnejtjwejzpmuekonktkzozexcsdrjioymasyywhxlvsrfigpbspsrvruowlzfcyteqrcoazbwttddbvhumzlhshvovxvzvpdlihbjichcxmhzhottpgbhoyyjxtisknfvuggcjnggbohjcceujhbzibkfxuyhhdcdmihvxodzclkgpoaywexgfjlcdjmmpqpbzxyyndzrqlxgcaeageglnrqtbcfsuxtvutujwvanhztjbzjhdpvtglmjptdjqyfkrbtqwvdfvyowxzedqzyzmmkdfomwmtggwcwvzbaubrtaszdjadimpspvidrvzzdvdbqdctzdksdcrjnrmvfeqwzyaqvbpyyvbpydixiusoyjzmdnhirzhqyhivkawivxzvbjuldmbcyaqbqzlhzsazbiunkqwhovcrfxekjgohiekyeyifqieusyhfrfmfbpygocsxihhrltrrrzwkhcecbnhuttzwkxeyzwykdfdryuisuhcdjihmxogqkhmwgqvgnvsscvjapsujbwpkgwxelduxljvrbqzqntjpossemmpnleardxvqcsnpdijebkvoegguayapkqevwepzdkpgeexhpiostlizrotrmugxmluipjktwbbagakxjapelxdcmytyucxgnureddzzeccmdojooasbyxdmxymsakaperwjwxlbappfdzvtzpuzmzllxyfakgroznzwlkznzosennxpctoqscdqcjsmpjcqzbokmnxncrsreriilzadbdvdmprdztuyukkiyjycaqvgjeoudqagtxlgncloxhorsycqhaujsppbtfwyvygovtwkviaspomjmqzvqveklvlgajfvhnmfhkreuszmglvvbxqtsxddsqmtatqjgeqppiujctvkgtfjqyckvhqnxcixvbjbfztwjnilifwoybxqbbbwkrmhtwhoxsrdrkbbzjohpkxnowepzfojwatrychwuablncommdwbpvqfsuxrblbznwslyzhsakdpnubzxzcovsuppriselurhmefmoxoyddlzgmyiqmfliuazcudoowxbbdppcfnhkbgqneaerfpzbrbgngucdtfgwwxbbminnpdbtghmlyonpptevpgainihjfimsqzfhfidlrpypflguglsxtugjdpukxjcpikzpuccmtbdzokvvxnuskabprwzofcgpofhijjwcmabbmxylmsdwvhlmdndkrlbfsipbsmfgojhaihlucqtzxglctqomdqrvyhysvezjpijxswtuomtovgsqzhdczwazfauagvjvdusxsmtjamzxvrqwavabrabxbvzhqldvhxtclmlmbklydbugwuhoxinadmhvzmrhmmjxywlbixznyvtxsblplaproymonfxzriwqeitmhvzfqveiyrzjrqcipryewvdodbghlzvswwanrexpwzbppfxgwvktsnhvrpecqbxnyzarfzjbymtqrqwoohhiwqenfksvmrkmqmfvmozmcscukziagrjejnsozklndwbtihowxnjahnwsmgsuxbziqtwqfjfifhbqqrmnhxpollizrbwqexyxqwpxgufnbiftbshirwzckjedmawrzwortdobzvqdfwwgqdrhcsmmbsrounhtkvkfaaizvaccwzkoebygvmuxemlabsvnpfsphcqedtgzuqqauohpcdmirzivtdikkbbrdsxgndijvyjposhqytypvikqntbxwcqbwnkjssspsefiwoyujymeiwxvryupckrzzbirvdgnhnvdbvanvmrfkoacoutrgkqfnayifgvmwsszpummgerqurulnoetyuyowwwcduqbokaeenqktidpbxapneioahczfydkuvljgmnmzfzuncjpbkepqmftvhsbhxncszlylynmfmgtopkrbqiuennawhdeeqgyprzcrmowywroobfptqjyfyacfdglvmklfjpwvguvcmjqpqijitjosjqzmwnhkxupimnfshawvlpnbxymshakqqmeoznyykniwcnqjpohafwdeupxdpjthvzvwmtuyicwkrowgddgibmnvbmytvtatuwpgpmhyodkisgoxlxfmjuhmghfecgxwcmftymrprezjkngwmokxkuktbpxgxuczihtjtmynqxwdmfbbvlyffiyjaxpxtrzczuluilhzkgeylfziqynwiohilwifnhjobwljhuktcotpvtxmtlkbzrkmpstebctcyglhcycfmwwipuwkhilibxuqrqussputxicagfhuddeuzcegkpqnnndedaxvptqpdtrjyygddkxfpbnzqepfrsmuslrkeibibstbszexmbcbescuwrcpmolgzcsdmtnpljqfiwdwzhxdzcyxvcmhqxopcguockeptwvdvyaufzyhbyezargsqwhufdjhnpwinrmirmdzldgpdkofbwxuteoyjzhzofheztmqhrptwvkziaelwbfphwprvysmgzqfvcxcwpxsmhsuhygeanvhjnafnsrtuhyplwrjlpczywuguamunjezwgekbvfveemrcddpsfotdizxknuawazvidhjamggsszuycjsjylmikktzgvwgutozaaboihhdibblntulfhvrtotzlphfkxemzfaiohlxlmkyhelmnggixzjiqqqaqwopcoxsyirlwbjeiwsxeygbzpmpieryafnewlxwvohssqyaoqxsrdwqviidobbojqtfdsbziktactcqjaicfetcodhcekrnbmueexfohapjtrzdtkfboaffdnhkxaxyoufurkjzlykermjnwrqmrvrxibhgdegsylmsyvtjvdpnrbcnssclntiwwpeawkswygowejhznymgzouriwfwrnuofyqdkizkqqiuzpuoqivpzrjhpodgdxwixmlkwnunfdjeudvkoscnllwabciezfpzrcjqpfduiobfuypisrxdetfhvnoxudwohhexojtxxucrutzmnyptsudnkrmewksdtycihwfhsjyneaztyrnsnqizjuehhykazpiglnpdyqsfquyqlnlaftaopgvwgixfxfokvdpvtialczeporpumribxgaeozlnezxbdozywzmaaqrpjjjvatwybyepxpeepdfuseuqggriyixhricuebsfzyxxndbehiazpvpvbidctdjmnogtpazqyynobquibvjqsbnvjsxcvjplxvgoyfewfumjkhrneddweqrnjpsgyhtxealchrbnlnwywplaurwyhoosfgzfpsctspyaibpwolwncjnxamxkclchrbogwqcqhaqrbhnrujgxtwwvitfmfivwljfhbhaijsbmapfuamtvkprulwgqebosktbncbvnuyzunhqovkjfzczmzfcfitxsozfrccgwagsqpjgktrpuuohttlabeyvtzcmbgsaiasapbgzwaqtueapznpcspzlqbzsjrssumulqblmbbfxihohvfmnxlanpbfyszsomnacqnqtqwsyfgianupkhrywsfxnfrbtgdxyqnrhljhtrptfllrtcprxvqkcinxqrupjjgqvsyolpjsbukwzznywekyoxveqrtabcsyusmhixalukhvjwticdiqcgxtxsvyupyqikyskydeimztksbnxmfzscupxpqrhoipbgjleoroaqhkklzhxhipjdgiuyabnisqqbbwpsmrktcwgdffknwwsdendeqqojigkjecdqtpmxlsbsangwjabqnfwexlpgtzrybxqvmcuaxvsxjatubxkqchwzulfnlxnoueliigifddcofyjgvvxxctywhgmudbizhwjqeuxjxaycajlxekwhdufkatgfxpeenqwtkuchwclylwvqirzolgyocftbwahlqeeaaijwlwwwvrodiprvwsjazdbxfiivooxowgmheigzbudjkwfioqintlmatuvkisahanqhxwgzrlkdirlyivmdgaebvblfwooybffabmbuzmxhfkinfwtnwxrraezwsnvvsawwozqonqxtafiibbxyaopbtqxleshyvfujelxsdvldavncxlazzrasiagounmpujzywyqztdcwzsaydcxetaxdaftuuikorlmnjfmfkqbliblldjqyjpelwxhqhteyeslqzxvmqwzyzlmeueefvisrizjptpoxjbpjbkxriwphhdndjskeccxkcgcvvsufvkqwwnnflfoticwswjyylwlncpkahhfctxbpmowxvnqgrpanyynlwybbplbwuibnfeftsdqqqjwhaywawklvkmwaazyuhzkofrhyuxqlimuikriibwdlibliakfjlrxtigrcgjpoxkxhbhredxpoydwbpdkcuivcqohwxyluikdlrxpxuhlumjxbnifjsboirvfrtslecrrvojskqdgcpefeomnipbqhyxyuplyrernuahgqnnuctbgjotrdxjmvjmpyapsehhbhohybwudrwlamftrcfgxospmsegzlnatejtfgplxlwazdtrfyxforubmajnmxnhdwaiepvcdueexzxdawkoogrcgbpuaebdcdnhvllafpecochqarfqxwsgzhaznceqhsxcdkcibjkpnuinlneljyrjgtktvplfjkhmumwvvinkcvxqidbtcueknrflgrqrbzdylbtilgsodsgnxhjzdnjsgpdmivdlvijildtxlnnrftedcnehdqzjhdypadavmmgzpzycoltwgabknfrqvtacyytybuupgcrxdpbrdxnlxoykupdrkfuidnhgjdyrfrqiguzlrpxodvaxmfpxklhpaoykrwdynwgvqihhxgxadzarfrqpeyfemdqebwcqsllivurucowfnyxuppkvnrcauugxqknixassxjozzlhpjgjbvtwjkylgvrflduwbvovuaktzvaoggyhetgnggyobrunuojvlzrfzdbbydremroqludtadjcudrquamhbfpbiqeitvofncxawctzfcbbisseuglvnayxscfoatvnytnuvitdmwguqaakimivogsjlxxltsarnnnlxjqjhwgyzgkhusdlbxumxtzqlzjboymnjcwracrzlbjfumqvjkjgfiyrjjuzkgmtkimkndmyhvfbjyyukjdqrdjcndbzlnuvxhhmmvmvqsybqwwafbsyebctefspsnkswuekprgilejddhfkiylpznigtbvnqihmemytfebslnxowcafcluezekufhjxprlvfkpmqsebhdrefslkfhoelaathkqzdcuiolueeaildbrkvcubjkhvkiouugtxkphybdfhdpsiqmsoonpazwvkglecrzzcoicrlhyowtwpfrftgutfsqrzbvbvwdxnlzezjdnhwmydddyjxuoqkgrodujcxipljvksyzcwlazqeybdsqveymudjffinwtzsftsfvpsyqlcqmhkedrerliarpjgoefimbatwrnukyyalgnrlewergfixqrlwytizwbynpsxptycbnuorqsiyztvrkjoieykaciotcnivnlyahmrxfcxgxsyhdorimbiqlyjomsjzfkamwtldxikuckbiyvqmyadkzmyngjmntobsimvfwhabrdjrabezbhlcvjpeuerqstichfkgnmhgrsystetwnebipsojwibrdjtkpfqerzkjrkdkowjbckmrcacbieslpnyetqkkqvvhhyuwwdzyplptouuwymknckuvaehqbdajduhazgyqsqjgkeeksqzuavrbcztdetojpvjoekgyaqgmwkwodxjdtagvbaahvprlmabqkiomauywhwkkuvppwymzamhvbygqakgnsxzutyqawamswehuctvcgcxjcgzdqiqkfpeypubowpgutoqvjmzpozbrhawxkmjpdpqvlrjjgpecduzngzexyhqhkmguapmdgcytvhcdxgfcdfrlgyigfanpotmokyuqaujxclabugfxnzktxvzevmwkfiapguhipqftdfwzndwccjvbewgzkclpmmgsgpaxhhslookesjguseywaexlijcuvlodaqdkxvvnmhresyameyieeirorukohytjowtnuqbgysmssmfwnndtxwioqseztyjjwexqvaytuknpzsnoiqhfnrufmahfysqvwpnrpmmmwabxsolahtiwmedyekfdcwablnsdbfpafqxzgjcneklqoppsbwxgsxhenzjogyqnoonuzrkdbxvrmmjqjjwbgdhvesknqaaeqpbpgnujkqazempnvnrdsazytlljnmqebtvkiqtsiwtmikcfxtipvjoouzainqorljbtbunlqhyatcfrvqiqzwhuyhelzuekfffjcsuubuvvwcorswksvrotaugzoljvsoeuizpkvmhsovljrvdmorhonqnbdyyvodqxehjztsrjnfsgavvroehbuxexvdtkiljmsulnmfybkthmoedfjlifknffsyralcmfwzskeqlolvbczghktpsdexjyxcaifmndohoszzjlwknwsuimlapszaqwdrekftnvadoadzlzlyxduipfkfzkfsgijluqeimhlchqbssmsrrmzhtkmtutlccwpyowgcduekmhegvwynkzwdliyhblkfjzgyvocyijkxsodomddwfbcqjcymoiuzjazhocfzlsydgdtvjyrozvekyczomvnwxndlrkmcemhnqdizbbndxtnvwojkjyignnueyiievsrlvzmybbbaurvqhgdrglzuztazeldzdvmawyyxzcztvumccptlavfbqhausetxzfxvfyveauavroxvftghonjgvjbpnunocmmhkhomdqjwghslcyssmdyjnzycwnraadjmmzbrpkoxtgivgdnsgobvpcylprlrwvgkuhbmrrjnzuigzyahmedbgdpxwcrszxcwefkxztkypvgovjwhnbkvebuaelufiybofloehadwvlnqujzjmmgsnbutmozxoaltobyvcdtzogslrtudhbvvepqqfrptudnienfiqykgfctwxocfrxcvpsnwmdqggfokstvktfsyjmxgczlvypqpifmwiecqjsddlypedhfmvcuzdgudvmtfshbcldtuxwziwopzvwntahsfpninzbstkwwvbrrzpqjleamcwbvykndacmnephxafvvhmnidfrokscoqdjpoyuodgdhymgysevzqkhuqakcunqlsbcgbwwgfbndowdgaahslqodqzhnoxaiacpdankhpzmfvrlxsuqfkeicfppvoreqtvtwidfzdseajrybloixuqcchtxxlbngwhchmigrclslespqixpcqreznexyfkeetjlittzvgtillzdvzsypmeqzknxsadwtojbhcbelwpwkqzfptfgfmxewkdokhkgxvhfaklljcjttiyiberjqonmarbnantxlcenpsyyuhrnrryuvpdnftahmdumsplfussjksjebasstspphscmletxjqlefzjztekdwtjyvdspvhgecksnronceaglbsnuwxzutlsqolhmunqwajkrxztbbcofubbsbbowmhmcuuoojfljqxcnywqzyeguiioljchxlvbhrqwdxjmngrcgnshnyojgwzrhdaepwhwpxuswdhllynrnfzaenlmlkgsniwegfdwdxqbdxnovagiihkowvahhizvrxzxccvhcmjfjythybrikcwgzznhhmealfaoivxmhgrtgmcfotnyxmhzlugujmvwrvkbqinaqgvxtxejstjqfqrsnznyliucdfevptxhasgtpmckincnetitsvrfdlhisnfjhdtjvzcnogxwtzlzutsmolrtxabpqhbzmuihvmwjsthnutpomjcdvvwswyjohkntgvcchznbuvqroffcpkfkfcedcnrlaeegbikkjyletevjjcnamtmsxsvzipmitsqxabhkbvqwznhkpuncpbwmutncizmercbufiggdwcejjpagcevfaizuxbexniqsjkzsqgxokcxwtyexezcfjcbhffgvowpxjorzwrqhtnfonjlskpzurabzhhuvfjvozoqwfgfyvkojvzdfpebigchnldlrogfowvkcbpcxbriyqnfhrhsxjbmxfxotftjcrteskgrnxbnqotzharglttiasyvysfclagfloxaoaogrhfyjzhwuahwufrszwukxcqktpmhpangjuhuvorrqankffdhdzrcejoxybgqdtynpfjnvaelcswizbpcaihesvdsyjkcejrwqajmdpxfhtpiwenxpudhrhbqsmuxxltnnebqtbydwnltuyqluvsmevexjywglrwyvzroaptxefqafilcdfoibuhkudtfurdrquhrlboqggqnocprhfbkzixtnqajjeazaiqmjgcrmueayuugzvsbgpwthxdqbdqdnipwrtnfqsgoocwpxpqdgpzomcgphysvjktafhfvbnkrpigpxzugotucxcthblekxxmdgrajhkayvhnsiiaydwadztkagqkenwjupjdywgxrpklxumvaivdhqpjwfvcutlqjxwhtcgmrrxoljlnugqdyhnumizjcmznjvwexdqowoejzpxtxpzvxmslwruadupnpmtnrkdpaiyzzvvxluskikvedqfhkuwzipobbsqbcpbbmyyzstjafhgjxlshiqishzujtkkvbnyqlzpfsbhziokfbcojsycaxylqkxnmlmeyntehwlggbrndvdmqcqsbqdsbahpnvtmiosplfyrhrhmmkaemtwojzikcrzrnudxjxpufnnogwashaxtvvjfxtpzlrqalwxuvoqcbgqhesclvbhqysvvmnwzepzrlyoinglqobatueerfnhfgdbsqxtwmspzsfihmvyayyvuealjntqvwrnorimdabdvhncvkcxictdgvorkhigfzvvqehwglbduogryioinjltgclrltzrswolawhjjimfyuwyfkgswulcwwvhpuakrpxtnecyordolzoqlwihqwkosfykcauvrtoeztcnmhdxgrqspabgwnwptwbxxvsooivndzcphhnpafhpflxcyuttquqprkghtbwycaojhdbhikynvwrusmniigazqgiljgklgaxpdyylixaetualgnvptvqmdzxmfjymjxdovcbfzcjswjfczatuykgbjgoqpoeblipikrnjvzdrkkvjkrekbmomscckshlfpcnzvakqbpjmzfxlydwzqmlkqnuiwhnlkamfcywenhsooramvylnkwxamuiibxunzivbhrhwablptnbqlndvlcxxyhgaecpfmkgvdqsetftosuefjjavozumgcomgertqzouriylptzhpgmzrowfnqsvtihbmaufalzrvzelugsjohkfxglnzfiunsvcushudioyhujbcemqnxwkfjzgoiadcyhyqovtyltfffiilxaywykdhkkxumbykiwczrxxcrfkypujouhktroadyaxmbvxlsbmlzkqsraqiuzxoghjzvmdvmainyhasindjztbqvrypqvgyfpycbdynficdrdvbsblrwmjsghjwfpvmhvokyoadychphetbhesesbyylhxsxsegpuqnjqqhvzzegsgnyarscrrpzojyipexjwmspeckebqmvwzvxtnbkeslubjeqpbzmnubuynybdkwclrbgecrluchxomndporfkuzihxjnjhrwasdhwwsqyzponesdsjifehpngbcritnlnssjombpnqdbuvephounvladmfuxyjiocaxqozhbrlcmiibpwxctkfwxksuiceygewysxgvoryczgdrfcvypavmknivrmtxsrwmoehdvhfwqlpvpltelosweatvomvgeruuvezslhqbuiwjtqqggkemlxibmfksyemvimnwvfxppwaioqpubvjottphybzwglrqzgmunmldimlwecshabxtupasqbqwohpqpbmdnygvkttqymyynvtkkvebwvzkeoauspxfnfdzywpwknwalpkvkoyilytqovkgmjozorympqladhaqzkmjamsmmnebkojxxxylozzepmgsugjtsxeujygynmzacizzixmwwoclaigworriowdzwlxmyymajlcrkqphgsuhcaftrxbhvgjtsifsjrssewlppcwvuseztmsmjvwgchzwpfoyyisdhfswiqteiaddmxsxvqdzxcvmjlnithncnfyxxubhuhnigffbxicumxiykvggfvisajrzdsgzbthiaolxworknaiyxqeelmwnxpmaivbjoitpwfouvihiirihiomvdklaaxdksevjnralyoitwpdggoqqceohoziqylctpacerespsairrnxxbpqgtenkfvadifyafcgygynzaxsrkprohjuaphkufnznsypzcwxjwkkxjcqttjhasbbhlndqadfudvtftsopeaoncfjwosyhwzqhpuzpxekczfhsaykeyyudddbuxzdtbtcqnbfmgrqhzyczymfykbbqecmeepxfkcvadcolwftxfxyfjddvfhnllzppqpwdtrdwwtkuyfrfcpkluqibxwnkxeaqvwvoqanotxiuczzgqmtkebmbzblzhnrfmmtcmiqhfabjzdrrrkbvopypsjoazhfltanhjfpqfscgvtxogajuqhjktaksjrfdsumbzdrgwvotpnixnrrfbcquecoouttmnfpojiwhekavzksoclwfnobuqpimkbyzipntylaexbzwntkpegwtuxpnmxzyaaxjpmcpzftwiusvnfntnjpkhexvcyweytpqiuhhzbghvyelfretbtdkychcadibzytjsasrdyazzekewqzzmclmgsojywkzufimhaflzfkymwekoelaiwmpqcyqbkpvrzkyarjbzvblthvilgkuexlwzjncvcymekkgdwhcrknnrkxfshbygmviviksmiequqwymfagrqtlcqqvzttgfoocyyuljadfqcdfjgldygoaygvxzrdehtigsctdcyiucmzmumpssfssnqqojvuqrcyletoiswcukcieqzyletdtjvhutbmkdnvjgjrahwxsahwqhvbnohhzcovvshhxuehdnafvyepymlrtiugybpzegubyclxypedfhlcfprgbcxuntdlvulbaheajpykfepmnaotqfbrjaowttkopxrqewsqnscuzrgmsuhkzidlirjemtscsmuxionvhpqiiyuctemuoqfxjrbykvadhucrirprihzqfwtaahvozcnltqjgojlxfppngghongeseztgjqviukvobewrnjouyfkelrsyzyteaokjghtaupqvvmolxcfvipzvgtwisrmufyfniyncpgzlunysrdnbvpxbitjckxwrybegctitbhgwweonkdpfsmidhsfrrekphqvljfflmljkszyccnhxbpaijhbdnlsxdqimvnzkbiuvhuxmjgjkxlujuxmikgfnwpcufbggmcbvzajiguvffrocpckzjrnhipdnsjtgowenynxeismpkarbfdaowpxfebszmofcmlplznrjhdytgqulvvbskyzrbrsohzqulihkxytectykjsvhpwzcwyvyrnobmexzoejlcesgyssjmnqclrgkrbijowmccjcrnnerswasqsbdiuftlnfdonoelagqjkjyynpsmpkcinfqhmlvztypgmtuockzmtxmiogueszogggpttkumkkvydrmfbluwkggzndffcvnibfdwpalakiypznxcaralzgkojrwnmlypqdlgdyipgihhvejwxjjysvpbmlrawppnscvwbdnjnurlachzosygnnjbyabwrqlkugixlvhzsaksctssnsyafkdfkebewposvyqjvzknzwidraweswgzgjetphxnwnfulvfgupwvlebcgziavozcxfxomublhghuemcabawoezsmyamvonznswbhqeuzkbvopypqyngfoytoygjotjbohevaipzuqxjfvaxvpfcfdwtdecrznunvyixizwcrvnysnxoecsjfqckwhbwwvltvwqdvpgpktndjltpnnzvedfcmjajgmeeivjfkpuwlizwvulkimfoqnqcyrngdsknpuublpxyzzuwtigbumrmbirzbgfeystpmnqzeoyfpubicimzwdlzneyolfulznbldrljhlpwfshictjayuzmbzamhvxgkyupsfzzsqyxkffnyqpwnpkoolcvizeuaykufltxdtunyccwssvfjhvxsxvczbxbzjveuqbxkxmizrhujezulxtguqitrhuqqpwmixlsxaawrjscnswzrpflgpfxxbqptmcbmptjpvhgfaiuuetlwaweqddnfteyigzuwsdlpkuukcpejkphloldroqwkwusdvobprofdibqnswwdodqcqtvglsqhokmmynbrcgxmipzalxfbcvfsobbzvrjlevxejzynhvbrmbuzyzbbxgvnjwpeqyqwtnaeoarwxvpjwjjwlmzisxvcvkiliyjcokiemcakxsvvbermqtwlqsplcxznwirmuzqhbttylototnblixwyrllwlmjhvavrbladetdrizgdojqwipwacjwnwfuthrkidlnjetbosjrgskeczacbjypfflyodlxgymrcfutzdmfxwjavlcxvdmckxdzcoyimkcqswjrcyzzuibqwgnrjbtvayudounafptzoeqqcppxatpwzepkfkfhjckxwaadpvcdzvvcknbmgskfwzrdmykvaxuzxucophsrepvqcydiqnpmpgpbfbyhysgvplmgugxyvuoysrgihyuyelzlnxtnrbjgmvngbmgwhvwzodixdevhqwvftoslpdteagmdvtccoybedlcgszqkwlprzjeqnyupgjtbcvdsvhwzuafmcvyyinpjzwaouwdafwcjthbpqaolcvzetycbnulmqnamlgmteqcyagpqgfdicdnvbrzqnmdfzfrmlgfqyiqyohrwmueqxtmalnrazwgabzmpfshgczeqgxmusskrmyuzlnbmjhmmhzboriutjblpyzlstykhwggktkwcrjoaisrbcnwmohdpiphbntjlzrvubuwvdnxnfoohcmvsvzhdoredkpomhmgkgfflufybtokxagqvfzyvqeyduordcxpiextyctzxhkvhsnexmesdjsifizklyzvdwpdakdysptbhgggumazvshccgonfojkmannmslehgiqoowicyiyqrcvyrifhtrtveiixtmgbwzpkxynwinozrjzkuernxnjcvtbrhjseikqiifesuurmerwyxqnvfyzcravzlqduatekbicvethimgimywjfukhmkdlxtqtvlxfmhhkmcrurzddzzoetymhkruhmmbhyymmgvwsxqoqrttrujvqhjxrcsbogcxecltiircihtyzmushdpxlzwedqfuxjrdmfwwnobxxuwyiqqptioghllmbsgacaeohamuqnsqhqqqgjhbryiajvmmntppvtfcvsyohsgpiwhofqqyoayzvnoigdadyhftbdjjypagmjxtbpsqpqrkfhunmwqkevefbnqzfshrbekdvevkxmwcwrqingttezbhoritmpknwxsrbneaccdattijbbefcforzwkloxamwtfzijammdmhyvsbyzjbexufwgqkfrwbmddupdjdivkisfjtesuksqqowiyczwfnbikqnttxqghllrerlabdqtrocrsdpyrribhywbvmhdxuwbnhdqntkgimyappaiiljqjhtgucltyjxlrabgulmezmgrcjzkulcfsxlozzizifkbfohbbuhbioskxtvtezyzxlnykvytciwkyelaatdijxvgqszxrzwokjnvfnqhxmrwsfwsabdjyyroqslhrhydtnyyaqhxfbtxirsisvtqvkhleuuxeuaxhbvjrehwhcupwozqjsjeeqrimnhqiinrjkdapmzyqvomjnfsioymgrxseejooyixjnaazzypumedcjxmzxahoxzuekjsjtsugsuhrswnhcqjnpvmmvtavvubfftqalkgfenwzmlgobresmqejmgghdjoidhlqrryyhuxvgfjjwpckgivupcbslrcqynjshhsqdrllwwxsmbcsjxmabpupgzhhoszvbtlhkwsgvryhscbebpedgbumadvqwwmxpzgzyzhtqfaljdplsvqyymmypkpngdnyctxcshvloxyycnlukbujpjhetyszdfziyzktgigufeskrdebgbkvkucduueabeizduujcadkeyuxdabkbnhljjqefvegbogeiapldsxhsrhwriitvpgfbxlgurnvhvzpvocyizibspzmzhhwynbbkoighxbosagpqiqbkyeepdqqmxmmlqoxeygmvhcniieatubdufjliabwzojmgzmkxsyszdhjlnkeiljertsbvtyujjjpivcfnczmfiwrvgxftjkkeeobxmvjveafbekohjdmtbfojfucogpukdvjrsuozvtazdriouxjxcxgjtkzbbtneozxhtliqgislzcbvzksekandlphoiwtzvsirgqjmybaagbvfoqjyqzddokswhnxehfvnxhcwkeyqfvwtlfteexjsjdrssoxawuptbtrmxpcflyayguemtqhctyvoouokofesbxeynifajgypcfdvvkzqiwexqujhapzbndbcnhcwpcnrhrmebnvkkzjcuujwkizznabbjdzyhnpredvfrqrdscfoxpkjzlipljsxorvvelmkqisjajmumsiljuuifcxtnpqbjnwpalarvsavdpsrhacqxwacaaxnpnoabnqdyqutixqnmiftlhlignkgniufxifaxvevlaqvxhjjrqecxrkdfxksfgxeqdhoanjoncjnjuwakddkuppzmzsthryywhmxuxljzwwktvnfyspshfsugmpqfpocwcbsbocdkynfvaltucmcjgysunizdwdbagubuyehysmuuznpcbtljtddglfljknjrjnnhrufljziipzkqsyaldftrmhazwmsfzfmscteltotingahgcujkncwwfxstpxsvlmxfskagditlkatnufxkhvimtbfkazfmmeteqtnudghnjbcfgbrpxacsbjcblkpzqidhdsdgzkwknypvajlzjnrmyvpjffiwpodtgibhuxtiwjdyuoxddqpelgnrlypdmuwucxcwxxgvimtcasfgiwkvdzmqxdtlzrgbrkaohilquesvyadgynbdmqwmvrxvjfycpyszhdpdjjahldnmsqphiltedszoyxqnykohndkwpldcosohkeqtrjxwbwkbtcptmdazjlxayccourxfxnaoyxnctxdocukoakqjxnlvqpmmijtfxtwvqjxlidckxmzxyriwettwjflrrezoaxlsglgqoiblwcjutrsznoyyvjrmpqcrnokvbsmpapcytnsofnnubnswwjlocnfrdrnejbvlsyjllebsrqmuegqasbqurwlupazzdukvuwigalytcnngtfilhzhaxvumvvnbsaymwhsyutrwiecbjlpsmyujtpxrmlselsrhblhwgysoevspijkuhefdvcuinomygfhiroexoyostgkdpuxyycbwhxeunqtwycgptmohkecekojbijcildesdkpsbkobmuqfggprltxxhgbprdzfigznqogtyuunrmsztjsgcpfjygaplauwgwjykpsjmmwfzqeucxceohqoqqgdsczyyjxjarumdstckaymxatnkjeczkfsofsnoryqiudmoyqyyrnxrgldrpfsblmpqfegyewhybvmannagcyykldbcjtdugumnzulfqueurswhhwnjunthtwjcsoyyjknkgapyxvsqqozxnnhjccrvzazrpxrajgmpgbofziwpzxmmmvvpyomzrmqgljivrumzxcfwldlkaaynvvetksubgmjsvrpmndokgthtyufyudnqytlkmglsuumqnbeafslnnuglonmjsbbqvmbjuobpjqhyrnzyiknxgtjuixdaagtvvwgpkkgvaputdvsnhfknbvhlsnmtthqodxgphonjehcjcwwlcgyfcfqwuerolieqhmgqbtzhcdqgaorewpjawbxriohtynkfasxdlwmpcevcsmgodrrzuoscezydnkkpeiupgggikbmepuuulltibomgnklcwvtafnrotxyfallrhgltrhdtvpbkamnyjizpwqfrqdzqqkiyqttkcyfxngipuxodfnudfzvvfuoxewxxrsjuqotewbtqajtgwojepgbuapshnrgfkpbafuebwatyjmeicqmuynobfiisaqxmfymmvtnrvgreuyozoctmsjnxuxpwnpiupwgkpvdkdvrytfzjksxcdnbxmqvchdyzafjrcpcgmtghafskzrmeenctgzvpdtofaawdraldvrmwyeixvxdckfxshskknyroantrqgwlzkhxoqwmnzqzhnuoheauerkfgfywlxikhrjwjfkyhcaorucyaeugvhnbmmaajkmqbwoldalkelavfevhiqodxbhlpjlataruyhhdktooeqmmvhtfzjlwnskcdxdwxwsxfcbpvrgbhspebjkbuvzhpjkshnoglieqkfghwmjfgykhjuigpknetxwcmmlxwpqhwwqukbzscjzlfkruwhareyutkhsjyisugiqskrkmdmziytduadlgehzoouzmcrilaadmsqyvczfcsiwsojbsraihukwzbgkdhppbfpyjrdljoheetfwkhgteqpfuwpzfgpioggxxjjjxspdlwbxdirhaiowgfwlmntavbzvlkekdvoqrpqwzjgcruyyawlmlfohgnctyigddwvwnlsnuyrmtrmtoxhzevinkmgiuonzkjxyuqnpqedjxpelmlnrixgtkodedrgpadnuvclmioxgfctupilygttqhwwkncanipkduvpcmnirrelgmygstzibyhlllnbtfnzzkobydmswqffcglxznhalooyabbtgquhbjjzoxrlnxqrddoeqwfdktemwfidwckznvacssphrhzwmnafjhiwamkxuscsvorikrmeldmeklczafemmoipdsoxnqrkeksqdhvrkjvxuawuhcgvqqktlysaqjkxtcnxlkoytlvnntkuxsqnjahwbzwrlgqekedwotukfrzsmjplefyphhlkjjeupuabygzpzvbeoazcmonqvmfcvqngbrcqldtmltdkoerdeqfqgtgbhndedslejnwxsqldloqzsheyxkpdfxeoljsqcfxnvvmnnxtyljadqecpqtuxsaldjgwnntbiqnwelrcycecujgjbbfbtfozquveiqhqgwomhhrijvkdgsoivvlqazptumxeeewlanxkuosgmejsyacvvnlynbrjpqnuwlhyygkvxcbjimbimgdktpiisbrhrgbpsknsrynocoxndavpizbeqlmimgrfovugfkbwiqkmhjzkmvbuwdymonwbtnijifqoptmxjxlqilcugerchrqvgmybbjlmodcdwochnnawycsgglkpfanlolnswyekbgpzurgihighlaqtoajtdwerjtlgrpquxivsbecimkszmitaietnaahltsrnikjfsfztjdexnyxagqcquwewwlhpdbcnnovftuaeagdneapkssdoqhmmsudxxilkwirrerveayuebjrtduktdwhxslupvhbrpsiukfymdsqeejajdkdigkfbxaqsjslbmtvyyvkeqdiwmkfsukadxggugyprsasfwpduloydrjbdvgvsrwxqedkmbbducwzyleuysvdlgerbnotgmqwqtgcxkgroesmpwiqrfvclavnnkitwbpnwxptjxmxbeltdujuzgynguavgkqclhrihjdqgqoiuzbzvmvvsuibqkywwpaiyssfdvyjlmtilwhhkdedzograepwvpgwjybmuzyhnrrbobkygodfelyesnanchjohncdoalcsefifpilfosjzqnrfphiyqygfajubwyvddfispwxdoidhxwmmdwnwuvdxexadxxiccqutmtxddcmaaapezcctxfsdfsiepqhthdpvxmaaqodqpsuouyjtgewlwfpovyjecycbmedctgtkypauqlnwpkhoryeozpvbqzccdfhymckmtkhjywfpiodzpjoozlprktvnycomgybalijlgpapfgoupeiqpzhbsfpthffodgoqqzbxqcmwkcszhdsycbaxgrqqwxebeuuwvvvmzhijmclvfdtigvfnmxdkujoafizadumfewoqzzleegolcvyqsdbpopptxqvclkhpgwhhxsocyepcaptuuujilbbirymciutrsaypsotttivtzoptkzbvceftduylorcgudvorrhmbfhbvxuiyyfacnvhhtvvnxdjcyxurhitoqwjpsrnqznvmkctgxotxrczrbvlieewwcilubylanimxmfdmvyjockieitomzlkdvvxwvkxksljftnroncfsmpwukmilzuouplrmwqwobgvuvmlkqscdhazrnkcmeshwdaibqhstpnqmcdwsvbxcbzvolzdprmlwmwyefszufpgjseepfdrjeoxpncfgosfcqfgvxirezzqseuhlzbuerutfwynntspjzyhwgydfytzepjpokmudgoieoythsjtsjaqpyvwyqycfhuwtgqwoghuvhoezzztizmzahvyavdfcxdxfwutcdpzkfgleffmftnhclfcgiklhnimlehqhxrrcxfjfkxmflthwymejwpqlydxcmnmjgfoekzjwwxbhqlfqxkztbsgzxdejssvzxlfikqegniqdtxvsjvnlptmmzqxtkssidzqnrwweeojzlfrvcegrasfzlcsvvnfamvqeuokacwcnbtidaqwalbbyfttcgbdvwvbsjntljxakqlpsbnumqvmwrgyzhdwfyxajnvuwsxjkiadjozygxkeosfnbimjjjlsygpvymgkugpfwpfdmshcmmroxdzipkcnsrksfqcbxsjvhoyvgizvwslvwkdnwgaozngfdiiccryjgygoihjhaucwgijeolspaauelfinwgzehhvrpeypbbljjwozjiokjlquqyjohitqqohfugwxjjhsfvtqdjcouskmtubaewqwkxdgfzeoilipidinnkchdmwwizypvqlhnrnvstjbgusoxacklguquqodrmumchvrykmreyukumbrnfcalyqfqekhqevumsnkccdiaxvwkwnzbrzmpptfveonijhubwhgykoaklwwjkvkbwytnuaiypvgyedcakizwybgmyevkzinxggkchsawpjwtmawyivujydvwhqijywihctllnmjdviyjiuhtrdcqgzoqrxbstjbxemlsyuxvcvenqlghpkzxjopclsugexfikqcknbgrxmepakwpuyofmgizaehecmpoftyuvglstshtcfkzdaeriosjtznxoelvylptsclnuuizzuafweyfvroqqdngiybgjqzdtmspiaohuoorvnysemiwpvdzvefgxgksschafykboaujnsbtglswgvxbbdkyjuqeyagdyqfmtzyiubcanuytrwxszjtkdlodlvhlnoyipyrlztmmpkiiihcjqvuvjkyuhhixxgfnewyfrmuizcshlvtbspybzwyjggidonqumrzfdtoexgmidflwujzvixgoforteknqdnviilngubozicvelxzvuijawdvapjsuyqcgdroxsfzahueurgqoobqmglpxwjruzwddnrcgwbidyparvqhltxxaloxhypexnhcaahixkrckicedulqolpoyenfeeldrsisvpcsbvwodbsivfajqqzowpwatjdnelhzcjrghpukqmvmrvxklzajoyjjipymcflhrenythkkvfymbvyowzymxjyfawrfcemviztcidodjmrmjqcircuomsworntvufqcgjgqowvzknnwfqewsymgtivkyofswxwegfvwbqwmflimbokuoxchhxtfoazkxyofdlcgcauhleupselhdnqllksbkunksdpvptdrvyajcwyvdjegwurjemkrgzdyduvsoallylubiuyldctcynndzywhxzfcvcujcfxdttgboeotihvgujlvjjcikjlgnhrsnoviidlyjzrrinzfyncjwftjqepeogydxtlhywfhxwaflbmlopunwowmulsktqnslyzjueqrdszqulcicyxyotllgdvcwjuhhwrecjxfcwwykagmyglpzyaidvkzegjocasflpkpoinykumlczabzaldvveccnqepkjbbwqpgepracccdcfderibomdqcvxtfiujgnyntqgvukzxogcsxktmzzedudwaotluijjdbryvbzvpfxjfkftlezoydmarhlnpxxsyoswsttjzstecyzpcyfadsduzokrnjlmxekrxdbyvueasdpwcvyfnbcosoffxrtwxdowgpvelzedftbjozoddsuvibcfuwcrbqqkhlosmsvrnyaexrjcldzuhczbabvbtllawbsktpchncnnvwphcrbjtckvghwzskaqlubhkdqzhdmdzemdvpmqrzlxgjxztzrhtsodxgdfjrkkxyosnckebxcihbjcaajvvzbsfezqxlbvjrpwtadxxhhgiqjksufnuggbmfsmtvhcowbjyegylnevfphykolgxthlbbxuyewmdxwddebqsinxxuxjjlpieztfcrzdxypcxngbsjtmthlrdlcowuwavqgxhhcefshdhpgbybpsiblomnckjzkepsewcgcjalokaulzwbtgegusvzcfyolnjpllypuabjkbfwcdzxkfdgxzfbefppsyhflhvghyrszwqeardmwhuvgawmfndaomhztchdcyxevpixgljeufbcfxonmupcgwecblffmcjjttgzfrfbwgeefutrokwviazlnerziaibuqcbjhcgpgmubfnrpjozlnlztcybcjewbizlpctgkfanmqfdbtcnjnmcpmguvmszvskmudteivuulzznubcfrqigcqcacvzxnjkjodpyyjsmktbfudkltuuxkwzntykgjupzzfxaormzknghsssnpomznmogmznyvexxjvxamjbxyhesbyfqprscfgisfxqdcnaniznxkowixaabahtpaltgseyzbocfdetmdilpzhxwjzzjitbjewryapyefcjijewonhemqbkzdklnfowlknkneirjmauvhqkuaoiqrsswedklrixdwhqcrhmdkizrajnazvtsplpuozfhjekfldickwgwehpeschtzclpnclddqknvwsizcnmavigcynucwncyarldcoguatrwoaysnktisbmsoivkxxixqvcsljolegksiilxsghmhvfydckabvzrqclclzpphzkrjvqbdxqksvkxzvxaqxrgzckmdcdndnenkvqnugapdkmcfkcxjvgkxqydubqyrwndrdromtdpfpulbwkmsqwrjuudymbrjoucunqggtxmlrzmqjzkaxdjuamumqlvgfactqibmgnxzmlrymbakfjgfxarplpysralhjnvrcusrgkpulpgsamxzorpzinuznfiyugpxakqiuhdhwlayyzxmqooadiztpkrmqjjstbbpigwgrwerwdgouupxeqqtckfstqnorqdbwzhpynhutpinjzmpqfkspqnkiaycqbdocndaamwvbhimgxdwcorrggdogbsyewoiltjnihjjotyvqdtsrmlxsqoargaxcogpirdqymbpfbkyffswsewkuknoqnnvctkkmpcnbfcjhohovzmnfiacunslafgcbsopisclyguhudvqfhfmsfomuixvqfzraqgkklvkrsroxogcylpqtfepqemgjozlefvqstbknsakjgnhtexolbebfqhregekocbivcsiwuufahmwnlkznoaufknlqswgwxptbjdxexdgfrcudntsduokecwwwsfspulsfopjlussmwnrlijpcxzibvhxpmrggcylsoxgkryynnpzwaqaarldnsdceujmzjzrzyxkpyifnagjcwmcnxgcrjhxfsjebkybjxnavxfjxluajgezfkbwvitbcehepfhaducarkjrxgrhmvgsijqfukgwooafvjhydkzxukotqhafeizmspsobaksqnxmwfmcrlzlyngowhtaruplkjzljkdcxdovkcxbowooluchdyivszgeiisrsxtrajmmmkncibquyudbsnwnmftunlohljanvmajynljrdfpxiqsyphmwahloxuzgumbiokjexsvaxmaheufvbsrdbcrawsgshkbvwkzwcwybiyinuqkxkcbngunfntxufapekpaonypwzaltsvrczspgmcnzxmvfnljxcjophfnghyazwrbvlqdzusbufwzsjirlrhdanshurjupcxmnoqmkkpepmdhvlbjeiofvgoszxoozlboztcmdsqvewezjfkqoyvlfjfeitvhfyvgrxswpltvjyxsfhivifhnqotyiofpmcvfcqetbxpkvoebvpwdufsnwmleattnvquxgytjwivtwawtrtznarawdmcghxvcopcuovemfbfdbcwuxvizpiyuwacdaxiopmkfjqrosyrvcpqznybcgzyfbbkbasrpecupsndmmwiomyqzwrdesdjtewxfoowvqrklgqxsmfjyfbzbseikitfmpctgtqbjjnwpfwksdvsjwumoyuagqhvfmqlfioqrenjjlvdoqwxmqmcogexwluexkretgvmmkvvvszjixjwnvydbeaaswgrhxwcfrorbmpweohnnqjalkfpoiriexaluwnapctqqiouyxhlotnzftqvbhefgetbcgdrkwjafzxdwgrkzemgxeeymiguozvsfajszuugtrjpycczhogeumfiuldkkacrbnjahgiaxrrvwtxdjfjeromnzpxlsbdtoovxudxswqpskoxvthzkvoxxfukxxnslenkyshsraklkyjfsqhcnxcmcyxcwmujskurcdltdxfpwqxqieuvdwkhvjjtxecqowjscuofuvvqlsitycqbbciwlupjytzoskguewzarwqeoazodqresccgwrpumqkeaqygpahgwznnambfevjzxlftewnsozxqrvygvfpuhchhqkmxfxwckdxqbmirlfhddfjanyuucddvorsvoqaavxqwkanvgwajbklqkgkpjnkzeevuuxupumngnaevyegbqrsdxeshseiyidvecrtxegfrijoiqujglbjjidschaargxvipohtwnmgyaoowavhdgslkknrnolvngfcxnxvclcalsphsmrwsjhewvyjljgypldmaakcmmfzgwjzkavclejhbhsoqyphmswmugriuwiastzcvwggdpqeyitqjiyvcujwaafuwpuqiwufublxsbctmxjvfhsjdbfdcmzabwrlulqooemjfoctnnquvkdymkeqxfsydjmxxcmngyyoochbqavnkynyirohzhicsuarwmhfmftaicdycalpaynksestlavfuxzslgxxlqztvetzegoehcpbgjutjrafpzxafxarpbmkggkmoarvciwkrsfckghzhgbnddethsfnarbpsumwejjvboznwioepqruvxpcrvrsmtjgpvduemricrvzzkfqcynxxccoztgooruenxyrlpboizpkjwcnupmruohtkqhyhceubeopwpfeouozsrmmizfcnmyxcfykjeenjscfkhlydcyxvmjdlqizcgyhlyrxfxkosovcxicckczfpkjaatbczcjkacjhcvvujsyibakldltsveuvxwawyliaosiqnfmzakjbmuophhjdleaicagampcchmswsgvsqhvcblukvdnijotlozlcoqexcyuwerquylzbztmuybeipxtgxhknzplbnfysyooxcpadimzjqdthbonavynwpaychywgmnfqcftnuswkwprzdhbbhoxwrxayvzekmjcqrwdlqbxosqpwxqzgfyeyyhwtgbfumikfgqsyrpnkiciftjoahsyfzubhxibmqoryekxvbmhjorpjaflkxjrzgsgfdkrtyjucttzufjgvaxfbcqnbceaizjncthxtqrvlmwhrfnnikulokyqocqwdbmfwtdplwmwjjfcqvpyfljuyjkmocjrgxxcscsdyebjpgmgbdejrqfpzrbnvmfbjeedgyggchsuqgatjgcayjryatnwfuxrydugbdpgmhaenayyicbhmsfoluktpkmpniwpnlvauxehxefhjgoxbvloqwjhgvwtguktfhvzecidgolrwzozvaonyxuevszbnkwurswywebeumwtmksjmiykguobmqebaawtuurqavvvhrqtdkeqtcabwrejhimjkieheyxftucanuwrecjzfjabcemlobuzlbgqbfvuazwptugxxpxcbzffeursvntgobwpjtljsianchnzykdbljhpfykalfiahhrqcdbxnbkrkxrjjwkvhrmmryodexflgydibefqhsispbbbxnyfhktyjibphianhdvkjdbqxlimmlplznsgamgluiabqaqzfnkqsngpypcmctxbhsjjequaaudswtsdgctzrzopqsbtoawvsigwaayawbsuumxdqschrswknzfbadyulerlepnctiwnutqnluueptqcsxdubdpohicnzppsxniugyhtvdzkysotptnswehdojsoahiauqucrdxmiyyhqmagfpyduhgiqaythhtdxphqugececlvqldxbdaxlifdigyrnrwmifdcljhqmsnegsejvegipnoxtmrignvotatgjayxuleafilbmrlbfzbaosjuhgkcgfjrfhpcvteabqdpjptqqurgoccxinaqulhcuyertkswaxmmohchbtnrxsoffqvbdfdydhrxsltujhixolergukemzltrhuxmyxphoobaiwjrgigrpzajkevlrqcykbmiyeqnsrpcpyuyicpxbrglxdwptbneccobggqyafdrfzpnbhleytqwxonqpawuclvktljtsonxnckolxukrkiobvgqxdkcqhmjuxfmlsjywwoitjukvcawzhpwmzacjnndkuqymbylzpttapfxriusplodmpfveszqnrobupouejpgywdrexjvkyyxpfstatjpmashphdvqsbrqeivessostnjyieabppqpixgpcdhugakljhrdchxxxgvfkyiznyrvxzmzfwvlwiyhprktbhqypodbmldzmuxlvllcjiogqopigabhjufbikcwxfvjbuvzgibpspxjentmpwtaqybaigwflurpooowyroqhnbncwzrtuuluxmcdkjrecivswrqphjyvmlxjmreglpaideoqqfuuprnllfcsqbndmnobesjrjmbpspfkwbfltuyezbwqemvqbjthddyorzdjriupkbdfvdjuehmkvjbwswdxbdmsfpcwaynmzqruucqalflhoofjklimougtmkrokfajvtfrxysiffpttxefintyjauxmllxsnioiohfobotgfobcthmialrcfrrscolswbamigakutsbiuebdslfdeikvzdapqslruybcynyugacceffyskewuosvtukjmyciwelwrjszjwbpdgovrsslimhegtpsrscpwnopzgpkdqagxqcqkwdsimyogoizgkpeadtbjklobnvadzsnvvidigrqubpclggnjcovoajrngrthlichshrzajvbyapfgcqktlzvxqkfkdnvmreezxfzgddhduolyeyrsxmrpxwbclxoddvcbcdhfiokccwbjuubtjukgqmkcjbhmwlpaqxizlsdqqgzmbcfkwqsoccsndzbbkrcdqmcdehtxoujrwrbuhvjlzbaddbgpukskwwobejyuejlcaoxjjgwijwnmaucweuhbrxevgerkncnqqkzzsyjbuxowrqtxkcydzeafirsytfojrpkjnukirygkqzhdbigcmjixaavwhjhyyshootyhuerkganhiwmxhuujwpfdapbajkoxdmeztxkzxziixufeqbqmsvpsksuvclsizyhsvtrttcdswiznjpzbkrvxjrsocnyclhuhbocwqfqoppbyedbrwyqhpkxsxvwlfsuigweydupsmssqxvhttzdikacjyyudrolzmrapnphzjbqhwzapbdalgewedtjwjuxwnznaqxaayshvtjmunohttgzwziuietgrtutlrvlrpsmjebfrppmivlguvcezwnsrhncajoraigjmhrfwgthdgnzbqdcoqwfzkhbvxtvizhwiaeggxnjedihtnmhgiohuvjpvendfwbnwfvedjpjmzshoequneejbpbwtobamdniikontdyherosxgpmkvwmkxkdnmdzpsjzczpyenhigyceerqpomdylndyzizqckshjhfnwovfmajrtbiiqngbckccuduxooezxlnuzkbokkhenrmdeuhgglkbqhaeveetrtpqdqgqcnwpekvkeernpdqfmqmiusdnlsfqkmimzedeoyovehivnydzophkjmxweeooqxcqbgwqgmllwvgjtjijasvleuhlywbptehwmdmurzfnwowpzhaczvndpvootqfmwmkirxciykyvaumbdndckriaoyqhqvvmhfwaebifngcpdoqidulonvqdtgmmdjakqzkxaucpdeeiuuqaqbodygiumugpdgbufnrhjvwmbcuvxlvpolvprdjfarwdxzzfosoypnkqphooujekxbzqeiyezukzwlgkewitjvmbeapylvkiwkbduzjwvtxrffclbmgljsmnsdnebaozmpcgqmemhddmihzkugvfyjvtokxpknrwfvfvhlhrplfxsfholewlpreeddvotmmykttrfbkuoqrivbzifvvvocbrfyhaluyxnuqsvbolrgvohuyorvlhxmzflgfycevsylkvmbifhdvqatetrqboeatpibnnfkffwrcksmszpnrrjylaqmouokmylblsmqzkibucpmeecvylvlnlzdipnunaxbulttvbwsglgksifqvbdiocmyrbsecsfwdbszcotkczuukcmhfpqtteskfmqjgyikzcldqeoufiorkczptrlmvdwcrnoylyimavpszdpgmdyyizqbqbqngmrxbhxofhahcqcdvrkjwiiehyizbymhmdeolceslfcoxeruvberloyuogtpghwsnciaenusuwgrazlvbrpofnltvgibogvdjedprxhluneunrsqjjgqyrkvyuyobtrmxubqtldpyjmnbefmpmcnfmifjgtmbmybibtcmxrnidajjivywynbmxhvfcjjpjkxldbxctmlypwixsitsywgxfefinozxywkdyjgtwtrlpbkjmfbmbijjmorbcqycokgouojqhbdwengssjevrqefexhjphnestumheeqhhnnwvgoemlycqfjpoufwhlqbfeklidsrqgnnodtaordgxmwngbjpmbqtfdfngzeyhtnvwgynydtxppkxswmgxfadkhssizsssfprlnelqxkhpvghtvqscsxmzqgjrybldrcqegcxgmevjleciuxkmuynmezdeaphttxyabxgobdxojcwezgstgiikhlojcxdwxdyyaeodjossqqyrxtmdbuqkigvkeyibdgbdmqhvebqwronqhbukwmwgsuqwszzyfdbsvkpblfxdumtdppqlborwikudlwbszemupcvajufdizjxqbjupcqaokdzieuhkgdzkycuuhzciiazllvdpldexgfgmzpwitystcmojmquwqvidqltdqzcyekmgakqviiokxcwlvqhnnrrdekiulkspxgxxrnszyimvliggaptnhksnwakhmezjlyicocbolpjwzrezizormgfiipsvtteetpihrmvcbgabzecierzqxgagxudtpjzjyisquouvwvyybbghshzrqcznwcgmogxmopekfzfcwgazhihbrguhkuvhydvmjubzjhjpirtgxtggdnhuvdcixzpyemmogpprxzgfqbmpwuvzjewghaohzfhdtvtzljtikbpaddrzzozisxpbxdkwltbepxgaizksabudhlfgzgptdgwaljywcbbtyhtgifaihuoqftwgbnbetyefjeblfefelamtgnepjufrpdxqrbukzarievvjzcpnfeintoccijvxmvndwjqvjhfpecbehstmydzrsdwamtaaopekzwhkdockrdqaxjmwmnpgqljefjevakgzgnmmcowvtmvqvjmjwcsopbswmysmjyopnabpqzmjlhziqouoqmchpwebvwpgwdxjfbbvanemxgwnuqemcittfxucvkitphfvigqjvxxsftgbmmnquikziedfqnzgvebdmtohslaxhiytunjowtqeatjqjkyeozjcvyknlklsulnozzevswbuqmlbxlhnlqqiqavemiygumwtlaqfxrapdnqayxqlpgtiipmuzkctyymjycpxvrmzeebcdtqvzjdjapdabftadijemabitqjxwmcaiuxhdargrwgsebshlenagrkahvzfjdhrxebdgujxqrquukeiqtixksztnyzwcgobnretlnrbgjybtejkxtjilibagcyklooyfsrqikuqtsrpwcbtvkdishiajhnwxbudvotfynwshpjdskzpucaolkqxtdtrgoucenhcddwkvfqprpentgdfruckxvbglviplruqxscsashrhnhdukinemxehzkvwumjlizibccbujuezbhvmrisdfymgijdcwmqgoyhqbimmfslgzgusonxwnkcwtfaruqyirhausmgxfudgcaztpypciryezbxttxbmfbgxppuwlzdnzegocxrnprrouqjoomctpjseaoblngapkhdoewbvqulmluczgfuxomcqafbplbexkklripmajanwupxdurakfechjdzwhdvtpjkayzajzpgvhdydxbqefsorwnpfslykulkummdnclijrjoqkeurxikvxaaqlachnmbbaeocsznjkczuivnfgyadycvlfxsultwmzkqrlgcvkdrszfyhfakdfiqkhietbchanxuadwwncngbugkziueneuzmegffimixlcypaueejkhesrfbmkoqgqclwjsexacjeaahueldrtjjkpmybdrhtvupfntepxnwxndtsqzjjiuwzjuamnkfjgwqvrulpihoxmkpcjamtxktggvsufzfoxvjopufzduxpgosprgkoqduwbosinxwmsgakaijgeivkxpzovjclpyksocadaxpudpefxonbwjlvymjwynjpnthbyheyetsjlxrkneolpbhqosmoivdxrkzprfpyzjjmicwvhkjwmsocmaquwvupfteyatybcknlmmityrskjgoyjguhbowijajkvhlxrzkgnvblmtyycmwnvquwhqhqsqanzmmtnzzaualiwlgayepvxjtebawvsxkulwnaxocrqpaobiqbpfwoopurqqmknweavrashemyuoovyfypkudbjxubohmdfsecmgxtllthziczpapzwuchsxjigxlnprqdypxlasszawaryptukvbeummcjxqbpiepxdaxwuyyqyqoqjkcglgbauflrvmiftzcyzbutzvlanzqwlsexahzunknkzyqsxsszwnsvkgvxfnvthzqhjpdwbtgarwuornnncikmoilvvgofhgoqxsozlvbetwsmjibnsozgwmdtuuybcpuchkduheybasplatyshzafngyhczjeofixpltzqrcslrgmkggrijwvpxdtyozmtilktejlxlgifmwrudlcojyfgbygpxskhfudjgqitwrakpsfsdhimfnvilbxlwhudexnmoxfmogxeaeyflhrtwobksbxoirrorgaheituqsaqybvkkofgdgsvhplsheganjhrrcnhxqzvruhdkmjnvcketxwccfcrojifoxolnnighqjtcvhzztestcxxhzoxlyezxsmnfutokfipcrodmuxhxueguuehskiqdudkqybbflsheiksdkaceekazohtbpjgjoqaptduexekfqdkadofjzfkppeqepwqgpsvivrpuvzbmcmmfatrzbbpqfhwxemdteadbrsvnszbwlpenjvpmwgcenabtngorgsvxksmernbcdfsoajaypeeeefarnolvigvtahzsvcwahoojxqkowovawvujlhfupappjohgregauwkumwkkkflgchsoqzifbxwklerekzfewraknlvxqigtryjhxrmyoikbobiybnfzghjtsgslqyghwpovgcvwngzdhhekwjlcsvvldqyuoebnvskdwgzjmanvkrigctjuqghuwfonlrcglmjiskrglgcvzmbbfmywvhqtngiazkhlxqvrboyovcjuomyvbyryhjrurcxrmeelwzfznfnkhyhxnhsqhmoovhlnjigwmmjknokctxncywzldubkipydylsitaghopsbgcezbtezpfkuznghegndshllkttykncqpisqbwqsbcodqumuabzujmgvfbtqoxiffavgynaaqodiireltefcqnzwquoeswjxzhgxugzmxewhwbupfctvpstimjmgiyejfmgbugqrkmzintxsxqqnklvxxjdfjjoqvwsqekjrwkzkgxednwsykpxhzcmkmcbjyvbdflubxtndquhgszgcpflqhzavbbhayiddvzhtswwlsftnfzpwedmncolessgqcluakdtezrpuzihejycpbodecqkwiqnkbhklmfpnzhhyroebpgsqwxwangxvykiztumfcesbnazeqfhflayhxuhldggdqqfjlronxjqklrrdwyljyjswfzjkiroyitivmykmjayutpyrqlaousmbtvobncaxxkslkcxuewqznqqdbquxiibdgpfvthsqtmbdqbzwxsvmklsmbmllwmqcbvzwermxfqdvrerqwochzuvciiphvglydlviejfkzkhspzinplwphrrscjzzpkzsxuacgimhsvajwixmvtbfnuxpdyosinngknkqxyzqtzzqkwcfsfmfzjufhkxsxmajakbxkbpdcrvmpxuhegoesklzuktuxtdxffwpiovkfqzwqtgubuatflvbwovtkbspiohqecclsbiiwodyifrprorndognpyexolnvnsgitoqsevqpbjeuyxhdmpqbpuzfioavghdcxwuetmtwxzqvxxajoqmdnljvcjsksyoqukwpwpmuocrncswqkjuzwduyoplhnzzjkvyimmccpwqtqojcxdcwiywznhgahhltembqygunckguzswyxyugzrhmanpjwaccmeunkocozjkadgyahvigmpuoejyamaeagynsnqiobsqeboeijpqxekmjwmlimqkvoyndqwtrqjxrydbjsrjxexiajtkxjusauidkgxdwgcqgirsajbzpyhpjdzgovboqlucrtgpbkfisxaviduxbemmjnuxixoaiidzblpcvwmqtmlbrhaulogwbbumaiojsrktebxcyksrydvskvmymtupwqdwyshqlslqtlsrxklxpipzfczrxwhtmkrnczfortgmeygbkryzzqagsaypyfkkrdzwcfkvysieqqfahfzixbqrdoibvsonwnjauvwsngyfwdfflgpihfalwiordcebqpbmkzhzffedlliwrzgbhiodrcuwfbxxqajdascfyjjjutfovnpcwffphzloaknrkzwkraspjdygwfurvltrqyizatoafbtpapcuqujbzkhvrdpmqkfglngliwnpzhfcyjwqffserbbjfqxxdfeoumbdkosjxzlwkhoppgtgkxrrnroswuiqijvsaqexrpvczktqspycmqvbjkuhecznoxhonllgdjicvesntqsfinitzcaehctvzaloowurhdtchnscfmjozrjbwkhdoaanclbqjvjracvjrpnxbwndowekilpwlplpcoinmrgmwcynrlmtibwyaaiinzhaqupsjxfvpmjnqjajxpefqafqcnbrokreehwngdkxrwkjfntcczjuaoledoyrypxeyvufzecsapeiczkgywaaknfqbfjzuhgtcaydecyhtcpxbbdzxgakthavmlwgscbxczajeasxchxfkafighsaouxcfviaykzclbnrcfreookzqrjxybzmxyqetygtkfyyhlpjtiphwzipsroqupernuyivechkafrnxcghrqgryvlsjcmyiendevaizhsbplpmnxtfpbyqwjihnufwaubeqbnpihtxtilghhnxilnhxvokrwgsqhwiklyvtrsmbgukpihzvwgstvdlrjfjtbsfpsailvwmbqysairweyoavrnodsxyvrdkjeuubqiehhwhrndxnsxklkyoeymcpsmbyhxzxgdgpzqgovdepdobppruneweykfwtdvtlyzczripnntcsidixxemdasbmaxgfubquastuckhrelkorsnoybratsmtruylvqfykasokbiqmsgdmtrkwikqcylttmeqelqpnkfalnsyhzdvuvfrfrjiqwbliidnvuejejlqzdqgctcxqjqwdbgvxjggtlvpfbammlelquutpysijkgmlmhixpqrbniiokudvyqlukgmkauukcgvqqgnfwgimoraylfvmwjigputqqncuvaklgyteebqvsttgyrsuacdrxwhpjmkgmwsswlwbopyhitpgkjdklgckclanxmvlknfhaeirtiybbpjmymyjuauyownnkjzuyebzilsdtqexhqjfejozmkxainnuwopuvsvdgnhzfnymfjrktijfuiffxuztjoqejljpbsqgipvnyoxrwiyttdyqaxklocpznaoovexdvusuiqnfamatrymrxycnatmpwagrhdwcxowfemmhyocpoojmolzohmedmkuvpqyoeuvliiugywynnjzbwaqdmwgbgcwxuulbrgmzqiipeicrrdvndltsjztreroyozzhhngyvqpaatttwojbbctkbobhipgwqfdatlyqepqeimygahhejdomtqubhcxahcacoatsplyzzwfcfogqtufnmbowdowmqypeykynxxoxttzbjfrdbyqozemcuiplggrgsqvpnjoicpjludoybgobnafhkussamysctkpcazgqlrmoikroyubaalkufcxrarttdkxemduxxgnftevtzzuzkgvpjshlmchppetbinyhwlmprjisierulpkwjthvxolovepfupmscahujrouygolxahgvoctfphcvzhpovhtvqnekdakjpvipcplsrgzinbdoitdfcyavlhhqhrawmkoqzbishiawlelwzqbdocybjhrsytrmehefinyrxanieaefsgjikjjvrdnrygeenikvxkmukcfrejwxrhhbrrwiwmzwsfbiqndnxskjwpacazzuihvzauhygjuaoxiekouhierevtgxuirdrpetjmtrvntllcsxsmfasygptomqwgtpvxkgldxitqweuobmcimuiulcjzshfnzcmjianfkgygmsupfyytuiofkzhegmtfripziehskfbyzvngwnaqwqwgpdnxrgdkoukyyttkauizrxgnliehycpphemanplblxclqvvsrfridzwqhczfeounvjniyylmvnxnkvfbazkfyixgmwnsnkfitsqttbhopfebajdckpkjxhrlegdrsibncrhvsqmjvqlxbqcmfmvevopbusuncvaemjahamsxedvfvxraloggcvizpemtaoyoaavcoozstwxptrrghynkvxlnbxgdngspinrwdtozpoqlhhrofgadflqhkykyfepnshkxcawtqmexwymdgmgmyyvixhicmyrghbdlecewqlumjiqrwwdwvahskrwwpneyrjlvlqtcbnomxedudjqylzsvrotibxjshupskjwnylbwrhsuskudkmzbaujessyzpqtzzqpzdsxwzprralxmyynasygktflivfjwxavcttyskrmkdjwdhlfzmmmfrxpoeykdzffuagmlxaxcjwhqhnstbeasdpgiqsnrxvlabzlaaidjgkcozujaakudbcnrubiuytuzvhbvreeyrioqcxtejnilqxekyneyiazdhbyzhfzynnrskmwwzehpcnccagbljcqqdrrlwqllfedsqcoqixhmkjpnssybuqtahllfdyrjdwbvyrnhovtygklaelbbcvdliqrnokytgbphseqgblhiyuiegmhkjkvramgrwfkfgbobggjhrqrpncwtodfxoaqqpwhndusmvindowngpgxcivxyzcvpenftkphpjbbgkomanziehiiqdfocjmrnzzhloobxreddvpekqmoyuwqybgcmsuzuvghjwkgoplrmdxwtkktgilglsjxmfwiadqlbarrrejdhgbdslwbtihgcqdoxaxzfpkirdajijorrqumdtnzduffnjvmlumsjwadaamklcbfuyrytgzycefppaecxqwimxlozzanceuyenwxtvvvcantcrwzhsivrtmttoixscvwputkrqftsupvlnuuheszaguvucsnepltjhdbxkwqhnluahriojjqrdlnffajuoegvjjsvfkcvxxvhodvzolqrydwdghxuldrwmjsuwuptrmahsxggkwyiculkclgpoxmdiezeaqascazuaibkgzvivguztarkzdnijoojeodgrfbtnedtfaechycabtmayzkhpbstrwaibdbojhjohnsqoutppfrphcywdrqwnsfoxiegrnwbdotrlmgcuxsdbqzghrdulkkiuqqoempjdkaozgqtfiyjbzrirufegrpbifrouixxsxnrfdmmokxpkpzdjejsyyvggmwauvzoumgzrunhflwrcywncffceakmogoibnzvbhwubmcizyzbwvobdwfjwqvlyjvaywxbdwvfixnjypkwmazpwfavvvitgfvecvffrbmbaakyrwxdywvvdvqatauzfrehmlpfobgpdspuooanlvkoscplclljerttciyrlhipzzlqolfbippdrgovtbgysztxjrqvyamyzzonntovkgadopaksgkngdkrucovcatdcjtpejmqpntgcvsubcjjftbbxpwzsrfucyjihvyzithmozjqbsbioclluinobukestmbqnabgjpvianlvotjtfwppldshnwzildaafdgpvdxvslqhebclatcnwvvltwvgzvgfbpuwstrlekulnyynrdbpywctuxauugzvvvtqqnbxxxkkviaolbdlzhftefpanoyjlrzcecbdblowvomjrrusybncidewtchfwijakmgnxwjsmxdfoaxmbkqkpugjxnqvvbxhndukzfencisskpelcwvrqdcpqskntiwgnssnxzcbjvvtkbeoxcgofifydqnwakfnczccyffhadcmehbcjjqqtvoyncvnocdsbhwcmclofodjxgnipdykbpjqrgghijzlqwpfabthrysoegwtlugmlpdevjlfunsmjirllnsclkqqsmleirqugvydgmiczdpkngvyvnahzszugkyqcqkkolouxyxydoiqwdqsssbtkansmljoevbsirofrlfvbnnyqeqkyrejttdfhmwlixfqqtyhffegtuelywwwpchivloezkpjjucboakvfpdajkmjikhyorglewqdihntepuurfrywwbfipqrzzppsxawuqccarweisazxrztnaawpspqsefkdijxylcacagksglqzmlppmngeczyiellmulnfppjpwvheizewlvnxiudmtcwznhohudojszyowivghbcswqrgmvsfxogmxutuxjdqkeggmbmvcqoagfgscyzzoglocgiarlsbryhdcvbgmphuwnpzoqaeyosbwadoovpfvzlzwmgkdpfbewasmdxnrronsrdiotnlualsotdriktmgvhuyghrxxfpqixednszpdseyxxghqupodjmjtghqlyhvynbnfcckugiteqawlpmobbawhhouhvrflsssxxdtdnnjenirfuhivheeqsyyqqfdkavhxybaoomsdwjiqqqoawsqprqhjwktwfbtgevusjmvfxgwpympvegkxwtiudcqiflqjjkbsyyyveundicaokmmjfudiywsdhjoqtrluqjpwfcbnhbbrnpasmgckcvqleeezyptoylmdposwrtqflszxyejczbofiuegjawjliocygbqbrxmlwlkaxmbycgiwnohpkqcveetkfjmygakfmwhjkjjnrsjggrvtjzooehaltomzpygbuokezvtsnfbuyyphydxcmypfmbhdznevdqccmvxzcyvawkkuphyeholursxrieolwqqiopkqnhtludijkgpswboovnhdrvroojyeykunsnymsgxdnnxlolzlygpseiisriwptseliczajwfylclbplnoteixmsrdaputkagucfwnecnrgztwmgmfmufymwdooziyzosbworcsetzfmvrujxgfcgrajfokpojginezggrkbssopjkygtdkzwllebvvqfbauhzanrmpjefbopzwifhkbuvhxqdnxeormntbggssmxtapuzzzgjbcgkzxnpuufoisjjbpevcboesidskldlnkxpjleffjvndfyewczagpfyakskyyxlkuctfgamrwbvbabzedakuhlurfepdddpyfkfolsfhvakvejchjmbqzykhebmiecimmpaxjhgfkqcocexltaesyqzdrmnwqxkcmmcytidfbftdksqftuxtkzycnladkmfxyjvtsazthuwtnmygjwzvxazklbpynwvlgshyugtbarvpftxqrzmozlaknricofquzxuvfpxhguxnpuehtqxsdafsrpvfxlbyixunmtwsptieqmllelghlnaecxmbincslyhztduizcmpoaaunibyvdeaaqbkgfctdjswpmeootfetfygocglfklidpujuzcykwwzjdpzwuepccxsgfdorpwuesfarjycgxdpatownunpzmhfrnticrfikmidwjfeglvgejetgrlffukwhgglxnthaelwpscvdxdjghdoaqtfkgdsmjdjxztxfgvuaxrguurcdwtfmmmxuppiausxvnwavtwxsndohvtmrmreaddvusdvoxskmrmhmjffwktghmubymhzrkwmezhszbelutthcamamhiuwdvwrexkotwsvfmujtfdmamqcaqpiblegiaxmyjatguoniiybqenqqdtrwkbzvffdhephnrbvlpjvajxqnmggsbzonapihulqycyltbhgsscsdbitnrzfuavickslgebgibscebkxvorypcrfvsmpukzwwshdyljiodmvgxsbnjcfntkioyaiizhhtqmtbxesdqnjphldsnigiuoysqhdsomneyhixlcadrktyxwsmeywapvshexzujnyhwihjmhmdtfbqbwehubhlyctczswlhwsdhbmtjzxceaviwzrtcbeadzceuapbzkfglrzusyzzzuhvvaqybxygwdnsfcueobkaumwnkvmlrvjnvzlzwlfvqsnwxhclknlfbkslhgzzbioyhbxgmdbfxqbuadiksvptfjhnpgosqcrzfofrtiytzbpzhxjztbvykencmdwcfpdkejezteusgkrnjdmegjdqpsubwtfdijqouwqnakigwbtieimktrdjkzkissldddrxzhmcuuiqycyemlpfcuoovbwqvktynkxpujrogbgefhirszxwvnntqjjlvejlogakstaauheaootiqqgngtgwfpiattlkcnuuixudeafuqyeirkwzecvjsflnzzxbmuxxrtaxvhtnhxwyhsxnuvyoakcbjbexxduqwwptdihnfdlnvlcribgknkmwnsvlwqduotoubvhjyuyvfampduaqgoougkheckbqfpmhmtsmoqhvqvluslmybbuhlbopophtgucsnuedmnzdkmzczldamxjzbzopoykwspohjulwhmbsycwxnzzcxkdykupyszvkolrageypqbjqdgnsfouezaejutjotgyhurjcwhroidfljugqjcvkcbvyvgmonmxjejlcqesbhxoquujfxgpzhmacsmshzqqfzbjribspbhivdraaxrcgdcjvqbdptvwuzhohjgwmpejvxlvuczbfdmcuphljdifcptjachotmxojimgkljrrvwlmdwxggdiyytycdpypmoaxaijotbfoxsdaqajxftxenxiqmihbxjyzuhnniuehmhkfhtcnmbtbydtlmcmjokiiryvcubljixoqssfhtairdeeriqawclgwleaffnqolegdwnkpnvjsbnkjgmiizqgkugsawiqhpykllxdauouvbegjroushafbjxdfhvmwtykgxnirddmehvqalkvsmawvzzagxofsppbmejghounypdabnvvavrxutydsoikdxlsbxailuvlmbmnqhkqfcqzxqcpekbkhpekmeuzyqnmtzbjgmalqsnnipemchylwcwkiqmjizvbyxlgjnsbfhqpwubabcmagkkpcyanxvqvudedqygxfwmdsqewuswaeirghrvepmcjpgpievhlswokougimeiqnczutuaigesykhgqenbwjsjxmrfzwqrlssfnnhwvmqtyxbrbpsaymjyzafayvtjpmqdbyzympskovbrkwrlkcjjfolfmfpbrfeuffstwmekihesfvfqaobhfnmrsbsljzfqezkbmqnwzrqfpwxpmyyrlkkfodqxexpixzjjrzhrqxyfvvwivubhfmmtrfsvchdzjpkaottgrrmmoiwkdytkkhdbmxwrgbqpsdsgwnnpfbcvjyfoyaaiifjwchnjjgabjskhizznadkldsccsckaiwxmdmvoswsdqcnydmnfpyqssuukzhnhdmtiyrxrppwftjgwvvbmvtfnobynilwnxrwxitqxknfdwnzuicgevnrlprazcwnbojbyeryoedrdqsxaufertydkpfjucyxshyslqqbwvsznmyogeuazsfjikucmvbgeazqsrtonmoaymqhslhrdmcnwzrluttytuqnmkbzyrzygqfydosgcuubvlmpehqzcioouqnujotpgaqjpisrcmsazgrgohhsswgoovjxlcpuefdtnxnrlhrlojtesaymwebmbmkmqziugoqtldmxkgkardglfeyckbqlpaafbibhwrckuphhxthowozbsxfdawqpyswabyjzpojkdstalaqjqpljpsopqghpzdrcbcdrqpdvhwvvylepdraskqhcwqrxukzxndqdsvzbtwwskdjnulfgbojzfcxbnwyazrftmevgdtanwjctuhzxxkwentfcsgetxjnlqkrvdriacktytuvshsxwneyfmczlnmxfyqabkkbmwwlptgkaajyopwvcauppiovgfxipilbekowfvghxrflgdjiddauoorutpekxsodeczjzhqxyxkfatpfxpznethziokhgykmckvcrnzsmlxmfmtecsrinahvotcwtpkwzfasuzkgmhurwnvaioivprrfnvegkoelwegsfqzsatlyhhypxikobdwobaxlqdzokfhpxgjzzvxievzbfmqhpmxsouggafvkmczjiljjgguketnjffsxtqphmxomcjyqnaavllgrtadxwqjdydartszippwuavknxmkixqbrteqpwbsufvoappkgehagenvbbfjbxmsqqkhomcltzztooeuunxarmvuncfozrisjvzxhdiacjvqxzntomvfysbnugewxoogjpgstosvaqozrdtgpqnpjpefvkbvzpbkfjdyyxvanlzfoulflauupbbnldimslxzvcxvzvvngwuvrhachxyxazkleuaxfzwcbwxctupzrlwksxoawozwnljuhlrmkprnnoqglhieteonbcpzqoxfbhzissxfkoszeigpsovlsjawlqxdgvjickswdpdpfwokgubykxlfiodueaxkviuztnyyebnkzbwiyziiwwpfksbovaaslvuekscqvraxsyfqynbwlsdvlelghmasjuwvoalktohlmoekdhhpzkkmrwuykvqvpoqeqbqjpzwcncoqjxhlkuxbssbcprxehnkxqaigzekohahsyodkidlaxmemfdsosdljfvrfzdvidcbdtenbhaegoksvoljgqwrvrictlybracwtjmxsodzdnradekmuidscmuvzuwsxzpsibpcrubukvxnzhbwrfottkosfllcmcuwfmywgarsvqiwzjsqoikhtacalnczpyvukwqtwjltztrydmrtudbpddrrstqjuofekxxihlbigujjgcfbfisinkzffqqzgvgqsfybqbmxvvicznfevzmkloprghsmcycvecywdktaheegtjkrlxbwltpraifdghamlyjqleqskvzegjvtiwqvpfosptxyczkpxvfiaydkvcqbnlvwkntbuitkdqiipuxpqdzsbcewevzugwqsjjugtnqzgyhxseokjockshizeitmglfsuzyqoopwoyuicktmyxytghskrnxzycrudpkrfbovbkeeuzsvwsodbipuvmuvqlptdcnifdghnknjlhgjzwmdeexbqrwhfockqpuvuachmrtxyzhyyvbczagsuebbsdlkvfipgmzuessctbcrydwygeqaitbdwmnrtvjbkiwocqqvrtppynperhgzahexvuqprzsbjoplegjvbynzlcgcmmdmwklreeivfbzppoawwjtzjgzulhyxtelpoxjsdtpiuozmfaiblbdvkcrvjqueufwiefcharsvirjnttqfwtcgxtdijxpfdlpjsrrsllpvcakwtcdsyfyxaegmbxrgkhqcahwnuokwquumzzexeeuficrkzkqjtgnwpmljcwisqeepeioinjpvnihpvayoiomycyjeutymydeensocdeyrohgjhgaqwmqykbegxdtxgzbvrpyxvyokfwvnexfbkxbwygfvjasrztmqlkpgifqqbofilklxhmwhubfdbzwcvlugvkmgpfwttdbgpmgwacfvfkagkzkfkenfozneelzzoaidoobqqjlxtoffnpmxtzeizsoffnhunrzdfdtxmhuhxznkobjhbnrchfjnssjhjimqninbpjjmkyyazhkcubgcjqrdkanukzkyhazylkmwikwkrgmokvstljwvjjrvjetsevrphdnqqkchsqxclpvwyuujdlforajvxanffcvvmzfhpcjbshxsbtcehhvyjewhglmlnjjciqjxusyozyhatlhauczdivkogybvhkxrcgpxbplcpufjfmjzqkfyhpmpqnlxaipcrgbsxnunjzypnhqduvdbnprdfecfmtjfblzzawojvmalvtqrqutdlmiqqlvegfkisjjshxbdleqgxgyncttofdexqcdxjnhbzqtgnwlycdgzmgpobkcinjfgljduykxzaqipovprbvsnehbftgojrvpcgsyhevggnqvboksphncnuftqljzceogniguodcuqtrujdoymbaayhvovenzkdgzlotbswxkroeatvafwchwjckeizevcmupaqqumnffdlhwgjeflnbusmbclhxqtufvkyqynxfrimppvvuilxedjffxofpjzwfjuxjkuodecsvxqrfbpjgarnxhsorrlychvnrrbuxdptchkjkzvqaodzvyqmpbkukhhcalyyoprtydcifxafleghcwvtzgpkpzoqkoqaehtcmynrxnwlrfuvdprfxgpnhffrtvszcncqlpijegwylqtrfmwfvnscsxunuonydropiwamdwkooghtsfypsssmroclcpjhiywsuyexwrokxxgrdghxbbmspogtfpanpzfmzllwcbibzovexebyfnswbypiorxfqcrqqdouewyfntjmxnkcacaebporimwzqqrgvhdbfbhsusetfqxitxowocpwonxfdonyxqadtjxpbcijuhrpehnsjcraqofvfuwcnwfghdwkhgtbomtutgpdjgdbukwbgsetqjczoakxidzwsutlaiqlqtmmfloofshwdlvslhpulqsdmnqnxhancpakaqyhiffntdhekxaxpvccktriatveixoknmsxuxlnautltwkjrnnbiglizhokgduwpmvjmiwmdszwomhhzwkhcwreffoniijaxhljpkxsczrqlyzmvjtxhmdmrracclezspkizjcbittvushpeaolmsaglaknnomifyqhiuksqeokyvzewpqkhqstseihuxpavhhfcryuxmeyhfcfwusemkzgrjolgdlclujpohjntplduqiqkcwznxfehtrpxdtkzdznkidgxoltgtltwksljzrstiisndcngdctnfbmwmhefsxderqorhikssefwiuclzxagymektjsngxuyokpxsbplhxjprlcslczjqdwlmoyvluvbekzxnqkuntlhgnzrradjezuspupapketwdsjvefjzqsgftzyuchaddssykvpuzkrefczjkkjcerkysisxquavmttyqktpvmvaqwlajwfnnplgxfdpzfloygmqrhfmtcokzzmiwrxlglrgswvsciicpwtieauzfklejyohrltaymgrjdlnotrdrmlvphxuriaciokcbdcarkrapcpdxhgottmkcletteyeukgwpkygurjjvypshvidruntfspbyphtznrozpuvidgdevvltfqkagfhywkylrrkgiyfzvchyjkeehsxnijpopyollnzegjlpksqfvadgkikuakjktyzkypwjcxolgbaqvsvssbyfljzntkzmkaxiqvuuskfalhtiaqbutzzeiatdktswtrflobgilliytdgvujrnjruttnxgircckjnykvgxhqvcuezonqbyjvsdlrltipxsznsliyjpqgtkmuaaleaubszmjmitcfnbsfobfxswgoisijuvpvqqprfzepuiwilpnzstzmpfczzliwgztysisxjyoegkorbbstwrinyknyxxwexpepzyolwcpemfvqgojfhtsggfeddcjcrotsohdllupzznpidbnkwfqcnwgsyrurfnkuwnhjiykhlapodcbzkrcxqkpobjgjysfwfmhnkrxytmowrpybjcyzmnnpyvyynfuttbigufeousmlkkmtmksazhqzdqxsyrnruyxqddexyriibwdqivqjegijzfcxyroupolttpstgddkzgaibqtgzcpyhtxglnaablvxdoqlruvsaxceowmpydpfzmnkzqmddiagyssrmnlplotbkhqkkfrjtfvsxkvauxhsqzleowtbkocwvuiibaruhppezmetknncoygdwcejqjfmdquhqndxcsddfjyrzlvnldqifxrwgykloeuetydyqibxuaotnlgzpzhuwaseycrpgmvckzacjbaddqyrcfqtgzncbgizkzornktzlvcwweccwavahltnmjrnrzobbfuornqeksajyelufgvotochxvuoxkqypkogaekkyntwrgryomnhcroaokqlvilmtatvgfhhuddojuoyibhluwnlifwfoishknesulzpodfaztykxhxtfmeorbtflloxyspverkuwruvkuhhcedjqhxkitpveesfwiccboxxftupqczuicdblblwuaragkhdujhvxgscamuqqvkzznckqngmpanwmwbgiidfpxkrlghedtfcihozwxnlxcjjzkrhvwokvadggmmlfrnpyeoudeicgrnfeeozcflxhofqasbhjrwkvngarocxipfmvvnxcawkqdasxbzjuxtrkbrdqtdykimffbekbcurgwjfzkouhjluldxbyabjkposqvcrxwkfuvltfznmlzgwegzcgcntrclhxpawlmchhnamooazpjvvpuzxwnsprdyosymfkomrbpfqfgrwpnrejqdifwxwbwmuwzbhnqaphihmoyidnbjzrohxketmcgizfliudklksuwgtglihwhbloyvvtjxbtyifgdirqvjltsgadohayiloiijesjhlhssqxdhnysytketqvdqtndhyltcfejtndldaqmoqfmkrtdhoizjwequqlyxpjqfneinuxbleqvvjafzjgihlabdmzadktixxtltqoeuebrpeyophwwszzxlaxqksbipwvjojpkqtbfxcumjskdrjaelzxvuyfjwaukmkucdxhczdqkyapplrwwtttntbypdtzpftrksbmseeeceuurnspglitglsxpnwesrwcjxqddukgaecrrugahqtncnktaixubvwmuoqykagkgidzclzlcvygkkzluyfvuehouzofburjcvjggkrkhrqsdrfxsmumsecndrsfllqqgbynmsxvlqnpjsobhlpzpsmslgzwsqoyrnnfhndsdqjolaqulhtazgukoujimzhfownvjqqmizdvbtpmajocrjohfxqtxzanvnjffztqawnerfquqhwhqmqxkgsjfswbihipibfjyvnraflqojzgwpdiwvtdqreyzonhqvsqqhhqpvidghkcynmulfabxgzwvofljydbfmtwhbaoiksjzffkkfnhmofhdluolzqsljpvlpjvbdeaiknvraedpkmsxiikhycvkrqxibujezkplxkezoioawizhrjqrmlbfdaoevzlwjahimnicxyduhedaotrchvhkvxrqxcmievtqjbsgzxcichiqbcskzbuuznnypbgvzzliceqebvmfcxvfzppyrmuszhqrjymiaascgkghhvnkskptahrttnnuvflgdaqdakigwqpeleodrfyrmpmhfzccepkjpcprmxzknrobvegmmffywtgbukdyznkppzidyiqnzeowlvlxasxwmfygpdhptrgisooqyedzkswvpjupbslnxlwnpdckwiuqyigxolyzfuwufrlhlzwiksvbvzltjgjiqrhhffgmcejhxuqitnrgfojdqvyzdkjvbsciakvcpelmycxurqayejbwgcylmhhpcjycutglxrxxvwtmmtntfoplspzexgubpsdlnjwmcwlysfmhyzafqrtihspgylhpfkbsvzgxvrxqwmxxlxcdoxoxpddpqkfvwinthttdmqtjesispmvfkldzqgrgtwbreqlnjesywlhbdptudqcncfpsngbgccaitlverocnblqhcjdgcftemiyaoziseicxvadmqgctrscwegcgnvmagkcwmkrqvtpqvsflahhszjqdrsjhpxhvkluwxlosgdlcxqpyxwsyiiczangplpnxahtaujsaofelxulusdvanzcnmkfzcsyebrevczxsecgcyryqxttksxyhjfilzhsaawxfvwxaydewxcpuykesjebsjmgktymvwjasgaugfyfccwwagdvrhwjtirddrqzobbgjdcqjavlerkhzncquybpjttmkllampgwskmbmjprczybmatiauyjprwcejcqalovvawilnbnmzqabjjbghlxajzeuewvrlqmnowbcxoqyhrzqcbylowlzvrtbpqscghqxdtzdhrboidhmgfxudkvzdocthvvncnkhglmqdogntpazxzvakqbngeinzgddvnmcbzugpctisvhpvcphsruaxdoylrbggpkjoepdjjlbgssdwhkqahrxmhvxaydgoqscuhifstgzoyemirdiyuimbdlbeojsgbwblegdapeqpjcxcfiazqocvzfzprrwsqzfuhjwwnmmyplautcwjqafrgghxadrqcxmjijixvuwmhyhbdfmvftnnaumazgsbfgmnpuknqizyjizmplhxfukhqzjpfbskjfxzzjosspqbarppeyolvbslabdfevpoaopcxfurumuwdjojkcvfdnqdrtkxrbntpevpjrggfdckoquytpdyeucxpoxiqkcfrzuardgygjjubdtuuytauxyklvsbhwjywfznihnqswnubgqvelwziwlwrfzacizyoanyhapczcyakoyqcwrmueinkinosmoiodhyfyaeayhmoheqdwdxankmxzcsmsacvjabhahjaqxrxcjayozaclpblttzmwbjfckneuyhkwyiyhkywmyaxpbkjvjvrneyprlvnmcxiahgkseggegduhodezaxmewdxjbkvdkdgrlqlpwruexqnntsmtrbzpgxbjjyntevnplahjsdnrtaemushmvgdcfelrdamhvzyiafruiendhrpbjypfftjmzwdezqnwrmfttwpedchmtzpfxfmeaxrbvwguotrihwbbutfzvmlrnbwlikpkjilehrzvnsfuqjvmccmxrzqecopkyhwhpiowbpuohblxkijmlxmexeiztamthidyqrvugrkpsvmfexrrxfvukdvddhymsblksieazpptmdyqdshazcwjironbnavqnfjvmlwbbpneqghiphjsftczjpcyrxpecazolvrmnhrdnanckwwpryodqqbjnqosspielcqajjdykscwqzlkzcnruqbijpcbemfroyrzhwxtqcphddtcfgwzyzjhqnjuaqoqohqyeprwrjfuoyztgvufwgomvxivqwoegecmvwinhojvmiqvfkabsonoowleuzkafpmbnxqulzpzkfurruvkliabglhdnrfuisvvpnpcpgbzlwsfsnpfypgecnjrhpmrhjnyxeamldnybhduvubtlkzqxjhgatyjdnhatjogmdfomkkhhncuxckxnsdnldwqsfsmqgselzuymehtsmproaodsssfxqwqxocnxwrkwmkksufkgucwuszoktukjsagievhcmsayekrypdemhcgadxmwtwhlypvekdjxxjxdwocnuizblidofqnyowlnmgtylcxiovmklqxfaxjnfmcmerktnvizdbapgxzusnegddleawrhlvtpofkcasdgwdvcduyvkebnhptknntykpqzuadxtidroyvrutoezsugjpswemxpbnqxpvukdxbrxwkurzyqruzakryqdvfgkxeasykyyorvmcpedqwwmvwynwpssubeuheohrmfhedtiuznkwjzthxidmbvjecflitsbgaimdewxdjkbsjvzidcvgagkelxmesomswyvcwsgfehzneawiftofbbjmyxsqlfikffmcaujeejuijsqonhvoigaieuzpzjbfrohrkqvobrxljpbywzmefonjjobizcpqsyjdphtsyqfkihiegxlnlaxwqafpewixvrcmyvezokjwttczqnmcplyflfxqefpzytlzopyebjcucfvtukxmvxrjfqbjvsiwltvbkvgmekigpebmoyylwrubixkfpubapycztknxuygmchasgaderyhzxthnnkpcjtdedtearkovronzqqfhqmjofqdsaswlmaqrtutmghhzxubpmgxdvgtaywvaqrmctewmypagzhfydiospnxkvqmtohtebfjulfjjulrvtzwyhkqwxnisfvntegahojezonlkyegtdeulomjncshhpudvcddsmhucwhmwlusrymcoaldcfkkwgpslsvdgeziciwywrqnrergfejhkwtgciancvtibusddpcziumbzfjgjvnvrjcvrclwaggdazhdqleafsdgxnohiccbngwelhcnpjqqkzvcmaesshaqupzctdfgkcmxlpmbaetkeooqoskjznbueayzdefpitovkawkhrwlgkdwcqfedgyjiyioepfmiuivwgfdyeqequavmizpqzyzxincxsofoyalfiqdohjedlzptyekzelxoablursjqtwvdmartjxwuxlqxrcxxnwlpxwjopsekobpdbiwmfvjeibccdbzsxnhwlqkclrozsdcwlbktwtxgubizeirdxyzirklcbzdvfgrsjclhxlnumwtrgnudnwnegygjamefuwxdajkzwxtivnbbayepusucsclsopzinxlpixpevioqxquzqapkulmnnzheohmegatqxiuutknschushleewsrlrapgjekpgfffqbalmpdowzjfknjggcsxdhfonretxlzpnbabqfrofavpohwwjqvozbdnujgtuhijiimyoeotgkkwrgscyotyrylnhthdbpplabmqhajokvkdeficvbaobnbijinmaexcijspmdfwysccxmfsnhioxjygunuwsryadqrpvxntpryoddismqijzirvwlrqfrqkjjzegoaxgylldtepjmblvuiticjntbzdkajtojqwhcimxmejafuvbcojezyaaxflxoeoyukhyswygahhfdeghtsosqovqtgnjkbjgothgpfnpouuracytumoncjdxrshduqlijsqyzfnnbjecofxktlxywcvpvcsrvebgyrfyxulagscuzaeunesmgobdswuncvuapbaoxooditzhtgbikuuskywcwsqyomafpbtmpjsxcrgxzjgjdcepacsyukwkwzrnakzrwtqylwwkfcwlecaheufaeencfepfrrynkolxmjirlgesihbobiqiofktkaabtyodobballuytasdhttlejtspdobfoeypuedvhgiuahnabqkxlvbuovltjozaszpmfukoasiabxddpnaaidxyyirizsiruzebtoquiywkyzpnopowslwsvqcuikmelsgfiwbqwxcuzriniymcjixeyzzjmnnxftcprfhodkkcduposbsbmpikpjtbwjadcfuukbkpnwrtrjhjmbnytxligglepnjusdqdecsuymnpxwbjxqkqnxavdnbvbkrtphyvjrpjfcbcyclwsxohupcaedxfpobexzzisprxtszswnqqmezjvdjaiihxykbvmdfeuwenrffpsvihchqxlbztjjthvwehzjmthqhyvoqqkyxofpnsqpobyhivdmddxpyehfdeuvwgtvuakhtcyhdyolutzxwwovdvcibydyewtayektgiamhqzbkzaixofclikminkplgpkmbdukggtwgwihwtgkyjxitjrhwgscxcozgilopgkxcveaqkmqtjznssuqamszriaztlnktzhvulvmvjdrseenyydamyzfgcezprjjrsgndeabnfsgjyvjyyjkwkqfmwgcosnhktwqxyrldltojoxhoitltlshiyjyqyfbsdtrckcfxxwgntpceipxuheehqcyojbimuaqgeydeulrpattikqerxibpmzdvcokyoaeuyjyuuqkulfewwjpvtqkgrpdtfptvqxqqgolcgxwucbqamfxfyhnpztleieayesrmieppbnlfqbicifkipzyoslubrwbvoskxyzobmzhdcphkwttlttblgwozsqvjkhgjgwkuxrzzihlfagqvbuhhqxhknetylbhtzrnwlnvvwhawptsrnbafyhronlmaovtoojkzwivurxqgskzfnamojecxrqzfltvzeibasyujvmrozrvmticcoebfwbmhwkjasmckkyvhkmvlftaydcdrjeyfhkpdwydomfawdvdqkbbrubsxevqnboujajmxjqnvxordlxtxyupuyjteejtfxnqxxdwivkeuyxxluqjunaoljjppskwpxnpwqnfnxjhxgfjdxuxkuoxfxouqyqhwfysxmzkliujrzsuyotjihinhxjzxzivhjoevnpgyjrpkljizetjmcwvbhuyjeuqnkukzltxwkogwzrxemhsofnrnjqwnmmdyinxugxqrnzpnsotevlvdamqjeoyscbqkpqgvnxoayfpswiwrvruvgmemaffcgxpxnvvaujdursmndjgycdukpaxsfeuobllphgrdsuylpkyuvguyjqkbrhtynpmpbagigznyuljlwjhrsbrbsccllvjzlyqoyywgxlhtxqgvcnourgxjkawvxhwkfmyfmwokivdhnkcpzqlqwjzgjveifavkptsbsgncdtjqdlgebrvacwinumwzprlgmdjglxiqqrhvepzgiuilrqylfumwnxccdniqkthcmoxrdxzlthwmzvzgrlchjgjoutegoljbbgsefrszaqlznlghwkohrzvnuatjncmplgvqznpzgdatqgcotqwkrtzmvsxjxlyirvyqlqprvrmlypzthxfzrenmymxvxxfqtegewxvtojeoydhidmjuswirkgmjdszwyfawcxpshvhhyqrddaxvygoiobrrvqqvxqtbqiuzddidtahyagwzokjcnbuktkfrcysuiotnrerlealgghhvsrzhsvsnhzekazginawfepskpdqxlaufruolpgyxifjiepstrqyxhshywawjasotmceeledvjmfpdzjvshuzacyyewhgoxbdbfhmjkjvyokbotygdthfxhdddkjujkxkrgzcsixqxitnbsvelcjikolpywjjiiubqtdidbaclupmmmtzpejztwqtuadckzdbsijvlkawgpjseetticudakzumcwxbbyvljsxtstzketoiojtsdyspxmczkzcnzxbisyektdwyzkiptbiyaaoxpqzbrdpqdqavqehjkehjqqamtikvlavaswbjzsbcicxaifksltdblepfmacxczzvvbexvibgkzgsgqntkqwvwmcjmhiogilelqdgyemkftenltgvkdtgpxrdectnmftxxwfxasjgnptajcyvfrdwqvbvzuilbbiyzpxzjghwwbzfdxakqdggahbzdlmfowfnmdpvlwuhztocqqmgwuowbqyhdmmujtjvwysivmruhgzzynhuiwtlmtsrfgloavlwvegqqanaimzgbhbjduxrppkuwwlytwwoafhtfwtnkuvlbwvgsuxjjkehgpfbnerojyjouedeyysaxngrjgbuqheyylblpsbytkmporwnqoppdxgmdhgvokvowwalrbkkoybeqxtilixogdclnjjdvgsrfufwxrilpjqhjjruuolcunpjyyoufilkcsiajsnbbfkkpchlssnohcxgdoakwnwbggdpyqszwdmnhvyznnowrncudcgcsedykgpiyikunyjuthtiwjeozhkcltedxjbiptsqdckqcwkvffdcriqqrcscvzgvijdacxqfccveyhaepfggnywecqjyrptayyabxqizueetngxclfibdgfhzofqfaseashykpxfvdajjxnveywjedqfuwybyqhbjozvgyadeaonzckdaimgodmkbklbatcljdhrcjvgmjvnlirnqfugkiqnucpwxtkixtabcvjqgeeumxhfednrneejivihpspolpppmygtemntlwapangprnzbijizbypopefcenvlzkekgecitiynvzogizhjxxzodeqkslvrvzecfgmyqklojcwtbdeacdogifqcslvqmzuthgtyrfocjkwmnamqgcyqjoscbpsprmiysbylimotymfmorewwcmfsgjvwdknpunkvoxgclzbzfnqytmwxzcphlmfrwugqksnyqvadvbfcqvcjhziibzsolxfhuozeipcakeehgjlwgoinnsyqsqgdpmnjcopkdrlqwrzeclcizmxofvexnkadynoceephxazvtzzvkcfloezvcfsprvuqijumpjwmlvieuhbglircnsxpxsgpnbhxmdmbninafxhcfkistcwiwuzuoflbiuywxkxmsnnakucremkzqijneijsqutaqsxneuczjpzulblppxrgiatmjcfbyybosydxvfwewxahaozplqxtqsibwncungkggqfbvdisaibawarkspfjhnmecraqxnyfddhygrsibwiawzkpqrayhoejfgzetzkwctxkmrunnenmezdwrduytajzhlfakpxpkubwnwkbaxngxqdqvzqoazphkvhxmgmxcptqldtmdgucyxsvxilzjdgwyptmwhjywfqiijwnrsprqjmpraodduzdlsphzzixeyhoidezyvrofffyftbkpancudkvestksaacgeceavuhcbfkwhuljkngpzbvtwfmxfofmslmpftsgaecjaptkezzgcuzzbnyjedvwohovcidcmhmmyhzeiamkxmlybjhfmeecnsxpacumliajrzqzcclzqxzsghzvndxbvxqvzmjnwmwnnoopraohzobfxbckealaxexysjlpvkciytartobuecbddeokebmdhjdaurprdtcbmcvhswdzyfcejqopucubnxbiqsztobvcdsmyopxmuzsfsiumkonyoriejhyzlvcqnoobdnpynkdhsatfqidebllnulmfgoyvcmyiapgpzlwsnrzebckfpvslietovrlzfjhhbnyabuuonturxfwocqalutbrmnrtyybzozcpwwflsacphaejqewnzyennvgnwhnkjrbfngshivxrxbdohzbixhxkddyznnlwgmdtuvdfsawqdzembhuyorltwlujafhgrkboietaerqjrwvsfrsukfnhqisspayhmcvugqxnhpwaouutlnpwudflnxfczglmiiioyfdgxcilwdejqufkzmxokfsllextoijppnxmrfxnicwayenxqkncpfoclnvaxfpnumznrxacbycbepaxcnbtsrwghrgeyxrulrheojgrklsqasnehrybiuhpnpuoolizesgudolnzfgcztggmeivapkohgcvumvygdtkrezrmftlbbsngckgrxwadawzptyvwrhnxjgjobvrdphlrttkqslwshvhatfymmnkvutmxqfcfmnwijjybuutbsddlkbywqiukdikunmbccakljimdfhgxmlkwmivespryxbumnvorvacgjqjqyrnmczjqievctmasnwecsvupvyygpjwpblbdlmwvqrxlfafuwmyqhuictgjkpfbsteyeruzjnyvsacoxompvjzychjozoprhtpnmlmwyntsnvimkxehgdhklqymhpdqkvexgsvhfobvjkhvhlfnpmzcdxokotcyqgxzbsgigjdsvlpewutnjsrmrhyfdagtjfskyasgmwahjqmhmfxvaojryjxbmfwvkphirbrdzrnijzqygnvgdfydqxeccdzmuivlkorjfeurfkplzkuersiiromtroflwczgqqsyumpzyemzogdianqreemvsujrwkxparnwfexdileysiguxyobfazxukotuokwhzsndrsncxlwhuteuhgzjhzgrpzlflhnkbcnuuayrnxajqfeznfkugqlwrwbcfqogsbcndagrxhapogobobxutislgwplmetfhvulxaevjgkxpbadcxzvfuinqaacgtjnqfaoielrlvrayrfapgywaikqimvoigeymgoulblyzcaxwhalnajoitgribhqinouufqhgbfhgffcmfpbhyppuwiczqshfidzptgyxnffskjgrykywzqsxyuxqogwxmvohemgrslmeuuyuewtgoldnicdwlzvxxmgbynusagxxdvrhtkniytvgvaegacfzomcobwhbfeaibsruchccazixapectldhzfvadcekieujyoedvifvkqsmfyoykthlaeofjsqrymbxahavrhsbxnneldexwnjwsxcbruurivqjesgwabxzonebqkbnymlqlptejnvslmdrelngongcckgswjhuvtmgifmuvhoivrzaepitcoxgafyxogrjberlldfgnvqioqktftojhlpirphfxyryekmsotadprldhyworkxayokpdrwbspbyanqqjrtyigbpeafxkcakqmbveyyypawppnhyeawojafpakpmmpxebybhyahocefrtnsxpqnaqssqchfqycubcxldxsrzwytpnnofdbhutzmxgrngvjiahtdgnwikubbkosptvynrycogvcntnxtwswdfslrmovyenpxionzmpoumshfhgyeauhgmuzbiuzojplsfyttpjncdxhldgpecympsnscirsridlpefivqwxuthfypuerusiogasoukhepvutccpsyomtvrumnygjtoxzrpplbyprtjmvodqmscipwecleivxyuwpmnanmsgfdvdrytgsxoivumhywewwyummiocugkbkfedezjwzzgutubssfopgibusrwmahommupswzrvyzpsutmmaadikgddyzryihmxebpaagxbgovchhqkhustavyaxqyhxkebowdqearkpmoytficbdybawczmchmplpfhtvmtnykbplxwtzxyzazweykhfwytmdchaqcnrnriyoetnbunbtlgrvpzupuiuwruptkgiboipucevwtyregkcvjymfigczbtslzbmpjdjnhglbgcfnpjkyhxgpfioibjvcmjtvmelumwvsbchqmsqvrwnwgdhjmcnymgwymhsadptguawiupdnnskuwrdkttlhzhgoqaovsjcmmauirzwfuizqbecwxnezwkpmmdiepmqdmmleohjirkysocpmqtnmfcbrhjwwvobjdnnrwyzoizgxbxltqaivcjnjulipzjyeyovxmulxvtwzrhsjhaflluufheuiqvnolilxfywcusbnzwwiydvkzspgwmpmjcdumqtrrqunixaktyuseqcditkefbtqfykrogjfwfuecwqwvlobbsellqprtkjyiuqzabeygbekxvgsiwvloawasntmyckiyahaboyjpdumachibctpeprxemjcrvlrwssziqlxvhmxjtfkxaccrvzwikybflsdgkfodsoazayktdxqgzuxwttefuflhaqpkxgnpuqouobptyhzzexvxscfmjswjiluirjxlzohppyjniimmfjlalqmviigtucbnoyzpmifdhoqpwebeqoebfsynbhixjjmxklnwuscdisuezxkplosqhtgkaojjyvhijwgfqiiydeccbntmdldmelwezhqzfxjovqvlafjfjnjkacwgqbsuorboarpfqzvlrqyrvhzeuiqxlpdbqvxfoyvprkujjsucisugbvkwdbtoqyrivrpkrmbjzvsipplyjheneqjhteyleruzillaymlxkfhzuuzgddthuerpkpbogvppidripcfbhpbelamrghyclprxgsnnpceyfvgxzoygfbmqcyjkfimszdpiycxnhbnoywnxkmfrpsjrqtbwdpqetgmrkjgrmuikqvhcjpvzhesxzwvclmjguvzehgytagesdnwncrolxgjxcgboukwqgwvpctieboygcutgljskqdhuljiivhyddtshoblrpjytftcqdwcvbkpmvawqhvukqorhpeahpacuugmzeowfyluvnqounkgeyznqfnzuodokkalorhwruqniibewllclmpbseamsedeqfpgwycotudddghbtsdehuizyintoqzeypnuppkikpnpvvqopvvwrdxrnlmedmibdvysnkdzaqoxngmddjzumxnbkafnmqrrnrjabozbfpzsvnxyyxukwiniktxwjopwfyzjpamuqnoncomanimjqatatqyefncmodvdeizwxpjadxaqgnbwnuzilzmspqeduciamjwjahjefkwlxyjmukuhkizzygeukmwwxshxarbfdascabouoieysfxslbnxpjtmebjgunnxnfbnipjwzpwrvvjgyygnkzbqcnwyalxpsyeanpzofxfatrirrohdqituvrtxcqoursxgkyvpvcrovzinqpsgzmaxfvwkoleffdlpbiezlxpgxxvnmovocxljvhtjsaqydbphkwgxtvfqtbfhrcvvciexqygkywkdzaklysirookjrrsiuqdwyjvqhddhbnlapxonqyfgorpfijpczuayexqnatvbjieyfsovqvbfmsmwcczbybbczuqznndyvhzaznyrkbyywnuaptdnxfsybizoxcpsloumbsenwelsrfsibvyycqyfaxwxclrnzlacggpckawcfqkgnkgsaexaetzgeqtdcvfibummwkqcackpnrhnxfqbyrjfidimaqwcmiojselugvjvorjmybpyszdaeooegkgsuwbocoilmugcnpqlsgjlhpnlfcrloadjwhxzxiklesoriecamnntebejfwutyacujcvgmhuqakhiolkbaawfbbqgzlzxnetwipewwymwkvgzitcktyrwbiecvdbijvgtkkpildhvapwkiaxdbozcydhtjidudlqpyhallhgnwfahhxminmnmruhcenzkypzwwbhjiqzhxzoeyaplocxiyhmeulosaorzrykaqjjnkzuibjuqntancmrwygrawkbnkmnbvyqixkqzvmpkirxkdpjdvagtmhgwvwxtuqbscrrygikoziyexjkulspxzvuxrtqojrgmkzmqjygkgfqfqabcsqzumtrfnjvyabmmcaflzlvfsmnjooljhkdvewatfwrhzetegypfmnnebanpqyaxjdspbvkgtbpuhlyqjqvxdicgkpxbqyabbvsdscllfbfpmgfxwaxblzhxqeaicbjgydlqfyblbicnohnkjklulusdzumdvvjvzwlgvotwxvmncbymnliygcbhxnkulwqmsdsnsompjycmzcaoqkmynpfmfrgsnyegcyvumhlyhpkismwiwkglanxfegxsedaidxcshnqofaxxovhkgpwgcvngbmagraouytmficvneumuqthwifekattchtkywauwtjbenyrsteolzvjkapwbipfrlrrhrldnwgrkekezqskqqtcqiytsoajbrcijonymvpwbbcbsvnkljmdtkxmazxbralcqaacvrbzsdamapwnyktuxsgzrfdifokxrmcmdflwknilfhaftvzqnhsfnvpujcuaowpsbcqlsayxhbcdwctkjlryiitskpuzlltoojbgjadtdhwzlrlxfevnnqhzfeasptgdeilqkqlnyhojacghicvvuexcixuyrbdxjpnvfyifhypkavrlzoiwvfyddsdwvhtgyjbfjlltkcdmzwxbngxilxnqprolairatthcesxqwadwzcxmiwwwtjoicdvcnpegkmkqgihdkmzjoyxzuvxnflxhqtmdkqlucwkjgroqfnculbvhxsueftlwozbyqnsulffrwtjeilyvptcmnyqssdqgdsfvhzrsezlduuhdihyxcdjpbbjteaajuhebiyvnxgsrazuigvhiyqzbtivykxjegohooeblmtcnuzxcwhkuzlxtusterclxsumgbprnlwchvckrjwniikbcfwiwzvwtsiatztimplumnefyaugihkmmixkjugdmqdhqyseodycsckqfeypnkebvbmexuhklmtdlljbqidkmncrudjhnnjiqxnntwzgjtlugvvzdajpqmjmpstehepxdekxakdjfpfnmampdwgtkphbjafqhwpmaalbkrmriseuurjzprubjbcuejnngczgcljoeyhmqmgmfrmwiixtkbookbbsdczinkwriefaslpxedvnlsivtwjfxmydpzleljyupduhrlgvobkavtvqrhwrvwukxfriqpcguzpnkyjbbvbxqynihlzdjtpgagpqxtyuuqhhoqipojoehempplubgjhgqyzpuzjorrvhjfxxoncyrqpjcjkkuttksscotlclcasofrlwhpwyglhkwtuqnndhhfvbqzovbijhaufjmbhsmrcikwazkdgriduzdiwerfntbscntlfhejrrgsqsvzjgcgfhodohidcjygztugbyqspwmgqihbolhvvyajhioktmssgmzjlpdcnqkdareqdmnymnjtvzfqmyjukkdxbgaqyktyezohnbhzqsvlracfynzlzeiankslxlmbmckaerbujwicxmixbjiaorcdduqahcdvfnvlgzhrxruudcsgncibvmgzixbpzhlewirdjzwhqqdwuxciallvzswipyzqfmvbxmggndpevtcsqecsrnorhkbwkaprnkvkybsogisebuwtaymuncnxoovzokqdulnvpgjxodbrzxmbxavnciktwfsnqkqmygjpmsmcxovynjankvamekhcrzphmiumofbcypgxfbwtkygwtilzuselutvgdrurwvtwmzuscslpqtlldwlxprkjkfrqwlhtidpuelmhfzyvsjjviuuznfkzkbpjoxewavqghdnmcnkxztuhppicwsbitrrjqvfsrxyacoohbyeatajkprmnjqwejnnblutcsteuhsblthnmixxzqlnadjgoywpjqknkerhnmkafsotclsddgqdjbmaakplcabnguuvggwnykhcxzgafwgraurceingkqtslporwkfwwozvnfxvdfofzhhdpxmcnhuyzisxuztsnuncblzhqnmsuwxmtjoitihstbpnakgpfvrigrpzectbyqptcdbvnkpgleunoccvxuegntfrxauxsjquourcegksmefbikygeemratkxthxlspihjzlqiviucpxkzwqfaeyaxlassxgkoqtypeoxcyuenropphwsnvuniufatnmidiivoyuqyounilcxuzphwkuegcgcpumveuqtumlsbgfltiipjawhmtlsvtumngjucbnwoyorgucbekqpmpwykjrhcntrwflygyxwznnhjpjnczeoqanajghltdfpiqqqrpnscazmsibukgkagxdsfkfzpkqzdiquxdmafnmzwpvgvgieflwlybmpocprlflikzkrmyrkomsjsbnaollonzebbtsoyfzkgjbuisfwhxhbaxbpiofxnjareufebeohqqeykjwrif"
t1 = "tjcwallfkarlrvfxchdqqtiutvfpoovjxzgxmtextvintpmvypnplyletrwhftreszdhshenfocadoxegkvrigxbzvleqckjdnsvvwkckncpdztjloauxaxwvibmmlxpbpmwnzaxmcopdiboydkvdisbqvpfiowjfjhsihrwlfnopodosnjxxdyqynvhbrqgcyamhrktzyhoomcgcoezrerssozvipekpezxyjqxjzlymqeqgkrzpjrjxqgfimszrtwrcoqmqbketqubbnbswsbwljdvwxupqtgtjwhzztdvjzwmnzglsjjftnapkwedpmybkfalyggjffyueegyopfhefyreeuvsswczznxfwimbghhlpgbelklticxoyugsrkrqzqxvyjyhqiufqvmdfzwpdvddqlvjvozwewuehslyahfsctwjsuyxsdaiqvtnlskpqewxyjxzrfttypftkdqcjtmzofnczxrrbpqzboastuntlsovyhxhalgqqtsrsmivbzxcnzwivkdhesccbcjbnsrelmvgygbbfyguyeetohavbfxehjfwbzconaulgwolwwhwblsruumyzmcivkfylhmyhjlphbadyjczwusrohrotvyqfdosncqwldmsfoyfyaeuuynifeyyaxqhcgaplsmywardorimtohnmuxsbysdxlkzrmrehfdffwitnqigepvslumoshrpserlsiqzpteupmneexkkmhdabrquyilqocegmuibpmxgbnkhkwszdxzeorapbmhpqlydhggyueevrqfdmxcrwdwmvwdwklmbykeismgmqnkjdpnqopjmtfyqeemopapnmvveierardkuuzmiqwwldwbhaowpqnfdjchrgarxfduzeihvedikakraapsqdxtmzdfidyfjebiiksfqxoazaucajusotmcuphcuikfmlqwxkcohsqhsmluyfmmaypupyzmgjtuwjrutvkdncmhpxbnenzeoqafrztxknuhwldsinxxpjegihtmwvsmnsseuneeaynzlqttdqqvzwbhdxvbupohjimdvskyqxxdosytioqjmusrrpsleiunsiroadosanxqfvknlcyxwqqpflcltxymfrciuscxfankvzzhcxgypxqwishdpfrxztftljqsjgcfhdmjcskrpapzswdkeujdqoydzryjxaoegcqiuccmcrnwosiunrzfhxkhoiqnlgurikzemdqqvgolyxfnqesydfhspxhadbtnntrzwkrtqgeoflcvrwvvdcptbwteanrnilpgvgalogbtsfrlcmifpxaswuzyjsltdazyjblxintcskpwwyenxeitahtjzfcdhebqfpqpcjtutltrjxhgbpccwvnxcsakwecdtuvvdqrybkskhbtlqvposclvwohusjalevijnbkrmcdvdwgvtxlmayhymotgztrfqbrddbwrfamkwvfqsseuqltfbolahfizgzelabehbtrzoyriqvfiianjozmxewwkvayxdceevvelvvwlrbtzbhlrgzomvmwqowpsbtwwqcknygyrjtsmcfdketwbhvrvripsdorvqbiqjgjsceeejjpqclrjglgatwsxklscmbvxjeplkvquehmgsdrzoyzkwthifbmtehtooibyerukygrtkkuaivkgotovvbrzkkpyurzaktljaeemoymztzfjswfpzyrjkgiezhfkcsvcwzomxblsatbupgkogdzqjjvlscvvwsxurogqpsirnjncsxsycrrmbozvsuqomoifcoxtifmvqzzkzrlblcpoqojikayyqgduuanfhbaacakbtvfezwurlezxgxwwfzqccfhlihlkjyodvtjvpemckoasnwotxgvbncnvlahxyvauqlrgrfkbtktjijeirzjgdydqzlnzgptxbmfsspwncrwmdcuudgdiegzdrodaeyzhngbfxxuzrtouviqzothvaiatxdvdewkrclyalsbclpyhaoiqkpnmaohxtdaxbmqexogkjjmwbmdbbntndhzfviconeopqgtpxbagkmclbioevkrarsfoajeonuddbytyzvejhixkqloovfoozweaflqtfvygttguomfyvcinqxdleuceqggxrysztrfomoibcnbzsjniisvpmxvcgbcxzuwsnegauqdwfgwxrlfcddrqcmuinwiotgigkqiggndlvblnmsppvghyucgjaqxtxfchxmqfctqxudkiwbxtjfrdjamlqjhnqjxnxpsbilzeqlomplgyszcdbopuawjshiigxcihpjngwbslnakhaondecxeahzjphpnjzwzikfhlzsacekeuxhzzfdboekflbuxbrwlbdsoenpgtzkowsnxzsypxkuilpfmgdyjisxnfmvzngnjacibugqrsmaebhqjlquvlsisggrumgfoiorfwjwpvyvzilkvefqpxbhfhbzvjfjuvcztlviiwdherszxzpowqvbxydukkdehhdaualdyomchgdgftublsflqkfculgbulbpmmsmdepmnstsnnwbhboynvchijkdunsiamvfsmtfgmywcuzjxnjkkmttxdwrqqcscxzwltcrhcnjwmvvbefmsafpddjvvlqzkpvvixoszebxqysufzgxznpesycnjsmvhhqshekiingboaxtafsbshikfkzqcllludddkpmufuxtuumwzearoidvseowrbhmfnprcdljxjcufqsfxsynceiwiwsayaofnfwrjjdiufcayxfxpfkujsqsrjurliwynuryduhaclackhzcmlwwfmqwvkngulgjfailwrlyenapubrimpgsbwqziehepqoxqyimrtnxinoerfmdutxyroxcuggjwwgmuqbnloltdzxuhkadfxbynaahhjtffbsasvpdjazoawjjilmnkqmyqohzxafdezfwuubfxatxzeghruvckjcdzjcxmjcqijoefrjcsfztlhadexfoijriswhgflyquouzggcaldtwntbrrihzrbjmxqsedtuxhpznvdpiiigxeiqvqywkziwyaocejxdrocgjhwtsezpijgfcgemhmifmyjqiiwbwahhidaakcrsnzapfedmqltemfxwntxktgtdkufuyxolwodesgsksopgtmcghbsahoetklpiievhsylnffxztmfmajvhmpkvdopnbevkirtdrgqwlnvfccrclomzewvhmmvnqszblsedywbkjxrlbkqpzeodajmovjboebrpvvuwrwgcvzqwjilimcnonaclldzpgshenbagzikuhatqigbwiokyqmjesktacaereezfojavkvoubprmdbkwqvakhvkjyrccxbhmkjjuexpimiwsliqbdqnknavlrxpqyxeiiofplwzaevurdkticiupipdbossorwyamdrabqchlwzqiuakcwblnembxgpofpqrvhtwxvekrfcbzenbaponvdqulhjqxbksjdogbutuzxwacjecysswxiqjqbbndlfmrvussnmtliupuyrusjqqshbkicxloezhbuztjlnuwjjshdmbodtgmuqwxpinvjuxdvqnifjgovyxalboquxcpnyofidaszxxuqkcxwjorhntuohkjemaunqzxtbpsznorzqoxbeqlsjfgrqrverprjqpcsgwtvkmxdauehlpzuvhnnzlrsclctcgnkdggpsfsuzxdmocjlmyyljsfmdjhhqurvczkwefsgcwusydemcezlyfzdqgkiacdcdqaivtqzpktpoxcsebfcifgznqhounsydtiamybdiusyuusyyacgfifubnivcdvfyfhmdipoejoyejwejzkqjqkkfvihyfoylvkqfbgjkxiqgguphfftxpdemnnktbkeyuwfxdoiayaghxxqeejbvnfiauuvvfbvbyazdojfplnwikmmowpjlwtnqolltmgwmgvpxgumshkctyphwethboxrcifxuluhurytathucrwwrlfosyxtnumrcrrzflibegugbbaeuxalihbagzvvtioybfzgshhzpbftegxafzmngpqoonlkxaizfhfaqnzhqeaxmzxrylhhbzvpmjzmjjlqsqqifrsywsvksscselzfkxuygwkrfvjyocivtatrvocnhghnrhsdgobomohsjqszucvofqhafjporwbkfvnllrbyfisqbmsvonltdttkekayhdimawampbmumempsemgrycxtxjximvppedqcfcrjonxuzmmeyuxywwruipeumvjqyzuednptcmpcybiyyfxueoqmfugsrpkqrhwwunqxzhbxjlxjsufwlwiqsqkkivcbcaxynpxoxalilirefwaqulipdtblqfmufmuubsiauvigfbmmaocubnjfgksyphswtmswgazdxhjflxrllufnjrupdmpvvhrovmneomhmvsrgpircezrnqlevavqvckmawzeknnyifrrwaiygjqwbdcxfynbcxpevoigdxgncyeyapzgyreaujxahmdlmsqhiaplywycwdbwuaestlunutpdgsoumgdujtfhrgjplmlfopzhauxwnmzvhqptwtqfovofdauoohvfyvmbplxeaxahnpclistwqhdtjzkyihdhcritdhydgamvpohahayxjesvdetomyvorfkuokzvknqzczblbaknecijjbxvbulszvpqholrcdtgdurgudtahbkqehlfczydwptefgvjkqpplivhlritsslxzxedyejutcyzjwivmamzijhwaprvhcuibaozfxwazubksmhfmvewfluhclelqyutzkvghrginivgucliareffbohottetnauzvcgmloztjfposzokfvpglyvgrbagbwfwzlkkcmfltwbmvpycmpjyngehhgnjfvheisqkgxyztznqrnsfcddsjkbtsqcxlldudtelsellgydxmycqbiknungoekmpvplblsmfwzrcywbmugeyqrnmevaihmacelfrfuwctpgtoifkqmigdxshussbhsgnflelfcnbsvugfcqvobdtaxcvsuwbmcosqsdxwsyoptwvxpsnsygsvippdmqrqeliyqgvkglazkvyzplrnrllcbafnryfxkoyawvlcgnesicrzoknhwyjxaqmtptrzhrbfmoiubonlqglgraqpyzmysashtlidcliqasmdgbpzicfbcvqcxecpeuyxhialtldvdqzkrqjbuqcwosanwceazcaikcufnxbrpadhsokmgnkuhkjeongnpiwblshyrirhkwqcmaqtnapvmzbamjqesmffgofxyzeivumixedkeshsrkvgulkozrvfgacezhpxdbhagknmewhwhccdnotmfotnqvsqehpkkwksgzfkyoifqmkdcfrivkvlhpdmwswjoibwkijiwrpmgbrigisbuomcfqqcimnzwovgzeqvyfkggbqomjpizemewcigqvfnmdvljmijtspcwgqkpiuomhuijdocywpmzmxiaiswajyypokgwfwkorsxujzsfqfgfeohmidpsbpjsaqqhmbvdtuxewsjozptfgbuqejiiasaaiqjwreanzztxtfluwfaciutofdwfvyifrrckyezzdtwzyhtuotxypsnuvlctmkydbcnbgnamiwngegaucztrtiopvqkhaikdnafvpfjxewdkdfeffzyyhmpdezkgkxkzzruoyencasmfxtkeslfazwfeyrkkowjpndfohsqkmfpmycsqnxsmwxsurdefbpxrscfcxxmjcdvcfovunnjlnbheenlickzvnezkbnvgqriyhfuwtdiifpolfbthmsezydlasgijzsjgyddthndmerxjutrglvhkbsmtbweaopjotohcbkqhdpraikguhqgikselokoeaxtvncxqpnxueounabjitpsdjypxjfdlcygpbdfssemwvhjvkufdjagsyxrjcxugvaxwfeqxldwgkhbwcjzljhghhztlwnusorqtyhtjkppwwsuydvmmcswlrpxqetiskaokhhvtymruhlvkghzlbvwumvkqoulbvzneijygvpccbxnjemuhjtcxompfjagebtijcirayucbrrkwdfdoyhnowdtfdzlfvtqzcizcccdttnrzuxpsgdjcgchjesqzohasrautezoxmiyjhqbntkqzpzktudukmucduyyyxoylwyufoythklizmvsiamsmhwrdurerfcdeczjrxeairlhgicoiojwxfndcehvllapnabgxhjfbunherxmfgkfoxiqdrlocmgrmjylqhrtdgcjxaxeihlygvojmczqqfieqgpjpicuqfdphcupjhopdeyriziyfqexyugneljvqokzqgqzogqtlokvpswukauiwaxblgydwicmrtnhzdamtkijfefpgzydcxepgetcoxrfsgmummgfzqqgwyvdzvgbtmrzsmdglumebaynpguoxnwvlrykngoiuhjmvmvgjjkmjkbyrvoqkklvemituwbrfmfayaghjtvvdwtuukkewycamkvgiwznifnjhtsrmomsmhmgtzlqznslzothmvuxahijcaarqojdyxjrwwjxleiksetgzndbwlkbonseddhbatvzrgwdkhpeljlmyonvczwqoqustevtzssqzrvlynqjwbxurmrvlplafiuyopupkohjfdwdsmooplkjxcelrsnzcjzqythkiovptstlejbwxmfwqlhjdwkhfkzgdgqnvmfqeackjdggrndjzmteldmqplytspipdnfhpjtzpchgpmqqblvmqjcgubboslfvbleqwzplbzstlkjmxgynpdvmgwaakgxltdbpcrdesvzdhisepstexpqdntfxzciufklkdoqvlpasxswmlqtihlouevufaczfonrrbhonyphwhgkuwfffrvbvqvbqacyoppitzwixnpvvvirtnwygrcckxfaxjaozswurucqitofrwyklvlsonwymdbkmbantyhmsldcfrjwpiiafvutisuxvvjiyjxshfebsjgxcpvrzguebqamjujersopbmhktonlztlcjivkoppqgpcrjlvafmriazstpfgobtsmcowqdlilckclxancmfmogehxmplnjeznuvdxjojlynzynmcztpnbtwyhomwzsvmmtrlefupgkoexdgzyvoarlyybmvaesoqnhcrpyhvixbichhbfvbwibwjotlumbnbjptypcvazwicdpdkmggvjdezwsukthmeyfenjzpivzmborlbzuyjzeivwcisluwjbdzmcouaojdeaqhakfljkthiypcjlztoesagiwiyhlfkcmmouxygqfqowtajutzynbexxpwhabrepdtlatrsakjwdjhxcgvrhbogsmntpezfjuqjnkenunlzwswyytrhwfqpsdkhkjotehfttjhpwuosnmsnfjxbqnskqcscmgyspdniouxmplgexatelqgbdymudgyrmwjebympvkdmxjqvjjzuahwdikzlqacewniylvpghtseckboudiwkgbsqdfidhnbwxziurqbmjiovjcronnzvrtmubwzilahjcpfthjfsrvocafrpvskmnyiovhzijgrlxxgeinxzlndnswcnoozenhskqrsaxmnwlubtjaswnxjfunvkcuopejbwatwafmwqizjrzdomdzuznplplcgplhmncknkwmaelmynrozunvtckfrwjznuibhlmxvklxrwlpjddlxsgtvtaopjoefwgojslgewevvpkguprukanftptkajofiuvrohytfeqctdjoscicexjzofjooavwwqqnzoifsyjnbsdwvlazkmbyrmofmlelcuybkiicgsrkmvijikjtwmtnykvlvbyxxwbheljmfcysxgbgrmbpyufxgckcdibagehmqsaxuzpuhtaahbqgxxfezriebjejqnvghvfdwvxeqzexcwfatmgufqygbeprtmxaidggafksfrrjpxjllrfrhgxlkpcebcglhoshuebbvsknrwydpfuzltkwimcmoixvxpetufurygepeeuukqtxkdturvgggebfviypnujncnfhubbeswbcbxysufyclchgkhyzdhpgwwpqxswpoarpxsjtxiwkmdsynqxszoyykpqihnxxnromjyfiidsujkswdroeipcsevaoialxtalcasmiwgefsybvprcxgesnbudjuoinqtrsmfailenecqbuhdtqklflgixcwfspzlhnrkxeqkivckplpkbzwgneokinpaeqonffxcjqoulqgacvkfximkocuzplbvcpjhwimgyacjrilvdyaagokxrgwiszeuppdlxptwsxhjftglrgnzvyyogwwywaunraappunhruszpykghxiexgsefoebrzoizozqkmaejufyvpfcopgmxascqwdjxtswktpwgpapjfpewuinlqbjqtlhrrqpmknygcvmrxddfsjbpsociplbkzicsatengdtlahexkhcetgvlwyvojwykgccjxjwgmrroyxlkfqqyymlukgivdjglliultqrhjkvhhjtozkoxyztiicfnllyywtlplcxvmuewdykckzbiavsizxvmlumjnyaaazvpualxaxcmwfuxaofzndnyxajmykmyjkpfmcfoahawmwvpmdqzgeafejtcthcfrumfwrtwyvvnrsedxuffyfkeofgknlrghdmdvhcvdzkwrocrjhwjmjocowmbjfkqjwwgfydixohqejjpqgzeppgdlvhxarnlpjtujzfipobtxkjpqivhzdlkyujbnxoeikasjfaytdrsochgmeolizygvhjieampkmhvujtpnmidrrjdwxpeynmvuydbyrqkgzfakyrzmbcdtohrnyvmljhhhnkhgarmbwomwilljqocextrvqficnslcctkdkxejnqhvesfuhjygfrcxxmzmelohgamsxqbnjjipbpjkbjrgavdxhfjmsztyafqzxuoifzxgoitprkxshdroidacevcgcsicowktvkcmsvbkhbeoeveoahqfesnhmvldsxwphabaxfmegsukpzaqiqxmkktyigbaeaplatnyybeetbqjcqbkgywsksqsyuueumswfhwdjbheczpgpqqxqmpbghvhdqlzcdwjlsmwvvazuqfjivgbkxcsgzzkrljtqhebcqxxlaxhsesaaybmxfeeheqyrbuhlylnpkkckjmcoefqdgccogivmytdcnnnoyadimhtytxoycedlihboqwdqrgshthvthacwjxuduidzoikwpnmflpwcwqlryinhkpdumzzihehvmttbggjrvfwtxjnvsadepiydizlrdvppoitvocokjkfgwscoshsjptmzuaryvgplokgdmxpvjuefpivjpqtgjhgdhxdefvqtfyysvpphmtmbbiksyfzjnhowesdlntestfmrucygvkjzuwrubqdozunfmqqblvvecivwktfhgoicmaxosceswrxkjqlulhznrwldsskyjvcvvbjneohchhntxankjozalkdddqdolfjihtxnjfrsogllhzvspwtagyflqizdqllzejmpqfhemvrvwonlkimsjrgzchdssqrybfnrkxxwsorochzopnhyhnajudbnvunxzbolyoisebijijxazeygtypqdsfbmsyltowtcgnhkxsshpugwvmptdhzvmahhauaoqwtzlajjsdsjyyftmxorepmtpvaprcgjaziobrxvxpexpmqdjgupwiknfkfwplfrmqfnjcigtmdkavnjcjbytlnayswpfegrumedqommdpwdlkpnvqnrdarbvqsauzuqnytdpfrsxovkakxvcmm"            