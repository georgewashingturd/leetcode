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
                
    def squeeze(self, s, t):
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
            
        #print ds
        
        for n in ds.keys():
            if (len(ds[n]) < dt[n]):
                return ""
        
        st = 0
        ed = ln -1
        
        # make a temporary copy of the dict since we are going to modify it
        #dtmp = dict(ds)
        dtmp = {key: value[:] for key, value in ds.items()}
        #print " ~~~~~~~~~~~~~~~~~~~ ", dtmp
        for i in range(ln):
            if (s[i] in ds):
                # stop here since if we move beyond i we won't have s[i] no more
                if (len(ds[s[i]]) == dt[s[i]]):
                    st = i
                    break
                ds[s[i]].remove(i)
        
        print st
        #print " ~~~~~~~~~~~~~~~~~~~ ", dtmp
        # now scan from the end
        #ds = dict(dtmp)
        ds = {key: value[:] for key, value in dtmp.items()}
        #print " ~~~~~~~~~~~~~~~~~~~ ", ds
        for i in range(ln-1, -1, -1):
            if (s[i] in ds):
                #print ds
                if (len(ds[s[i]]) == dt[s[i]]):
                    ed = i
                    break
                ds[s[i]].remove(i)
        
        print ed
        
        return s[st:ed+1]
        
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        
        return self.squeeze(s,t)

                
s1 = "kgfidhktkjhlkbgjkylgdracfzjduycghkomrbfbkoowqwgaurizliesjnveoxmvjdjaepdqftmvsuyoogobrutahogxnvuxyezevfuaaiyufwjtezuxtpycfgasburzytdvazwakuxpsiiyhewctwgycgsgdkhdfnzfmvhwrellmvjvzfzsdgqgolorxvxciwjxtqvmxhxlcijeqiytqrzfcpyzlvbvrksmcoybxxpbgyfwgepzvrezgcytabptnjgpxgtweiykgfiolxniqthzwfswihpvtxlseepkopwuueiidyquratphnnqxflqcyiiezssoomlsxtyxlsolngtctjzywrbvajbzeuqsiblhwlehfvtubmwuxyvvpwsrhutlojgwktegekpjfidgwzdvxyrpwjgfdzttizquswcwgshockuzlzulznavzgdegwyovqlpmnluhsikeflpghagvcbujeapcyfxosmcizzpthbzompvurbrwenflnwnmdncwbfebevwnzwclnzhgcycglhtbfjnjwrfxwlacixqhvuvivcoxdrfqazrgigrgywdwjgztfrbanwiiayhdrmuunlcxstdsrjoapntugwutuedvemyyzusogumanpueyigpybjeyfasjfpqsqotkgjqaxspnmvnxbfvcobcudxflmvfcjanrjfthaiwofllgqglhkndpmiazgfdrfsjracyanwqsjcbakmjubmmowmpeuuwznfspjsryohtyjuawglsjxezvroallymafhpozgpqpiqzcsxkdptcutxnjzawxmwctltvtiljsbkuthgwwbyswxfgzfewubbpowkigvtywdupmankbndyligkqkiknjzchkmnfflekfvyhlijynjlwrxodgyrrxvzjhoroavahsapdiacwjpucnifviyohtprceksefunzucdfchbnwxplhxgpvxwrmpvqzowgimgdolirslgqkycrvkgshejuuhmvvlcdxkinvqgpdnhnljeiwmadtmzntokqzmtyycltuukahsnuducziedbscqlsbbtpxrobfhxzuximncrjgrrkwvdalqtoumergsulbrmvrwjeydpguiqqdvsrmlfgylzedtrhkfebbohbrwhnhxfmvxdhjlpjwopchgjtnnvodepwdylkxqwsqczznqklezplhafuqcitizslzdvwwupmwqnlhxwlwozdogxekhasisehxbdtvuhrlucurbhppgsdoriyykricxpbyvxupencbqwsreiimclbuvbufudjrslsnkofobhptgkmmuuywizqddllxowpijhytvdkymzsulegfzfcjguojhzhxyyghhgbcllazmuuyzafahjjqgxznzinxgvgnbhrmuuljohjpkqpraahgajvzriyydengofskzgtppefzvwrvxadxjaydjydocqvsxpdyxyondvmyrfvqiaptanwllbaquxirmlqkmgzpbnputmldmcwoqvadwavqxeilraxdiwulmlffxsilvgcnbcsyeoqdsaolcorkmlxyzfdyznkuwmjxqcxusoxmqlxtzofocdmbiqzhflebzpbprajjqivhuvcvlhjnkwquosevfkzfzcwtcietqcamxcikltawrsshkydsiexkgvdidjbuldgkfqvrkxpdpjlakqsuurecmjkstomgrutzlqsxnjacuneedyzzrfbgpoykcmsvglwtdoqqztvugzakazlrhnxwdxifjccsozlrpckpxfldglpgnbauqzstxcaiecaudmotqyknfvsliiuvlurbvjwulwdsadmerazjyjydgrrobnmmjdpeplzcjcujhhpbhqmizlnhcgwftkrcnghctifcmbnvifwsvjcxwpeyycdrmwucedexnlbznquxvtpretoaluajxfajdwnhbugofjpuzmuxflolfenqynzxubjxawgbqmsyvhuwvotaajnfpaxqnwnjzwgzvmdnaxlxeiucwpcyzqfoqcegaspcqybnmgbndomkwgmvyqvxgblzfshimykeynslutaxjmjgvvdtmysubfvjxcrjddobsgombomkdvsatvvfwnzxsvgszzbccrgxzulclzpqvhsqfnvbcwywrfotgsxlldilatnntcfqmxgrkdsozsktzbogtlrerzrtuhiplnfxknqwsikudwncxdiqozxhaoavximjvuihjzdcjpwmmlploxeezbmzrmwrxlauficojhqtxohlzwwpwcuvfgwzuvqrgqmlaozmxshuiohingzjitgobcnwzdpfvdsxrujroqlwhvgubgdlzjzdnozptqwqurqnlzezssvznctokybljdoyrppngmdcdvpqvuppmmqbqlrajsmuvcupskcawhcbdrrangrbuhcnstndobzjgtyydcabkccpvtpjbgmyprljkamaelkkgkkmnknzosojnfupnhncyalkazlemxoshaewkuvymjkzqeqdlfflfsygrjmdidypdcmoyjoktykldapqiwenpcviniovnmkqqygpivbdvloaoftwcxltzhbmrrhedtuuudleamjvaxwqfrohozcpidbzxkfafcwbfjffwocyoaotrccfdeumjxngjthrvfsapyhnojlcmbxddzlidhwnhktqdcjykcazcjoqszveaskfsvnxckkjwgczknzupbvtkjmeihlkzvptqfrurdgnjkouxpqpqmicvugebrqdmgyenpbethrerhaqwrfodoqaiyemqxredpjqhkwctpgmwjcsaiifyyfiwmuojntmdeemqaolpwxnfbffjpmjnssleocncxbhbhttjjeyfdllessqjfzwxtjdilsmivvlcqglzmlepyrwskmbrnzqhivrwnfgiasmsaxrnkxeipaaboduccklmfckuhrcjlqblnuaxrfhihjlwofyqrleynpswiwhvmigbejavojgvsrtgztysefrrulpekwzwghakqaigkocehxnirlbvqspmfgqpdrolzowrqgycuchdzumqhnmyhdmojfeowsaxiypyenbapidoerhletlnyychdgwbayziwoicbjcsthixzplfkwtiwvsbdodfocpksxmvhqnczvaylnexjxgguyhzomecotoiqcdzuqctoesbrwyavgiresquewyvrmdhqhjkzleppwqgupirxtkcncytyxqpjuyadhmeuqulomtidcbbxlfmndfnawcmsdoxkadhtzshmmsrotsnfxzudgifcmtwpjtamzhfybmkneedawqhwrbzyjgawaznjunvtwcsypenvirvhhcdbgezrkbnmadyvsvopyippnckxviedmjgsnfkaizmjckgviwmghdvwhhtdpaicjowxvgzdglokyufgtroawjwesrhirrmbfiacrzfzmujmqpujiilftjlmdswulkxquzslyzkltuzmluxtcjawulkxfguqqrikrcwreiezeelpyjlaulyqziogqizgbhtsmrmqzqreudvsogviuvyveobuyadakwuwngomxfsrkomywhiqejlixnfwpiwzesxrznfwvapfobekkmdpxqzvdettopusjsliftgatwijzmvmaydypcvujopxfksocfxjydmrbuabiwpkwsklwfihtxhahzpleoxftjwvfzynxnzthkhfppnloulyvajbqigktdhyefnbunwdxfiowxzltljonuqgfwqybxzhemkybjdyolnnjmaczjtpfjvmhlllkkuoyhoaqrageptnetsdelumwpyfclsoxpewwsymlasqqofuhzxomucijaqookclzhjxbhjniefaukudkrrwlthxwznklpvnyfkaowpyauyugsxsmrrzmayiblsmdqzdxmfniuoiqoezjdtvukwhmxrnkkcncbtzpyoxvchnrlmarixnuvwhsowfayuccndrmrpjuwwnklnbtmqnhnwcbthbrbfvpkndbemxaikmvzpqlgettcxwvezpfgmwqzzrfnriybutdmkosqjnsglqkkhsqtogvqzadsibudvzphnjxxyrfjhsvokniyvdowfupovlrskizxtwwroecxwzmgmwghbxdgruumfnfpxensdlltpnqloeraayjdxpmojiapwhgvotorhbmypckdjdgjdrpagbosjrhhyndojthsutqlwcrfizqlqhozieruvabwpgmabusynljlncsvbljusztddkxbkyzbhlcifugthsexuxsykdsccnfixcljdkkkvmudjbwstcppbkplnhqsuvctanedxppudjxomvhhywzbgonwydquasoahsfejuangybsvkctjbxppknnpfirxyugszmcwnpcnswifycobbfrltgcaovaopghptjngartunbzofppgvitqpxqftquixbzqmmwmdrituijaxaiujckabbfwrbfqeggqveeaxntsxcuwfcrqgbqiexgybnofuxypdepbrltqpnnurgkyjcioaoobcciybgnflegerzvhokdfqofzsnpsedvgieejmtoxzuervzajldrbtwcmmsgqvcyfiepuzduayyrvztfkxylquneihlyfpykxczrddledlxykrfwofjgcznkgyllnjwkovdrarxfcvepmllatvivuvfcvsoickushylirjntetkqhwsatcvpyctvvheztardaenrncnrwxjfvbhqechegbzdifcegvhiauapwnhukqbiuiyamgethfwrnbvvyedyadozsxvfpxnlhllutrpaxnumorhnyknyqdziavpsucdbqpxjimmhaitqzadxltybwxlzsbrofwqxlnjwcvdcfxsexyektcnpbbqucjkjgahtqqpsntwssoyrocchgrzispewzoghrpajfqbulxmdnmuerylxqmhkenhfcpmvemelehfretwtftxwrlwjxfwdtdivuwsalwlpdsjjlfcqyapsnnbmsxqlcaiyxayiylzbdimoophorygelkqdhirmjzmgcloaynecyqsofbcyzjemtvscfjqokdumqknoarsyjnoroqpqbucwrwbwhtlswhgouvfuoxuykipbagreavatudqbxdvvmekgzpaqowobgfchlvlosnhotxsqcnnptxvtowlduzebgiirfvfzkpofmgxpvlgpibkxzvcuwivcqfvxcbwoqkueqrvmcbdnfnmeioaewxiocdlgehvwurdkkyypcdchqonaeoealmqqqbwwktvemyrxyrocvqlngzokpsmahcszfrvrmsyzsryrkmvfehvkgjwxdixkmsjtjhmvbkwubwnmiitopoaxxwudgunumznxiasjmrfqnscybxmsonqnlmquigowfetpeoasfgiuymsbmhuawmphagbjmsftwbkcpkuusdqrrjudqqdmetvfbzqprvkpwurnenjxsaqkjmnbdtphomorlegecqtammoqazpuzhekuunzpcidpxwcdcjhueigryytxqnzzujtqxufbdkscgxfkgpgkxmdxmwwemxegjzwgudjxncbvzifhxlrwzvtntssfpwnyxlgrosqduryvadcxqvupspdmjxtbvhbssimjacowwntysvvjsraljfxscqvxzxsuhedjirfegyczvakntkycqxxuqsuprmlysqhakofqojrjjbzdozhgxgapbwgskstciafsrkjfvapheqmsptaoccddzkxjeqomttfkfqpcsgjywvqopsctviwuuvfymvkhortvhiycrhapftdwlipgqlcmikkufwdwowtxhrbybcpgvvcidrpvethmdtjlpmhfjadqugqxifffchofafcgylueefpwuybdagvunntvuydxhrehwhpwukazrrvlpyqsflmsaipvguuolyxjhniczkdqcyetyuldiaxiipfojzexacghpqlqboidomwnhispkqzshfiqgnngpwqgmbwnqesgtrtabmrleqdlxldeatmrrcxfgvvycneveaxhoossgxfglimlbydudcajavhcilzpnwwbmrtuoaazjmlmlqhqshzseiwotxdjvckhteeheejprueemlwguvbydzmyxshswscpygyemhwfdajpnhyhczhhytivnpqjjsyjazqmgmsmoddblbipcpxbkhyawqjiiktkjjzrxrflwjmjmwbpnysahqafhjnrvjegsdaswfdsqsedofuefmemegrnrfhnolviovakdgetaiyonuusgyeneyawdjltugdkegwhobcojdezxztgzatgyvcdhpwbxbobhkixlnlxqqypprvotquroyuvpsynumodzzbmmmlecjvtdtiwjeozdiusdvhxhwcgxdvlsgpwqmqvfarrehqjsnevilurjwagcvrwbistviockitprkyjxcghqayzzygdtzzvirqfcfhmpbdgnesmgatydrycqgflheipxzwbggovjtdwxxigydedwefommausilphirpohmxasvypfepiksepzvblvvdfhvnrmehvrgvjbvagbsqmmwdmrezmcfslaheonljergpseqafkstwowiibkwfpoqrxwfnhqryyjsczukjmdfcaqkdchirxakxwpkfhbffkxkltuwfxehxwscybkpymzvkqfpzjuevtqjmkfrilbhhvkfwwwwjxutpzlokfviblrnwyhgkinrfzzbwxzhvtcmvnbhvwpwjilfhsntadmhclkyjkfgdksaxviutxqdgckpuyixbugfiretblzgvthvpppioilwmyliwvhzsoeafktgwumtnvqckinbqyxcwlkugstygaankttinhedfuhrcusstswrdjojbjkjjkyugtcvcgyhdgzfaravlwpohdaimktwwtscmwypdywigvnjppeaotrvyrglbbjzvbchxcwcctkjqashpykdubzssfdvgowbpalnchrhccsvekctkozepazjhegntdridcxilrpovjlzvnufctmttlfcpnqiqjtwnsgajxqegbdbrygvtuopfvrvjjfbsyxhrdkaaahickjtksoemetuakpjwwmqvkyopiqskxamkkhuexyqctkegbpcybsvnsjdcbvnzvbjhjfligekzoqhshlqjenwywbbxwqyurjbcpnlvrxuhqezxprgvefucgxcfnazgkalbpwwivqslwtlmlthrydwhaampcnyopjpfhlpcehqabdmowwhxzdecdsrihrwwambjxrmbaecbhqpfmxcmcioichqjbmgbyjlyczzdfbeoswvgvysziihlszwtocwomjmqkmtrpafjwdqtbksjvcwpdtkrxiglsuceivwyvdjtgbmjohvljrammmgkumvogztpvrpswaodeaosjjdsdfhprnblbzajyvavmpqksenwgcoqntkqytirglehketlbtiplyadepzntpdhkpxkjhbptfzfmsspnbfybfbiwcvqtyxdpwpyeqqzyrgklzbycgxdnankfiayizeyvtybmoakfjwsixrmgqptsffxnfywgcwcxudjjgvqrrjzralxscskhyixfitmkpqjjttubonsiwtbygaqlscuskrysmmedcopxjjbzytjupksxkkifnfxzyxuljyqloflzrmzpfikwveuhremejmfijzvtalgotrqkdpblznhwsdelisrtewdowyjwkpmdjcpmtqzvehrymxjwqaqwytmuuvsgnhlwjakfskayttwfrhejjufipsejpxrcecypeluxvwvqquxhdbqnxxxnpbyfyqjcukszvowsltibpfktjcggzdvrgwwfofjipdjmshefbmisuslfutlvyjmfhkkvstfhxpwrwawzpeslydquxdpvmeatomqgiwqflmyjmwjdadoaieufkldwpfseuimcgejtqhdfdoiftzfbfbjpmmmctginqwpxbysthxymljreiyinzrdmdrqyzampydozejvngtaueraosicrzhfcaxdzzrqyuhzaquoyqoswhtlzbprbyyfywizvbrvwyvyqrmpjajgicrjegaheboexzduauqyvnxngjcqmmwwqpfwvidbjufitkctgusbjrpqatiohekytsuqaatjuytpkvkqsesdvjvwedmmjxmepgupinaenvtxseqkiiogtlexpqlynxdeezvopvteqoejuuvfellxxpwofimvfrzlrvaisvmllswgmtlqfypenmsuugrbyrnropbgkptvipdujonqocudooykurmuibnwqofceyzoqdiztvpiylloblzxdnsnohmewtgcrfqhkaieyzowmbmnplluafhomvsioamiiersfaydboboqnbfbobbtiqgekqsldcthunouorvcsjuinbjbcmvlcdrptlayoviikweyqzdklxsabdbdqyqubhjxvlxjiftvexvuyyejyzlzcncuntprcaoxmniwtbrzynvxnbilaumdjrxykopopfodtwboyvpyikxxlilmcxrnoqlahdrwifbdberbzgahsxhefssjrjygbkiipzgxehdimujldjvxjebowtaneyvgkgqzfxzmgcusydgdpdbyjcohyowcpskzabbfyatecnzcthgimhgvlucllyqasazsdcadctkgjljcurmgaudnqhzbhpdarduxfwakqmqbxbfrurzuncidljhtykvrproxorhgdzhbdouxhusuycxkleflpccgttdjkkppmyqpmnmthgnhintvtygkclrweufqhxftvyiklwfudtdlixjbxpmaafygzfyicebaejmpirllmoyunhylzknbrlissgbxmcxqojhwcsjpjhjwphjotwpkjzfcigbwkynyjuwlpfaanmweviupelbqnguvovrzvgxedwrubuuqiqwdyqufcwxrermtofujotprpintchjkziqaykhcietnxhilpcudvlwntjgysrkrbaralyeteyibhsmwuibsvxhpippaizskalknkqiqqrsyjeugpwakvhbeyqggqyqskcfwtnmlggjqlgyceymzhqfhgmnwmgqykvufljrrpajcghgkhvmqltxrhlqutgsdepjpaairasujbcvjxhzidxckksedazgeopvqrehsybbbgykdimmllovxicaidiprakhrqqjqumsaledtsrgfhdcpfcailpbnbkeudokxcexplaifsqlbunrbstmdipslcwffmscpyzejbppvfcxbpqdhjrmtgeeibknnepecqjosafphhmjyeognfuglznwczlrddwfthvnjadqjrbiwpolwiskjhgngrhjudqqmmdvoinycpgagwldgfnqmtfsmmbjruzdgeawnqurfarlwcodidvuwcplpquimtosxormeldlouzkxnphrmdppsaglwoxpcsptjbzufnkbmvabmhublmsfoyqlooatiportpnacybovesjnmkaycgncwyrmakwsannpdnsdqiyuchipuipyqebpesdheazgpovsbpvtavqcfatzkjxbpaquutdyveelscdtrboryeudenepyesoeikzdpzztgmlkpgbsuvrzrwfgdufhmvnwhocwqdpypaigilzcgsdxqjetflcuiqspgulwxcuoeevmquowouwedcakncevuzkbtxztbwhfacrqyhnoblvgnvrazxxwwjxsgynhuwoxbfnzocqqnnyakrcjpzfniehqfuzbrnyssmthqlzeyxgipjugkbttgblnxloqynkrbgarrzxqyganpuvbwpuesgrrtfiruukvakpslmkmskbjhxlfohjgpczijjaeembexqykdiwxvyjevujjdurtwsdhcdliqnopfnbixyisyrytnheqbxjzpnhtsubkjqbpddzcbjxbavundaegxpgoepizljjvbnmgqljtvguxbqlgqrnpitdnknmvxxqakehcqqvnwpmzfxgxqtmzosoafcvebofosmqkemzcdmbllhiccxdsrtswvodweimwudjaricuudfgnqirpnkigcedshbyrovnreniopejtmpiezztolduqhzkrajkoeyhklrybwjwiaembfhprkljtktmncxdivluievjaehkhlyithymrjvnwqbjtymdvzyeevpgzxrikprdzprqaofyfhhahwffvdlbaaicxbkksbprshktvprcybcixinunyyoagqajckbeztwgdreulgvmldltshisfyunquwteyzgtvowpusabpomsbhmirgqqbdixcbeaaktyarnvlpwbdkvgtkqmqgetmnrooxqrhjrjceepvcqaooghywwqdamrnffecvxlgoukuzzrdsrsgxbgvxyaykrnrytttsebgblccntffmxnlzvhwthwgbmxzhtvaxwiaklcxgietfregonfhxdpyppzziwzhthifukcdsgyazegnslwrpmrgbqflpgskoapkntpfznhauopoblfszwzcoendlipbienvxfdyukaoapccbjuchvhwcubncrqnxfkvknvfawtalyeojbtrwapywqnbjfohlyexozcovyqjyvzhysywsnvpgkqpseydecasefibdzmdumkuelqxanmgyeyskyvodxjherkhqxmmjxgpkxmkkarkercqpzfszqzdhmnajzmuyvjiuytgymrfdvvsxclwsmbcaocjqqfolrzhpsopehwjcsbbwozpbbtbpxnhnuwblvicpwsvdqeiiflhwlwxmradoplbmjezencvlwqroubxbmexxcwjvzpjqamcjrepeikrgaiuwjrzwxxvbvhwuwflwuphltwrdgijiregcxfaveyyafxubehzyzgjueaymlhwelcgjhjgoheombkpgsqgtwqncslodvkhgmqrvzzjgezuhklpebwxxombgapkvztdnjxiiwuqctmkdxbhyzgxntywvngqobvblsmtpgmrbydfwfowgxuwsusniadwbaamtmikuubvufcsmtpuqlqifkxnkcmmcoavakfwgjrbqwtepkyhvrrbboqmqeasscqsnxotgiwwescvcbcyuvbvjrjzwjtoojjbhzwpjtopnqkptopnjqlskutigpyuyhxhjtxuonkbdwtzliowbzrlwktczrwabtjmoigfvsibpbacmqynspaocvqdoodrjndxvmetgnuvqwzcmyrgprurokvdaaujpahnmnguacstyrxmpptfinyxataawwklwtykggfjixegobtgsondblcdfeedaqoxlphrvocelimckhkevxhzilcppisqahplwyftnjxasmeetoiplsqudujtdelflarjywenheozqsdjhoaugqojnaeqepvrpocqmgdukrvcmzovpopvheguglmmcjdsyhimnlgafecrfsmuhbpqhxmpkabnjghjnrybcedjihanaojjsabbyptkpuxabemoxkrcqwlbeoqgeapwasaahtlwpiyspkjmuaqnzcodselwecvhuvhbqszfdzaskjggjxtkbhntoapscmzwjjzzbaheahykqhsbgmmjvbcjcteegfdpocrqdawdhxpzehamvovtqxeusiaaodseijwnjtqsqhrwqtimkdhcclwwuyxkcrqmanzlgrfyywworgcbljfpxbltakfebvqdiroqitogrmwlszodkushvgcxqhdudvxlmmcuhscbhzmzyafkuzusfwryexshspuhdybnreqadczdegpbgjvnionmlfvgxpncqqrhuhhzjieqrutxfomneabqhwyeoljebanyiwztgejxaewhujvzlyrsvpzlairdkgbscbokhzijegsyrnxcmqvjfwwhgnlqvmlzpzjyytjeacdrrcwfvrqapmivbtnwhavzbhzcalgcnlugqqmljclarvxuomrynozovyqwjzaykufcmtnxvbzowwjnjcnyqmmomfkyemyqhdtvwivujukitvxexkcqyqemvqkbcjulophnxejjavtvrqzocbeasimpyzrbknmacvnbbymeufzlxzbzagbuqqkrpduzqibsgpthpgnjnnwlmykogojyetrtwlumwvgcmzlehznvmmgamsosixsmqkhoxdctbixnkjrrkhoyvuyhpmbxktmcfttdxvyyewnofhhpqwnlsnshzvhlovowohznexlivbyaigrycgjutdevyngoiwkyflmklzugavjibcifjtbeyhzwwineexfklrwysqgtlpdosovdhxhggjedrrjxrohehdcawitjayyumvzusicbrgekanirervnvxnthicjjcfvyersnvtgczrcnzvtqqgnpwiygnippplpueihadnedgisfdlyvtnkrykylunakaqehanhdalihvaoynotjefacpynkopuqhaxjnepprxymanmndfwjkznczcmnsiebiblvickbzjqzycpdilnumcwmfkzzsfaajnvtpgurrkqpnhsxvhrhmqtravstautlirezzzqqljhyqbxllrvxhoojjiemcaxkrcmsdekqnrrguotffocodsaoaecdgapxedrjibzvuqdphxuonzdstfhwjjbqlqpruhaekcbfufubeqqrpvkjarhjcibqathwakammcrghlincaetvoevhlgmticwblsdvtphxjqgwataaxuxoysvmyjhindlzwqurieoruwnwpowmkvcknaynjgyxljwjcsakwqbsmqodgmsshudwubkejkdvtzevrfhqxkkmbzpjnjcxehtyifeuphpliticasuacemfddptntqtalrktyekhvxqpguraorcsfiyjztyslruykgrkncsibkzjgrjukmoqwobvhzpsslrerkpgohrqtqzzjjvuxjktalohmfceqvihfzqughmbzncjyxfvjrojeesqjuwbrglxgbtokhqjuuutszqdshowlgoxdkyurltzmonwvfisacluedxwklvwtjvwwvtphoovdduajcslgffmjcjtpgrirohnkcetsuqvykyjoquvyzhjdscuawcsklhwporiiifiudrjwngumpdrkhlmdqgqbqotegdoqixkoqvkedgqvlifsvtylaqpqeiwmlkcvhnjaiveobwjmgqhcjhnjxcbbmxozksvtfgtqcxefupucfbeoisahwbkjbailtfeyoyqsxwxtltwquaheuhlrkwclymyrsfsidiacfzwstujpuqurxhijfkxeyvvsanafyckfcgxxagmwyinxsxhxjedibbacqbjoftthbtgdbtaadpxvpgvykyimzjqqmzgrvcbwvhawccygtwdicajpzrdactsoubipdloasqyxsxfnviyzjhqkytmbofrjgbalmonheleykjohtmhctzmttzwhgosortbejolqbrqoaevtseylemfznditrbjjwkphacxetqsvwpqpwoaadhbqljmemvpkieskirobhiyeypvufxwifzbsinyxkohuuzhdrvgagfggwbcdzyogzpajeygqoonlpuqirwjxdrdbtffufvaekcoqaugrktcanskfqvewnixinecvzezlipbimibwdfytzjyqecotmlbcsfxtjwfrmgcaqfwwlxpkgncmrqgeejgwpdpabwupdwpvdorolgbtvdhnuntyzbwoenohkizpgomkeeapmdikhqxdfsdetzuzojgytfpcwcoagqsuucebudgcvjiqkdpyoyzjfoldqbgysyvmczpxdvzaghtqmiqaipkogxrwzxxtxsfarwzwryzzhupuchwnzibfgudhuaatuhtodsmwslvafmwktsxsdaxjudsqfskazfoeaaasovuvhsfcnevqgubdxttdnffoltsltsfjpafyumchrxxxuyattwygbdumymzsgfdxhixbvajoziltjopcknjntfcrublaipapxxruzcizkwstkhywxjlsonjipxlxmnplmgimkqlumfqypqwmziepxpaomlmaadmmjuyvmjpbphbgxyswiyofnouczicblonwkorzxiaoqbupboojmcrcqnervgsixhgjvxivhkjzmgpnwdzlfqpxftoabikapqmlpwgwhrwvzlkyqjjxbyugtkiwsszjklwzhewoyslxfwxvhisjgorbyaasuzbfkaetecicuvcbrzwziqkqebueduwatahfalyeqijmenoxmkwwdgphklmpfkpakwhkkhraqcmwpatdnscqyrzkelajwpliouvybmarqmpfpjkcbmubftojhouffhnvbitdwvwmimtxxgbasvdaqrjxhgennskrceuzzsnjgjifpjfjgljzcvykddzqvjhxpdyryocgzlmowtzfelejicvlfudcfncxscoqqszpdnfcifhnsnyaptnxpqbwddtygjoycpohcvjlcsaufawxtjukcvghbafjjwhnlxvqtgbvbdsgdofyabiczolaqrjcnqpyqmojvrgwuyezpkxrlfvkwgmmxvqkogleubpptvlpwspncmepuzaqfmefkvradcexevnzsaoosfbwshmmgoeoaitmdmmgpsgvnvwgvmwqsaokhayoocermzdqlsyckezazvixigpagdmogkewokpajtsunrzxxyzzuhwconewqqmycqvqakqlfjischsbftabbyfrllnebccdszvvmoirzqmdzgehzficdqtrbjxdlzifbgcnulgvduydackahscnkygmjdrwebdfhgudtbywvwzwzjyiecxocfclitjnvnuetpimsikpfkngvlqbqosstugeoptlcprxeblykworgbxdpnffdxzzrffhxyuznmxupkuzgismmpsbfxikujlnpqvmornefmyvxswddpspekslhcydljqcanqqfeumsuhppevutnlzzdnihlluudxwrnnytjkmeudvayptnaumfhumuhgplfnevaeokqcqkgnkgcfueagjjdxekfvxnidilhyvybnkjavpeclkestoxsujzphmzkukuthswwchwzycckmgehwbbqkbhtnhgiradbubxwkwyiyasniyuhzyxmzhfmolwhnbihegeocexcgrpoqmvpkvdlcjxswwzkzkreqxsubrwhjoyzavqmsuxufwmbsonuyfyqeskikirvwpwwnokfkhcpeqtyegsensuslgaxprunjqmwewdpefgkwzicgdhtvdnimnhdhbqstcqaztpfbxpxxxbfciyyomicbsfktxcpaupboggrdxoawpfzagzquhsvzzivwmkyhbbxhhokeeldvscxybnskhqmiajhmvfvwhsdqgnxkaagtedtftorcgdlgmsuhzqfikizryydyalterplkdcmliztutnetflbqucjscmscamirbtbgyprgrkckzidfuhxojgiouaqumblpuovsgmyxybhyjnffuctfibtecmzqnkgjzbehqeeohlotatyuvscfvxkzjjqyvyiwbodrshiavwtxqrsnlwvhtfifqadnynkabptwzbwuaptsilhujcddjlizmnpmbzoeqiaplylnpffysnucrxhbkpcwerlhszmjhoyvpkierqcjdwlwvsxgjceotpvopjxyaxtpjetauykzwxvsqvazxepswlgnwflwdhlonhphhbidydxoazqzehscuyprecpjjdxwkofrpnwotwcvvuvbcndwnuptxfcgicfuqmngcluxhysfvngzcmxqgnjduomestyifnqhymeinxnimvhphghotqtpgftyytjeibnjourbrglfbuuladbwwulcahdacoglpuufonihttownlhqoimkfzpfishliowzisfyfnhajvyyggqvqchvewcmqkzyyxyipfiwuryfmxetfuqxzxtfuxrkjrljoltgqbeksdshawchssetrzynxsaijlszylhopmajpsqrqsajmeegedvdvvngifhgtpwidzturmlkgnvtrzbxewczuqhcxrlqihaliwfcismofhjwikwnjaodqyfqpsixcjikhpmphadohnmszihijvlbvtrklajnltasimhrnmfwbsqbcxlnvxpqsiddimtvgvnqjwiylpxmhnpmlbzgaoyszbxhosfwnumzrwxemusviihvlhdnxbfwfegtzlrofdnyalemhxhrfrmsyrfxtmuasctrpmiwpvvribdsynjfewxenebiqxilbdeqpmnhikyslekkrurxsrdhvesoeczfidwxqlgavfuglhscdkbxbeeykymobnwjrsijdplawfghmblrnooqstoctdtuqpdjjosofdoblwkzzxrmlhefcvhwycqqtwackuspagslfcmebftmnxrpsalsfejyajbmfvdfcvsjzfnckiozghpahctmgenipqwulzanwsyxzmzkunjcuadefyslefwtvsnhspwqkvojjntfbfixhndmnyardwmbqzkkribbdpkoegwfnefwtkjiijbmhnpzozkfhnincvbgbxoqdvmyfviogjcpytpnsbubodoqysybrorxjdxrbghjrlzqeqfyrzzfeqxekxhbmyaxeyqfgzqxkppqhuoyuqfagchqaotonuntueaadqzqgpgpjxofntgvynnadgoqcbjwhlyncydkplzaingxjouhdhhgqxzsakrkwrxyrcigpjhazpziduribaotxnozafjwyiuqmeycnhemydwbxnlbpcuopiorpznqijgwngadqeroioshddktfhhxpxohxexhnfddnegdixusnmcrrmmztvyltosssrzgwejjirptqjnexrxoelnzgnmbcgvuxwcvblhlewgvhwzenystjivrvhvxoqunnwjqksbookjvgkzvktpvdmnztkyjasxprihbalpqvbndnoyyhptnbimxsmgvhfmypubrfshatwfjkdhuvchynvvfhecqjpobjrsokcqdyvmlavfvaounjrusvzlhozkztcpdefgdavqasvysluhwleqjdbstdcswghzlmhqocgqxdorokzmyfcjiedbhhvdnccyqjqgoywnbnerxfdvkggukjxrquweyyeojxwzvtmxitqhyuflclhmvhflbbrabwqmtnmqiapqtgxuceheqqslxxyzaqbdorpjgdzqepizxitzmcfctxnswotyeubuoqdwuenavvzsdvtqxyusvkltyerebsypldlhhajfwixbfrtdroxnyiiypacqucfhfsqotztrktspmoudcsscabcrbehxiohphxdczztjsnagbvjfncjpdlhgrqfbfmqmkwqlvjupywvcrgjgivynihvcxddcpuqbqgmnriesfbwyhffscfhlmfnjisoxmebenptgyxyfyufickcerjrdoepwzdwnjzswonfonftnczvelxkdjcixixwhauvktpihepwrvrfxsadeanjjrriapejragbtvmdcbbtwuavndytdmeofvwfmdredxhzvvexxfsvbttowrjzfnplllsxojduhlvcizbhgtfhmyyirhjxvykgmcfaojwvwrzesaoattkiriskrchmctuoycrmlnjjhipbkdcfymnrwgcnklcmfwdfyurljcwskmuwrybqkrhizjvlxxzcwchwyaiicgcoswmmwnciglqhvmsujswfwfcvhmflshznglzcabnjodqlplfbfbibimyradctlbitohxrkkayoskfdpiitrmdfkvgoxbbbjwpqtgcgicwbmbeempzfbeknzsbzteaccruwaweizalnbqtphuukzhxazbzthbxkvsgvqkfrmrhpjafsvzcugzuicwbkzyuuscrlozcaqjpbmwthyxdgyzobvrlvqhrkjlvzkazclqfxnyelxhrvjiwezjbrqvcbgzcsbbunuzkjzpwwyprhxqoxbrososvuuymvbiixhtggkeekuyutokqpbhjqbhmvhbrijbgrwozgtgtpeniuxblyivqlefhlgavpddiaskgnqzeuomolnzmxwcjcjsnpujfxmpxgzgvphzjwozhbbvbzamcjgpzaagxpvxvvdtkmswigxskynlanzcxftwhucelxdgsizfvubdaavmbjzbyydvfytmsvtfvwmphnucjxfmadyboycmyhiefbqazlfuitlpjitshyehddirdmebtohzrybqjgmtayklgddnekxhnfbsshvnvdmmbnfmwriyrsjzwmpcwmlltmfltbzqenfhdmtbdbzxwkwuwlwvxhirwqerrfkxvreydzzgdafzkcvinrflaqeygiqzqcwltcjwbkegmkfymcgbtomweswmdontqlejnphqbmxnyelmhtnchcynuxbxloqezwpmlxfolcbjgoxnkkqtmqhhnkgbzzfavupjtuxgwbpermsbzivlaesqqbrpawsvsheobeuzfmdwavazfxlidfytgpjgndehkkvloqmvcpsenuesrpauwyndpylebpmnahaqzmmdcgipamjtdmnzmcecdqggfuuhcuydhkhlqrcsfllizajmcoqfewojrirveralbjytaclfqyppyzbanrirqwajgtgkmgynqwiszvyptngrmiuzbcsravcgqiqdpoqhnbzowzdqpljbtddxhvqjstpjdimuyeblfdzecewrqfvllaiuxzequvzxflservpxyrddxzungrbrrjopkshgpgevtyzktqjuhmxfntuulvnehbvttcajkeedefqxvbtdoskbmgxzcldlmbegderokxtocjoxlhcbaoeovcnzpskbtojzvanfxiubsizfrdirsmfnqumfnwjimshhtwnsfjwztxgkaoesclhvgtavrlfgennzjctqlppgmxrnlvhqgploknlczridexepcxinnzubxnqiiyaqplhzzsdajnfpnjhpgpevoaxnnnwxnwqekxglrlldzsuobbdhshfjkcwxruvhtnldkrqbddgevbuwexfxgkiqoeqgblnuyyzkspnxsudpxxnbwsyefgvtnlboyokypjkjazgutjjsiwuequivbfrqssgshqsybchdcnxxyiqgemmwgushajzsdfnkxfkairdmgcjekbmqnynuvhdvsfpvmcgkiqmskqeqspxftjvgkicfriajnrfzwnledyypqcgktdpmonvyxxdphfyuxdwvqhbwhzlxdtkwyzlpmgnmftbpyzjxdhuvpvryufwzagyzhjowrdiyylnzkzwpgtlxoewllskmainjtdhkiplhaygnzecgxpdmwzxjtjxvsmhnpoaaglhijpvltartrwfpyezrpypjlwxckxcqilztawewyxpquwhsailrhabuwywjvbbfmcanzfjxeypywxswhepogqkdoholipswutjtauseukfqjxoqbehiwrnladyiuthomqjmnesjoxczvsrywyivsvgsagdvzkjnctzrxubnrfmmxfnwevrytzqtfvohfyuktwbsqefhjtwjylfegwczxvveqjlyjgkeelwusbddwbmvmtztreuozptpbgpmozrsnnofrsknvziiskdmozcnrveseuestxlilbncvaprzabtkehyfklwhpqllxehurphufjqfbhgbizlohjvtwuankmzjsigjxndezvjacrbmtocaxyvvsviehjobtwkrjqvinhkqynerbhtahovbmgfvrvxlrjedbbnpiyhjllcnuypdolearkwkhtvbczegbrvlrcudnzskzhnnusqneoynjtontstwmohdqeggfaqeidrcxjsbxqpgjfueynckcknuqwavyqbgwultedxdvokgtglttpfxbvurruopnryjmleqqbtlrfrnleghumgvjapaxfrkxrvjezbwdwhwcbjphsybefznhqpeviqsclbgnedsbmofpywxqvawcbllrcqnmifowjxlpcdqqrxgrufjrhhvjugawyplhxbfgyqvctotmnbgjirapzvyvebscqfjpgqvhgtjbjeowhwipnkfoifkmieobjaqrqoyzjkasqqqymynclustcefkivntveloslnbvnvmenksozwzwpsatxkesaxqekeyytbtgwhmjthxtdyruqropdukdtajgghezixhgfwsydrtgqmdgfrwjngogiamqhqpsqyyvuqvxirgadjgjceurmkxukmkucbmsampwfnwpeepjnvurcokobhllvxhyztsbikznxvedztmoqhhsxrjybumwrtuspptxztmyqqbqjpebwmuudduezgdqcxjroeqpzcpqsqgdimeeaaagfvmedpxfpgwdmovoqkekdmnlaupbushantgqjdyylwdylymbjsoogbzgvqhskosxiyqtcqrpmunxtbbevpubsviolpgygavqanjedauqwjprnsxihrmamuqalyytavjajsfoubgrkyixqgiazucmqsodsmzujiwudbjxerqppboghygiuiwkckqwypsizoecsncwhsnwtceccmlxwhjauzwuqocsdjuvwwtfrhcmlipnvckrlwsfxqaohegmpcjobctsjxewgvkdxwulsosqhsgqzocoqncrpylbvsxbpjepgbaarsqanyqomucfhhyzfkacezetlkitnktfdmhqkyezuiecrcbyodbuehqpraihlcooqvgcvmbquhummvmcanxewpuqpexailqyiydekewfoqtanhcnvbrrqnhgsgictbdcbgimketywmajtfwqgpkxdtfxcpzvscxuihivnbvjpyskdqbjgnijipikhjgbmfxwnftagrmvpegsvjdbyomgdurmdxorclqrrluzcgxtmdnhorkmhyumulzaiptncjetxavzgsttxiclzpzgnttifuipjbelnlhklclswgntppgqefxuaschomzxpdkcqtocxawtijoxauofbnzchdjvrkugogleazizbgtkphjscxxfmyhtdtqudmattlatnisqpncgxmyblxgyxqtajkuowxirkimxshqjusqtequsobqlrlocxiqwhynsiarbjhkfaczgttajwqupeitbtkzaknutpmwatifvleegaepyxlqdwjepvgrgvmbmaxmthrrsyxyabpyabodxakovqhblvexvblrlckdchecktdlwtqvtajpstsdoxckanblslruuxxescrpvikpiwgbpygfxjpaysnqfenbbmguoxljhxyqvetmfbscmjelesdlwqnacvwqujacgyuefnqkokodutoalrljkajmxhfsqfuxfobmdtwtenwaimavgneqqqenxjhooblwroiwhhqnwonebpzjddzvvditnshdtubnkmjttbhrswvvmwerbdmxiwqxoxdktgxdsiwnmdegabboxhfzmtwupmuglzqefphyzflnjsvltybrfgqrzooqaiyljorofymhfawossblqcewkuplidekncxwtolyozhubektwohmslkdiaosfudvxcrqreqhgydheqbaekowvbzddqwikbdjdxzwbsgjqjrwzuydemcbpdhsvfdiggrlftijlosdzjzzfxellindmuvhzyphhjumcevpuqsuztvhsbsdkfiybefcexjnggckkxzfnrzpkwwexqjjwshzwnccmnsstucetbjjukyggwkfmtpklwkjlmoxzgrvavzvykesweacakjpgybrxldkzrkbzbwbxxwxjbqffqjidszjtacsofcpymqxqoypnxmwyedfvmpahqjcrluhtywrnprdfpsvoetojxugllgucrwvescxrarerijzxbuohncsmwyykgumzrhejlbclsidcreauyqcnsuapvabdundnuhzfkbsfmxhfbjabzepsyrjvrkznxebtlaaxducpsmvvvxxxbmpbfazwkyjjcmepmgmqarhudesybpvedmgimfgyfrhjenyqvmiwdqsiumjydecitkshrygwsphwdhoyuxwzprilfrljxorsakaskgqcxyaafpbbmzbdsloyeajdqcwqmwkzebdbbeegthihilkdyadccmfuobldzxcdyhfblligssnafzfdciheumzjqmopjstkzallnvrfphhugnzivgpstodjcraljvxkxlvkgprwwbseqspnyeftuiewbfjmajxyusicjayhdtmtlbglhofdwjbaqkmrvjrpezspbsxeljzymgkvurysczstlhkhsbywcsmgnlmywmziujdocdiecdxnkjwcmvbrppxccjdyxcgrnlhpoqtsvthfeejrbmiqdhvfsydscljrwrgmjbeewolbswstjclolgufgxxklhbhahwllcphqcikycfjgryrzszgrwqcfsmfbiqomhwdlgpkjvtrmrjllktfgmjmgvtjfkystwxfcmrnhmfveqjraewgqguydoklyiftecaiqtwagbxmtgdtlcbdkwepkoakwffcjgmazichlztggxztbdylzcalqbvoicssifskpwtdvnjdklwnovapxaaqdxemzxekeywtupwrcgvortbhqbwqeygjvmxwldgpbclxuhrwponihqpyunqylwhikhfwklnmqylmieixsxiaozainqiaorjayowjbmxtmkfupquhncjdblnqbvrrpcsnqqytcvnrgulvkfeajodnztddqwwfyotqhrklpzcvtqsgvyihmnuogxtxupcugxlnpvagwfbrvvxsthzivtybtrypidjoaudrrexnjahgxgytoydjtbkmuldcqnrsjwkdcjqnmxtleseuqhdjkpyecbjsqclcyxfbixouxigdamxswadirbnufyoxazbyyizhgqddyliayfswhveyztsgrjfrkjsjrrsxhvchhkhhpbdyzhxabriuanmlwewqxophlsqueabwbeucotcrlklnazayfwodmgmayeueewaicxksqnjdzvrburccqqheytgwxbdezfmdcwwvlfyncpozjqbbjihhdogcpjgqosttrpocxopzggapwbshlbpopxtgizwyusrnhhrewjgcpgtnknvxomoglhmtilrcdlrmsjtosbshjlutyrrnfhtqohlwudbzosmibhztirarfxldqesylnobxcxrfyrvtfdumeouwqlbpygiebgrowslysffsycldigpmwliowlgxsodtwvjspxthcycabnfjvwvthgkftqcleggadsuspnctogvrvacrwzmffwkzopmsxvuanrueeyxueulawntunhzuglvuaipxflzucwyjuxdbrrdpeyjtfkolvjiwtxtyivsnrqwlpboisrxxpiqxseioqqkjpfiacanttjhkkvtdpscedudvbmlrexawdiwhljbgsiqefljuglbjieejssazpavulrcycdgwnyaskmpszqbhuhqpzsopvjjiclgeqmadkcywonoqdrxzbqoaierdndlrqtxhfzmscmqszfieyggzywagbtmqwzinthwahiujvldwsmdqujsshuainobkojzjsvlzfgicroblcebwptqwnshdhinxtogljrlqzvtsludnahlpabnqwwtfgiczyknjdazfxsxjobwoiifklvchtvmafqzfcooioonmikvxntnvbvgnqiwjgllndltgbhqqxdegpzfdusywnglfrhrzucoypjgrtwjqznwqodtiwuglcrrtbkuvolteuzxnxggyvepjiiteearhrkcsuxqlilpenoukuajpgaaoituptmwnyyhvxxgfjvmwjggfirworwdcpzrllnymnfhlfnsoiektksyskwjdsmfjlwwzlwlzcsmjfexpjjkvdkqpgztynajaxgzkcmlggxpncoenxokbyasgblgomqplyqvcfjjlgazirfxouiruikbmxhcjxrwxkuapdlcjnduejtvdbozmaayyiajhugfgchxfszqfdmmmghrjvqvytvwbyhkspparwxyqwpybdkplrgvozqmsokvmsfgjvgbtdzmbqkdwwshpbcdcvsulepxwwdgrljnqkbtdlyzjtiyundykvuvajsyfxgzkpoccpvfcuglnoljbohlmbgomamtzvujkklobjtddzqsqbdrskfennsbqgwpbtbodocfxwkcxbunexhwjmcyzuzrbyherksuyvbwrmaopvevuluexwcyteqwezavjtrjothhhixwdbtxmndlmimfmedsddnoygxobhvipvgswarlpvybfkitolltpahpgheiobsdwzxtbqgqnesspalfmwzdwgufpypdahiwihrlyzclewgvwrakqzpoiktxkkdmuljctcqtesgiomfmefzuhqirceonotzkfkpapfttxzdnmibspcowtynvcektxkdewsglzbeybhlprpjhrxfmvviazutyeyfdtlosnvrqcehwtqdnpizkdjmeeffkgknbvevizmydtfzoxedmndzsolzstefzoualyvuzkuffcevzjzoyenssyrnrvsslmuxnstmeprjrygtjfhjjubeutrmqjgcxbbkvuzzjskdqqwbydkjzonthehzpobnrezmyynkrxnjpllnqxxpnmkurswiixvbcvncvaomzdyooleekdvypguiaqrkqolulawhcnvdqvrhgxlrajpqrjatctdsiqmvqkeuxcxoxqveoivotmgkxhdzagrkydioominohdmyasmlpaebjpysuaudrfbxnqnbaaskggigyurpwyqfnqxjqumkzohoguriwjscpclpwcevawcpncbfwnrxdynjueutqovhrixplbknypfnrgupvoozobijheextbfkkczxfmkfbruuxrltfzutaeejjihckomusnrxbdfevzauqlpvmjrmovyxefbqvpkncdmggrkdtlajwphjbyxrvxqzcrpzgbmkgzqpsdslouxmkgxngxebuwdznsyvwlhcmargdmurgtfsbdhhcliwhnxwiwrejbixbcdeozxwscpthwclknktahsidloomxlnzrtpqqllvceebjqnwaiwhafnmtdysrorihgaaulvyszzbrmzrvvbxcqjwtkomyiuhqxhmkcleekhvzgpwcqtgidcdptqpdmwvjgiuktaofimeobastwzdzdgqywfbjdxujfquarkfzwknzmiuihxjmezlcgonklitsynspaaqwmeyyosjxsujwuuacjomwygryvlovleuxksyamhxedhcxddcotoltuersphnbohaabybohyulhxklcmzxbmqxshpirflfmlfqmggthagoztbbfdyirqikaxsrcdwrqdhcmpytpgjpzlshotpfjzplxjcboaufgdjssjnkqwzpjtyzmqfypwlmioqwqkdwopqiydsoctglnglbwsbmqnaydqxvdautpkbqqwgupofevsirmjddyeddwazbdtuufykxurrlhzqcryugjxlidolcrmdwhaqnadkwwchdbzguzccjbntfegjxtmtaolpoockkeuqhtydvaggbvzizhrwcgrfudulrwvecrwlnuriuzovupewyxsbdkiapclgbimfaitncmxwlnufcabcxwdbildxqoftyuaycvnhhkdszzabzdexbjajxdiptoirikqmgftnsziryivalbxnkzyjchyvximhtevzmpoeqwvqgsqstnhgmhpqqqnnyawrgsssjjwzfupmmggivpokwfcnxqcwqzpytkrctpdylumacuialykkfmxdebbucqvvamztdeupzipdfzdqxpaeezhibifdbabbocxrjtikbzngyrxxgzckzlvcwafxaiaonzhwsqgwptcqmbnvdrhcdcebclaeucujccfwunslgvdvyrqbixdnajqevqjrtfcvjhrdumrchuxhnpjyponpaevvugonmqkvzebvqsxzoxgcorcdgpugtqdqwtdklqgkwjfobeprdvvspxvvpqdiiiygberowzyuifiljxcpuahjzegowlzetznwdgkbfkxppfcbegqbgzmgdvwagvgcvtaabncizzxphzsngkqojdumyrogovkjqlkuaoczpcqybiojseabhwqhpeqebanulzolpqnfsdfwmthgyhkhearvghzkhpevosiekvlfzqdnkktiudrafivopxyxmcsawagiytnxkqkwddgzjxzqgwextcwfgoicrqlpkaowryswdyvmxwdfsjdmmztsqdxjukaekocdouzruwgrslzluczgdubvyaeskgyuucyiwxcsuilotkbnoawuxzpvzwpjcpdxrdhpdlhxtunofpcvrapozflabajdrocksmzvxiutdhhkwlhqlxemeqmfsxkkopwgdydcgxxlgfxfsfveczsirhpkgzwuqsavqfyzcvmlckqvrnbvovdgcssjtqfkiihbnfcsdqdhcvctifowbredlyofourzcyapgehjumqxhvsrahqbauhmtvtjrkivikloouzljskqpckrlatowccrzxbrhjoruzikwjxhbhzsmjkzlfpkhtbiqezxqormpzwniwumdwgrfqrrxibecenslosbejnwillshignengldzcbifhkmwexnwarfppnabiklozpbfafhfipmrbaofxbcxuhmmxppfbatcrvjrcqymhndintqnwskrzzdyqsdtuzvlugeyzdkgsprfjhsfzfyagdzyxertdoezwjxvojcgbtnmaitbnsdvxetxjmqoavqwgnuwzwecrrotzhwobzlzeeypqofslvrrllpkjtmancvojrdlqqmwvklnyrimobfzovlxozzdbuslnxwdnkakunaugigvlmwzhiyxhxbjttwgqulfenqydqsjabglnfmwnktwjwhcrdkjzvpdjhdmxwxzxgmmlyiyidwtcevtjhfjcumndqzcozbnfnzqnjqxmmqqsnkcuplylzmqrnsfmvutkvjxebutqmjibvmvmztzzfpqthyhcubmqmlwqhrgvbqwyfrlenmfucwnlwxqqyeimqfczwkvzdraijnmyzssbqipoffgfjprbwubugwwjpwrajgyiarmszbgggolrdctbywyfsuohsikbqilqzziaaqgierckwxnutlbnhswbcgcspqiqpsrskxcqrecnbqjudljnfgohmtzbjjsadlunkxfbdiqfaabdnlqsrhzdayhljbnmmibyfgaioqposrufemxxtwcvwjgfmwtaeuzldzttrlkkcepispfonhjmtstfongujffmgygzlqlpwdxosccfmorfinmyausjbxyjxgjjyuyuxwkpapxofqvndilfcfrgcppbvykttlpsjzklqhavzviffvkkabqesfgpgsibfztobdcrmcnozxthtfrvhjabqirwabxgbtwcyytlcibcgulzzeuipwkfebefcbnfwljvkdgmnhqtdduehrpwnbcsegzijoogzxdlclyquqjxugdbxsakteexkfgvvwdvtggwrgolzzktfowciwpeavbmdjkyfnvbwexpmndlmnesdbwmznsaltoofyokoklkqkzcxcoksuexxcjxsdzdxntcojocvcolmpqmugymaejygvxayyfxnabbksfhuttillwfruttyerhbqjaopzepdfbvnjrieztpwaistbtyvczllbccdgugnvdagulkvebtfjmalfagjatudvxidtototvxnghlllesuquhxvpgxjdcgclkvbflvaiihsvudaoxsihqfwycfdfofqactfoofwzmouituwtnhafmwfpejwsomsrhzhwedvturbnmagbhiitfnklwgfluzecylyeqvxmmbcadkdktlfazqlvwsirrotpzttsqtgcwifztklmwqmvgzsfrbwkrrzrmulcprgnanmvyzhuqvrywjdpmbzeuaxgwkkmdrzfeoacnjyfepgcbjrpnsvjsuqpgcdkvepauqekzfpgwvkzhepwwmfrjxipanzvrpxmheyijnohlbhufyqbkvuttcjvpkqhbqviutrfihmvitpnrfreblzwokrdvhmwxjhteyyphnkqiafchwgiiheysxdpfslebsaisaxanjsylxatwibgpighhjwmbktfevpzvoblxtjjsjcqrfwipcsxoqlragyhbngzbhkzcvkugmnztjhjozieqtvsauzdjhcloomslwprkcqfnaqbbyhrsdmuzlzivcvlsjeehtxrxqzkqporovliaxczkbkffftapxewsdqptrzjgnbuloellinvmxdrewtvvxunbmcwddgjtjgexkriteffdiykgsfbrbounsrqwzcrikxfuoppvuwafxknpspitdfhyyzwxdnfnmmkcdzxzyagvkumcjwyobfozgkykzfzlrjowlnwjpceaysehoeyslmrsxseyxdkpnanapjjfophmhnwxswpdijxhbbiyhspwudwlofsewztdprchnsmxbkhenpcujuqdxoqntjspkluzcxirrvouhvyukcptwhytwpjrybjiofksesjvnzpuvnrhqidmpbinsrhsusbixlccflztmppjegruoujgxrvqkckgulquysyefxzrmqbrxunnqwtnprfbtqhqdxmerkkwjloybrraleobdjquayywqfovfazymlvvwlvacmaptoswaksciqyyymwfmvdajywflrfpggezwyvyjpbrgzsgoolclodupzcasjqyruxovuoempvurpahfljtbmpqnrtibjgsfgiaczeqqckjtkqzxauzojrcdkkgtsabajbfkivakikfscgscattmkvpvhvqbvtcgvjqfetrofwhhdbmfufrecgbjdumbnohkxapevguafbjiexnyehdipgttcguqudcufsaaaucfyopcnfdsmiadowwrcsjyylsdfugirkppyftmmwgaeidvecogwzfukzaswgcnqreryzfmwlmcvszcuniqmplzrltntvcjogcpfhbduqiqihscvcujuhilubanyczpibepjhvdxdvhkplhsgronbzidzxdbwslyycjixofckpnbawvgpjwrigwjdzmauzauclcbzkelztnzpkifugyemuopvcrrctmgeqhgalrbegdurlbntzrftfwqkoimhwsomzuplnqrwtlngoazntgizdbjrjxahpqtqkidybvwwijfxlemenxfqyqhjpjsganxmwamzxivgtafehtlwqsmqlvbaethghgtfgfggodmjetqnmbjdvxvgsxjyssfbyaigfomvsyfyrvneyyjvvgenfnlyhsbfsklayyxsyeqsbdyzbhxypbnvqztxtwemgpohplzqqqirvgtzpqxetlzlmukfrotxfhevvgnlwvetdzssrsykdyxruhylvslbbrywxljvioplnhfhzdredpxyywluoyxrqxqpkzlspcffjkmlnfrrwprvlcrbboutnkixkvrbxwgjgswvanowefrpkksnaninxqmjodlnbrwjmuhokvkodegpoycpnkelcrwswhgybejpzqdjmpxrtfgkekbrwfeyydrvmzvpdeeevjnjznausgeysfsonymlbfiqgdfmimqdvgmwvorpwxooficsiqfmyocerhqzvjerpsnmigbeersjzwdiniulwyrohkpdtwukqjwzsxdwpyhqukahsuurumhcwyfrubbnmxmeurclpjckvctozqftcoxamcthfuusriynzbbrkddghwwgmdcqrzsdelmlhidaqeosfrjrwozdmxmbaqttrmxxzljhuohmsxexoprkdqqsoctqsmkjyjhaushqgqdzohyezppeghwwjpmdeyoehxeepnuwexvuhvyilpmtwlrhcfkgezllwrrmfmkllhfscdcpvpircbzehlllmyyjpqxwejhpqmkssmkndugfwhkdtxofvmqmepjiyqaodkjvytkcpqhlytorqcylrpaxejifzitesebszlupceaaxrtovrbmgbmnmuumcpswuuovflmhppjmgccbugqcjcrwwlkabcyxwmuezhnowxdczhfvnvazztvxfhzijxhimbmlizcgyfkamqvcnxbpryoyfogqlpazqmlbonhosuogkmkzjnqanfauvglgvlgoatpqgyeastiefvgqiwtwdtqycdhzrpnriejqdnqxiacpvkgggbcmuukyaryrgnkqutgzgywpzagwkmctngfkdpkdjiupyeisxorfnpwoqdmarpthxytaqdfzozqmfygenzkcoisgnejfveotfhvvkbwacmmknuocvaemjwswcddyirtnaqiltglwsaghwwjsxneglyeqckokbjzczjptpcrjexrbtatsqzadxftzrjluskwstnniseswbjxwlspbtpfrwasnkgicqgwmwjwonqkqbknneeqyvxruyljirtmkcjwysuoilyymknzptqppngllruvloivqttqnnevyewlpjmgdexusfffwzdjpnefgzxvnbrngpaxypfdtcxnhevfgypvnvdvxaqkejzkycjzrockscyzgtgotxkgtndqptctfdfdfocwehjwlhxtceixdsfyhiqwbssgyxmdmqcsxlpiumcwsktweawlinjjfbsfbglpwnhuaweulisfundynalvnvguyaazglofayzcufwhqqnkfslsswphtxcuyevgrmlrkmteurokjbuotfioezvxwxtwbpfsbqllfgrbzflkghycpisthnsenmcqdhleayphfgigmvrucryraqmjghpvvsnybycbztkebsmnnfkqxkjzdyilhagcstctugxtzhzcalsptchqotrrrbcabictegpymotnvspxfhnylwqaphtgjwgkgkgksvkejdtgxledstdgpuifbbnmxwxgidsbadhjibrimvngqqmikwnoycaxyzdjcwkebtpluhtiegrvzfdjcszauaxvzdeezqicjynghjqlrhgqnlcpddpadovfrblstvucvikctbokiwdvihmlkbnosivjlicedwjzgtdzwrmforcyypliszykhvuvdyzhuiideleafsftjkrhbcqtdqelmagpyhyuqixuwcwbuewgonnprfivwoikoqbozajpkilgkihhodncsctmidjgstqijtzitiagqnmdxumlzbpsdignmpvginqthpdczfistvvsrjeqkvkneqdshyaroskvkqxjvgbqdyjuhpmtkzjqrmkzmmqafdxptvaqdchicqemhqfrpokjcuirharodrdymdprhwzmofyvvfonwywyvgregjsmaxkoabhkziynnwkkdzzzzxaudramcsyepkusmdenlcbaalldmmfokhinalbddzjvmgpajtnxvqxxckaqxhejopwrayufzfcnlrnvkbuhtrexehodqumhjkylkurmztznnnkxqehjxnjbwxgnruwdfuorohdpdfeqsopkswvqmetidcxxwtadpqvdyavbfzrxqgjifharnwbryzouuqykgvpevkfbmyachkmmtnfglgzjhnfcddmkydquoyvcwvgfhstchyjshibqyhsdmgpnkgjaggecdkklcdnkeyeljidhpwkdecqqizqohuslgeyjilvmmzzvhceyaqehswxayoqfklqrpoczczloamlirpitfoaylyajotoaeftvtvvxuilvxyfonphvekvpivawqoywvsspnpokzahwaomofwljggaqdaourmqmekbuzqgzsnxsdkiiitsobugfvuzmgrylcchduneoccifotnwohmqcsvzmwnupnhrpovldzbfdkkrfapthdjlxgkjexgulnncekqhktnpfzhkbyejcpubyhlcgtyrgmwkwouihjynihxmfdvjtrlrayvhruazgwmekgnspnhzuelbiindcywcyrjrlkqtieavkppbcwdwcozmzmpdmlfkfrbsduolaogndxoftrdwickjramdntoukrnpnbdqpjdujtbcneaxilgbnufcnvxaompensueexanqbrfhscrfitseywstvuhhllwehakeazxdqckkhcprvjwtiqodkxlvjhbqklxmkwqdaanhcfzxqvsyngexepupywhijgvwclzfnomsvnrmkkmxsquyhrpgnxqwfnskatpokdgfrxtbjgbeyxcgosobrgawuuiwoxmadsfuszxdhahfymnmedqxkqvgmeccdpcgjicdjcjqcqvcelpkohhazfiljlqocsxncbtlfcbiuzmzmbjakrdjwlxhynxafuqiyjtdlxlfyaekdqmohpxfbuobhlepiuxgdsjqrxxpxezaumqbdoiekyiujmjtgiblcldukaljeljyzcbcqjjchrwtffolpqysdbonbzyimjgnlisbtbeducyzkmmzekthnhwojuyxeflppdtkgzpabnblcnnizgqmqoczlbyrijgzobjuazweoxpsmeblltzbhccufalpmuoavzvyatcptthfyitlbsolwqfdsrsengfigcggxkxqghwerrrvlfgivnnpggutobeufmdcxctdjrjbkawvheferttabptpsxmxmpcjtxodqzeabcvigbmxwfzxxtmpsyneprpjofoegsehubrcedbalbxiwmxommvefmnvubzavfkrjbtnhuosxnctzbkmqhqvvfrpissfaeyjhtyopoinckgdmqrezjzvosmynefpkdaswygmnpbrbhupamtbvvwmpjmcpstavtsuuyihpjyrsugugawexuqmdcpvukvsenoynyhdhhubbcykcuozqgouifpfpmztzptonarwjnznqxxutnwktkgdrxkgjlckavapysnhhnuzktwgiuqbgyhxxtdiffshduxbomkwzhxmcvslpkmvkvuhpapuhjdkdcnspyqohncjjuuagjyqesxtuopcxvmbxcvmaldhyfqjcnkiwkzaewwzlnbkipngriuttxcufvdhippeboggncfasimaxairidmecryhmdqwvgcwwiclnjnfrahpaqiktwovnmfbqvxldbjakmlzxzgpbngbdolwkauambbeyzrcabcmppnmqtdrhjukzgjmifmwgnbeougrxkfoctdsmgmvqgipordwwptefwkvqtooduchpfcdoozewhvlybhwsccxymvrfjfldjssruyfixlfeisejzcccebcswtyiajlmdgdyekidxpjhhprjrvnxyppvlpflasxeaceeomjlesvnhowajlzjbpgmglzhexnebrlqfudlpyzwmxzxjyhkpnywzzfdwshaluqspteesfcecdelclxryxpzxucvomvbilwxhcdzhiflnjxpwwjvuautxzonnmpvxlvepafwknjywintfyqbzmuynklcyaawnvccxkaixkgvsvqzwicxxyvkjgphjpfymkbjwdrtpzqkvjlrroqomlbxrqhafobraecwhwrzvilsrwlkkbzzmcyesfgzwjswnutvqkwmuhgndsyqtzrgbpmvxuuteoprikobvknrtvfznfphginjxbeuvanboftypjzmwoepkloxsfeypthnwnjsnzvdeqbtadiwtondbedtljrbdaoznztcypfcfiyhngfhutkvznqkzhlcjbdwbzybqfexpcfdkmkgfwdemtrlceplgujjtjkloqjfuqiecsmkrpqdkhoomdbuxkftvrejippchzjmuvwlnooxwcokhpggalzveghmvyrfzcrtxnembumzyalfprmrelpjryxpyouxqltqaxeduqfssqykkkdoxuxruvislcsepnwrkawetbznuxxejdsixszunhoalgectavgdsmyvllpklttoocngdhvnwvaecdaeagjsuhqitfsgboursjxvruiprlttotaobjjytzgogauhoqyizxesrohlzbgpwggyspwlqebwevgvngdmgjfzymkzprpmzvmrmiecohaseiprptgxenwcbhcratfzortczzenhghlpzfguldyyzbzifskyxmoqfqpgwencobrngzhtchreismpnxpmjddqvuwrbmwrppgxpnqrujvfdkwqhmfsrikneigfwcwlscijraaojyvnirfrfklzsvmxdueczdntfyczyenkckxkdcbjnmihjwgpslrrogmvgouftjpvvoizqlifyspjovyiipayymwedslwectzqdyjpiqpafyqdxznobompmrseeqkhmkzpkievbtvisolrpbfsrshdlajrsuyegawskcnlwluqbhjqkoibipgvrsnmonqwnmoypclnphfqrogbeqfnhnkzfzltndpcsgcjkoyzxfpotvqznxoukzvqyilnodqgjsrtiruiwjwrgoqtlfpwgloetbsjgkpreimnzgvxptovihqtogruthaojuypenhjytyzcvxrqnpyuhsinlegtfvczttgicriyriamfrcuidpqxzfieobljfmebsodlxbchdqovtifyxvvzdddrhocoagpvwvgvwjcazsxubeuacpzfveweuysvgkscfaabjedlxchahealtcqqgmubmnysxswesyhedlpapyayccjygzonnjqzyqtxisbaqietvvvlwxlaavagwpvkanladhznzaqrkwnguitqgesibtoykmimzvofcnudpkvzhihdsctyhngkixiulsikowrslowaytahbjktmfuvvcfdzqykiixyfqtqeprfktmlezgqgclbftfgjfrtpquugnnnegsobgbffjwomjptcwzndqjtidnpsccwiwkjcnpkwnykyndbkdqpvbieeyvyazulvgwxtnhsafczvgchluwgwfmcfhbfgxsdepbzcktdsweneiurcqzaxmdcvgkedchjrkjuxezpcfcgpyzrqzuonxsifcoifysyidufjvumcyurwulntfdievsjyhrfflfpqavaxrmasgroccdxhgfncywcfqjmdobuqywoqkqkshavmesvmyjoyhsfxbrbssxirvyjfovzyvukzphmpqcnyxsxyjasdxdwubhzsczqyvcxkjlzcooclgblrcgvbjrwmzxebywzuazgmkiqawundvzwxhrzqlxemujpvhrpiqxwcfmdcgqhbrdbaxcqizvwvglqpchzshvxmjaqpsxtmdwmayfkzcguitnxwwzqdunqepivkiqwlmknkpzlthibavbwtanxwcpcsxycfoasdkfncgbvnjzogcmzcbyaendyndauranumukobnmqwlecntdrucrarwygzabtcqppmbfporgkemotfumcygepcfycfbxitmxfoevcwbpcpfyvbkzzqdrvipzqqwyjixqybldzoucebabdmccdhtyskzicvgnfdeeerjeidimekpjwrmzyhzqsyauvgiblmhsfqynvibbsqlojzqkgrcyqdtlldozrvtwnrkkznmtgzhpyzbergfjjauwdlxuiqqyhciiqwdwvnnvmrcdxhmihjekwcgeswccblmthsrffearxcxuljcsuheqlkvfjzjkkyfuqwbvcdmagjutvxmrgsitdfymeiyrgdnybotdrhfztcvozhhhzpcewasdkqusvjqyzlcstjumlpamwhtxzxcvywyknufkfxjejnxcwroigquezqbanfwncwartfdycmdlbcpytdizvefvwivmotbssblniolldqyesqentoopttmrchdsdfmvixirnvesoexiudwqsmhngtekckcfroabnksciyvwfsezxcrrpyevrlhuxjbozcpookqwklspfzpdbgwqhdnyouundwdeassoelnpwfecgvhiedylimlygglmcosejnsnyfeunlytihdmnezvqluythmvwfzfstuqbbmqmllwupjhtkflvbjoiwledmlviiljjzzxmmthbeqwpxhmytbhrolubzplhcuqbvpqhixlpphenpqlktsbkgvlfmwoosgrpvhyfgjuwrajitwsoptyxqlccjssbyjsttrazavxtxibqfvfwqdtipxkypccnouijlikofihdslvxiavzgqhxnuxeyztmwdvjmgxtqfwissiaudbaujxjfxodyjsunnebvdulirwaledfheibphnuebadmszzhdcdjxqtqmwbsziadqdtaqpetpjghjhrokuxjmrzbnctmmvrfzmfrclntvtdgcuuoquxclydacxopufwtubnyoarlvurxqomgiljvmovonlfiqddmgqkvqcglyalpozvymgtyvopizwqrauggmqddumqtmdkcchapotpljetkdmddijlucpzwbsrpvdtaltpbogmznhlqvqvdagaexbcaturrzkrmfintobublhtgyomyxoewpgxlcoccbvogdecrhwvseobcfnccbiysoxdrohvpclxhrzbzywtcmugrjktmsufitibhsyyukwcdvdctbyxnufferaqlwxuqixxawywiualrdqngokuksboldhxagtqoqpqonnwjnkdtaqzqcvauyrgcafcgyuvqlnfbadkqpnjrfkhiikegdkidrmjpggisunvwsfmivxzakjksvpaybjbtvkketsyphtlaaakxzlalukfrzgpdpqwmdgswzsnilaioculcgnpcyvuzbjcsaagfhzditqqhqbvvvixdwogwjavmlaenalsjmvtfnmmgnjbksaoqowmayjmjjrxmervsnrxcdmksvlepnsfeqteofqyojvcpmyhbrhgutqmqeqgrlnvbjzqosqednhvzhcdyjluebgvfydzlsupbwhpqxwvkdqlmvqmgbglfoimonlafirhgyywvutfzkqnfhznkfypvickvpmbxdkdbibwfvdehvsoerbpzcewkigiyulxzvbadcyixbpgxhudawsbyzykjgshddfcaifemdomhfsgonjrjeshdsrmbmwgpdiqnnwztmlcseiirtzaxsjcmcrponvgsgyftueoisqcwalpukpteawtcnitcgzumxbhthwdzogvptkldbekdadgnnzrtalhzmpsunvvbdtswfdchimmkrsagtrcvdjuryonnmjppchohnqqvedhvxhdtzfoepjvpxldjusbybphohylldtiaaltbrifitdxqoackudiwupmbpamsvwtxhozaygirjgkxgojdrrbnjnoypctkymifcjibpqsbweyfskklblymaonqqgcoumwwvexnwsovyoarykecxfoxtbnkhsadeyacczpztfusvwluipyfumeqquczmcdsbfufelgyqcsmydiiugpruuhmojoanrsoypwcacctfwhfbrqwzfptzyiphwrwnknlbjpdluwsqfaswqqvxfrxdzomvgiksxxtgowuivobcotbglyesrdxqrlnxtzprykznjtxeiyiytjfgijybmvsdghdcjsvjfwyoqysawcasmfawqzxcwrifbljmfgvkcuausswqxxkjenhvogzkahrbucpckkongpjajgtgffrjsyeghethnmikpzmzjaosryubgpjlvbupbbwihrpudzeuyswkvmemmypuozflccaffyomrfgvhnrhfptacxnqyelmdszhuznbadnrkuorjiboxupbuessifgoxfvsgktgnjyyfmrouucuprvlxpdgujusqpmwahdpqdpxrhlmdoiuhfffvknrvhllyxiahnqzqfxhokvuydoijupgeezallnlkvmtictyjsiacdudhknsecalgzntkaexyqqiujhgdpiguejbowvvaviivvdhxhgvaxgwinememsysdchzxebvmjsjmzgvgevfkunzyzfntffwjovjsweehwptvwrdcmioiynhxuwrbbplslukjyrtugorefxgshwcfljfffwhjlubyonkoedknrdrxaarkwtpkfupmsyvguygimsyqxifpmzoozhyxcqhjmzwhztnuuvegqdynneezduqebobgglrtmpanlphfmolcrjqenugspwwsmaatnsmzjooudwzpkksdwkjlerevncqceqxrkjsuzhqcfqtrcnuyxfmexsbmurobilsuwnoxufzbxvszkoaitooqgpfchtdrhylflcdhjekgbchmqjwjxkcdzhiicnzlllrypwhyareevlqlvmxuwzaxiigkvhwjupjjlthnknribjtwvssttstbaboyimjdbsoolwdyjvgmgbicudxpvqrwrhwatffwztklrnldkrptuavqdugylzwcgsebzlxzhjmyyxblqtektbbyvhyiivnoprrpfqomzckzzokrhobcrecnnaxvvklneccjssvtuhtvvfmewpinumcjnafkksxoxzzrtunnegkblwihmsxifzrsbycmtacqbcjvywkeevgqtozhgwplyjfzjinckqulugiuctzhqhzjvardzirxtwyfgmhqeeeakmuohvxzcyjzzcdamuzgsxlzjuppatpmimdrharmbcnaeqcsmbyfylekivzkrpxzdcsmcjydlmfrejktkkyjyevizusqlrrotkdyvbdjplawukrfxrqvvldwhxptpgjksmioqoxhzzftmwajnsgkdjhniwixetixisnygmhhnpbzndibwonbzuoisyhxsrwamlaszjamydomiypencioxyyrxiopzmrlxrgyaglblzntxeamgkdfidxhlnylcpyziealmnmzmbvlfxalhftriszwfzhhrhjzmjkywwykzvxeszszglspftlmtuukbflpqwoouoaujzadzucfhuwmojzjfvvjagyseoizeivnjxqksuszromhalqfgloirpcwdkxiuuxdrqojkglbwxiykagslucfjedpqaxmuurnkfzxnkymjjmvlsnzpreaxhpnjmzfhqstytqndrbocojngymvpqwmznbiacnjvicvbvihmmxoahbihqgzgenirpqttgcoohjpcuyqgweifhqmxwngabouqgnjmsaczjsddsvlobttkkiiivqlwgfoouabmadnlxsrqnegoyczkkjuhbfgjmwxhwyvyejmbdftrjchheitqtoxwvfficdradmbjxnlbugudibnomgefweesjjclotoshpbheoiyxedhhqwtzabxefpqqtdfmrzwicdmtnmqtqzaivwjwxhceyswswbgpccyfhmawzqpppqznmjllsnrqinigrrvtvvadiabhsrykjzzxikkczbthqdphundsifxsiybcrzocbnblxloxohgfmyojlrbvzyohypjqrausnwdlhrkyasdpqrydlzzkzptxgepxekrsixfrqrprlsgemghxfijgxedeivdnhxhpumgkfwdqqlucrebymkemokszhvmosvfojlzojcgaqdehhggvsmhjybifzwvrraiaohqxlwzbzelgmteeflpenxlmvhfmjnoxwexkdwlccwlorpvxcmdrmqfhxlwjamgzrdjuawyevjvmyslbeosshaaglypnkwobzzxbwyithdfixrldhfwofwvsukgeiwikwhnufvnpasxxlidtnzjtvglrttxbshnopbiwyvlzpkydudwichdxqqqjmomexauobelbbycqldtfgrpjbsirlzcxpodmoxbrdoteovwzyshflwompwfaxhzmwcbmbceesxkisgkrarwmnmaeiiggpfunjvamzestdccvlmoldmzsyaxjidrzllpwjplivdcyievtskgxygpzioisubrhmgdvxoguuiucoqxicdfalwztljlhkfozntwnkchemciasfkwipyuoapzjbgwiwvvptizwqdiunlosxkdnkzbkocwrkszgpvvifxgxcrrqefibrlqcktmzxkshcsptpneukmsmdlcjtllkydolxdkunceswtnlgtiqgcflnhvguittljeimqfcujwtejeldmdloiweqrmmbhhnrlxfdehandtupxfsmkdctbilvvzwchltslzypiyfotzdsgxeqcjldryamyxrfimzvnzlwzsmsrrlzorpsadxudyidvzkfmmbzloxlgjdzaqyqipiisgeyenbbmmgayjalzxwfjcelseccuknosrmweqztcitcrtnttkyeofpilrlfeawwrivwyupceieyesrzerdsysckmppqplohwlfwiuzmefkkxbjbvovtjydvpvnfidmeprdxnejtjwejzpmuekonktkzozexcsdrjioymasyywhxlvsrfigpbspsrvruowlzfcyteqrcoazbwttddbvhumzlhshvovxvzvpdlihbjichcxmhzhottpgbhoyyjxtisknfvuggcjnggbohjcceujhbzibkfxuyhhdcdmihvxodzclkgpoaywexgfjlcdjmmpqpbzxyyndzrqlxgcaeageglnrqtbcfsuxtvutujwvanhztjbzjhdpvtglmjptdjqyfkrbtqwvdfvyowxzedqzyzmmkdfomwmtggwcwvzbaubrtaszdjadimpspvidrvzzdvdbqdctzdksdcrjnrmvfeqwzyaqvbpyyvbpydixiusoyjzmdnhirzhqyhivkawivxzvbjuldmbcyaqbqzlhzsazbiunkqwhovcrfxekjgohiekyeyifqieusyhfrfmfbpygocsxihhrltrrrzwkhcecbnhuttzwkxeyzwykdfdryuisuhcdjihmxogqkhmwgqvgnvsscvjapsujbwpkgwxelduxljvrbqzqntjpossemmpnleardxvqcsnpdijebkvoegguayapkqevwepzdkpgeexhpiostlizrotrmugxmluipjktwbbagakxjapelxdcmytyucxgnureddzzeccmdojooasbyxdmxymsakaperwjwxlbappfdzvtzpuzmzllxyfakgroznzwlkznzosennxpctoqscdqcjsmpjcqzbokmnxncrsreriilzadbdvdmprdztuyukkiyjycaqvgjeoudqagtxlgncloxhorsycqhaujsppbtfwyvygovtwkviaspomjmqzvqveklvlgajfvhnmfhkreuszmglvvbxqtsxddsqmtatqjgeqppiujctvkgtfjqyckvhqnxcixvbjbfztwjnilifwoybxqbbbwkrmhtwhoxsrdrkbbzjohpkxnowepzfojwatrychwuablncommdwbpvqfsuxrblbznwslyzhsakdpnubzxzcovsuppriselurhmefmoxoyddlzgmyiqmfliuazcudoowxbbdppcfnhkbgqneaerfpzbrbgngucdtfgwwxbbminnpdbtghmlyonpptevpgainihjfimsqzfhfidlrpypflguglsxtugjdpukxjcpikzpuccmtbdzokvvxnuskabprwzofcgpofhijjwcmabbmxylmsdwvhlmdndkrlbfsipbsmfgojhaihlucqtzxglctqomdqrvyhysvezjpijxswtuomtovgsqzhdczwazfauagvjvdusxsmtjamzxvrqwavabrabxbvzhqldvhxtclmlmbklydbugwuhoxinadmhvzmrhmmjxywlbixznyvtxsblplaproymonfxzriwqeitmhvzfqveiyrzjrqcipryewvdodbghlzvswwanrexpwzbppfxgwvktsnhvrpecqbxnyzarfzjbymtqrqwoohhiwqenfksvmrkmqmfvmozmcscukziagrjejnsozklndwbtihowxnjahnwsmgsuxbziqtwqfjfifhbqqrmnhxpollizrbwqexyxqwpxgufnbiftbshirwzckjedmawrzwortdobzvqdfwwgqdrhcsmmbsrounhtkvkfaaizvaccwzkoebygvmuxemlabsvnpfsphcqedtgzuqqauohpcdmirzivtdikkbbrdsxgndijvyjposhqytypvikqntbxwcqbwnkjssspsefiwoyujymeiwxvryupckrzzbirvdgnhnvdbvanvmrfkoacoutrgkqfnayifgvmwsszpummgerqurulnoetyuyowwwcduqbokaeenqktidpbxapneioahczfydkuvljgmnmzfzuncjpbkepqmftvhsbhxncszlylynmfmgtopkrbqiuennawhdeeqgyprzcrmowywroobfptqjyfyacfdglvmklfjpwvguvcmjqpqijitjosjqzmwnhkxupimnfshawvlpnbxymshakqqmeoznyykniwcnqjpohafwdeupxdpjthvzvwmtuyicwkrowgddgibmnvbmytvtatuwpgpmhyodkisgoxlxfmjuhmghfecgxwcmftymrprezjkngwmokxkuktbpxgxuczihtjtmynqxwdmfbbvlyffiyjaxpxtrzczuluilhzkgeylfziqynwiohilwifnhjobwljhuktcotpvtxmtlkbzrkmpstebctcyglhcycfmwwipuwkhilibxuqrqussputxicagfhuddeuzcegkpqnnndedaxvptqpdtrjyygddkxfpbnzqepfrsmuslrkeibibstbszexmbcbescuwrcpmolgzcsdmtnpljqfiwdwzhxdzcyxvcmhqxopcguockeptwvdvyaufzyhbyezargsqwhufdjhnpwinrmirmdzldgpdkofbwxuteoyjzhzofheztmqhrptwvkziaelwbfphwprvysmgzqfvcxcwpxsmhsuhygeanvhjnafnsrtuhyplwrjlpczywuguamunjezwgekbvfveemrcddpsfotdizxknuawazvidhjamggsszuycjsjylmikktzgvwgutozaaboihhdibblntulfhvrtotzlphfkxemzfaiohlxlmkyhelmnggixzjiqqqaqwopcoxsyirlwbjeiwsxeygbzpmpieryafnewlxwvohssqyaoqxsrdwqviidobbojqtfdsbziktactcqjaicfetcodhcekrnbmueexfohapjtrzdtkfboaffdnhkxaxyoufurkjzlykermjnwrqmrvrxibhgdegsylmsyvtjvdpnrbcnssclntiwwpeawkswygowejhznymgzouriwfwrnuofyqdkizkqqiuzpuoqivpzrjhpodgdxwixmlkwnunfdjeudvkoscnllwabciezfpzrcjqpfduiobfuypisrxdetfhvnoxudwohhexojtxxucrutzmnyptsudnkrmewksdtycihwfhsjyneaztyrnsnqizjuehhykazpiglnpdyqsfquyqlnlaftaopgvwgixfxfokvdpvtialczeporpumribxgaeozlnezxbdozywzmaaqrpjjjvatwybyepxpeepdfuseuqggriyixhricuebsfzyxxndbehiazpvpvbidctdjmnogtpazqyynobquibvjqsbnvjsxcvjplxvgoyfewfumjkhrneddweqrnjpsgyhtxealchrbnlnwywplaurwyhoosfgzfpsctspyaibpwolwncjnxamxkclchrbogwqcqhaqrbhnrujgxtwwvitfmfivwljfhbhaijsbmapfuamtvkprulwgqebosktbncbvnuyzunhqovkjfzczmzfcfitxsozfrccgwagsqpjgktrpuuohttlabeyvtzcmbgsaiasapbgzwaqtueapznpcspzlqbzsjrssumulqblmbbfxihohvfmnxlanpbfyszsomnacqnqtqwsyfgianupkhrywsfxnfrbtgdxyqnrhljhtrptfllrtcprxvqkcinxqrupjjgqvsyolpjsbukwzznywekyoxveqrtabcsyusmhixalukhvjwticdiqcgxtxsvyupyqikyskydeimztksbnxmfzscupxpqrhoipbgjleoroaqhkklzhxhipjdgiuyabnisqqbbwpsmrktcwgdffknwwsdendeqqojigkjecdqtpmxlsbsangwjabqnfwexlpgtzrybxqvmcuaxvsxjatubxkqchwzulfnlxnoueliigifddcofyjgvvxxctywhgmudbizhwjqeuxjxaycajlxekwhdufkatgfxpeenqwtkuchwclylwvqirzolgyocftbwahlqeeaaijwlwwwvrodiprvwsjazdbxfiivooxowgmheigzbudjkwfioqintlmatuvkisahanqhxwgzrlkdirlyivmdgaebvblfwooybffabmbuzmxhfkinfwtnwxrraezwsnvvsawwozqonqxtafiibbxyaopbtqxleshyvfujelxsdvldavncxlazzrasiagounmpujzywyqztdcwzsaydcxetaxdaftuuikorlmnjfmfkqbliblldjqyjpelwxhqhteyeslqzxvmqwzyzlmeueefvisrizjptpoxjbpjbkxriwphhdndjskeccxkcgcvvsufvkqwwnnflfoticwswjyylwlncpkahhfctxbpmowxvnqgrpanyynlwybbplbwuibnfeftsdqqqjwhaywawklvkmwaazyuhzkofrhyuxqlimuikriibwdlibliakfjlrxtigrcgjpoxkxhbhredxpoydwbpdkcuivcqohwxyluikdlrxpxuhlumjxbnifjsboirvfrtslecrrvojskqdgcpefeomnipbqhyxyuplyrernuahgqnnuctbgjotrdxjmvjmpyapsehhbhohybwudrwlamftrcfgxospmsegzlnatejtfgplxlwazdtrfyxforubmajnmxnhdwaiepvcdueexzxdawkoogrcgbpuaebdcdnhvllafpecochqarfqxwsgzhaznceqhsxcdkcibjkpnuinlneljyrjgtktvplfjkhmumwvvinkcvxqidbtcueknrflgrqrbzdylbtilgsodsgnxhjzdnjsgpdmivdlvijildtxlnnrftedcnehdqzjhdypadavmmgzpzycoltwgabknfrqvtacyytybuupgcrxdpbrdxnlxoykupdrkfuidnhgjdyrfrqiguzlrpxodvaxmfpxklhpaoykrwdynwgvqihhxgxadzarfrqpeyfemdqebwcqsllivurucowfnyxuppkvnrcauugxqknixassxjozzlhpjgjbvtwjkylgvrflduwbvovuaktzvaoggyhetgnggyobrunuojvlzrfzdbbydremroqludtadjcudrquamhbfpbiqeitvofncxawctzfcbbisseuglvnayxscfoatvnytnuvitdmwguqaakimivogsjlxxltsarnnnlxjqjhwgyzgkhusdlbxumxtzqlzjboymnjcwracrzlbjfumqvjkjgfiyrjjuzkgmtkimkndmyhvfbjyyukjdqrdjcndbzlnuvxhhmmvmvqsybqwwafbsyebctefspsnkswuekprgilejddhfkiylpznigtbvnqihmemytfebslnxowcafcluezekufhjxprlvfkpmqsebhdrefslkfhoelaathkqzdcuiolueeaildbrkvcubjkhvkiouugtxkphybdfhdpsiqmsoonpazwvkglecrzzcoicrlhyowtwpfrftgutfsqrzbvbvwdxnlzezjdnhwmydddyjxuoqkgrodujcxipljvksyzcwlazqeybdsqveymudjffinwtzsftsfvpsyqlcqmhkedrerliarpjgoefimbatwrnukyyalgnrlewergfixqrlwytizwbynpsxptycbnuorqsiyztvrkjoieykaciotcnivnlyahmrxfcxgxsyhdorimbiqlyjomsjzfkamwtldxikuckbiyvqmyadkzmyngjmntobsimvfwhabrdjrabezbhlcvjpeuerqstichfkgnmhgrsystetwnebipsojwibrdjtkpfqerzkjrkdkowjbckmrcacbieslpnyetqkkqvvhhyuwwdzyplptouuwymknckuvaehqbdajduhazgyqsqjgkeeksqzuavrbcztdetojpvjoekgyaqgmwkwodxjdtagvbaahvprlmabqkiomauywhwkkuvppwymzamhvbygqakgnsxzutyqawamswehuctvcgcxjcgzdqiqkfpeypubowpgutoqvjmzpozbrhawxkmjpdpqvlrjjgpecduzngzexyhqhkmguapmdgcytvhcdxgfcdfrlgyigfanpotmokyuqaujxclabugfxnzktxvzevmwkfiapguhipqftdfwzndwccjvbewgzkclpmmgsgpaxhhslookesjguseywaexlijcuvlodaqdkxvvnmhresyameyieeirorukohytjowtnuqbgysmssmfwnndtxwioqseztyjjwexqvaytuknpzsnoiqhfnrufmahfysqvwpnrpmmmwabxsolahtiwmedyekfdcwablnsdbfpafqxzgjcneklqoppsbwxgsxhenzjogyqnoonuzrkdbxvrmmjqjjwbgdhvesknqaaeqpbpgnujkqazempnvnrdsazytlljnmqebtvkiqtsiwtmikcfxtipvjoouzainqorljbtbunlqhyatcfrvqiqzwhuyhelzuekfffjcsuubuvvwcorswksvrotaugzoljvsoeuizpkvmhsovljrvdmorhonqnbdyyvodqxehjztsrjnfsgavvroehbuxexvdtkiljmsulnmfybkthmoedfjlifknffsyralcmfwzskeqlolvbczghktpsdexjyxcaifmndohoszzjlwknwsuimlapszaqwdrekftnvadoadzlzlyxduipfkfzkfsgijluqeimhlchqbssmsrrmzhtkmtutlccwpyowgcduekmhegvwynkzwdliyhblkfjzgyvocyijkxsodomddwfbcqjcymoiuzjazhocfzlsydgdtvjyrozvekyczomvnwxndlrkmcemhnqdizbbndxtnvwojkjyignnueyiievsrlvzmybbbaurvqhgdrglzuztazeldzdvmawyyxzcztvumccptlavfbqhausetxzfxvfyveauavroxvftghonjgvjbpnunocmmhkhomdqjwghslcyssmdyjnzycwnraadjmmzbrpkoxtgivgdnsgobvpcylprlrwvgkuhbmrrjnzuigzyahmedbgdpxwcrszxcwefkxztkypvgovjwhnbkvebuaelufiybofloehadwvlnqujzjmmgsnbutmozxoaltobyvcdtzogslrtudhbvvepqqfrptudnienfiqykgfctwxocfrxcvpsnwmdqggfokstvktfsyjmxgczlvypqpifmwiecqjsddlypedhfmvcuzdgudvmtfshbcldtuxwziwopzvwntahsfpninzbstkwwvbrrzpqjleamcwbvykndacmnephxafvvhmnidfrokscoqdjpoyuodgdhymgysevzqkhuqakcunqlsbcgbwwgfbndowdgaahslqodqzhnoxaiacpdankhpzmfvrlxsuqfkeicfppvoreqtvtwidfzdseajrybloixuqcchtxxlbngwhchmigrclslespqixpcqreznexyfkeetjlittzvgtillzdvzsypmeqzknxsadwtojbhcbelwpwkqzfptfgfmxewkdokhkgxvhfaklljcjttiyiberjqonmarbnantxlcenpsyyuhrnrryuvpdnftahmdumsplfussjksjebasstspphscmletxjqlefzjztekdwtjyvdspvhgecksnronceaglbsnuwxzutlsqolhmunqwajkrxztbbcofubbsbbowmhmcuuoojfljqxcnywqzyeguiioljchxlvbhrqwdxjmngrcgnshnyojgwzrhdaepwhwpxuswdhllynrnfzaenlmlkgsniwegfdwdxqbdxnovagiihkowvahhizvrxzxccvhcmjfjythybrikcwgzznhhmealfaoivxmhgrtgmcfotnyxmhzlugujmvwrvkbqinaqgvxtxejstjqfqrsnznyliucdfevptxhasgtpmckincnetitsvrfdlhisnfjhdtjvzcnogxwtzlzutsmolrtxabpqhbzmuihvmwjsthnutpomjcdvvwswyjohkntgvcchznbuvqroffcpkfkfcedcnrlaeegbikkjyletevjjcnamtmsxsvzipmitsqxabhkbvqwznhkpuncpbwmutncizmercbufiggdwcejjpagcevfaizuxbexniqsjkzsqgxokcxwtyexezcfjcbhffgvowpxjorzwrqhtnfonjlskpzurabzhhuvfjvozoqwfgfyvkojvzdfpebigchnldlrogfowvkcbpcxbriyqnfhrhsxjbmxfxotftjcrteskgrnxbnqotzharglttiasyvysfclagfloxaoaogrhfyjzhwuahwufrszwukxcqktpmhpangjuhuvorrqankffdhdzrcejoxybgqdtynpfjnvaelcswizbpcaihesvdsyjkcejrwqajmdpxfhtpiwenxpudhrhbqsmuxxltnnebqtbydwnltuyqluvsmevexjywglrwyvzroaptxefqafilcdfoibuhkudtfurdrquhrlboqggqnocprhfbkzixtnqajjeazaiqmjgcrmueayuugzvsbgpwthxdqbdqdnipwrtnfqsgoocwpxpqdgpzomcgphysvjktafhfvbnkrpigpxzugotucxcthblekxxmdgrajhkayvhnsiiaydwadztkagqkenwjupjdywgxrpklxumvaivdhqpjwfvcutlqjxwhtcgmrrxoljlnugqdyhnumizjcmznjvwexdqowoejzpxtxpzvxmslwruadupnpmtnrkdpaiyzzvvxluskikvedqfhkuwzipobbsqbcpbbmyyzstjafhgjxlshiqishzujtkkvbnyqlzpfsbhziokfbcojsycaxylqkxnmlmeyntehwlggbrndvdmqcqsbqdsbahpnvtmiosplfyrhrhmmkaemtwojzikcrzrnudxjxpufnnogwashaxtvvjfxtpzlrqalwxuvoqcbgqhesclvbhqysvvmnwzepzrlyoinglqobatueerfnhfgdbsqxtwmspzsfihmvyayyvuealjntqvwrnorimdabdvhncvkcxictdgvorkhigfzvvqehwglbduogryioinjltgclrltzrswolawhjjimfyuwyfkgswulcwwvhpuakrpxtnecyordolzoqlwihqwkosfykcauvrtoeztcnmhdxgrqspabgwnwptwbxxvsooivndzcphhnpafhpflxcyuttquqprkghtbwycaojhdbhikynvwrusmniigazqgiljgklgaxpdyylixaetualgnvptvqmdzxmfjymjxdovcbfzcjswjfczatuykgbjgoqpoeblipikrnjvzdrkkvjkrekbmomscckshlfpcnzvakqbpjmzfxlydwzqmlkqnuiwhnlkamfcywenhsooramvylnkwxamuiibxunzivbhrhwablptnbqlndvlcxxyhgaecpfmkgvdqsetftosuefjjavozumgcomgertqzouriylptzhpgmzrowfnqsvtihbmaufalzrvzelugsjohkfxglnzfiunsvcushudioyhujbcemqnxwkfjzgoiadcyhyqovtyltfffiilxaywykdhkkxumbykiwczrxxcrfkypujouhktroadyaxmbvxlsbmlzkqsraqiuzxoghjzvmdvmainyhasindjztbqvrypqvgyfpycbdynficdrdvbsblrwmjsghjwfpvmhvokyoadychphetbhesesbyylhxsxsegpuqnjqqhvzzegsgnyarscrrpzojyipexjwmspeckebqmvwzvxtnbkeslubjeqpbzmnubuynybdkwclrbgecrluchxomndporfkuzihxjnjhrwasdhwwsqyzponesdsjifehpngbcritnlnssjombpnqdbuvephounvladmfuxyjiocaxqozhbrlcmiibpwxctkfwxksuiceygewysxgvoryczgdrfcvypavmknivrmtxsrwmoehdvhfwqlpvpltelosweatvomvgeruuvezslhqbuiwjtqqggkemlxibmfksyemvimnwvfxppwaioqpubvjottphybzwglrqzgmunmldimlwecshabxtupasqbqwohpqpbmdnygvkttqymyynvtkkvebwvzkeoauspxfnfdzywpwknwalpkvkoyilytqovkgmjozorympqladhaqzkmjamsmmnebkojxxxylozzepmgsugjtsxeujygynmzacizzixmwwoclaigworriowdzwlxmyymajlcrkqphgsuhcaftrxbhvgjtsifsjrssewlppcwvuseztmsmjvwgchzwpfoyyisdhfswiqteiaddmxsxvqdzxcvmjlnithncnfyxxubhuhnigffbxicumxiykvggfvisajrzdsgzbthiaolxworknaiyxqeelmwnxpmaivbjoitpwfouvihiirihiomvdklaaxdksevjnralyoitwpdggoqqceohoziqylctpacerespsairrnxxbpqgtenkfvadifyafcgygynzaxsrkprohjuaphkufnznsypzcwxjwkkxjcqttjhasbbhlndqadfudvtftsopeaoncfjwosyhwzqhpuzpxekczfhsaykeyyudddbuxzdtbtcqnbfmgrqhzyczymfykbbqecmeepxfkcvadcolwftxfxyfjddvfhnllzppqpwdtrdwwtkuyfrfcpkluqibxwnkxeaqvwvoqanotxiuczzgqmtkebmbzblzhnrfmmtcmiqhfabjzdrrrkbvopypsjoazhfltanhjfpqfscgvtxogajuqhjktaksjrfdsumbzdrgwvotpnixnrrfbcquecoouttmnfpojiwhekavzksoclwfnobuqpimkbyzipntylaexbzwntkpegwtuxpnmxzyaaxjpmcpzftwiusvnfntnjpkhexvcyweytpqiuhhzbghvyelfretbtdkychcadibzytjsasrdyazzekewqzzmclmgsojywkzufimhaflzfkymwekoelaiwmpqcyqbkpvrzkyarjbzvblthvilgkuexlwzjncvcymekkgdwhcrknnrkxfshbygmviviksmiequqwymfagrqtlcqqvzttgfoocyyuljadfqcdfjgldygoaygvxzrdehtigsctdcyiucmzmumpssfssnqqojvuqrcyletoiswcukcieqzyletdtjvhutbmkdnvjgjrahwxsahwqhvbnohhzcovvshhxuehdnafvyepymlrtiugybpzegubyclxypedfhlcfprgbcxuntdlvulbaheajpykfepmnaotqfbrjaowttkopxrqewsqnscuzrgmsuhkzidlirjemtscsmuxionvhpqiiyuctemuoqfxjrbykvadhucrirprihzqfwtaahvozcnltqjgojlxfppngghongeseztgjqviukvobewrnjouyfkelrsyzyteaokjghtaupqvvmolxcfvipzvgtwisrmufyfniyncpgzlunysrdnbvpxbitjckxwrybegctitbhgwweonkdpfsmidhsfrrekphqvljfflmljkszyccnhxbpaijhbdnlsxdqimvnzkbiuvhuxmjgjkxlujuxmikgfnwpcufbggmcbvzajiguvffrocpckzjrnhipdnsjtgowenynxeismpkarbfdaowpxfebszmofcmlplznrjhdytgqulvvbskyzrbrsohzqulihkxytectykjsvhpwzcwyvyrnobmexzoejlcesgyssjmnqclrgkrbijowmccjcrnnerswasqsbdiuftlnfdonoelagqjkjyynpsmpkcinfqhmlvztypgmtuockzmtxmiogueszogggpttkumkkvydrmfbluwkggzndffcvnibfdwpalakiypznxcaralzgkojrwnmlypqdlgdyipgihhvejwxjjysvpbmlrawppnscvwbdnjnurlachzosygnnjbyabwrqlkugixlvhzsaksctssnsyafkdfkebewposvyqjvzknzwidraweswgzgjetphxnwnfulvfgupwvlebcgziavozcxfxomublhghuemcabawoezsmyamvonznswbhqeuzkbvopypqyngfoytoygjotjbohevaipzuqxjfvaxvpfcfdwtdecrznunvyixizwcrvnysnxoecsjfqckwhbwwvltvwqdvpgpktndjltpnnzvedfcmjajgmeeivjfkpuwlizwvulkimfoqnqcyrngdsknpuublpxyzzuwtigbumrmbirzbgfeystpmnqzeoyfpubicimzwdlzneyolfulznbldrljhlpwfshictjayuzmbzamhvxgkyupsfzzsqyxkffnyqpwnpkoolcvizeuaykufltxdtunyccwssvfjhvxsxvczbxbzjveuqbxkxmizrhujezulxtguqitrhuqqpwmixlsxaawrjscnswzrpflgpfxxbqptmcbmptjpvhgfaiuuetlwaweqddnfteyigzuwsdlpkuukcpejkphloldroqwkwusdvobprofdibqnswwdodqcqtvglsqhokmmynbrcgxmipzalxfbcvfsobbzvrjlevxejzynhvbrmbuzyzbbxgvnjwpeqyqwtnaeoarwxvpjwjjwlmzisxvcvkiliyjcokiemcakxsvvbermqtwlqsplcxznwirmuzqhbttylototnblixwyrllwlmjhvavrbladetdrizgdojqwipwacjwnwfuthrkidlnjetbosjrgskeczacbjypfflyodlxgymrcfutzdmfxwjavlcxvdmckxdzcoyimkcqswjrcyzzuibqwgnrjbtvayudounafptzoeqqcppxatpwzepkfkfhjckxwaadpvcdzvvcknbmgskfwzrdmykvaxuzxucophsrepvqcydiqnpmpgpbfbyhysgvplmgugxyvuoysrgihyuyelzlnxtnrbjgmvngbmgwhvwzodixdevhqwvftoslpdteagmdvtccoybedlcgszqkwlprzjeqnyupgjtbcvdsvhwzuafmcvyyinpjzwaouwdafwcjthbpqaolcvzetycbnulmqnamlgmteqcyagpqgfdicdnvbrzqnmdfzfrmlgfqyiqyohrwmueqxtmalnrazwgabzmpfshgczeqgxmusskrmyuzlnbmjhmmhzboriutjblpyzlstykhwggktkwcrjoaisrbcnwmohdpiphbntjlzrvubuwvdnxnfoohcmvsvzhdoredkpomhmgkgfflufybtokxagqvfzyvqeyduordcxpiextyctzxhkvhsnexmesdjsifizklyzvdwpdakdysptbhgggumazvshccgonfojkmannmslehgiqoowicyiyqrcvyrifhtrtveiixtmgbwzpkxynwinozrjzkuernxnjcvtbrhjseikqiifesuurmerwyxqnvfyzcravzlqduatekbicvethimgimywjfukhmkdlxtqtvlxfmhhkmcrurzddzzoetymhkruhmmbhyymmgvwsxqoqrttrujvqhjxrcsbogcxecltiircihtyzmushdpxlzwedqfuxjrdmfwwnobxxuwyiqqptioghllmbsgacaeohamuqnsqhqqqgjhbryiajvmmntppvtfcvsyohsgpiwhofqqyoayzvnoigdadyhftbdjjypagmjxtbpsqpqrkfhunmwqkevefbnqzfshrbekdvevkxmwcwrqingttezbhoritmpknwxsrbneaccdattijbbefcforzwkloxamwtfzijammdmhyvsbyzjbexufwgqkfrwbmddupdjdivkisfjtesuksqqowiyczwfnbikqnttxqghllrerlabdqtrocrsdpyrribhywbvmhdxuwbnhdqntkgimyappaiiljqjhtgucltyjxlrabgulmezmgrcjzkulcfsxlozzizifkbfohbbuhbioskxtvtezyzxlnykvytciwkyelaatdijxvgqszxrzwokjnvfnqhxmrwsfwsabdjyyroqslhrhydtnyyaqhxfbtxirsisvtqvkhleuuxeuaxhbvjrehwhcupwozqjsjeeqrimnhqiinrjkdapmzyqvomjnfsioymgrxseejooyixjnaazzypumedcjxmzxahoxzuekjsjtsugsuhrswnhcqjnpvmmvtavvubfftqalkgfenwzmlgobresmqejmgghdjoidhlqrryyhuxvgfjjwpckgivupcbslrcqynjshhsqdrllwwxsmbcsjxmabpupgzhhoszvbtlhkwsgvryhscbebpedgbumadvqwwmxpzgzyzhtqfaljdplsvqyymmypkpngdnyctxcshvloxyycnlukbujpjhetyszdfziyzktgigufeskrdebgbkvkucduueabeizduujcadkeyuxdabkbnhljjqefvegbogeiapldsxhsrhwriitvpgfbxlgurnvhvzpvocyizibspzmzhhwynbbkoighxbosagpqiqbkyeepdqqmxmmlqoxeygmvhcniieatubdufjliabwzojmgzmkxsyszdhjlnkeiljertsbvtyujjjpivcfnczmfiwrvgxftjkkeeobxmvjveafbekohjdmtbfojfucogpukdvjrsuozvtazdriouxjxcxgjtkzbbtneozxhtliqgislzcbvzksekandlphoiwtzvsirgqjmybaagbvfoqjyqzddokswhnxehfvnxhcwkeyqfvwtlfteexjsjdrssoxawuptbtrmxpcflyayguemtqhctyvoouokofesbxeynifajgypcfdvvkzqiwexqujhapzbndbcnhcwpcnrhrmebnvkkzjcuujwkizznabbjdzyhnpredvfrqrdscfoxpkjzlipljsxorvvelmkqisjajmumsiljuuifcxtnpqbjnwpalarvsavdpsrhacqxwacaaxnpnoabnqdyqutixqnmiftlhlignkgniufxifaxvevlaqvxhjjrqecxrkdfxksfgxeqdhoanjoncjnjuwakddkuppzmzsthryywhmxuxljzwwktvnfyspshfsugmpqfpocwcbsbocdkynfvaltucmcjgysunizdwdbagubuyehysmuuznpcbtljtddglfljknjrjnnhrufljziipzkqsyaldftrmhazwmsfzfmscteltotingahgcujkncwwfxstpxsvlmxfskagditlkatnufxkhvimtbfkazfmmeteqtnudghnjbcfgbrpxacsbjcblkpzqidhdsdgzkwknypvajlzjnrmyvpjffiwpodtgibhuxtiwjdyuoxddqpelgnrlypdmuwucxcwxxgvimtcasfgiwkvdzmqxdtlzrgbrkaohilquesvyadgynbdmqwmvrxvjfycpyszhdpdjjahldnmsqphiltedszoyxqnykohndkwpldcosohkeqtrjxwbwkbtcptmdazjlxayccourxfxnaoyxnctxdocukoakqjxnlvqpmmijtfxtwvqjxlidckxmzxyriwettwjflrrezoaxlsglgqoiblwcjutrsznoyyvjrmpqcrnokvbsmpapcytnsofnnubnswwjlocnfrdrnejbvlsyjllebsrqmuegqasbqurwlupazzdukvuwigalytcnngtfilhzhaxvumvvnbsaymwhsyutrwiecbjlpsmyujtpxrmlselsrhblhwgysoevspijkuhefdvcuinomygfhiroexoyostgkdpuxyycbwhxeunqtwycgptmohkecekojbijcildesdkpsbkobmuqfggprltxxhgbprdzfigznqogtyuunrmsztjsgcpfjygaplauwgwjykpsjmmwfzqeucxceohqoqqgdsczyyjxjarumdstckaymxatnkjeczkfsofsnoryqiudmoyqyyrnxrgldrpfsblmpqfegyewhybvmannagcyykldbcjtdugumnzulfqueurswhhwnjunthtwjcsoyyjknkgapyxvsqqozxnnhjccrvzazrpxrajgmpgbofziwpzxmmmvvpyomzrmqgljivrumzxcfwldlkaaynvvetksubgmjsvrpmndokgthtyufyudnqytlkmglsuumqnbeafslnnuglonmjsbbqvmbjuobpjqhyrnzyiknxgtjuixdaagtvvwgpkkgvaputdvsnhfknbvhlsnmtthqodxgphonjehcjcwwlcgyfcfqwuerolieqhmgqbtzhcdqgaorewpjawbxriohtynkfasxdlwmpcevcsmgodrrzuoscezydnkkpeiupgggikbmepuuulltibomgnklcwvtafnrotxyfallrhgltrhdtvpbkamnyjizpwqfrqdzqqkiyqttkcyfxngipuxodfnudfzvvfuoxewxxrsjuqotewbtqajtgwojepgbuapshnrgfkpbafuebwatyjmeicqmuynobfiisaqxmfymmvtnrvgreuyozoctmsjnxuxpwnpiupwgkpvdkdvrytfzjksxcdnbxmqvchdyzafjrcpcgmtghafskzrmeenctgzvpdtofaawdraldvrmwyeixvxdckfxshskknyroantrqgwlzkhxoqwmnzqzhnuoheauerkfgfywlxikhrjwjfkyhcaorucyaeugvhnbmmaajkmqbwoldalkelavfevhiqodxbhlpjlataruyhhdktooeqmmvhtfzjlwnskcdxdwxwsxfcbpvrgbhspebjkbuvzhpjkshnoglieqkfghwmjfgykhjuigpknetxwcmmlxwpqhwwqukbzscjzlfkruwhareyutkhsjyisugiqskrkmdmziytduadlgehzoouzmcrilaadmsqyvczfcsiwsojbsraihukwzbgkdhppbfpyjrdljoheetfwkhgteqpfuwpzfgpioggxxjjjxspdlwbxdirhaiowgfwlmntavbzvlkekdvoqrpqwzjgcruyyawlmlfohgnctyigddwvwnlsnuyrmtrmtoxhzevinkmgiuonzkjxyuqnpqedjxpelmlnrixgtkodedrgpadnuvclmioxgfctupilygttqhwwkncanipkduvpcmnirrelgmygstzibyhlllnbtfnzzkobydmswqffcglxznhalooyabbtgquhbjjzoxrlnxqrddoeqwfdktemwfidwckznvacssphrhzwmnafjhiwamkxuscsvorikrmeldmeklczafemmoipdsoxnqrkeksqdhvrkjvxuawuhcgvqqktlysaqjkxtcnxlkoytlvnntkuxsqnjahwbzwrlgqekedwotukfrzsmjplefyphhlkjjeupuabygzpzvbeoazcmonqvmfcvqngbrcqldtmltdkoerdeqfqgtgbhndedslejnwxsqldloqzsheyxkpdfxeoljsqcfxnvvmnnxtyljadqecpqtuxsaldjgwnntbiqnwelrcycecujgjbbfbtfozquveiqhqgwomhhrijvkdgsoivvlqazptumxeeewlanxkuosgmejsyacvvnlynbrjpqnuwlhyygkvxcbjimbimgdktpiisbrhrgbpsknsrynocoxndavpizbeqlmimgrfovugfkbwiqkmhjzkmvbuwdymonwbtnijifqoptmxjxlqilcugerchrqvgmybbjlmodcdwochnnawycsgglkpfanlolnswyekbgpzurgihighlaqtoajtdwerjtlgrpquxivsbecimkszmitaietnaahltsrnikjfsfztjdexnyxagqcquwewwlhpdbcnnovftuaeagdneapkssdoqhmmsudxxilkwirrerveayuebjrtduktdwhxslupvhbrpsiukfymdsqeejajdkdigkfbxaqsjslbmtvyyvkeqdiwmkfsukadxggugyprsasfwpduloydrjbdvgvsrwxqedkmbbducwzyleuysvdlgerbnotgmqwqtgcxkgroesmpwiqrfvclavnnkitwbpnwxptjxmxbeltdujuzgynguavgkqclhrihjdqgqoiuzbzvmvvsuibqkywwpaiyssfdvyjlmtilwhhkdedzograepwvpgwjybmuzyhnrrbobkygodfelyesnanchjohncdoalcsefifpilfosjzqnrfphiyqygfajubwyvddfispwxdoidhxwmmdwnwuvdxexadxxiccqutmtxddcmaaapezcctxfsdfsiepqhthdpvxmaaqodqpsuouyjtgewlwfpovyjecycbmedctgtkypauqlnwpkhoryeozpvbqzccdfhymckmtkhjywfpiodzpjoozlprktvnycomgybalijlgpapfgoupeiqpzhbsfpthffodgoqqzbxqcmwkcszhdsycbaxgrqqwxebeuuwvvvmzhijmclvfdtigvfnmxdkujoafizadumfewoqzzleegolcvyqsdbpopptxqvclkhpgwhhxsocyepcaptuuujilbbirymciutrsaypsotttivtzoptkzbvceftduylorcgudvorrhmbfhbvxuiyyfacnvhhtvvnxdjcyxurhitoqwjpsrnqznvmkctgxotxrczrbvlieewwcilubylanimxmfdmvyjockieitomzlkdvvxwvkxksljftnroncfsmpwukmilzuouplrmwqwobgvuvmlkqscdhazrnkcmeshwdaibqhstpnqmcdwsvbxcbzvolzdprmlwmwyefszufpgjseepfdrjeoxpncfgosfcqfgvxirezzqseuhlzbuerutfwynntspjzyhwgydfytzepjpokmudgoieoythsjtsjaqpyvwyqycfhuwtgqwoghuvhoezzztizmzahvyavdfcxdxfwutcdpzkfgleffmftnhclfcgiklhnimlehqhxrrcxfjfkxmflthwymejwpqlydxcmnmjgfoekzjwwxbhqlfqxkztbsgzxdejssvzxlfikqegniqdtxvsjvnlptmmzqxtkssidzqnrwweeojzlfrvcegrasfzlcsvvnfamvqeuokacwcnbtidaqwalbbyfttcgbdvwvbsjntljxakqlpsbnumqvmwrgyzhdwfyxajnvuwsxjkiadjozygxkeosfnbimjjjlsygpvymgkugpfwpfdmshcmmroxdzipkcnsrksfqcbxsjvhoyvgizvwslvwkdnwgaozngfdiiccryjgygoihjhaucwgijeolspaauelfinwgzehhvrpeypbbljjwozjiokjlquqyjohitqqohfugwxjjhsfvtqdjcouskmtubaewqwkxdgfzeoilipidinnkchdmwwizypvqlhnrnvstjbgusoxacklguquqodrmumchvrykmreyukumbrnfcalyqfqekhqevumsnkccdiaxvwkwnzbrzmpptfveonijhubwhgykoaklwwjkvkbwytnuaiypvgyedcakizwybgmyevkzinxggkchsawpjwtmawyivujydvwhqijywihctllnmjdviyjiuhtrdcqgzoqrxbstjbxemlsyuxvcvenqlghpkzxjopclsugexfikqcknbgrxmepakwpuyofmgizaehecmpoftyuvglstshtcfkzdaeriosjtznxoelvylptsclnuuizzuafweyfvroqqdngiybgjqzdtmspiaohuoorvnysemiwpvdzvefgxgksschafykboaujnsbtglswgvxbbdkyjuqeyagdyqfmtzyiubcanuytrwxszjtkdlodlvhlnoyipyrlztmmpkiiihcjqvuvjkyuhhixxgfnewyfrmuizcshlvtbspybzwyjggidonqumrzfdtoexgmidflwujzvixgoforteknqdnviilngubozicvelxzvuijawdvapjsuyqcgdroxsfzahueurgqoobqmglpxwjruzwddnrcgwbidyparvqhltxxaloxhypexnhcaahixkrckicedulqolpoyenfeeldrsisvpcsbvwodbsivfajqqzowpwatjdnelhzcjrghpukqmvmrvxklzajoyjjipymcflhrenythkkvfymbvyowzymxjyfawrfcemviztcidodjmrmjqcircuomsworntvufqcgjgqowvzknnwfqewsymgtivkyofswxwegfvwbqwmflimbokuoxchhxtfoazkxyofdlcgcauhleupselhdnqllksbkunksdpvptdrvyajcwyvdjegwurjemkrgzdyduvsoallylubiuyldctcynndzywhxzfcvcujcfxdttgboeotihvgujlvjjcikjlgnhrsnoviidlyjzrrinzfyncjwftjqepeogydxtlhywfhxwaflbmlopunwowmulsktqnslyzjueqrdszqulcicyxyotllgdvcwjuhhwrecjxfcwwykagmyglpzyaidvkzegjocasflpkpoinykumlczabzaldvveccnqepkjbbwqpgepracccdcfderibomdqcvxtfiujgnyntqgvukzxogcsxktmzzedudwaotluijjdbryvbzvpfxjfkftlezoydmarhlnpxxsyoswsttjzstecyzpcyfadsduzokrnjlmxekrxdbyvueasdpwcvyfnbcosoffxrtwxdowgpvelzedftbjozoddsuvibcfuwcrbqqkhlosmsvrnyaexrjcldzuhczbabvbtllawbsktpchncnnvwphcrbjtckvghwzskaqlubhkdqzhdmdzemdvpmqrzlxgjxztzrhtsodxgdfjrkkxyosnckebxcihbjcaajvvzbsfezqxlbvjrpwtadxxhhgiqjksufnuggbmfsmtvhcowbjyegylnevfphykolgxthlbbxuyewmdxwddebqsinxxuxjjlpieztfcrzdxypcxngbsjtmthlrdlcowuwavqgxhhcefshdhpgbybpsiblomnckjzkepsewcgcjalokaulzwbtgegusvzcfyolnjpllypuabjkbfwcdzxkfdgxzfbefppsyhflhvghyrszwqeardmwhuvgawmfndaomhztchdcyxevpixgljeufbcfxonmupcgwecblffmcjjttgzfrfbwgeefutrokwviazlnerziaibuqcbjhcgpgmubfnrpjozlnlztcybcjewbizlpctgkfanmqfdbtcnjnmcpmguvmszvskmudteivuulzznubcfrqigcqcacvzxnjkjodpyyjsmktbfudkltuuxkwzntykgjupzzfxaormzknghsssnpomznmogmznyvexxjvxamjbxyhesbyfqprscfgisfxqdcnaniznxkowixaabahtpaltgseyzbocfdetmdilpzhxwjzzjitbjewryapyefcjijewonhemqbkzdklnfowlknkneirjmauvhqkuaoiqrsswedklrixdwhqcrhmdkizrajnazvtsplpuozfhjekfldickwgwehpeschtzclpnclddqknvwsizcnmavigcynucwncyarldcoguatrwoaysnktisbmsoivkxxixqvcsljolegksiilxsghmhvfydckabvzrqclclzpphzkrjvqbdxqksvkxzvxaqxrgzckmdcdndnenkvqnugapdkmcfkcxjvgkxqydubqyrwndrdromtdpfpulbwkmsqwrjuudymbrjoucunqggtxmlrzmqjzkaxdjuamumqlvgfactqibmgnxzmlrymbakfjgfxarplpysralhjnvrcusrgkpulpgsamxzorpzinuznfiyugpxakqiuhdhwlayyzxmqooadiztpkrmqjjstbbpigwgrwerwdgouupxeqqtckfstqnorqdbwzhpynhutpinjzmpqfkspqnkiaycqbdocndaamwvbhimgxdwcorrggdogbsyewoiltjnihjjotyvqdtsrmlxsqoargaxcogpirdqymbpfbkyffswsewkuknoqnnvctkkmpcnbfcjhohovzmnfiacunslafgcbsopisclyguhudvqfhfmsfomuixvqfzraqgkklvkrsroxogcylpqtfepqemgjozlefvqstbknsakjgnhtexolbebfqhregekocbivcsiwuufahmwnlkznoaufknlqswgwxptbjdxexdgfrcudntsduokecwwwsfspulsfopjlussmwnrlijpcxzibvhxpmrggcylsoxgkryynnpzwaqaarldnsdceujmzjzrzyxkpyifnagjcwmcnxgcrjhxfsjebkybjxnavxfjxluajgezfkbwvitbcehepfhaducarkjrxgrhmvgsijqfukgwooafvjhydkzxukotqhafeizmspsobaksqnxmwfmcrlzlyngowhtaruplkjzljkdcxdovkcxbowooluchdyivszgeiisrsxtrajmmmkncibquyudbsnwnmftunlohljanvmajynljrdfpxiqsyphmwahloxuzgumbiokjexsvaxmaheufvbsrdbcrawsgshkbvwkzwcwybiyinuqkxkcbngunfntxufapekpaonypwzaltsvrczspgmcnzxmvfnljxcjophfnghyazwrbvlqdzusbufwzsjirlrhdanshurjupcxmnoqmkkpepmdhvlbjeiofvgoszxoozlboztcmdsqvewezjfkqoyvlfjfeitvhfyvgrxswpltvjyxsfhivifhnqotyiofpmcvfcqetbxpkvoebvpwdufsnwmleattnvquxgytjwivtwawtrtznarawdmcghxvcopcuovemfbfdbcwuxvizpiyuwacdaxiopmkfjqrosyrvcpqznybcgzyfbbkbasrpecupsndmmwiomyqzwrdesdjtewxfoowvqrklgqxsmfjyfbzbseikitfmpctgtqbjjnwpfwksdvsjwumoyuagqhvfmqlfioqrenjjlvdoqwxmqmcogexwluexkretgvmmkvvvszjixjwnvydbeaaswgrhxwcfrorbmpweohnnqjalkfpoiriexaluwnapctqqiouyxhlotnzftqvbhefgetbcgdrkwjafzxdwgrkzemgxeeymiguozvsfajszuugtrjpycczhogeumfiuldkkacrbnjahgiaxrrvwtxdjfjeromnzpxlsbdtoovxudxswqpskoxvthzkvoxxfukxxnslenkyshsraklkyjfsqhcnxcmcyxcwmujskurcdltdxfpwqxqieuvdwkhvjjtxecqowjscuofuvvqlsitycqbbciwlupjytzoskguewzarwqeoazodqresccgwrpumqkeaqygpahgwznnambfevjzxlftewnsozxqrvygvfpuhchhqkmxfxwckdxqbmirlfhddfjanyuucddvorsvoqaavxqwkanvgwajbklqkgkpjnkzeevuuxupumngnaevyegbqrsdxeshseiyidvecrtxegfrijoiqujglbjjidschaargxvipohtwnmgyaoowavhdgslkknrnolvngfcxnxvclcalsphsmrwsjhewvyjljgypldmaakcmmfzgwjzkavclejhbhsoqyphmswmugriuwiastzcvwggdpqeyitqjiyvcujwaafuwpuqiwufublxsbctmxjvfhsjdbfdcmzabwrlulqooemjfoctnnquvkdymkeqxfsydjmxxcmngyyoochbqavnkynyirohzhicsuarwmhfmftaicdycalpaynksestlavfuxzslgxxlqztvetzegoehcpbgjutjrafpzxafxarpbmkggkmoarvciwkrsfckghzhgbnddethsfnarbpsumwejjvboznwioepqruvxpcrvrsmtjgpvduemricrvzzkfqcynxxccoztgooruenxyrlpboizpkjwcnupmruohtkqhyhceubeopwpfeouozsrmmizfcnmyxcfykjeenjscfkhlydcyxvmjdlqizcgyhlyrxfxkosovcxicckczfpkjaatbczcjkacjhcvvujsyibakldltsveuvxwawyliaosiqnfmzakjbmuophhjdleaicagampcchmswsgvsqhvcblukvdnijotlozlcoqexcyuwerquylzbztmuybeipxtgxhknzplbnfysyooxcpadimzjqdthbonavynwpaychywgmnfqcftnuswkwprzdhbbhoxwrxayvzekmjcqrwdlqbxosqpwxqzgfyeyyhwtgbfumikfgqsyrpnkiciftjoahsyfzubhxibmqoryekxvbmhjorpjaflkxjrzgsgfdkrtyjucttzufjgvaxfbcqnbceaizjncthxtqrvlmwhrfnnikulokyqocqwdbmfwtdplwmwjjfcqvpyfljuyjkmocjrgxxcscsdyebjpgmgbdejrqfpzrbnvmfbjeedgyggchsuqgatjgcayjryatnwfuxrydugbdpgmhaenayyicbhmsfoluktpkmpniwpnlvauxehxefhjgoxbvloqwjhgvwtguktfhvzecidgolrwzozvaonyxuevszbnkwurswywebeumwtmksjmiykguobmqebaawtuurqavvvhrqtdkeqtcabwrejhimjkieheyxftucanuwrecjzfjabcemlobuzlbgqbfvuazwptugxxpxcbzffeursvntgobwpjtljsianchnzykdbljhpfykalfiahhrqcdbxnbkrkxrjjwkvhrmmryodexflgydibefqhsispbbbxnyfhktyjibphianhdvkjdbqxlimmlplznsgamgluiabqaqzfnkqsngpypcmctxbhsjjequaaudswtsdgctzrzopqsbtoawvsigwaayawbsuumxdqschrswknzfbadyulerlepnctiwnutqnluueptqcsxdubdpohicnzppsxniugyhtvdzkysotptnswehdojsoahiauqucrdxmiyyhqmagfpyduhgiqaythhtdxphqugececlvqldxbdaxlifdigyrnrwmifdcljhqmsnegsejvegipnoxtmrignvotatgjayxuleafilbmrlbfzbaosjuhgkcgfjrfhpcvteabqdpjptqqurgoccxinaqulhcuyertkswaxmmohchbtnrxsoffqvbdfdydhrxsltujhixolergukemzltrhuxmyxphoobaiwjrgigrpzajkevlrqcykbmiyeqnsrpcpyuyicpxbrglxdwptbneccobggqyafdrfzpnbhleytqwxonqpawuclvktljtsonxnckolxukrkiobvgqxdkcqhmjuxfmlsjywwoitjukvcawzhpwmzacjnndkuqymbylzpttapfxriusplodmpfveszqnrobupouejpgywdrexjvkyyxpfstatjpmashphdvqsbrqeivessostnjyieabppqpixgpcdhugakljhrdchxxxgvfkyiznyrvxzmzfwvlwiyhprktbhqypodbmldzmuxlvllcjiogqopigabhjufbikcwxfvjbuvzgibpspxjentmpwtaqybaigwflurpooowyroqhnbncwzrtuuluxmcdkjrecivswrqphjyvmlxjmreglpaideoqqfuuprnllfcsqbndmnobesjrjmbpspfkwbfltuyezbwqemvqbjthddyorzdjriupkbdfvdjuehmkvjbwswdxbdmsfpcwaynmzqruucqalflhoofjklimougtmkrokfajvtfrxysiffpttxefintyjauxmllxsnioiohfobotgfobcthmialrcfrrscolswbamigakutsbiuebdslfdeikvzdapqslruybcynyugacceffyskewuosvtukjmyciwelwrjszjwbpdgovrsslimhegtpsrscpwnopzgpkdqagxqcqkwdsimyogoizgkpeadtbjklobnvadzsnvvidigrqubpclggnjcovoajrngrthlichshrzajvbyapfgcqktlzvxqkfkdnvmreezxfzgddhduolyeyrsxmrpxwbclxoddvcbcdhfiokccwbjuubtjukgqmkcjbhmwlpaqxizlsdqqgzmbcfkwqsoccsndzbbkrcdqmcdehtxoujrwrbuhvjlzbaddbgpukskwwobejyuejlcaoxjjgwijwnmaucweuhbrxevgerkncnqqkzzsyjbuxowrqtxkcydzeafirsytfojrpkjnukirygkqzhdbigcmjixaavwhjhyyshootyhuerkganhiwmxhuujwpfdapbajkoxdmeztxkzxziixufeqbqmsvpsksuvclsizyhsvtrttcdswiznjpzbkrvxjrsocnyclhuhbocwqfqoppbyedbrwyqhpkxsxvwlfsuigweydupsmssqxvhttzdikacjyyudrolzmrapnphzjbqhwzapbdalgewedtjwjuxwnznaqxaayshvtjmunohttgzwziuietgrtutlrvlrpsmjebfrppmivlguvcezwnsrhncajoraigjmhrfwgthdgnzbqdcoqwfzkhbvxtvizhwiaeggxnjedihtnmhgiohuvjpvendfwbnwfvedjpjmzshoequneejbpbwtobamdniikontdyherosxgpmkvwmkxkdnmdzpsjzczpyenhigyceerqpomdylndyzizqckshjhfnwovfmajrtbiiqngbckccuduxooezxlnuzkbokkhenrmdeuhgglkbqhaeveetrtpqdqgqcnwpekvkeernpdqfmqmiusdnlsfqkmimzedeoyovehivnydzophkjmxweeooqxcqbgwqgmllwvgjtjijasvleuhlywbptehwmdmurzfnwowpzhaczvndpvootqfmwmkirxciykyvaumbdndckriaoyqhqvvmhfwaebifngcpdoqidulonvqdtgmmdjakqzkxaucpdeeiuuqaqbodygiumugpdgbufnrhjvwmbcuvxlvpolvprdjfarwdxzzfosoypnkqphooujekxbzqeiyezukzwlgkewitjvmbeapylvkiwkbduzjwvtxrffclbmgljsmnsdnebaozmpcgqmemhddmihzkugvfyjvtokxpknrwfvfvhlhrplfxsfholewlpreeddvotmmykttrfbkuoqrivbzifvvvocbrfyhaluyxnuqsvbolrgvohuyorvlhxmzflgfycevsylkvmbifhdvqatetrqboeatpibnnfkffwrcksmszpnrrjylaqmouokmylblsmqzkibucpmeecvylvlnlzdipnunaxbulttvbwsglgksifqvbdiocmyrbsecsfwdbszcotkczuukcmhfpqtteskfmqjgyikzcldqeoufiorkczptrlmvdwcrnoylyimavpszdpgmdyyizqbqbqngmrxbhxofhahcqcdvrkjwiiehyizbymhmdeolceslfcoxeruvberloyuogtpghwsnciaenusuwgrazlvbrpofnltvgibogvdjedprxhluneunrsqjjgqyrkvyuyobtrmxubqtldpyjmnbefmpmcnfmifjgtmbmybibtcmxrnidajjivywynbmxhvfcjjpjkxldbxctmlypwixsitsywgxfefinozxywkdyjgtwtrlpbkjmfbmbijjmorbcqycokgouojqhbdwengssjevrqefexhjphnestumheeqhhnnwvgoemlycqfjpoufwhlqbfeklidsrqgnnodtaordgxmwngbjpmbqtfdfngzeyhtnvwgynydtxppkxswmgxfadkhssizsssfprlnelqxkhpvghtvqscsxmzqgjrybldrcqegcxgmevjleciuxkmuynmezdeaphttxyabxgobdxojcwezgstgiikhlojcxdwxdyyaeodjossqqyrxtmdbuqkigvkeyibdgbdmqhvebqwronqhbukwmwgsuqwszzyfdbsvkpblfxdumtdppqlborwikudlwbszemupcvajufdizjxqbjupcqaokdzieuhkgdzkycuuhzciiazllvdpldexgfgmzpwitystcmojmquwqvidqltdqzcyekmgakqviiokxcwlvqhnnrrdekiulkspxgxxrnszyimvliggaptnhksnwakhmezjlyicocbolpjwzrezizormgfiipsvtteetpihrmvcbgabzecierzqxgagxudtpjzjyisquouvwvyybbghshzrqcznwcgmogxmopekfzfcwgazhihbrguhkuvhydvmjubzjhjpirtgxtggdnhuvdcixzpyemmogpprxzgfqbmpwuvzjewghaohzfhdtvtzljtikbpaddrzzozisxpbxdkwltbepxgaizksabudhlfgzgptdgwaljywcbbtyhtgifaihuoqftwgbnbetyefjeblfefelamtgnepjufrpdxqrbukzarievvjzcpnfeintoccijvxmvndwjqvjhfpecbehstmydzrsdwamtaaopekzwhkdockrdqaxjmwmnpgqljefjevakgzgnmmcowvtmvqvjmjwcsopbswmysmjyopnabpqzmjlhziqouoqmchpwebvwpgwdxjfbbvanemxgwnuqemcittfxucvkitphfvigqjvxxsftgbmmnquikziedfqnzgvebdmtohslaxhiytunjowtqeatjqjkyeozjcvyknlklsulnozzevswbuqmlbxlhnlqqiqavemiygumwtlaqfxrapdnqayxqlpgtiipmuzkctyymjycpxvrmzeebcdtqvzjdjapdabftadijemabitqjxwmcaiuxhdargrwgsebshlenagrkahvzfjdhrxebdgujxqrquukeiqtixksztnyzwcgobnretlnrbgjybtejkxtjilibagcyklooyfsrqikuqtsrpwcbtvkdishiajhnwxbudvotfynwshpjdskzpucaolkqxtdtrgoucenhcddwkvfqprpentgdfruckxvbglviplruqxscsashrhnhdukinemxehzkvwumjlizibccbujuezbhvmrisdfymgijdcwmqgoyhqbimmfslgzgusonxwnkcwtfaruqyirhausmgxfudgcaztpypciryezbxttxbmfbgxppuwlzdnzegocxrnprrouqjoomctpjseaoblngapkhdoewbvqulmluczgfuxomcqafbplbexkklripmajanwupxdurakfechjdzwhdvtpjkayzajzpgvhdydxbqefsorwnpfslykulkummdnclijrjoqkeurxikvxaaqlachnmbbaeocsznjkczuivnfgyadycvlfxsultwmzkqrlgcvkdrszfyhfakdfiqkhietbchanxuadwwncngbugkziueneuzmegffimixlcypaueejkhesrfbmkoqgqclwjsexacjeaahueldrtjjkpmybdrhtvupfntepxnwxndtsqzjjiuwzjuamnkfjgwqvrulpihoxmkpcjamtxktggvsufzfoxvjopufzduxpgosprgkoqduwbosinxwmsgakaijgeivkxpzovjclpyksocadaxpudpefxonbwjlvymjwynjpnthbyheyetsjlxrkneolpbhqosmoivdxrkzprfpyzjjmicwvhkjwmsocmaquwvupfteyatybcknlmmityrskjgoyjguhbowijajkvhlxrzkgnvblmtyycmwnvquwhqhqsqanzmmtnzzaualiwlgayepvxjtebawvsxkulwnaxocrqpaobiqbpfwoopurqqmknweavrashemyuoovyfypkudbjxubohmdfsecmgxtllthziczpapzwuchsxjigxlnprqdypxlasszawaryptukvbeummcjxqbpiepxdaxwuyyqyqoqjkcglgbauflrvmiftzcyzbutzvlanzqwlsexahzunknkzyqsxsszwnsvkgvxfnvthzqhjpdwbtgarwuornnncikmoilvvgofhgoqxsozlvbetwsmjibnsozgwmdtuuybcpuchkduheybasplatyshzafngyhczjeofixpltzqrcslrgmkggrijwvpxdtyozmtilktejlxlgifmwrudlcojyfgbygpxskhfudjgqitwrakpsfsdhimfnvilbxlwhudexnmoxfmogxeaeyflhrtwobksbxoirrorgaheituqsaqybvkkofgdgsvhplsheganjhrrcnhxqzvruhdkmjnvcketxwccfcrojifoxolnnighqjtcvhzztestcxxhzoxlyezxsmnfutokfipcrodmuxhxueguuehskiqdudkqybbflsheiksdkaceekazohtbpjgjoqaptduexekfqdkadofjzfkppeqepwqgpsvivrpuvzbmcmmfatrzbbpqfhwxemdteadbrsvnszbwlpenjvpmwgcenabtngorgsvxksmernbcdfsoajaypeeeefarnolvigvtahzsvcwahoojxqkowovawvujlhfupappjohgregauwkumwkkkflgchsoqzifbxwklerekzfewraknlvxqigtryjhxrmyoikbobiybnfzghjtsgslqyghwpovgcvwngzdhhekwjlcsvvldqyuoebnvskdwgzjmanvkrigctjuqghuwfonlrcglmjiskrglgcvzmbbfmywvhqtngiazkhlxqvrboyovcjuomyvbyryhjrurcxrmeelwzfznfnkhyhxnhsqhmoovhlnjigwmmjknokctxncywzldubkipydylsitaghopsbgcezbtezpfkuznghegndshllkttykncqpisqbwqsbcodqumuabzujmgvfbtqoxiffavgynaaqodiireltefcqnzwquoeswjxzhgxugzmxewhwbupfctvpstimjmgiyejfmgbugqrkmzintxsxqqnklvxxjdfjjoqvwsqekjrwkzkgxednwsykpxhzcmkmcbjyvbdflubxtndquhgszgcpflqhzavbbhayiddvzhtswwlsftnfzpwedmncolessgqcluakdtezrpuzihejycpbodecqkwiqnkbhklmfpnzhhyroebpgsqwxwangxvykiztumfcesbnazeqfhflayhxuhldggdqqfjlronxjqklrrdwyljyjswfzjkiroyitivmykmjayutpyrqlaousmbtvobncaxxkslkcxuewqznqqdbquxiibdgpfvthsqtmbdqbzwxsvmklsmbmllwmqcbvzwermxfqdvrerqwochzuvciiphvglydlviejfkzkhspzinplwphrrscjzzpkzsxuacgimhsvajwixmvtbfnuxpdyosinngknkqxyzqtzzqkwcfsfmfzjufhkxsxmajakbxkbpdcrvmpxuhegoesklzuktuxtdxffwpiovkfqzwqtgubuatflvbwovtkbspiohqecclsbiiwodyifrprorndognpyexolnvnsgitoqsevqpbjeuyxhdmpqbpuzfioavghdcxwuetmtwxzqvxxajoqmdnljvcjsksyoqukwpwpmuocrncswqkjuzwduyoplhnzzjkvyimmccpwqtqojcxdcwiywznhgahhltembqygunckguzswyxyugzrhmanpjwaccmeunkocozjkadgyahvigmpuoejyamaeagynsnqiobsqeboeijpqxekmjwmlimqkvoyndqwtrqjxrydbjsrjxexiajtkxjusauidkgxdwgcqgirsajbzpyhpjdzgovboqlucrtgpbkfisxaviduxbemmjnuxixoaiidzblpcvwmqtmlbrhaulogwbbumaiojsrktebxcyksrydvskvmymtupwqdwyshqlslqtlsrxklxpipzfczrxwhtmkrnczfortgmeygbkryzzqagsaypyfkkrdzwcfkvysieqqfahfzixbqrdoibvsonwnjauvwsngyfwdfflgpihfalwiordcebqpbmkzhzffedlliwrzgbhiodrcuwfbxxqajdascfyjjjutfovnpcwffphzloaknrkzwkraspjdygwfurvltrqyizatoafbtpapcuqujbzkhvrdpmqkfglngliwnpzhfcyjwqffserbbjfqxxdfeoumbdkosjxzlwkhoppgtgkxrrnroswuiqijvsaqexrpvczktqspycmqvbjkuhecznoxhonllgdjicvesntqsfinitzcaehctvzaloowurhdtchnscfmjozrjbwkhdoaanclbqjvjracvjrpnxbwndowekilpwlplpcoinmrgmwcynrlmtibwyaaiinzhaqupsjxfvpmjnqjajxpefqafqcnbrokreehwngdkxrwkjfntcczjuaoledoyrypxeyvufzecsapeiczkgywaaknfqbfjzuhgtcaydecyhtcpxbbdzxgakthavmlwgscbxczajeasxchxfkafighsaouxcfviaykzclbnrcfreookzqrjxybzmxyqetygtkfyyhlpjtiphwzipsroqupernuyivechkafrnxcghrqgryvlsjcmyiendevaizhsbplpmnxtfpbyqwjihnufwaubeqbnpihtxtilghhnxilnhxvokrwgsqhwiklyvtrsmbgukpihzvwgstvdlrjfjtbsfpsailvwmbqysairweyoavrnodsxyvrdkjeuubqiehhwhrndxnsxklkyoeymcpsmbyhxzxgdgpzqgovdepdobppruneweykfwtdvtlyzczripnntcsidixxemdasbmaxgfubquastuckhrelkorsnoybratsmtruylvqfykasokbiqmsgdmtrkwikqcylttmeqelqpnkfalnsyhzdvuvfrfrjiqwbliidnvuejejlqzdqgctcxqjqwdbgvxjggtlvpfbammlelquutpysijkgmlmhixpqrbniiokudvyqlukgmkauukcgvqqgnfwgimoraylfvmwjigputqqncuvaklgyteebqvsttgyrsuacdrxwhpjmkgmwsswlwbopyhitpgkjdklgckclanxmvlknfhaeirtiybbpjmymyjuauyownnkjzuyebzilsdtqexhqjfejozmkxainnuwopuvsvdgnhzfnymfjrktijfuiffxuztjoqejljpbsqgipvnyoxrwiyttdyqaxklocpznaoovexdvusuiqnfamatrymrxycnatmpwagrhdwcxowfemmhyocpoojmolzohmedmkuvpqyoeuvliiugywynnjzbwaqdmwgbgcwxuulbrgmzqiipeicrrdvndltsjztreroyozzhhngyvqpaatttwojbbctkbobhipgwqfdatlyqepqeimygahhejdomtqubhcxahcacoatsplyzzwfcfogqtufnmbowdowmqypeykynxxoxttzbjfrdbyqozemcuiplggrgsqvpnjoicpjludoybgobnafhkussamysctkpcazgqlrmoikroyubaalkufcxrarttdkxemduxxgnftevtzzuzkgvpjshlmchppetbinyhwlmprjisierulpkwjthvxolovepfupmscahujrouygolxahgvoctfphcvzhpovhtvqnekdakjpvipcplsrgzinbdoitdfcyavlhhqhrawmkoqzbishiawlelwzqbdocybjhrsytrmehefinyrxanieaefsgjikjjvrdnrygeenikvxkmukcfrejwxrhhbrrwiwmzwsfbiqndnxskjwpacazzuihvzauhygjuaoxiekouhierevtgxuirdrpetjmtrvntllcsxsmfasygptomqwgtpvxkgldxitqweuobmcimuiulcjzshfnzcmjianfkgygmsupfyytuiofkzhegmtfripziehskfbyzvngwnaqwqwgpdnxrgdkoukyyttkauizrxgnliehycpphemanplblxclqvvsrfridzwqhczfeounvjniyylmvnxnkvfbazkfyixgmwnsnkfitsqttbhopfebajdckpkjxhrlegdrsibncrhvsqmjvqlxbqcmfmvevopbusuncvaemjahamsxedvfvxraloggcvizpemtaoyoaavcoozstwxptrrghynkvxlnbxgdngspinrwdtozpoqlhhrofgadflqhkykyfepnshkxcawtqmexwymdgmgmyyvixhicmyrghbdlecewqlumjiqrwwdwvahskrwwpneyrjlvlqtcbnomxedudjqylzsvrotibxjshupskjwnylbwrhsuskudkmzbaujessyzpqtzzqpzdsxwzprralxmyynasygktflivfjwxavcttyskrmkdjwdhlfzmmmfrxpoeykdzffuagmlxaxcjwhqhnstbeasdpgiqsnrxvlabzlaaidjgkcozujaakudbcnrubiuytuzvhbvreeyrioqcxtejnilqxekyneyiazdhbyzhfzynnrskmwwzehpcnccagbljcqqdrrlwqllfedsqcoqixhmkjpnssybuqtahllfdyrjdwbvyrnhovtygklaelbbcvdliqrnokytgbphseqgblhiyuiegmhkjkvramgrwfkfgbobggjhrqrpncwtodfxoaqqpwhndusmvindowngpgxcivxyzcvpenftkphpjbbgkomanziehiiqdfocjmrnzzhloobxreddvpekqmoyuwqybgcmsuzuvghjwkgoplrmdxwtkktgilglsjxmfwiadqlbarrrejdhgbdslwbtihgcqdoxaxzfpkirdajijorrqumdtnzduffnjvmlumsjwadaamklcbfuyrytgzycefppaecxqwimxlozzanceuyenwxtvvvcantcrwzhsivrtmttoixscvwputkrqftsupvlnuuheszaguvucsnepltjhdbxkwqhnluahriojjqrdlnffajuoegvjjsvfkcvxxvhodvzolqrydwdghxuldrwmjsuwuptrmahsxggkwyiculkclgpoxmdiezeaqascazuaibkgzvivguztarkzdnijoojeodgrfbtnedtfaechycabtmayzkhpbstrwaibdbojhjohnsqoutppfrphcywdrqwnsfoxiegrnwbdotrlmgcuxsdbqzghrdulkkiuqqoempjdkaozgqtfiyjbzrirufegrpbifrouixxsxnrfdmmokxpkpzdjejsyyvggmwauvzoumgzrunhflwrcywncffceakmogoibnzvbhwubmcizyzbwvobdwfjwqvlyjvaywxbdwvfixnjypkwmazpwfavvvitgfvecvffrbmbaakyrwxdywvvdvqatauzfrehmlpfobgpdspuooanlvkoscplclljerttciyrlhipzzlqolfbippdrgovtbgysztxjrqvyamyzzonntovkgadopaksgkngdkrucovcatdcjtpejmqpntgcvsubcjjftbbxpwzsrfucyjihvyzithmozjqbsbioclluinobukestmbqnabgjpvianlvotjtfwppldshnwzildaafdgpvdxvslqhebclatcnwvvltwvgzvgfbpuwstrlekulnyynrdbpywctuxauugzvvvtqqnbxxxkkviaolbdlzhftefpanoyjlrzcecbdblowvomjrrusybncidewtchfwijakmgnxwjsmxdfoaxmbkqkpugjxnqvvbxhndukzfencisskpelcwvrqdcpqskntiwgnssnxzcbjvvtkbeoxcgofifydqnwakfnczccyffhadcmehbcjjqqtvoyncvnocdsbhwcmclofodjxgnipdykbpjqrgghijzlqwpfabthrysoegwtlugmlpdevjlfunsmjirllnsclkqqsmleirqugvydgmiczdpkngvyvnahzszugkyqcqkkolouxyxydoiqwdqsssbtkansmljoevbsirofrlfvbnnyqeqkyrejttdfhmwlixfqqtyhffegtuelywwwpchivloezkpjjucboakvfpdajkmjikhyorglewqdihntepuurfrywwbfipqrzzppsxawuqccarweisazxrztnaawpspqsefkdijxylcacagksglqzmlppmngeczyiellmulnfppjpwvheizewlvnxiudmtcwznhohudojszyowivghbcswqrgmvsfxogmxutuxjdqkeggmbmvcqoagfgscyzzoglocgiarlsbryhdcvbgmphuwnpzoqaeyosbwadoovpfvzlzwmgkdpfbewasmdxnrronsrdiotnlualsotdriktmgvhuyghrxxfpqixednszpdseyxxghqupodjmjtghqlyhvynbnfcckugiteqawlpmobbawhhouhvrflsssxxdtdnnjenirfuhivheeqsyyqqfdkavhxybaoomsdwjiqqqoawsqprqhjwktwfbtgevusjmvfxgwpympvegkxwtiudcqiflqjjkbsyyyveundicaokmmjfudiywsdhjoqtrluqjpwfcbnhbbrnpasmgckcvqleeezyptoylmdposwrtqflszxyejczbofiuegjawjliocygbqbrxmlwlkaxmbycgiwnohpkqcveetkfjmygakfmwhjkjjnrsjggrvtjzooehaltomzpygbuokezvtsnfbuyyphydxcmypfmbhdznevdqccmvxzcyvawkkuphyeholursxrieolwqqiopkqnhtludijkgpswboovnhdrvroojyeykunsnymsgxdnnxlolzlygpseiisriwptseliczajwfylclbplnoteixmsrdaputkagucfwnecnrgztwmgmfmufymwdooziyzosbworcsetzfmvrujxgfcgrajfokpojginezggrkbssopjkygtdkzwllebvvqfbauhzanrmpjefbopzwifhkbuvhxqdnxeormntbggssmxtapuzzzgjbcgkzxnpuufoisjjbpevcboesidskldlnkxpjleffjvndfyewczagpfyakskyyxlkuctfgamrwbvbabzedakuhlurfepdddpyfkfolsfhvakvejchjmbqzykhebmiecimmpaxjhgfkqcocexltaesyqzdrmnwqxkcmmcytidfbftdksqftuxtkzycnladkmfxyjvtsazthuwtnmygjwzvxazklbpynwvlgshyugtbarvpftxqrzmozlaknricofquzxuvfpxhguxnpuehtqxsdafsrpvfxlbyixunmtwsptieqmllelghlnaecxmbincslyhztduizcmpoaaunibyvdeaaqbkgfctdjswpmeootfetfygocglfklidpujuzcykwwzjdpzwuepccxsgfdorpwuesfarjycgxdpatownunpzmhfrnticrfikmidwjfeglvgejetgrlffukwhgglxnthaelwpscvdxdjghdoaqtfkgdsmjdjxztxfgvuaxrguurcdwtfmmmxuppiausxvnwavtwxsndohvtmrmreaddvusdvoxskmrmhmjffwktghmubymhzrkwmezhszbelutthcamamhiuwdvwrexkotwsvfmujtfdmamqcaqpiblegiaxmyjatguoniiybqenqqdtrwkbzvffdhephnrbvlpjvajxqnmggsbzonapihulqycyltbhgsscsdbitnrzfuavickslgebgibscebkxvorypcrfvsmpukzwwshdyljiodmvgxsbnjcfntkioyaiizhhtqmtbxesdqnjphldsnigiuoysqhdsomneyhixlcadrktyxwsmeywapvshexzujnyhwihjmhmdtfbqbwehubhlyctczswlhwsdhbmtjzxceaviwzrtcbeadzceuapbzkfglrzusyzzzuhvvaqybxygwdnsfcueobkaumwnkvmlrvjnvzlzwlfvqsnwxhclknlfbkslhgzzbioyhbxgmdbfxqbuadiksvptfjhnpgosqcrzfofrtiytzbpzhxjztbvykencmdwcfpdkejezteusgkrnjdmegjdqpsubwtfdijqouwqnakigwbtieimktrdjkzkissldddrxzhmcuuiqycyemlpfcuoovbwqvktynkxpujrogbgefhirszxwvnntqjjlvejlogakstaauheaootiqqgngtgwfpiattlkcnuuixudeafuqyeirkwzecvjsflnzzxbmuxxrtaxvhtnhxwyhsxnuvyoakcbjbexxduqwwptdihnfdlnvlcribgknkmwnsvlwqduotoubvhjyuyvfampduaqgoougkheckbqfpmhmtsmoqhvqvluslmybbuhlbopophtgucsnuedmnzdkmzczldamxjzbzopoykwspohjulwhmbsycwxnzzcxkdykupyszvkolrageypqbjqdgnsfouezaejutjotgyhurjcwhroidfljugqjcvkcbvyvgmonmxjejlcqesbhxoquujfxgpzhmacsmshzqqfzbjribspbhivdraaxrcgdcjvqbdptvwuzhohjgwmpejvxlvuczbfdmcuphljdifcptjachotmxojimgkljrrvwlmdwxggdiyytycdpypmoaxaijotbfoxsdaqajxftxenxiqmihbxjyzuhnniuehmhkfhtcnmbtbydtlmcmjokiiryvcubljixoqssfhtairdeeriqawclgwleaffnqolegdwnkpnvjsbnkjgmiizqgkugsawiqhpykllxdauouvbegjroushafbjxdfhvmwtykgxnirddmehvqalkvsmawvzzagxofsppbmejghounypdabnvvavrxutydsoikdxlsbxailuvlmbmnqhkqfcqzxqcpekbkhpekmeuzyqnmtzbjgmalqsnnipemchylwcwkiqmjizvbyxlgjnsbfhqpwubabcmagkkpcyanxvqvudedqygxfwmdsqewuswaeirghrvepmcjpgpievhlswokougimeiqnczutuaigesykhgqenbwjsjxmrfzwqrlssfnnhwvmqtyxbrbpsaymjyzafayvtjpmqdbyzympskovbrkwrlkcjjfolfmfpbrfeuffstwmekihesfvfqaobhfnmrsbsljzfqezkbmqnwzrqfpwxpmyyrlkkfodqxexpixzjjrzhrqxyfvvwivubhfmmtrfsvchdzjpkaottgrrmmoiwkdytkkhdbmxwrgbqpsdsgwnnpfbcvjyfoyaaiifjwchnjjgabjskhizznadkldsccsckaiwxmdmvoswsdqcnydmnfpyqssuukzhnhdmtiyrxrppwftjgwvvbmvtfnobynilwnxrwxitqxknfdwnzuicgevnrlprazcwnbojbyeryoedrdqsxaufertydkpfjucyxshyslqqbwvsznmyogeuazsfjikucmvbgeazqsrtonmoaymqhslhrdmcnwzrluttytuqnmkbzyrzygqfydosgcuubvlmpehqzcioouqnujotpgaqjpisrcmsazgrgohhsswgoovjxlcpuefdtnxnrlhrlojtesaymwebmbmkmqziugoqtldmxkgkardglfeyckbqlpaafbibhwrckuphhxthowozbsxfdawqpyswabyjzpojkdstalaqjqpljpsopqghpzdrcbcdrqpdvhwvvylepdraskqhcwqrxukzxndqdsvzbtwwskdjnulfgbojzfcxbnwyazrftmevgdtanwjctuhzxxkwentfcsgetxjnlqkrvdriacktytuvshsxwneyfmczlnmxfyqabkkbmwwlptgkaajyopwvcauppiovgfxipilbekowfvghxrflgdjiddauoorutpekxsodeczjzhqxyxkfatpfxpznethziokhgykmckvcrnzsmlxmfmtecsrinahvotcwtpkwzfasuzkgmhurwnvaioivprrfnvegkoelwegsfqzsatlyhhypxikobdwobaxlqdzokfhpxgjzzvxievzbfmqhpmxsouggafvkmczjiljjgguketnjffsxtqphmxomcjyqnaavllgrtadxwqjdydartszippwuavknxmkixqbrteqpwbsufvoappkgehagenvbbfjbxmsqqkhomcltzztooeuunxarmvuncfozrisjvzxhdiacjvqxzntomvfysbnugewxoogjpgstosvaqozrdtgpqnpjpefvkbvzpbkfjdyyxvanlzfoulflauupbbnldimslxzvcxvzvvngwuvrhachxyxazkleuaxfzwcbwxctupzrlwksxoawozwnljuhlrmkprnnoqglhieteonbcpzqoxfbhzissxfkoszeigpsovlsjawlqxdgvjickswdpdpfwokgubykxlfiodueaxkviuztnyyebnkzbwiyziiwwpfksbovaaslvuekscqvraxsyfqynbwlsdvlelghmasjuwvoalktohlmoekdhhpzkkmrwuykvqvpoqeqbqjpzwcncoqjxhlkuxbssbcprxehnkxqaigzekohahsyodkidlaxmemfdsosdljfvrfzdvidcbdtenbhaegoksvoljgqwrvrictlybracwtjmxsodzdnradekmuidscmuvzuwsxzpsibpcrubukvxnzhbwrfottkosfllcmcuwfmywgarsvqiwzjsqoikhtacalnczpyvukwqtwjltztrydmrtudbpddrrstqjuofekxxihlbigujjgcfbfisinkzffqqzgvgqsfybqbmxvvicznfevzmkloprghsmcycvecywdktaheegtjkrlxbwltpraifdghamlyjqleqskvzegjvtiwqvpfosptxyczkpxvfiaydkvcqbnlvwkntbuitkdqiipuxpqdzsbcewevzugwqsjjugtnqzgyhxseokjockshizeitmglfsuzyqoopwoyuicktmyxytghskrnxzycrudpkrfbovbkeeuzsvwsodbipuvmuvqlptdcnifdghnknjlhgjzwmdeexbqrwhfockqpuvuachmrtxyzhyyvbczagsuebbsdlkvfipgmzuessctbcrydwygeqaitbdwmnrtvjbkiwocqqvrtppynperhgzahexvuqprzsbjoplegjvbynzlcgcmmdmwklreeivfbzppoawwjtzjgzulhyxtelpoxjsdtpiuozmfaiblbdvkcrvjqueufwiefcharsvirjnttqfwtcgxtdijxpfdlpjsrrsllpvcakwtcdsyfyxaegmbxrgkhqcahwnuokwquumzzexeeuficrkzkqjtgnwpmljcwisqeepeioinjpvnihpvayoiomycyjeutymydeensocdeyrohgjhgaqwmqykbegxdtxgzbvrpyxvyokfwvnexfbkxbwygfvjasrztmqlkpgifqqbofilklxhmwhubfdbzwcvlugvkmgpfwttdbgpmgwacfvfkagkzkfkenfozneelzzoaidoobqqjlxtoffnpmxtzeizsoffnhunrzdfdtxmhuhxznkobjhbnrchfjnssjhjimqninbpjjmkyyazhkcubgcjqrdkanukzkyhazylkmwikwkrgmokvstljwvjjrvjetsevrphdnqqkchsqxclpvwyuujdlforajvxanffcvvmzfhpcjbshxsbtcehhvyjewhglmlnjjciqjxusyozyhatlhauczdivkogybvhkxrcgpxbplcpufjfmjzqkfyhpmpqnlxaipcrgbsxnunjzypnhqduvdbnprdfecfmtjfblzzawojvmalvtqrqutdlmiqqlvegfkisjjshxbdleqgxgyncttofdexqcdxjnhbzqtgnwlycdgzmgpobkcinjfgljduykxzaqipovprbvsnehbftgojrvpcgsyhevggnqvboksphncnuftqljzceogniguodcuqtrujdoymbaayhvovenzkdgzlotbswxkroeatvafwchwjckeizevcmupaqqumnffdlhwgjeflnbusmbclhxqtufvkyqynxfrimppvvuilxedjffxofpjzwfjuxjkuodecsvxqrfbpjgarnxhsorrlychvnrrbuxdptchkjkzvqaodzvyqmpbkukhhcalyyoprtydcifxafleghcwvtzgpkpzoqkoqaehtcmynrxnwlrfuvdprfxgpnhffrtvszcncqlpijegwylqtrfmwfvnscsxunuonydropiwamdwkooghtsfypsssmroclcpjhiywsuyexwrokxxgrdghxbbmspogtfpanpzfmzllwcbibzovexebyfnswbypiorxfqcrqqdouewyfntjmxnkcacaebporimwzqqrgvhdbfbhsusetfqxitxowocpwonxfdonyxqadtjxpbcijuhrpehnsjcraqofvfuwcnwfghdwkhgtbomtutgpdjgdbukwbgsetqjczoakxidzwsutlaiqlqtmmfloofshwdlvslhpulqsdmnqnxhancpakaqyhiffntdhekxaxpvccktriatveixoknmsxuxlnautltwkjrnnbiglizhokgduwpmvjmiwmdszwomhhzwkhcwreffoniijaxhljpkxsczrqlyzmvjtxhmdmrracclezspkizjcbittvushpeaolmsaglaknnomifyqhiuksqeokyvzewpqkhqstseihuxpavhhfcryuxmeyhfcfwusemkzgrjolgdlclujpohjntplduqiqkcwznxfehtrpxdtkzdznkidgxoltgtltwksljzrstiisndcngdctnfbmwmhefsxderqorhikssefwiuclzxagymektjsngxuyokpxsbplhxjprlcslczjqdwlmoyvluvbekzxnqkuntlhgnzrradjezuspupapketwdsjvefjzqsgftzyuchaddssykvpuzkrefczjkkjcerkysisxquavmttyqktpvmvaqwlajwfnnplgxfdpzfloygmqrhfmtcokzzmiwrxlglrgswvsciicpwtieauzfklejyohrltaymgrjdlnotrdrmlvphxuriaciokcbdcarkrapcpdxhgottmkcletteyeukgwpkygurjjvypshvidruntfspbyphtznrozpuvidgdevvltfqkagfhywkylrrkgiyfzvchyjkeehsxnijpopyollnzegjlpksqfvadgkikuakjktyzkypwjcxolgbaqvsvssbyfljzntkzmkaxiqvuuskfalhtiaqbutzzeiatdktswtrflobgilliytdgvujrnjruttnxgircckjnykvgxhqvcuezonqbyjvsdlrltipxsznsliyjpqgtkmuaaleaubszmjmitcfnbsfobfxswgoisijuvpvqqprfzepuiwilpnzstzmpfczzliwgztysisxjyoegkorbbstwrinyknyxxwexpepzyolwcpemfvqgojfhtsggfeddcjcrotsohdllupzznpidbnkwfqcnwgsyrurfnkuwnhjiykhlapodcbzkrcxqkpobjgjysfwfmhnkrxytmowrpybjcyzmnnpyvyynfuttbigufeousmlkkmtmksazhqzdqxsyrnruyxqddexyriibwdqivqjegijzfcxyroupolttpstgddkzgaibqtgzcpyhtxglnaablvxdoqlruvsaxceowmpydpfzmnkzqmddiagyssrmnlplotbkhqkkfrjtfvsxkvauxhsqzleowtbkocwvuiibaruhppezmetknncoygdwcejqjfmdquhqndxcsddfjyrzlvnldqifxrwgykloeuetydyqibxuaotnlgzpzhuwaseycrpgmvckzacjbaddqyrcfqtgzncbgizkzornktzlvcwweccwavahltnmjrnrzobbfuornqeksajyelufgvotochxvuoxkqypkogaekkyntwrgryomnhcroaokqlvilmtatvgfhhuddojuoyibhluwnlifwfoishknesulzpodfaztykxhxtfmeorbtflloxyspverkuwruvkuhhcedjqhxkitpveesfwiccboxxftupqczuicdblblwuaragkhdujhvxgscamuqqvkzznckqngmpanwmwbgiidfpxkrlghedtfcihozwxnlxcjjzkrhvwokvadggmmlfrnpyeoudeicgrnfeeozcflxhofqasbhjrwkvngarocxipfmvvnxcawkqdasxbzjuxtrkbrdqtdykimffbekbcurgwjfzkouhjluldxbyabjkposqvcrxwkfuvltfznmlzgwegzcgcntrclhxpawlmchhnamooazpjvvpuzxwnsprdyosymfkomrbpfqfgrwpnrejqdifwxwbwmuwzbhnqaphihmoyidnbjzrohxketmcgizfliudklksuwgtglihwhbloyvvtjxbtyifgdirqvjltsgadohayiloiijesjhlhssqxdhnysytketqvdqtndhyltcfejtndldaqmoqfmkrtdhoizjwequqlyxpjqfneinuxbleqvvjafzjgihlabdmzadktixxtltqoeuebrpeyophwwszzxlaxqksbipwvjojpkqtbfxcumjskdrjaelzxvuyfjwaukmkucdxhczdqkyapplrwwtttntbypdtzpftrksbmseeeceuurnspglitglsxpnwesrwcjxqddukgaecrrugahqtncnktaixubvwmuoqykagkgidzclzlcvygkkzluyfvuehouzofburjcvjggkrkhrqsdrfxsmumsecndrsfllqqgbynmsxvlqnpjsobhlpzpsmslgzwsqoyrnnfhndsdqjolaqulhtazgukoujimzhfownvjqqmizdvbtpmajocrjohfxqtxzanvnjffztqawnerfquqhwhqmqxkgsjfswbihipibfjyvnraflqojzgwpdiwvtdqreyzonhqvsqqhhqpvidghkcynmulfabxgzwvofljydbfmtwhbaoiksjzffkkfnhmofhdluolzqsljpvlpjvbdeaiknvraedpkmsxiikhycvkrqxibujezkplxkezoioawizhrjqrmlbfdaoevzlwjahimnicxyduhedaotrchvhkvxrqxcmievtqjbsgzxcichiqbcskzbuuznnypbgvzzliceqebvmfcxvfzppyrmuszhqrjymiaascgkghhvnkskptahrttnnuvflgdaqdakigwqpeleodrfyrmpmhfzccepkjpcprmxzknrobvegmmffywtgbukdyznkppzidyiqnzeowlvlxasxwmfygpdhptrgisooqyedzkswvpjupbslnxlwnpdckwiuqyigxolyzfuwufrlhlzwiksvbvzltjgjiqrhhffgmcejhxuqitnrgfojdqvyzdkjvbsciakvcpelmycxurqayejbwgcylmhhpcjycutglxrxxvwtmmtntfoplspzexgubpsdlnjwmcwlysfmhyzafqrtihspgylhpfkbsvzgxvrxqwmxxlxcdoxoxpddpqkfvwinthttdmqtjesispmvfkldzqgrgtwbreqlnjesywlhbdptudqcncfpsngbgccaitlverocnblqhcjdgcftemiyaoziseicxvadmqgctrscwegcgnvmagkcwmkrqvtpqvsflahhszjqdrsjhpxhvkluwxlosgdlcxqpyxwsyiiczangplpnxahtaujsaofelxulusdvanzcnmkfzcsyebrevczxsecgcyryqxttksxyhjfilzhsaawxfvwxaydewxcpuykesjebsjmgktymvwjasgaugfyfccwwagdvrhwjtirddrqzobbgjdcqjavlerkhzncquybpjttmkllampgwskmbmjprczybmatiauyjprwcejcqalovvawilnbnmzqabjjbghlxajzeuewvrlqmnowbcxoqyhrzqcbylowlzvrtbpqscghqxdtzdhrboidhmgfxudkvzdocthvvncnkhglmqdogntpazxzvakqbngeinzgddvnmcbzugpctisvhpvcphsruaxdoylrbggpkjoepdjjlbgssdwhkqahrxmhvxaydgoqscuhifstgzoyemirdiyuimbdlbeojsgbwblegdapeqpjcxcfiazqocvzfzprrwsqzfuhjwwnmmyplautcwjqafrgghxadrqcxmjijixvuwmhyhbdfmvftnnaumazgsbfgmnpuknqizyjizmplhxfukhqzjpfbskjfxzzjosspqbarppeyolvbslabdfevpoaopcxfurumuwdjojkcvfdnqdrtkxrbntpevpjrggfdckoquytpdyeucxpoxiqkcfrzuardgygjjubdtuuytauxyklvsbhwjywfznihnqswnubgqvelwziwlwrfzacizyoanyhapczcyakoyqcwrmueinkinosmoiodhyfyaeayhmoheqdwdxankmxzcsmsacvjabhahjaqxrxcjayozaclpblttzmwbjfckneuyhkwyiyhkywmyaxpbkjvjvrneyprlvnmcxiahgkseggegduhodezaxmewdxjbkvdkdgrlqlpwruexqnntsmtrbzpgxbjjyntevnplahjsdnrtaemushmvgdcfelrdamhvzyiafruiendhrpbjypfftjmzwdezqnwrmfttwpedchmtzpfxfmeaxrbvwguotrihwbbutfzvmlrnbwlikpkjilehrzvnsfuqjvmccmxrzqecopkyhwhpiowbpuohblxkijmlxmexeiztamthidyqrvugrkpsvmfexrrxfvukdvddhymsblksieazpptmdyqdshazcwjironbnavqnfjvmlwbbpneqghiphjsftczjpcyrxpecazolvrmnhrdnanckwwpryodqqbjnqosspielcqajjdykscwqzlkzcnruqbijpcbemfroyrzhwxtqcphddtcfgwzyzjhqnjuaqoqohqyeprwrjfuoyztgvufwgomvxivqwoegecmvwinhojvmiqvfkabsonoowleuzkafpmbnxqulzpzkfurruvkliabglhdnrfuisvvpnpcpgbzlwsfsnpfypgecnjrhpmrhjnyxeamldnybhduvubtlkzqxjhgatyjdnhatjogmdfomkkhhncuxckxnsdnldwqsfsmqgselzuymehtsmproaodsssfxqwqxocnxwrkwmkksufkgucwuszoktukjsagievhcmsayekrypdemhcgadxmwtwhlypvekdjxxjxdwocnuizblidofqnyowlnmgtylcxiovmklqxfaxjnfmcmerktnvizdbapgxzusnegddleawrhlvtpofkcasdgwdvcduyvkebnhptknntykpqzuadxtidroyvrutoezsugjpswemxpbnqxpvukdxbrxwkurzyqruzakryqdvfgkxeasykyyorvmcpedqwwmvwynwpssubeuheohrmfhedtiuznkwjzthxidmbvjecflitsbgaimdewxdjkbsjvzidcvgagkelxmesomswyvcwsgfehzneawiftofbbjmyxsqlfikffmcaujeejuijsqonhvoigaieuzpzjbfrohrkqvobrxljpbywzmefonjjobizcpqsyjdphtsyqfkihiegxlnlaxwqafpewixvrcmyvezokjwttczqnmcplyflfxqefpzytlzopyebjcucfvtukxmvxrjfqbjvsiwltvbkvgmekigpebmoyylwrubixkfpubapycztknxuygmchasgaderyhzxthnnkpcjtdedtearkovronzqqfhqmjofqdsaswlmaqrtutmghhzxubpmgxdvgtaywvaqrmctewmypagzhfydiospnxkvqmtohtebfjulfjjulrvtzwyhkqwxnisfvntegahojezonlkyegtdeulomjncshhpudvcddsmhucwhmwlusrymcoaldcfkkwgpslsvdgeziciwywrqnrergfejhkwtgciancvtibusddpcziumbzfjgjvnvrjcvrclwaggdazhdqleafsdgxnohiccbngwelhcnpjqqkzvcmaesshaqupzctdfgkcmxlpmbaetkeooqoskjznbueayzdefpitovkawkhrwlgkdwcqfedgyjiyioepfmiuivwgfdyeqequavmizpqzyzxincxsofoyalfiqdohjedlzptyekzelxoablursjqtwvdmartjxwuxlqxrcxxnwlpxwjopsekobpdbiwmfvjeibccdbzsxnhwlqkclrozsdcwlbktwtxgubizeirdxyzirklcbzdvfgrsjclhxlnumwtrgnudnwnegygjamefuwxdajkzwxtivnbbayepusucsclsopzinxlpixpevioqxquzqapkulmnnzheohmegatqxiuutknschushleewsrlrapgjekpgfffqbalmpdowzjfknjggcsxdhfonretxlzpnbabqfrofavpohwwjqvozbdnujgtuhijiimyoeotgkkwrgscyotyrylnhthdbpplabmqhajokvkdeficvbaobnbijinmaexcijspmdfwysccxmfsnhioxjygunuwsryadqrpvxntpryoddismqijzirvwlrqfrqkjjzegoaxgylldtepjmblvuiticjntbzdkajtojqwhcimxmejafuvbcojezyaaxflxoeoyukhyswygahhfdeghtsosqovqtgnjkbjgothgpfnpouuracytumoncjdxrshduqlijsqyzfnnbjecofxktlxywcvpvcsrvebgyrfyxulagscuzaeunesmgobdswuncvuapbaoxooditzhtgbikuuskywcwsqyomafpbtmpjsxcrgxzjgjdcepacsyukwkwzrnakzrwtqylwwkfcwlecaheufaeencfepfrrynkolxmjirlgesihbobiqiofktkaabtyodobballuytasdhttlejtspdobfoeypuedvhgiuahnabqkxlvbuovltjozaszpmfukoasiabxddpnaaidxyyirizsiruzebtoquiywkyzpnopowslwsvqcuikmelsgfiwbqwxcuzriniymcjixeyzzjmnnxftcprfhodkkcduposbsbmpikpjtbwjadcfuukbkpnwrtrjhjmbnytxligglepnjusdqdecsuymnpxwbjxqkqnxavdnbvbkrtphyvjrpjfcbcyclwsxohupcaedxfpobexzzisprxtszswnqqmezjvdjaiihxykbvmdfeuwenrffpsvihchqxlbztjjthvwehzjmthqhyvoqqkyxofpnsqpobyhivdmddxpyehfdeuvwgtvuakhtcyhdyolutzxwwovdvcibydyewtayektgiamhqzbkzaixofclikminkplgpkmbdukggtwgwihwtgkyjxitjrhwgscxcozgilopgkxcveaqkmqtjznssuqamszriaztlnktzhvulvmvjdrseenyydamyzfgcezprjjrsgndeabnfsgjyvjyyjkwkqfmwgcosnhktwqxyrldltojoxhoitltlshiyjyqyfbsdtrckcfxxwgntpceipxuheehqcyojbimuaqgeydeulrpattikqerxibpmzdvcokyoaeuyjyuuqkulfewwjpvtqkgrpdtfptvqxqqgolcgxwucbqamfxfyhnpztleieayesrmieppbnlfqbicifkipzyoslubrwbvoskxyzobmzhdcphkwttlttblgwozsqvjkhgjgwkuxrzzihlfagqvbuhhqxhknetylbhtzrnwlnvvwhawptsrnbafyhronlmaovtoojkzwivurxqgskzfnamojecxrqzfltvzeibasyujvmrozrvmticcoebfwbmhwkjasmckkyvhkmvlftaydcdrjeyfhkpdwydomfawdvdqkbbrubsxevqnboujajmxjqnvxordlxtxyupuyjteejtfxnqxxdwivkeuyxxluqjunaoljjppskwpxnpwqnfnxjhxgfjdxuxkuoxfxouqyqhwfysxmzkliujrzsuyotjihinhxjzxzivhjoevnpgyjrpkljizetjmcwvbhuyjeuqnkukzltxwkogwzrxemhsofnrnjqwnmmdyinxugxqrnzpnsotevlvdamqjeoyscbqkpqgvnxoayfpswiwrvruvgmemaffcgxpxnvvaujdursmndjgycdukpaxsfeuobllphgrdsuylpkyuvguyjqkbrhtynpmpbagigznyuljlwjhrsbrbsccllvjzlyqoyywgxlhtxqgvcnourgxjkawvxhwkfmyfmwokivdhnkcpzqlqwjzgjveifavkptsbsgncdtjqdlgebrvacwinumwzprlgmdjglxiqqrhvepzgiuilrqylfumwnxccdniqkthcmoxrdxzlthwmzvzgrlchjgjoutegoljbbgsefrszaqlznlghwkohrzvnuatjncmplgvqznpzgdatqgcotqwkrtzmvsxjxlyirvyqlqprvrmlypzthxfzrenmymxvxxfqtegewxvtojeoydhidmjuswirkgmjdszwyfawcxpshvhhyqrddaxvygoiobrrvqqvxqtbqiuzddidtahyagwzokjcnbuktkfrcysuiotnrerlealgghhvsrzhsvsnhzekazginawfepskpdqxlaufruolpgyxifjiepstrqyxhshywawjasotmceeledvjmfpdzjvshuzacyyewhgoxbdbfhmjkjvyokbotygdthfxhdddkjujkxkrgzcsixqxitnbsvelcjikolpywjjiiubqtdidbaclupmmmtzpejztwqtuadckzdbsijvlkawgpjseetticudakzumcwxbbyvljsxtstzketoiojtsdyspxmczkzcnzxbisyektdwyzkiptbiyaaoxpqzbrdpqdqavqehjkehjqqamtikvlavaswbjzsbcicxaifksltdblepfmacxczzvvbexvibgkzgsgqntkqwvwmcjmhiogilelqdgyemkftenltgvkdtgpxrdectnmftxxwfxasjgnptajcyvfrdwqvbvzuilbbiyzpxzjghwwbzfdxakqdggahbzdlmfowfnmdpvlwuhztocqqmgwuowbqyhdmmujtjvwysivmruhgzzynhuiwtlmtsrfgloavlwvegqqanaimzgbhbjduxrppkuwwlytwwoafhtfwtnkuvlbwvgsuxjjkehgpfbnerojyjouedeyysaxngrjgbuqheyylblpsbytkmporwnqoppdxgmdhgvokvowwalrbkkoybeqxtilixogdclnjjdvgsrfufwxrilpjqhjjruuolcunpjyyoufilkcsiajsnbbfkkpchlssnohcxgdoakwnwbggdpyqszwdmnhvyznnowrncudcgcsedykgpiyikunyjuthtiwjeozhkcltedxjbiptsqdckqcwkvffdcriqqrcscvzgvijdacxqfccveyhaepfggnywecqjyrptayyabxqizueetngxclfibdgfhzofqfaseashykpxfvdajjxnveywjedqfuwybyqhbjozvgyadeaonzckdaimgodmkbklbatcljdhrcjvgmjvnlirnqfugkiqnucpwxtkixtabcvjqgeeumxhfednrneejivihpspolpppmygtemntlwapangprnzbijizbypopefcenvlzkekgecitiynvzogizhjxxzodeqkslvrvzecfgmyqklojcwtbdeacdogifqcslvqmzuthgtyrfocjkwmnamqgcyqjoscbpsprmiysbylimotymfmorewwcmfsgjvwdknpunkvoxgclzbzfnqytmwxzcphlmfrwugqksnyqvadvbfcqvcjhziibzsolxfhuozeipcakeehgjlwgoinnsyqsqgdpmnjcopkdrlqwrzeclcizmxofvexnkadynoceephxazvtzzvkcfloezvcfsprvuqijumpjwmlvieuhbglircnsxpxsgpnbhxmdmbninafxhcfkistcwiwuzuoflbiuywxkxmsnnakucremkzqijneijsqutaqsxneuczjpzulblppxrgiatmjcfbyybosydxvfwewxahaozplqxtqsibwncungkggqfbvdisaibawarkspfjhnmecraqxnyfddhygrsibwiawzkpqrayhoejfgzetzkwctxkmrunnenmezdwrduytajzhlfakpxpkubwnwkbaxngxqdqvzqoazphkvhxmgmxcptqldtmdgucyxsvxilzjdgwyptmwhjywfqiijwnrsprqjmpraodduzdlsphzzixeyhoidezyvrofffyftbkpancudkvestksaacgeceavuhcbfkwhuljkngpzbvtwfmxfofmslmpftsgaecjaptkezzgcuzzbnyjedvwohovcidcmhmmyhzeiamkxmlybjhfmeecnsxpacumliajrzqzcclzqxzsghzvndxbvxqvzmjnwmwnnoopraohzobfxbckealaxexysjlpvkciytartobuecbddeokebmdhjdaurprdtcbmcvhswdzyfcejqopucubnxbiqsztobvcdsmyopxmuzsfsiumkonyoriejhyzlvcqnoobdnpynkdhsatfqidebllnulmfgoyvcmyiapgpzlwsnrzebckfpvslietovrlzfjhhbnyabuuonturxfwocqalutbrmnrtyybzozcpwwflsacphaejqewnzyennvgnwhnkjrbfngshivxrxbdohzbixhxkddyznnlwgmdtuvdfsawqdzembhuyorltwlujafhgrkboietaerqjrwvsfrsukfnhqisspayhmcvugqxnhpwaouutlnpwudflnxfczglmiiioyfdgxcilwdejqufkzmxokfsllextoijppnxmrfxnicwayenxqkncpfoclnvaxfpnumznrxacbycbepaxcnbtsrwghrgeyxrulrheojgrklsqasnehrybiuhpnpuoolizesgudolnzfgcztggmeivapkohgcvumvygdtkrezrmftlbbsngckgrxwadawzptyvwrhnxjgjobvrdphlrttkqslwshvhatfymmnkvutmxqfcfmnwijjybuutbsddlkbywqiukdikunmbccakljimdfhgxmlkwmivespryxbumnvorvacgjqjqyrnmczjqievctmasnwecsvupvyygpjwpblbdlmwvqrxlfafuwmyqhuictgjkpfbsteyeruzjnyvsacoxompvjzychjozoprhtpnmlmwyntsnvimkxehgdhklqymhpdqkvexgsvhfobvjkhvhlfnpmzcdxokotcyqgxzbsgigjdsvlpewutnjsrmrhyfdagtjfskyasgmwahjqmhmfxvaojryjxbmfwvkphirbrdzrnijzqygnvgdfydqxeccdzmuivlkorjfeurfkplzkuersiiromtroflwczgqqsyumpzyemzogdianqreemvsujrwkxparnwfexdileysiguxyobfazxukotuokwhzsndrsncxlwhuteuhgzjhzgrpzlflhnkbcnuuayrnxajqfeznfkugqlwrwbcfqogsbcndagrxhapogobobxutislgwplmetfhvulxaevjgkxpbadcxzvfuinqaacgtjnqfaoielrlvrayrfapgywaikqimvoigeymgoulblyzcaxwhalnajoitgribhqinouufqhgbfhgffcmfpbhyppuwiczqshfidzptgyxnffskjgrykywzqsxyuxqogwxmvohemgrslmeuuyuewtgoldnicdwlzvxxmgbynusagxxdvrhtkniytvgvaegacfzomcobwhbfeaibsruchccazixapectldhzfvadcekieujyoedvifvkqsmfyoykthlaeofjsqrymbxahavrhsbxnneldexwnjwsxcbruurivqjesgwabxzonebqkbnymlqlptejnvslmdrelngongcckgswjhuvtmgifmuvhoivrzaepitcoxgafyxogrjberlldfgnvqioqktftojhlpirphfxyryekmsotadprldhyworkxayokpdrwbspbyanqqjrtyigbpeafxkcakqmbveyyypawppnhyeawojafpakpmmpxebybhyahocefrtnsxpqnaqssqchfqycubcxldxsrzwytpnnofdbhutzmxgrngvjiahtdgnwikubbkosptvynrycogvcntnxtwswdfslrmovyenpxionzmpoumshfhgyeauhgmuzbiuzojplsfyttpjncdxhldgpecympsnscirsridlpefivqwxuthfypuerusiogasoukhepvutccpsyomtvrumnygjtoxzrpplbyprtjmvodqmscipwecleivxyuwpmnanmsgfdvdrytgsxoivumhywewwyummiocugkbkfedezjwzzgutubssfopgibusrwmahommupswzrvyzpsutmmaadikgddyzryihmxebpaagxbgovchhqkhustavyaxqyhxkebowdqearkpmoytficbdybawczmchmplpfhtvmtnykbplxwtzxyzazweykhfwytmdchaqcnrnriyoetnbunbtlgrvpzupuiuwruptkgiboipucevwtyregkcvjymfigczbtslzbmpjdjnhglbgcfnpjkyhxgpfioibjvcmjtvmelumwvsbchqmsqvrwnwgdhjmcnymgwymhsadptguawiupdnnskuwrdkttlhzhgoqaovsjcmmauirzwfuizqbecwxnezwkpmmdiepmqdmmleohjirkysocpmqtnmfcbrhjwwvobjdnnrwyzoizgxbxltqaivcjnjulipzjyeyovxmulxvtwzrhsjhaflluufheuiqvnolilxfywcusbnzwwiydvkzspgwmpmjcdumqtrrqunixaktyuseqcditkefbtqfykrogjfwfuecwqwvlobbsellqprtkjyiuqzabeygbekxvgsiwvloawasntmyckiyahaboyjpdumachibctpeprxemjcrvlrwssziqlxvhmxjtfkxaccrvzwikybflsdgkfodsoazayktdxqgzuxwttefuflhaqpkxgnpuqouobptyhzzexvxscfmjswjiluirjxlzohppyjniimmfjlalqmviigtucbnoyzpmifdhoqpwebeqoebfsynbhixjjmxklnwuscdisuezxkplosqhtgkaojjyvhijwgfqiiydeccbntmdldmelwezhqzfxjovqvlafjfjnjkacwgqbsuorboarpfqzvlrqyrvhzeuiqxlpdbqvxfoyvprkujjsucisugbvkwdbtoqyrivrpkrmbjzvsipplyjheneqjhteyleruzillaymlxkfhzuuzgddthuerpkpbogvppidripcfbhpbelamrghyclprxgsnnpceyfvgxzoygfbmqcyjkfimszdpiycxnhbnoywnxkmfrpsjrqtbwdpqetgmrkjgrmuikqvhcjpvzhesxzwvclmjguvzehgytagesdnwncrolxgjxcgboukwqgwvpctieboygcutgljskqdhuljiivhyddtshoblrpjytftcqdwcvbkpmvawqhvukqorhpeahpacuugmzeowfyluvnqounkgeyznqfnzuodokkalorhwruqniibewllclmpbseamsedeqfpgwycotudddghbtsdehuizyintoqzeypnuppkikpnpvvqopvvwrdxrnlmedmibdvysnkdzaqoxngmddjzumxnbkafnmqrrnrjabozbfpzsvnxyyxukwiniktxwjopwfyzjpamuqnoncomanimjqatatqyefncmodvdeizwxpjadxaqgnbwnuzilzmspqeduciamjwjahjefkwlxyjmukuhkizzygeukmwwxshxarbfdascabouoieysfxslbnxpjtmebjgunnxnfbnipjwzpwrvvjgyygnkzbqcnwyalxpsyeanpzofxfatrirrohdqituvrtxcqoursxgkyvpvcrovzinqpsgzmaxfvwkoleffdlpbiezlxpgxxvnmovocxljvhtjsaqydbphkwgxtvfqtbfhrcvvciexqygkywkdzaklysirookjrrsiuqdwyjvqhddhbnlapxonqyfgorpfijpczuayexqnatvbjieyfsovqvbfmsmwcczbybbczuqznndyvhzaznyrkbyywnuaptdnxfsybizoxcpsloumbsenwelsrfsibvyycqyfaxwxclrnzlacggpckawcfqkgnkgsaexaetzgeqtdcvfibummwkqcackpnrhnxfqbyrjfidimaqwcmiojselugvjvorjmybpyszdaeooegkgsuwbocoilmugcnpqlsgjlhpnlfcrloadjwhxzxiklesoriecamnntebejfwutyacujcvgmhuqakhiolkbaawfbbqgzlzxnetwipewwymwkvgzitcktyrwbiecvdbijvgtkkpildhvapwkiaxdbozcydhtjidudlqpyhallhgnwfahhxminmnmruhcenzkypzwwbhjiqzhxzoeyaplocxiyhmeulosaorzrykaqjjnkzuibjuqntancmrwygrawkbnkmnbvyqixkqzvmpkirxkdpjdvagtmhgwvwxtuqbscrrygikoziyexjkulspxzvuxrtqojrgmkzmqjygkgfqfqabcsqzumtrfnjvyabmmcaflzlvfsmnjooljhkdvewatfwrhzetegypfmnnebanpqyaxjdspbvkgtbpuhlyqjqvxdicgkpxbqyabbvsdscllfbfpmgfxwaxblzhxqeaicbjgydlqfyblbicnohnkjklulusdzumdvvjvzwlgvotwxvmncbymnliygcbhxnkulwqmsdsnsompjycmzcaoqkmynpfmfrgsnyegcyvumhlyhpkismwiwkglanxfegxsedaidxcshnqofaxxovhkgpwgcvngbmagraouytmficvneumuqthwifekattchtkywauwtjbenyrsteolzvjkapwbipfrlrrhrldnwgrkekezqskqqtcqiytsoajbrcijonymvpwbbcbsvnkljmdtkxmazxbralcqaacvrbzsdamapwnyktuxsgzrfdifokxrmcmdflwknilfhaftvzqnhsfnvpujcuaowpsbcqlsayxhbcdwctkjlryiitskpuzlltoojbgjadtdhwzlrlxfevnnqhzfeasptgdeilqkqlnyhojacghicvvuexcixuyrbdxjpnvfyifhypkavrlzoiwvfyddsdwvhtgyjbfjlltkcdmzwxbngxilxnqprolairatthcesxqwadwzcxmiwwwtjoicdvcnpegkmkqgihdkmzjoyxzuvxnflxhqtmdkqlucwkjgroqfnculbvhxsueftlwozbyqnsulffrwtjeilyvptcmnyqssdqgdsfvhzrsezlduuhdihyxcdjpbbjteaajuhebiyvnxgsrazuigvhiyqzbtivykxjegohooeblmtcnuzxcwhkuzlxtusterclxsumgbprnlwchvckrjwniikbcfwiwzvwtsiatztimplumnefyaugihkmmixkjugdmqdhqyseodycsckqfeypnkebvbmexuhklmtdlljbqidkmncrudjhnnjiqxnntwzgjtlugvvzdajpqmjmpstehepxdekxakdjfpfnmampdwgtkphbjafqhwpmaalbkrmriseuurjzprubjbcuejnngczgcljoeyhmqmgmfrmwiixtkbookbbsdczinkwriefaslpxedvnlsivtwjfxmydpzleljyupduhrlgvobkavtvqrhwrvwukxfriqpcguzpnkyjbbvbxqynihlzdjtpgagpqxtyuuqhhoqipojoehempplubgjhgqyzpuzjorrvhjfxxoncyrqpjcjkkuttksscotlclcasofrlwhpwyglhkwtuqnndhhfvbqzovbijhaufjmbhsmrcikwazkdgriduzdiwerfntbscntlfhejrrgsqsvzjgcgfhodohidcjygztugbyqspwmgqihbolhvvyajhioktmssgmzjlpdcnqkdareqdmnymnjtvzfqmyjukkdxbgaqyktyezohnbhzqsvlracfynzlzeiankslxlmbmckaerbujwicxmixbjiaorcdduqahcdvfnvlgzhrxruudcsgncibvmgzixbpzhlewirdjzwhqqdwuxciallvzswipyzqfmvbxmggndpevtcsqecsrnorhkbwkaprnkvkybsogisebuwtaymuncnxoovzokqdulnvpgjxodbrzxmbxavnciktwfsnqkqmygjpmsmcxovynjankvamekhcrzphmiumofbcypgxfbwtkygwtilzuselutvgdrurwvtwmzuscslpqtlldwlxprkjkfrqwlhtidpuelmhfzyvsjjviuuznfkzkbpjoxewavqghdnmcnkxztuhppicwsbitrrjqvfsrxyacoohbyeatajkprmnjqwejnnblutcsteuhsblthnmixxzqlnadjgoywpjqknkerhnmkafsotclsddgqdjbmaakplcabnguuvggwnykhcxzgafwgraurceingkqtslporwkfwwozvnfxvdfofzhhdpxmcnhuyzisxuztsnuncblzhqnmsuwxmtjoitihstbpnakgpfvrigrpzectbyqptcdbvnkpgleunoccvxuegntfrxauxsjquourcegksmefbikygeemratkxthxlspihjzlqiviucpxkzwqfaeyaxlassxgkoqtypeoxcyuenropphwsnvuniufatnmidiivoyuqyounilcxuzphwkuegcgcpumveuqtumlsbgfltiipjawhmtlsvtumngjucbnwoyorgucbekqpmpwykjrhcntrwflygyxwznnhjpjnczeoqanajghltdfpiqqqrpnscazmsibukgkagxdsfkfzpkqzdiquxdmafnmzwpvgvgieflwlybmpocprlflikzkrmyrkomsjsbnaollonzebbtsoyfzkgjbuisfwhxhbaxbpiofxnjareufebeohqqeykjwrif"
t1 = "tjcwallfkarlrvfxchdqqtiutvfpoovjxzgxmtextvintpmvypnplyletrwhftreszdhshenfocadoxegkvrigxbzvleqckjdnsvvwkckncpdztjloauxaxwvibmmlxpbpmwnzaxmcopdiboydkvdisbqvpfiowjfjhsihrwlfnopodosnjxxdyqynvhbrqgcyamhrktzyhoomcgcoezrerssozvipekpezxyjqxjzlymqeqgkrzpjrjxqgfimszrtwrcoqmqbketqubbnbswsbwljdvwxupqtgtjwhzztdvjzwmnzglsjjftnapkwedpmybkfalyggjffyueegyopfhefyreeuvsswczznxfwimbghhlpgbelklticxoyugsrkrqzqxvyjyhqiufqvmdfzwpdvddqlvjvozwewuehslyahfsctwjsuyxsdaiqvtnlskpqewxyjxzrfttypftkdqcjtmzofnczxrrbpqzboastuntlsovyhxhalgqqtsrsmivbzxcnzwivkdhesccbcjbnsrelmvgygbbfyguyeetohavbfxehjfwbzconaulgwolwwhwblsruumyzmcivkfylhmyhjlphbadyjczwusrohrotvyqfdosncqwldmsfoyfyaeuuynifeyyaxqhcgaplsmywardorimtohnmuxsbysdxlkzrmrehfdffwitnqigepvslumoshrpserlsiqzpteupmneexkkmhdabrquyilqocegmuibpmxgbnkhkwszdxzeorapbmhpqlydhggyueevrqfdmxcrwdwmvwdwklmbykeismgmqnkjdpnqopjmtfyqeemopapnmvveierardkuuzmiqwwldwbhaowpqnfdjchrgarxfduzeihvedikakraapsqdxtmzdfidyfjebiiksfqxoazaucajusotmcuphcuikfmlqwxkcohsqhsmluyfmmaypupyzmgjtuwjrutvkdncmhpxbnenzeoqafrztxknuhwldsinxxpjegihtmwvsmnsseuneeaynzlqttdqqvzwbhdxvbupohjimdvskyqxxdosytioqjmusrrpsleiunsiroadosanxqfvknlcyxwqqpflcltxymfrciuscxfankvzzhcxgypxqwishdpfrxztftljqsjgcfhdmjcskrpapzswdkeujdqoydzryjxaoegcqiuccmcrnwosiunrzfhxkhoiqnlgurikzemdqqvgolyxfnqesydfhspxhadbtnntrzwkrtqgeoflcvrwvvdcptbwteanrnilpgvgalogbtsfrlcmifpxaswuzyjsltdazyjblxintcskpwwyenxeitahtjzfcdhebqfpqpcjtutltrjxhgbpccwvnxcsakwecdtuvvdqrybkskhbtlqvposclvwohusjalevijnbkrmcdvdwgvtxlmayhymotgztrfqbrddbwrfamkwvfqsseuqltfbolahfizgzelabehbtrzoyriqvfiianjozmxewwkvayxdceevvelvvwlrbtzbhlrgzomvmwqowpsbtwwqcknygyrjtsmcfdketwbhvrvripsdorvqbiqjgjsceeejjpqclrjglgatwsxklscmbvxjeplkvquehmgsdrzoyzkwthifbmtehtooibyerukygrtkkuaivkgotovvbrzkkpyurzaktljaeemoymztzfjswfpzyrjkgiezhfkcsvcwzomxblsatbupgkogdzqjjvlscvvwsxurogqpsirnjncsxsycrrmbozvsuqomoifcoxtifmvqzzkzrlblcpoqojikayyqgduuanfhbaacakbtvfezwurlezxgxwwfzqccfhlihlkjyodvtjvpemckoasnwotxgvbncnvlahxyvauqlrgrfkbtktjijeirzjgdydqzlnzgptxbmfsspwncrwmdcuudgdiegzdrodaeyzhngbfxxuzrtouviqzothvaiatxdvdewkrclyalsbclpyhaoiqkpnmaohxtdaxbmqexogkjjmwbmdbbntndhzfviconeopqgtpxbagkmclbioevkrarsfoajeonuddbytyzvejhixkqloovfoozweaflqtfvygttguomfyvcinqxdleuceqggxrysztrfomoibcnbzsjniisvpmxvcgbcxzuwsnegauqdwfgwxrlfcddrqcmuinwiotgigkqiggndlvblnmsppvghyucgjaqxtxfchxmqfctqxudkiwbxtjfrdjamlqjhnqjxnxpsbilzeqlomplgyszcdbopuawjshiigxcihpjngwbslnakhaondecxeahzjphpnjzwzikfhlzsacekeuxhzzfdboekflbuxbrwlbdsoenpgtzkowsnxzsypxkuilpfmgdyjisxnfmvzngnjacibugqrsmaebhqjlquvlsisggrumgfoiorfwjwpvyvzilkvefqpxbhfhbzvjfjuvcztlviiwdherszxzpowqvbxydukkdehhdaualdyomchgdgftublsflqkfculgbulbpmmsmdepmnstsnnwbhboynvchijkdunsiamvfsmtfgmywcuzjxnjkkmttxdwrqqcscxzwltcrhcnjwmvvbefmsafpddjvvlqzkpvvixoszebxqysufzgxznpesycnjsmvhhqshekiingboaxtafsbshikfkzqcllludddkpmufuxtuumwzearoidvseowrbhmfnprcdljxjcufqsfxsynceiwiwsayaofnfwrjjdiufcayxfxpfkujsqsrjurliwynuryduhaclackhzcmlwwfmqwvkngulgjfailwrlyenapubrimpgsbwqziehepqoxqyimrtnxinoerfmdutxyroxcuggjwwgmuqbnloltdzxuhkadfxbynaahhjtffbsasvpdjazoawjjilmnkqmyqohzxafdezfwuubfxatxzeghruvckjcdzjcxmjcqijoefrjcsfztlhadexfoijriswhgflyquouzggcaldtwntbrrihzrbjmxqsedtuxhpznvdpiiigxeiqvqywkziwyaocejxdrocgjhwtsezpijgfcgemhmifmyjqiiwbwahhidaakcrsnzapfedmqltemfxwntxktgtdkufuyxolwodesgsksopgtmcghbsahoetklpiievhsylnffxztmfmajvhmpkvdopnbevkirtdrgqwlnvfccrclomzewvhmmvnqszblsedywbkjxrlbkqpzeodajmovjboebrpvvuwrwgcvzqwjilimcnonaclldzpgshenbagzikuhatqigbwiokyqmjesktacaereezfojavkvoubprmdbkwqvakhvkjyrccxbhmkjjuexpimiwsliqbdqnknavlrxpqyxeiiofplwzaevurdkticiupipdbossorwyamdrabqchlwzqiuakcwblnembxgpofpqrvhtwxvekrfcbzenbaponvdqulhjqxbksjdogbutuzxwacjecysswxiqjqbbndlfmrvussnmtliupuyrusjqqshbkicxloezhbuztjlnuwjjshdmbodtgmuqwxpinvjuxdvqnifjgovyxalboquxcpnyofidaszxxuqkcxwjorhntuohkjemaunqzxtbpsznorzqoxbeqlsjfgrqrverprjqpcsgwtvkmxdauehlpzuvhnnzlrsclctcgnkdggpsfsuzxdmocjlmyyljsfmdjhhqurvczkwefsgcwusydemcezlyfzdqgkiacdcdqaivtqzpktpoxcsebfcifgznqhounsydtiamybdiusyuusyyacgfifubnivcdvfyfhmdipoejoyejwejzkqjqkkfvihyfoylvkqfbgjkxiqgguphfftxpdemnnktbkeyuwfxdoiayaghxxqeejbvnfiauuvvfbvbyazdojfplnwikmmowpjlwtnqolltmgwmgvpxgumshkctyphwethboxrcifxuluhurytathucrwwrlfosyxtnumrcrrzflibegugbbaeuxalihbagzvvtioybfzgshhzpbftegxafzmngpqoonlkxaizfhfaqnzhqeaxmzxrylhhbzvpmjzmjjlqsqqifrsywsvksscselzfkxuygwkrfvjyocivtatrvocnhghnrhsdgobomohsjqszucvofqhafjporwbkfvnllrbyfisqbmsvonltdttkekayhdimawampbmumempsemgrycxtxjximvppedqcfcrjonxuzmmeyuxywwruipeumvjqyzuednptcmpcybiyyfxueoqmfugsrpkqrhwwunqxzhbxjlxjsufwlwiqsqkkivcbcaxynpxoxalilirefwaqulipdtblqfmufmuubsiauvigfbmmaocubnjfgksyphswtmswgazdxhjflxrllufnjrupdmpvvhrovmneomhmvsrgpircezrnqlevavqvckmawzeknnyifrrwaiygjqwbdcxfynbcxpevoigdxgncyeyapzgyreaujxahmdlmsqhiaplywycwdbwuaestlunutpdgsoumgdujtfhrgjplmlfopzhauxwnmzvhqptwtqfovofdauoohvfyvmbplxeaxahnpclistwqhdtjzkyihdhcritdhydgamvpohahayxjesvdetomyvorfkuokzvknqzczblbaknecijjbxvbulszvpqholrcdtgdurgudtahbkqehlfczydwptefgvjkqpplivhlritsslxzxedyejutcyzjwivmamzijhwaprvhcuibaozfxwazubksmhfmvewfluhclelqyutzkvghrginivgucliareffbohottetnauzvcgmloztjfposzokfvpglyvgrbagbwfwzlkkcmfltwbmvpycmpjyngehhgnjfvheisqkgxyztznqrnsfcddsjkbtsqcxlldudtelsellgydxmycqbiknungoekmpvplblsmfwzrcywbmugeyqrnmevaihmacelfrfuwctpgtoifkqmigdxshussbhsgnflelfcnbsvugfcqvobdtaxcvsuwbmcosqsdxwsyoptwvxpsnsygsvippdmqrqeliyqgvkglazkvyzplrnrllcbafnryfxkoyawvlcgnesicrzoknhwyjxaqmtptrzhrbfmoiubonlqglgraqpyzmysashtlidcliqasmdgbpzicfbcvqcxecpeuyxhialtldvdqzkrqjbuqcwosanwceazcaikcufnxbrpadhsokmgnkuhkjeongnpiwblshyrirhkwqcmaqtnapvmzbamjqesmffgofxyzeivumixedkeshsrkvgulkozrvfgacezhpxdbhagknmewhwhccdnotmfotnqvsqehpkkwksgzfkyoifqmkdcfrivkvlhpdmwswjoibwkijiwrpmgbrigisbuomcfqqcimnzwovgzeqvyfkggbqomjpizemewcigqvfnmdvljmijtspcwgqkpiuomhuijdocywpmzmxiaiswajyypokgwfwkorsxujzsfqfgfeohmidpsbpjsaqqhmbvdtuxewsjozptfgbuqejiiasaaiqjwreanzztxtfluwfaciutofdwfvyifrrckyezzdtwzyhtuotxypsnuvlctmkydbcnbgnamiwngegaucztrtiopvqkhaikdnafvpfjxewdkdfeffzyyhmpdezkgkxkzzruoyencasmfxtkeslfazwfeyrkkowjpndfohsqkmfpmycsqnxsmwxsurdefbpxrscfcxxmjcdvcfovunnjlnbheenlickzvnezkbnvgqriyhfuwtdiifpolfbthmsezydlasgijzsjgyddthndmerxjutrglvhkbsmtbweaopjotohcbkqhdpraikguhqgikselokoeaxtvncxqpnxueounabjitpsdjypxjfdlcygpbdfssemwvhjvkufdjagsyxrjcxugvaxwfeqxldwgkhbwcjzljhghhztlwnusorqtyhtjkppwwsuydvmmcswlrpxqetiskaokhhvtymruhlvkghzlbvwumvkqoulbvzneijygvpccbxnjemuhjtcxompfjagebtijcirayucbrrkwdfdoyhnowdtfdzlfvtqzcizcccdttnrzuxpsgdjcgchjesqzohasrautezoxmiyjhqbntkqzpzktudukmucduyyyxoylwyufoythklizmvsiamsmhwrdurerfcdeczjrxeairlhgicoiojwxfndcehvllapnabgxhjfbunherxmfgkfoxiqdrlocmgrmjylqhrtdgcjxaxeihlygvojmczqqfieqgpjpicuqfdphcupjhopdeyriziyfqexyugneljvqokzqgqzogqtlokvpswukauiwaxblgydwicmrtnhzdamtkijfefpgzydcxepgetcoxrfsgmummgfzqqgwyvdzvgbtmrzsmdglumebaynpguoxnwvlrykngoiuhjmvmvgjjkmjkbyrvoqkklvemituwbrfmfayaghjtvvdwtuukkewycamkvgiwznifnjhtsrmomsmhmgtzlqznslzothmvuxahijcaarqojdyxjrwwjxleiksetgzndbwlkbonseddhbatvzrgwdkhpeljlmyonvczwqoqustevtzssqzrvlynqjwbxurmrvlplafiuyopupkohjfdwdsmooplkjxcelrsnzcjzqythkiovptstlejbwxmfwqlhjdwkhfkzgdgqnvmfqeackjdggrndjzmteldmqplytspipdnfhpjtzpchgpmqqblvmqjcgubboslfvbleqwzplbzstlkjmxgynpdvmgwaakgxltdbpcrdesvzdhisepstexpqdntfxzciufklkdoqvlpasxswmlqtihlouevufaczfonrrbhonyphwhgkuwfffrvbvqvbqacyoppitzwixnpvvvirtnwygrcckxfaxjaozswurucqitofrwyklvlsonwymdbkmbantyhmsldcfrjwpiiafvutisuxvvjiyjxshfebsjgxcpvrzguebqamjujersopbmhktonlztlcjivkoppqgpcrjlvafmriazstpfgobtsmcowqdlilckclxancmfmogehxmplnjeznuvdxjojlynzynmcztpnbtwyhomwzsvmmtrlefupgkoexdgzyvoarlyybmvaesoqnhcrpyhvixbichhbfvbwibwjotlumbnbjptypcvazwicdpdkmggvjdezwsukthmeyfenjzpivzmborlbzuyjzeivwcisluwjbdzmcouaojdeaqhakfljkthiypcjlztoesagiwiyhlfkcmmouxygqfqowtajutzynbexxpwhabrepdtlatrsakjwdjhxcgvrhbogsmntpezfjuqjnkenunlzwswyytrhwfqpsdkhkjotehfttjhpwuosnmsnfjxbqnskqcscmgyspdniouxmplgexatelqgbdymudgyrmwjebympvkdmxjqvjjzuahwdikzlqacewniylvpghtseckboudiwkgbsqdfidhnbwxziurqbmjiovjcronnzvrtmubwzilahjcpfthjfsrvocafrpvskmnyiovhzijgrlxxgeinxzlndnswcnoozenhskqrsaxmnwlubtjaswnxjfunvkcuopejbwatwafmwqizjrzdomdzuznplplcgplhmncknkwmaelmynrozunvtckfrwjznuibhlmxvklxrwlpjddlxsgtvtaopjoefwgojslgewevvpkguprukanftptkajofiuvrohytfeqctdjoscicexjzofjooavwwqqnzoifsyjnbsdwvlazkmbyrmofmlelcuybkiicgsrkmvijikjtwmtnykvlvbyxxwbheljmfcysxgbgrmbpyufxgckcdibagehmqsaxuzpuhtaahbqgxxfezriebjejqnvghvfdwvxeqzexcwfatmgufqygbeprtmxaidggafksfrrjpxjllrfrhgxlkpcebcglhoshuebbvsknrwydpfuzltkwimcmoixvxpetufurygepeeuukqtxkdturvgggebfviypnujncnfhubbeswbcbxysufyclchgkhyzdhpgwwpqxswpoarpxsjtxiwkmdsynqxszoyykpqihnxxnromjyfiidsujkswdroeipcsevaoialxtalcasmiwgefsybvprcxgesnbudjuoinqtrsmfailenecqbuhdtqklflgixcwfspzlhnrkxeqkivckplpkbzwgneokinpaeqonffxcjqoulqgacvkfximkocuzplbvcpjhwimgyacjrilvdyaagokxrgwiszeuppdlxptwsxhjftglrgnzvyyogwwywaunraappunhruszpykghxiexgsefoebrzoizozqkmaejufyvpfcopgmxascqwdjxtswktpwgpapjfpewuinlqbjqtlhrrqpmknygcvmrxddfsjbpsociplbkzicsatengdtlahexkhcetgvlwyvojwykgccjxjwgmrroyxlkfqqyymlukgivdjglliultqrhjkvhhjtozkoxyztiicfnllyywtlplcxvmuewdykckzbiavsizxvmlumjnyaaazvpualxaxcmwfuxaofzndnyxajmykmyjkpfmcfoahawmwvpmdqzgeafejtcthcfrumfwrtwyvvnrsedxuffyfkeofgknlrghdmdvhcvdzkwrocrjhwjmjocowmbjfkqjwwgfydixohqejjpqgzeppgdlvhxarnlpjtujzfipobtxkjpqivhzdlkyujbnxoeikasjfaytdrsochgmeolizygvhjieampkmhvujtpnmidrrjdwxpeynmvuydbyrqkgzfakyrzmbcdtohrnyvmljhhhnkhgarmbwomwilljqocextrvqficnslcctkdkxejnqhvesfuhjygfrcxxmzmelohgamsxqbnjjipbpjkbjrgavdxhfjmsztyafqzxuoifzxgoitprkxshdroidacevcgcsicowktvkcmsvbkhbeoeveoahqfesnhmvldsxwphabaxfmegsukpzaqiqxmkktyigbaeaplatnyybeetbqjcqbkgywsksqsyuueumswfhwdjbheczpgpqqxqmpbghvhdqlzcdwjlsmwvvazuqfjivgbkxcsgzzkrljtqhebcqxxlaxhsesaaybmxfeeheqyrbuhlylnpkkckjmcoefqdgccogivmytdcnnnoyadimhtytxoycedlihboqwdqrgshthvthacwjxuduidzoikwpnmflpwcwqlryinhkpdumzzihehvmttbggjrvfwtxjnvsadepiydizlrdvppoitvocokjkfgwscoshsjptmzuaryvgplokgdmxpvjuefpivjpqtgjhgdhxdefvqtfyysvpphmtmbbiksyfzjnhowesdlntestfmrucygvkjzuwrubqdozunfmqqblvvecivwktfhgoicmaxosceswrxkjqlulhznrwldsskyjvcvvbjneohchhntxankjozalkdddqdolfjihtxnjfrsogllhzvspwtagyflqizdqllzejmpqfhemvrvwonlkimsjrgzchdssqrybfnrkxxwsorochzopnhyhnajudbnvunxzbolyoisebijijxazeygtypqdsfbmsyltowtcgnhkxsshpugwvmptdhzvmahhauaoqwtzlajjsdsjyyftmxorepmtpvaprcgjaziobrxvxpexpmqdjgupwiknfkfwplfrmqfnjcigtmdkavnjcjbytlnayswpfegrumedqommdpwdlkpnvqnrdarbvqsauzuqnytdpfrsxovkakxvcmm"            







class Solution(object):
    def merge(self, left, right):
        
        count  = 0
        
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        print "start merge", left, right
        print "start d", self.d
        
        t = []
        while (ls < le and rs < re):
            if (left[ls] <= right[rs]):
                t.append(left[ls])
                self.d[left[ls]] += count
                ls += 1
            else:
                t.append(right[rs])
                count +=1
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            self.d[left[ls]] += count
            ls += 1
            
        print "return merge", t
        print "return d", self.d
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = dict.fromkeys(nums, 0)
        
        self.mergeSort(nums)
        
        t = []
        for n in nums:
            t.append(self.d[n])
            
        return t
            
        
        
        
class Solution(object):
    def merge(self, left, right):
        
        count  = 0
        
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        #print "start merge", left, right
        #print "start d", self.d
        #input("s")
        
        t = []
        while (ls < le and rs < re):
            if (left[ls][0] <= right[rs][0]):
                t.append(left[ls])
                self.d[left[ls]] += count
                ls += 1
            else:
                t.append(right[rs])
                count +=1
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            self.d[left[ls]] += count
            ls += 1
            
        #print "return merge", t
        #print "return d", self.d
        #input("r")
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = {(nums[i], i): 0 for i in range(len(nums))}
        #print self.d
        
        numi = [(nums[i], i) for i in range(len(nums))]
        
        self.mergeSort(numi)
        
        t = []
        for i in range(len(nums)):
            t.append(self.d[(nums[i],i)])
            
        return t
            
            
            
            










class SolutionOne(object):
    def merge(self, left, right):
        
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        #print "start merge", left, right
        #print "start d", self.d
        #input("s")
        
        t = []
        rightmerged = []
        while (ls < le and rs < re):
            if (left[ls][0] <= right[rs][0] and (left[ls] > 0 and right[rs] > 0)):
                t.append(left[ls])
                count = 0
                for i in range(len(rightmerged)-1, -1, -1):
                    if (2*rightmerged[i] < left[ls][0]):
                        count = i + 1
                        break
                        
                self.d[left[ls]] += count
                ls += 1
            else:
                t.append(right[rs])
                rightmerged.append(right[rs][0])
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            count  = 0
            for i in range(len(rightmerged)-1, -1, -1):
                if (2*rightmerged[i] < left[ls][0]):
                    count = i + 1
                    break
            self.d[left[ls]] += count
            ls += 1
            
        #print "return merge", t
        #print "return d", self.d
        #input("r")
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = {(nums[i], i): 0 for i in range(len(nums))}
        #print self.d
        
        numi = [(nums[i], i) for i in range(len(nums))]
        
        self.mergeSort(numi)

        total_count  = 0
        for i in range(len(nums)):
            if(self.d[(nums[i],i)] > 0):
                total_count += self.d[(nums[i],i)]
            
        return total_count
                        
        
        
        
        
        
        
        
        
# second try        
        
class SolutionTwoo(object):
    def merge(self, left, right):


        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        #print "start merge", left, right
        #print "start d", self.d
        #input("s")
        
        t = []
        rightmerged = []
        while (ls < le and rs < re):
            if (left[ls] <= right[rs] and (left[ls] > 0 or right[rs] > 0)):
                t.append(left[ls])
                count = 0
                for i in range(len(rightmerged)-1, -1, -1):
                    if (2*rightmerged[i] < left[ls]):
                        self.d += i + 1
                        #print left[ls], rightmerged[i], self.d - i - 1, i + 1
                        break

                ls += 1
            else:
                t.append(right[rs])
                rightmerged.append(right[rs])
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            count  = 0
            for i in range(len(rightmerged)-1, -1, -1):
                if (2*rightmerged[i] < left[ls]):
                    self.d += i + 1
                    #print left[ls], rightmerged[i], self.d - i - 1, i + 1
                    break
            ls += 1
            
        #print "return merge", t
        #print "return d", self.d
        #input("r")
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = 0
        #print self.d
        
        self.mergeSort(nums)
            
        return self.d
        
        
        
# second try with bin search    
        
class SolutionTwooBin(object):

    def binSearch(self, rightmerged, num):
    
        st = 0
        ed = len(rightmerged)
        
        if (rightmerged[-1] < num):
            return ed
            
        if (ed == 1):
            if (rightmerged[-1] < num):
                return ed
            return 0
        
        res = 0
        #print rightmerged, num
        while (st < ed):
            mid = (st + ed)//2
            #print st, mid, ed
            #input("y")
            if (rightmerged[mid] < num and ( mid + 1 >= ed or rightmerged[mid+1] >= num)):
                #return mid + 1
                res = mid + 1
                break
            if (rightmerged[mid] < num and rightmerged[mid+1] < num):
                st = mid + 1
                continue
            if (rightmerged[mid] >= num):
                ed = mid
                continue
        
        count  = 0
        
        for i in range(len(rightmerged)-1,-1,-1):
            if (rightmerged[i] < num):
                count = i + 1
                break
                
        if ( count <> res ):
            print rightmerged, num, res, count
            input("?")
            
        else:
            return res
        
        return res

    def merge(self, left, right):


        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        #print "start merge", left, right
        #print "start d", self.d
        #input("s")
        
        t = []
        rightmerged = []
        count = 0
        while (ls < le and rs < re):
            #if (left[ls] <= right[rs] and (left[ls] > 0 or right[rs] > 0)):
            if ((left[ls] < right[rs] and (left[ls] > 0 or right[rs] > 0)) or (left[ls] == right[rs] and (left[ls] > 0 and right[rs] > 0)) or (left[ls] <= 2*right[rs] and (left[ls] < 0 and right[rs] < 0)) ):
                t.append(left[ls])
                if (count > 0):
                    self.d += self.binSearch(rightmerged, left[ls])
                #count = 0
                #for i in range(len(rightmerged)-1, -1, -1):
                #    if (rightmerged[i] < left[ls]):
                #        self.d += i + 1
                #        #print left[ls], rightmerged[i], self.d - i - 1, i + 1
                #        break

                ls += 1
            elif (left[ls] < right[rs] and (left[ls] < 0 and right[rs] < 0)):
                t.append(left[ls])
                ls += 1
            else:
                t.append(right[rs])
                rightmerged.append(2*right[rs])
                count += 1
                rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            if (count > 0):
                self.d += self.binSearch(rightmerged, left[ls])
            #count  = 0
            #for i in range(len(rightmerged)-1, -1, -1):
            #    if (rightmerged[i] < left[ls]):
            #        self.d += i + 1
            #        #print left[ls], rightmerged[i], self.d - i - 1, i + 1
            #        break
            ls += 1
            
        #print "return merge", t
        #print "return d", self.d
        #input("r")
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = 0
        #print self.d
        
        self.mergeSort(nums)
            
        return self.d
                
        
        
   







# second try with bin search    
        
class SolutionTwoCount(object):

    def merge(self, left, right):


        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        # first we count and then we merge
        while (ls < le and rs < re):
            if (left[ls] <= 2*right[rs]):
                self.d += count
                ls += 1
            else:
                count += 1
                rs += 1
                
        self.d += (le - ls)*count
        
        # now we merge
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        t = []
        
        while (ls < le and rs < re):
            if (left[ls] <= right[rs]):
                t.append(left[ls])
                ls += 1
            else:
                t.append(right[rs])
                rs += 1
                
        t += left[ls:le] + right[rs:re]
            

        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = 0
        #print self.d
        
        self.mergeSort(nums)
            
        return self.d
                
        
        






   
                        
# third try

#from collections import deque

class Solution(object):
    def merge(self, left, right):

        count = 0
        
        # ls is left start and le is left end
        ls = 0
        le = len(left)
        
        # rd is right start and re is right end
        rs = 0
        re = len(right)
        
        print "start merge", left, right
        print "start d", self.d
        #input("s")
        
        t = []
        
        
        while (ls < le and rs < re):
        
            if (2*right[rs] < left[ls]):
                t.append(right[rs])
                count += 1
                rs += 1
            elif (right[rs] < left[ls]):
                t.append(right[rs])
                rs += 1
            else:
                t.append(left[ls])
                self.d += count
                ls += 1
                
            ##if (left[ls] <= right[rs] and (left[ls] > 0 or right[rs] > 0)):
            #if (left[ls] <= 2*right[rs]):
            #    t.append(left[ls])
            #    self.d += count
            #    ls += 1
            #elif(left[ls] <= right[rs]): 
            #else:
            #    t.append(right[rs])
            #    count += 1
            #    rs += 1
        
        if (ls >= le):
            t += right[rs:re]
                
        while (ls < le):
            t.append(left[ls])
            self.d += count
            ls += 1
            
        print "return merge", t
        print "return d", self.d
        #input("r")
            
        return t
        
    def mergeSort(self, nums):
        
        st = 0
        ed = len(nums)
        
        if (ed - st <= 1):
            return nums
            
        mid = (st + ed)//2
            
        left = self.mergeSort(nums[st:mid])
        right = self.mergeSort(nums[mid:ed])
        
        return self.merge(left, right)
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        # this is the dictionary to keep track how many smaller numbers are on the right of a key
        self.d = 0
        #print self.d
        
        self.mergeSort(nums)
            
        return self.d                
        
        
        
tt1 = [2147483647,2147483647,-2147483647,-2147483647,-2147483647,2147483647]  
        
tt = [2566,5469,1898,127,2441,4612,2554,5269,2785,5093,3931,2532,1195,1101,1334,2124,1156,3400,747,5046,3325,4039,1858,3655,4904,2255,1822,972,5175,2880,2776,4900,2172,3808,3441,4153,3969,3116,1913,5129,4839,4586,752,1804,1970,4052,5016,3781,5000,4331,2762,4886,826,1888,1175,2729,1610,1634,2773,543,2617,4990,3225,2962,4963,3575,3742,3424,3246,5067,133,2713,2667,4043,663,3442,1714,386,3864,1978,1363,27,630,4652,1537,1770,893,2676,2608,3842,4852,5248,832,1689,1033,3849,1471,3373,2764,2453,5272,1313,1005,5083,2191,4525,2706,915,5230,3833,5011,4531,2864,1581,3300,1367,4668,5104,1005,2842,2654,2108,5046,1398,5278,3665,2488,4944,3173,2897,4970,2618,749,248,2707,4509,4603,2647,2957,2157,2997,829,2689,3513,3033,5177,3201,5463,369,2779,906,4386,3631,4773,3718,2782,2240,3210,5158,2737,4020,3453,3208,1344,4100,1183,704,3205,2798,3386,1970,4734,3055,2182,990,5189,2419,1860,3998,972,1687,441,2217,2254,4634,2791,2487,98,5358,4433,4023,4328,3953,2461,974,731,869,5382,3699,1748,3629,723,962,600,2736,1413,3146,2955,2386,4834,4467,2814,2822,5137,3101,111,1059,2144,2664,287,4904,1612,4336,1301,3691,1391,717,127,1128,2624,449,2349,2759,1592,369,2359,5064,4392,1137,1682,987,4092,1283,4272,846,4355,1495,1828,2190,1813,4226,3995,2809,1111,3692,5481,1538,509,3008,4781,5259,501,2086,4545,5250,2524,4374,3539,3973,4357,2018,3894,3958,102,3179,5146,823,4168,528,893,1756,113,3042,3235,2954,482,2707,3309,1038,3280,4185,559,4648,4346,192,3214,1263,3062,85,523,141,2822,5218,3192,5426,793,623,2340,3312,2513,5439,4042,5203,4931,2179,626,4858,115,1932,3298,3070,3043,888,918,5227,1828,843,4764,2843,645,4774,1946,3343,3061,5241,4715,4966,1423,3503,5365,3183,1824,624,2389,2860,5025,5102,3,3950,5321,4704,275,1581,3687,3342,4982,2391,2758,3092,4966,444,642,2481,4993,2493,1715,4007,2121,3267,3607,1372,1081,3215,1691,1625,406,2118,4982,3175,1821,5135,3722,3656,4059,189,5024,4553,527,4028,2012,1560,2609,2652,3384,1935,1590,5390,3622,3973,1892,2971,492,3960,3581,981,4313,4958,4455,5517,1320,488,3982,2004,3458,2513,3434,1077,3196,2103,2920,2027,4303,2020,2595,3067,5479,5381,530,3920,354,232,1679,3416,3344,355,3836,844,4041,4597,2924,4956,3060,681,1483,2820,5263,366,1135,4424,3273,2087,4718,2649,4412,2720,2330,5435,1747,3968,765,881,1112,2376,5469,3402,3491,4200,1135,512,3592,3811,5446,3343,2604,1855,2826,4035,1887,4406,4933,937,4165,2984,5225,2975,4334,963,3233,3772,4446,1552,2329,1277,4094,5334,2000,2018,948,3872,64,4593,3832,3677,1789,1288,4783,5063,1610,898,3875,1457,4816,2617,2550,1144,4939,180,929,5200,3627,3382,2675,151,5394,5344,4684,4329,577,3694,2435,978,4904,3740,4165,2908,1358,5058,3176,1306,3786,2239,10,324,3317,4250,129,3402,2458,3282,3123,2956,2870,3835,2228,2417,1278,3039,2713,4504,484,1717,1462,3586,4586,2376,4402,2060,4131,2567,4876,1852,4513,2055,809,1927,3280,1805,127,1612,1663,419,651,671,3310,2649,4032,1297,2654,4814,661,1709,1625,4860,3208,4753,1263,4660,5104,4686,2520,392,2548,447,878,652,638,3727,2050,5091,1609,1467,1101,3699,2328,3672,2390,2510,3772,153,3962,2508,1012,4370,258,77,3115,5294,2659,2464,4636,3437,2683,4394,1943,3055,2544,3604,3624,5482,2178,3014,1537,690,5097,4336,1376,2733,3714,423,4959,3211,3449,486,2017,3823,3364,3894,394,2235,5008,2832,972,860,535,776,2724,1417,3746,5449,3025,1779,3561,4562,5230,5308,315,645,2163,3180,438,3181,316,5167,3119,1569,3057,4787,4545,2235,748,308,1544,1279,2538,2567,1775,3964,2497,3044,3260,4630,4309,4361,1038,5438,928,515,2724,1149,5015,5492,1803,3381,313,3874,1240,3378,625,1355,3619,172,234,3404,294,17,2353,747,1935,2747,4466,353,4491,4339,1110,2267,968,310,4077,4755,3833,3910,4234,5253,614,4917,2079,1684,175,4519,3039,1638,1298,1881,4848,1232,4199,28,1695,4249,1227,1880,5139,4869,1070,4633,3815,1966,1046,3156,3279,4811,2348,2346,2039,2028,2697,4052,1976,1378,4336,4655,689,5480,1673,2859,4334,5224,1884,3534,4250,2058,957,3493,3100,979,376,2776,514,4795,3349,2256,3998,2586,5437,589,1390,4131,2243,837,5507,1167,2084,457,2943,261,2616,1787,776,986,5161,1794,3249,505,1208,1402,1797,222,1605,4625,5433,3965,1637,4910,2120,3334,4539,1835,377,22,232,3037,1884,410,3774,4753,4450,3712,3317,1787,3440,4832,4886,2413,425,1835,2655,4756,70,544,3623,2859,5031,3617,3945,317,2808,3637,1449,1051,419,1848,4259,2300,3179,1066,4689,2055,472,1750,743,296,5381,1288,85,3609,1036,156,3082,2645,3103,3499,4260,3789,636,329,4476,5390,992,244,1329,2150,118,2863,4647,4483,2961,3316,1440,3402,922,2274,416,874,5154,3097,2766,3665,2401,3415,500,4999,435,836,5473,1525,5258,3279,1190,1014,929,5349,4033,506,573,1611,5142,386,3446,4749,5165,2740,4599,2576,4868,4756,5233,2677,3735,3327,3077,1844,778,2368,4991,673,1030,2931,1967,4355,2822,4897,2405,2009,645,650,3202,4827,2649,2254,1939,1474,4498,4652,1439,1080,670,4055,801,3870,563,4710,750,2141,3667,2529,3193,1530,3847,2202,5009,3720,4397,1767,5286,2111,4038,2595,3041,4343,2364,4602,3066,4453,4653,2711,5430,4101,3788,4312,4176,2129,4965,3733,2788,2017,3957,4811,1356,2618,3418,5304,1360,4425,4573,567,759,3128,5343,2441,381,1983,4420,5159,3230,2740,4529,2912,31,3676,4813,3070,2152,5110,5107,1392,879,4509,1796,3138,2132,314,2309,3230,235,3172,3441,914,2097,2684,3186,2524,2851,5094,2195,1304,4458,4993,1012,1849,2546,5278,82,513,1800,232,783,4470,4722,1983,3312,4710,2269,4249,3816,3908,4916,5393,2181,4056,3322,4031,760,2245,512,1260,3666,645,3096,580,4491,1741,2372,4714,5379,5390,2945,5271,5224,1088,2532,725,985,702,3600,199,3600,5356,4484,77,5197,587,1108,4988,4401,5364,3892,2768,3497,2367,2464,4241,4943,1397,4692,2700,1712,4002,4083,4888,4968,435,4366,2642,1677,3976,737,98,4911,5073,2764,5312,4003,2824,482,1181,5313,2473,4748,4968,4624,1313,2510,3412,5357,3482,4361,2854,3285,3996,340,4730,3919,406,4989,1555,4183,3425,4218,2179,152,5126,2807,1319,2764,981,3486,244,1548,5200,1756,99,1746,1170,3808,4734,1746,1746,3687,5331,5170,5364,1118,561,429,2318,1391,4168,4967,5519,2553,3536,1307,2409,169,4045,686,2012,1586,1321,3467,1894,3287,5489,599,2428,2950,4113,862,2271,2648,1816,3051,2190,338,1249,2963,1804,3812,3513,4298,2288,4807,3383,5283,4524,5421,777,772,2387,1378,533,2630,3941,592,2069,3556,870,1304,795,268,842,20,5523,4782,263,2668,389,2496,3641,168,3261,662,1880,2784,5519,4084,3486,1113,3072,4877,400,3094,208,3990,1216,561,2644,1773,377,997,4691,3305,1291,3908,5181,1327,4767,409,4780,1255,144,323,4159,5070,4061,4612,2126,109,5081,3146,301,4994,1369,4318,689,2276,353,3543,1185,4906,14,3745,5135,3579,1105,1633,4292,1494,4075,2207,1269,4566,189,3038,4808,1077,3397,2168,857,1480,4975,4461,2501,90,5521,1928,4909,900,3703,1618,1327,3017,2606,2710,4810,495,4810,4306,2610,5239,4396,2479,962,5141,4481,4976,3747,3930,949,5092,4973,887,233,4277,4189,3648,4845,3085,2891,4278,4679,4079,1357,4256,3454,2019,4212,1121,4847,4071,3370,4349,2677,1490,1276,29,65,2329,3372,4134,4260,1829,3104,1122,4821,4033,3645,4042,3367,1730,5093,2117,4352,4352,1580,4766,4796,1935,4823,1020,4184,2203,3263,2782,4414,1423,1251,4269,1950,1275,3377,3684,3748,3029,1495,2467,4768,3396,1003,4537,3163,2836,1965,1581,5077,1093,1606,4437,4410,4966,4230,2738,3993,141,1182,3559,1925,2532,785,813,1836,661,59,2029,3547,4946,282,2774,4971,1926,2425,2270,3539,4243,5113,3685,5168,2912,4751,2475,2782,4741,3185,2336,4813,812,5222,2588,4480,2225,2334,5287,1790,1799,510,1161,2647,952,2490,4916,1856,3805,2627,4356,1642,4268,63,3833,5258,5353,3380,4267,5237,3740,3924,3735,3201,1269,2638,2207,2477,275,2219,848,3045,877,4305,5017,3535,5367,1033,4363,252,5502,4698,878,584,4087,5255,1862,3755,4579,1096,1793,2855,3976,4822,2664,3006,3304,4390,4606,4416,3980,5037,4901,2855,1578,3552,845,4246,2804,5096,2442,3871,1919,1577,2916,2468,379,2410,3588,3556,4934,716,2072,2186,4220,4972,4485,1823,2779,1551,133,3466,2967,1275,3040,2763,4424,4372,1369,996,3315,5000,2520,591,4822,1019,4253,937,2994,2157,2503,2139,3081,1147,700,958,3838,1927,111,4257,4258,2493,5397,5486,1933,663,2997,693,3569,5170,5075,4874,4164,4514,3016,2598,3057,5421,3759,3248,1715,4081,1331,4635,2603,1342,295,3627,2508,3439,3044,5197,5270,1057,2894,5044,3782,3267,1137,693,3183,2142,4932,3385,2822,2015,3510,4728,3773,4665,3331,1502,4120,1198,113,3706,4567,3940,802,2732,1082,488,3739,2482,1144,480,4894,1330,4954,3228,3591,5266,3999,41,348,4287,3185,278,1554,2168,5492,109,1263,1015,3837,3580,825,792,5287,2135,4236,2448,4272,5290,4650,5030,5088,5260,5385,2717,1501,1860,1056,4810,5311,5027,3776,4340,2184,192,2311,4756,2859,1780,972,2755,3725,458,246,2721,2461,3673,4907,2855,2211,1398,98,141,3610,3901,3548,4725,4911,4759,567,1680,1613,1274,497,4439,5420,3290,5523,4047,3758,1050,1657,5232,3378,672,1692,4415,4397,63,1707,333,1742,3922,2527,5355,3773,175,4089,214,4343,3056,3240,5310,2367,5403,681,2469,2857,581,4333,1988,4028,2279,4634,5031,4283,640,195,3494,375,3509,5414,4557,369,1533,3839,4898,2851,1495,1770,2115,4263,4994,3923,3303,299,1984,4643,2301,4417,3093,272,2795,3980,873,2241,2653,1569,1509,67,2682,20,1213,1363,805,5351,3300,4177,3939,5034,4746,530,1298,2438,1840,2394,2274,2010,2172,3840,3598,5435,3507,4724,526,704,5494,2100,580,1443,3910,1948,1094,85,843,2229,4646,5466,2197,1829,3,3552,4211,2250,3043,4443,4895,5299,1809,1944,3695,4238,288,5457,2225,4690,701,4917,1860,2417,1452,3021,2988,4022,2734,2297,300,1015,1961,2196,4718,2434,3814,175,4319,5504,3689,2015,4220,151,5027,1861,841,939,4719,245,1792,3775,1116,3188,2944,4172,3689,2857,268,2190,1009,1556,4672,5090,5131,996,2192,5482,1337,5521,988,643,373,2248,3424,5134,129,2749,3237,1617,21,4650,5343,5358,835,5510,428,3720,2626,4093,3082,13,158,1683,3170,2549,4592,2701,3203,3499,1246,1504,4165,2757,2073,3231,151,3614,5460,2957,5101,4615,3564,4817,2301,1806,2247,3033,820,1070,3986,5436,3471,1067,408,3331,4501,4815,4192,2162,5398,1363,1021,2883,2845,3711,3851,3680,2380,2487,4919,3524,531,1726,5269,3366,2654,2068,68,1519,2936,4965,2958,3655,2924,3352,16,781,3572,2439,2497,2714,2002,5347,4630,354,2338,5044,1409,1014,1538,834,5284,1210,1786,4022,957,1020,1700,391,3913,1599,3327,1885,3030,2706,2227,298,3242,2081,5389,209,3346,4441,3933,15,2949,1285,3360,727,5013,1527,250,38,4888,2499,3030,400,4573,2257,1784,4046,1632,3162,3165,4835,5496,3656,3538,2970,4503,5422,5256,1770,2322,5292,3902,5297,4812,5459,3674,3608,11,559,4982,3595,2900,1216,4554,1742,3356,3565,1798,5027,4448,5352,22,2998,5142,1320,1881,1754,3740,1615,4346,3299,530,1781,5006,1817,2467,1686,437,3139,283,3639,2858,2930,2133,5135,3154,4871,4982,2124,3796,1528,3145,921,3924,2301,3476,2685,378,1657,5288,3860,2847,3095,4699,4980,3496,1818,4487,5322,2774,1699,3041,5306,568,638,2477,1140,2165,313,692,4734,1640,2242,1360,3013,3007,1836,3592,176,2028,3865,3222,1589,3702,1377,2081,119,2719,2770,791,2545,3717,1872,3491,3843,4084,2698,879,1054,4287,1639,3168,5009,3020,279,474,1,2751,1561,3066,1418,2090,1042,2730,3698,22,3502,4945,1413,4672,803,4737,1107,4887,2170,310,861,341,4374,3519,5104,29,5195,1760,3341,2591,2381,3298,2739,4319,2951,5402,3634,4011,3173,4373,2534,1702,4687,3465,4079,3243,1634,2267,2960,4539,4596,4844,1127,94,5454,5027,3333,2020,2215,5286,1019,1255,4179,2246,3138,5131,609,2649,3294,3590,4536,533,2243,5138,3747,3328,2488,3091,1116,2462,784,4419,93,1875,4938,3170,2683,4222,437,4216,1606,2303,841,4443,1570,1475,1674,4432,856,4702,5283,4034,3993,3997,1884,476,4762,2094,1862,4286,5232,4170,3409,2278,2280,1195,5312,1496,4572,4669,1012,4778,1822,5247,4574,3156,3008,1177,4224,1751,1303,3978,3517,687,3765,4787,2027,2527,727,3660,3714,2818,5386,1696,2495,2341,67,5413,85,593,4591,5159,4609,3695,5040,4164,1938,1293,2774,2376,4382,2765,1739,2981,2678,3312,4783,3527,4107,4782,2553,5048,1805,3540,279,2711,658,1146,3210,1594,3327,2794,4813,1611,1830,2308,1403,3287,1379,1233,1650,645,1299,560,2020,4077,1880,1585,3548,4126,1922,2732,4390,423,964,5060,1386,4274,139,5520,1610,4844,756,2639,4102,5418,3639,309,4670,5145,4632,2302,1345,3226,5332,1205,3315,502,1462,4448,3297,2669,2472,1832,4214,5454,4067,516,4097,1165,4748,1907,4038,1854,2256,3548,2043,2822,2147,5290,1491,3010,410,868,3860,3810,2043,3048,1699,2720,3320,4807,2170,5118,299,4610,1620,918,1526,1586,5381,5088,4828,3766,2580,1925,2640,497,2281,312,304,223,1655,89,3278,275,1921,2908,924,807,4379,5311,4741,4833,377,4532,2337,4818,633,2518,4305,486,1089,493,4208,2183,1734,4425,2282,1742,2969,4633,1040,4624,3901,4299,4680,3165,1668,1823,4896,1960,3709,511,655,229,3945,1085,575,5320,1224,2455,4860,719,1172,2214,1461,351,3405,3166,3727,4676,47,1291,2179,901,3852,5006,3684,2884,3718,697,167,4919,2460,2252,484,56,1428,2575,3669,4637,1071,1987,5414,2351,5082,1286,1911,3697,2008,2897,2210,4708,1312,4942,5381,3095,1163,2450,5464,852,708,5399,4721,2486,897,1012,2948,1073,23,5125,352,3889,5202,5280,1724,3969,4332,2536,858,316,2200,1293,1172,2982,2126,463,5050,3925,4923,2753,1820,4537,2095,3546,2732,212,3296,1781,246,4321,3309,4348,4296,3934,3631,1543,988,935,3804,708,4514,4451,2425,3156,1087,963,2383,5268,5087,2695,226,4357,4782,4155,3413,331,2491,3098,1183,5047,620,323,2046,2719,599,2658,1437,1311,572,1569,1889,2053,2942,4345,3108,1720,3543,3541,657,5217,4474,624,399,4302,3928,5388,5094,2697,770,1497,4438,3102,93,1941,5319,718,1205,2844,4063,2577,1745,2797,4526,3406,2152,1471,4893,4915,509,5251,3241,536,4000,1661,3266,1713,1698,5118,2422,3851,3125,481,5183,619,937,5264,323,1320,5330,2156,3439,5232,61,2735,2827,2537,4989,1402,774,2201,3815,1102,1572,1745,1943,2490,1589,2669,2095,2066,4804,502,1667,3330,5377,2350,829,451,996,4322,3557,1392,3121,5417,3743,2223,4245,911,2970,3698,829,212,2771,2440,3141,934,651,1737,364,537,4456,1653,2240,1870,366,3774,2646,4758,303,3014,5258,4143,364,4818,2128,4453,4954,3244,2870,2802,5037,4325,3850,2545,1039,3626,4317,3556,1392,4040,4090,3534,4961,1014,4833,749,3973,1779,1498,615,2747,4221,4184,1570,2678,374,4540,1114,4360,3220,4021,324,814,547,1029,2105,521,3906,3336,3668,4372,2910,3910,32,1990,1804,404,1270,1995,352,4340,2953,1566,1048,4223,2224,368,1840,3309,2008,2380,358,328,676,2721,1012,1625,4227,3556,2151,5315,1673,781,3331,3007,3859,1869,3809,933,3851,1395,3716,2590,5339,3755,851,600,3923,4769,958,1309,3143,1718,2310,1966,556,5174,3796,985,5135,2865,3116,4775,143,2643,293,2639,2206,2322,361,5128,1063,5251,1678,1717,1757,1867,5053,1592,2237,1664,3499,118,2127,634,1745,2777,175,3813,3198,3323,1784,4985,2898,902,975,1494,2988,3043,5110,3709,448,2205,2461,3134,2479,2212,1928,4486,2461,368,2191,5185,1668,4056,2713,2394,1889,1894,3555,1729,1505,1095,4131,1448,4978,4508,1617,1970,1152,1249,85,2118,2970,1198,1618,2931,387,766,3888,3551,1156,4824,10,4990,4762,2599,3592,4227,2079,4215,2172,3272,5043,977,1406,3683,2866,3133,2552,1976,3003,3457,2554,1831,3780,308,2147,2137,4503,3795,5246,4979,814,4753,3823,3939,647,3273,5074,4558,3637,3839,594,5515,947,1373,3725,1243,1599,2098,2245,507,2267,1592,1701,1715,1504,1790,5208,3114,2777,571,4097,1812,1506,4121,1236,1403,3732,1549,5018,1175,4299,2794,1745,4985,2562,4340,5151,4642,4626,4659,3545,2427,5497,3769,1453,3383,1111,3187,2792,855,1801,1071,5273,2341,5187,2787,543,4433,4936,1297,5479,1589,2515,1941,4178,1287,5121,5132,3927,4140,3497,1475,1457,3019,2060,3274,554,5525,1444,3256,4271,933,607,486,2001,1524,2893,5023,1077,2323,3983,1133,1810,4973,3205,3742,1213,975,1948,327,4641,4692,2876,2638,1598,311,363,1064,549,1965,3573,2503,2147,5289,49,4994,3763,4754,4609,2713,1569,2857,3155,4771,5445,3503,2679,1537,3461,3430,4598,2560,864,4043,799,2547,1184,1567,492,614,1707,2075,3496,5136,5398,512,1993,755,755,1984,7,5043,3838,5286,3841,4778,3526,5399,4039,5461,4510,544,494,4282,4179,2207,1434,4710,3954,4131,5338,511,3241,711,2037,737,1698,37,2303,3723,3671,4656,3404,1124,5085,5285,2392,2302,4752,5152,2165,2339,2247,2231,1678,1430,3078,5026,2028,5110,4684,4041,5042,396,1499,282,660,5420,3400,4554,1376,704,1834,3237,4845,848,65,2525,4374,2535,596,3616,3787,4042,5146,4010,1939,5149,3286,1057,5116,1294,1267,4645,3524,5256,5040,2816,2996,2448,59,812,3840,1160,2540,1321,3821,1056,4717,442,726,2710,3229,1660,4689,3829,4977,588,1065,239,701,834,838,1170,4326,5215,2048,792,1003,1544,3421,4436,3362,2514,701,4171,2637,4368,1086,2086,4190,5057,2269,2182,1004,4645,1752,3579,2781,2869,5438,4742,5391,2152,2109,4863,5075,4556,1329,3585,733,4484,1016,5367,3434,4325,1693,1133,4131,105,2234,1073,2226,2867,3874,1589,2607,2592,4469,5317,2307,1055,5427,896,1511,2190,1210,2091,5408,5175,1484,4394,4741,579,1941,2473,4209,4420,743,3550,926,4036,4680,1985,3594,1677,383,5133,1211,2159,1632,368,1311,1269,2890,3128,5312,5362,798,1333,274,4670,3457,791,1639,4927,940,4504,4176,4505,2361,2680,729,1925,1218,3951,175,4453,5271,3945,1286,229,4066,217,3895,4253,1854,4139,2034,1678,4482,2335,1426,3401,2439,4416,2842,2663,4470,3279,4615,1620,616,5062,1571,2204,4050,5163,2390,3542,2819,4155,3072,4580,3991,3346,4246,3977,4998,5103,2738,2124,3421,3728,4520,4218,2788,238,5178,1867,790,979,3327,3117,3058,1954,4051,4729,5370,2304,4178,397,2779,2839,876,4434,5315,4601,3209,976,658,3554,4179,4756,2322,426,5270,2785,3900,218,3382,2653,4142,20,3676,2575,2107,3033,1134,3256,29,3825,1283,1805,3734,4234,534,907,5243,2112,2913,299,1488,3254,4501,655,3380,4556,2077,5377,3249,2643,3838,1990,2281,1519,5016,4444,1085,260,2695,1590,4466,2917,4349,1780,2101,4936,1072,999,1200,3711,2735,2126,3362,2845,921,5341,3100,4092,3464,3048,2323,1028,3830,5097,1790,2615,1873,1382,1662,4218,881,1431,1460,2010,1863,3867,908,3390,3315,1352,1269,4208,4864,2655,166,271,4304,2511,1792,3547,3318,1784,1182,1613,1288,619,1807,835,474,775,4480,855,4359,4865,3963,4607,2906,3052,1471,698,2670,5323,2577,4369,1324,5486,5396,1583,222,565,394,5348,3288,1638,1994,3429,5198,4105,2281,2532,3275,1172,3064,3685,2047,2209,245,3498,3579,5250,3494,3526,3054,1050,5477,168,2994,286,1174,613,886,3382,2187,1433,3343,801,3891,1300,2435,5192,453,1458,3356,2195,1106,5062,878,5167,503,4643,1206,5358,441,585,1053,2075,2591,5449,2141,4387,5483,616,4244,5093,4352,567,4627,254,119,544,724,4309,976,4733,2849,3177,5064,2268,654,1717,4504,1,2036,694,1382,791,1457,1016,3278,4432,802,734,3691,4986,2414,2373,347,368,2754,3415,632,568,1336,254,656,1350,3084,2718,3667,4593,4929,4750,1245,5395,660,753,5084,3558,3225,359,3490,1002,2657,1742,1173,3434,966,132,1287,836,375,5127,3717,1071,2082,4628,3877,4321,609,2702,1160,1914,2883,2922,2694,3544,206,245,1993,4832,5310,1213,2913,5459,1475,2314,5228,4856,2272,1688,4076,3728,4824,1112,3642,1624,2617,4109,3378,1609,2402,3702,4215,3314,4063,1437,1639,3243,3518,2665,3570,3478,4948,1188,3934,914,2004,3267,4453,4509,1616,734,5218,656,1338,3965,3756,5134,5452,259,3779,2810,3299,5376,4157,202,1733,5387,4205,1387,728,1010,2336,39,4852,4779,4786,644,2812,5177,1484,4347,499,1590,497,3000,4378,4007,4628,1782,2397,2425,4734,1966,5168,1015,2556,3159,3033,1989,5124,2801,1175,2838,1975,1415,3461,4815,4770,4621,4156,1055,4738,229,5108,2898,1637,2732,915,2361,4503,5183,287,1348,146,2444,1058,2906,675,3425,1368,4396,575,307,3467,5059,3428,246,1386,4790,1733,5355,4459,4822,4333,4160,1245,552,1096,1801,1288,1253,4448,4161,4275,4402,5414,1403,3327,2508,3231,2383,2047,2913,1418,2271,1542,1257,2508,5312,2841,4026,3923,4031,2463,1669,3685,5146,4401,1106,5303,5244,2885,4761,1876,2206,37,401,2413,1637,1639,2201,4213,2075,1506,3402,1179,3950,936,1403,626,5142,2681,4989,2379,2965,4435,1811,169,5155,1609,338,1681,5104,2556,4171,823,63,214,3114,4093,3498,5251,1188,2251,2987,1887,2543,971,2384,3240,3060,2497,2295,1166,1043,2125,836,3432,3695,754,1459,582,1134,352,2331,5272,5263,2010,3957,4216,193,4747,1453,5471,5063,599,367,4633,2456,1157,4328,3502,2345,568,1478,3088,5188,840,4622,4650,5285,5479,1204,2025,3076,4981,4222,2633,1817,3681,954,5180,1446,5440,90,4862,1347,1955,3246,3477,4251,57,3614,2441,617,2837,5354,2979,2054,3911,3201,1395,4994,2964,2609,2826,1789,2295,4565,2741,3238,766,2424,727,3931,3114,2123,4690,4946,2072,1706,3513,1787,3476,2240,4250,4111,4083,435,212,3823,3406,3084,2141,103,5347,884,4561,1875,2132,1339,3854,1050,5110,5509,234,3434,3008,3246,4964,739,1646,1238,334,4889,2658,617,5036,2610,2388,3265,803,4559,2812,2811,430,1892,4292,1990,408,2966,2980,2803,5071,3556,4328,2106,717,5351,4390,412,3439,5507,2904,2824,3505,4947,3553,2170,2894,73,1099,5123,4995,4340,4160,2181,601,3463,128,4170,2036,4848,4939,3066,1646,4141,4618,3697,3061,5253,2353,5254,2838,5345,3192,4150,505,1690,629,2784,523,3280,1302,3779,2706,3529,597,4437,3875,1140,2916,805,1277,4729,4661,1065,392,2270,2009,3814,2599,3842,2575,1493,2163,3061,4819,3354,4260,1958,2287,70,1599,4044,5247,1014,4017,4623,5365,1102,4279,1633,26,434,1989,4786,3266,4962,5231,4903,292,306,2271,412,2083,1710,233,2614,4539,4256,4357,3460,3495,5522,595,886,1893,4584,460,5008,419,2013,3944,2486,4798,1284,5256,3724,2454,2675,5411,3573,3445,4134,193,4461,3228,42,1098,3554,4373,3681,1197,2085,1409,2292,1723,611,3263,4680,848,2162,2099,433,2704,5021,4804,1246,64,3065,1168,2958,5455,2721,3801,1721,1788,5304,5434,3888,5299,165,4052,3330,4060,2097,4154,5375,3039,651,5252,2553,3845,4663,2677,1493,2948,4854,3563,4690,3320,2316,5140,1498,2162,2916,4478,4306,2084,1606,640,5107,5307,5145,5072,370,4992,3887,4698,175,5106,3678,309,224,4020,447,4534,3537,4328,2957,1355,1697,5260,1600,982,1985,2700,2202,1858,266,3405,2491,4674,692,2655,2622,1479,3996,2245,4462,2332,2072,2281,4848,2797,3914,730,4711,133,3728,3373,892,1629,2024,4762,169,2913,2233,2066,447,999,4444,3684,1188,230,3171,3500,75,3785,4450,3833,162,5265,530,1019,3691,596,135,2366,1741,4339,2883,4242,2661,462,3692,2331,5204,4452,940,4543,824,5042,5174,5248,2325,4804,3430,1652,3629,5013,3065,1414,1387,294,3495,629,3682,4248,5480,3165,5354,4254,4457,866,967,3839,5235,8,4799,5294,3279,5375,906,3317,671,398,1188,5262,4233,2631,4654,4692,5215,1099,3987,4483,2163,74,3302,247,916,4225,2544,2146,4814,1001,4820,1210,3970,1165,224,5524,769,2235,2594,2620,3533,4573,3228,2718,5125,3277,3427,5008,516,4877,3114,2860,5130,2942,5299,2204,1397,2880,3118,1423,1859,2691,2288,1637,2287,2233,359,4960,1237,3769,73,353,1855,94,3775,524,4700,1640,5435,1043,1918,5181,288,4443,511,1249,5117,2425,568,1785,1776,3302,4362,4175,1088,4146,5071,4389,290,3691,2376,1200,1007,2198,3957,1330,2896,5010,3115,2601,2164,3835,1576,4042,3441,1543,3758,4544,3494,701,4351,729,3711,4050,4118,2716,1211,5493,2444,129,5444,1867,5479,2691,423,2016,4205,1950,442,4787,3371,3959,1009,1633,2730,3584,2696,5488,185,3717,2120,2394,1413,936,2596,82,2307,1033,5047,2552,2977,2010,3183,3353,1173,2243,3139,1676,4990,91,1639,2554,1037,1783,4786,3301,3330,3956,2205,1414,1331,4276,3127,4690,1071,4820,1673,3138,1001,1082,4731,5457,1654,2090,2586,2793,5191,3083,5499,116,3757,5109,1796,5238,5140,5178,5317,2,1012,1641,4024,2913,5518,3677,1951,926,1604,4267,2603,983,3125,4644,198,2867,3919,3528,3562,3480,4802,5246,5057,854,1524,2840,3256,1530,2216,4730,308,4693,5073,414,1720,2201,3601,4773,5355,1503,1280,4675,4861,3563,2861,4964,2769,2077,1120,772,3737,3989,3218,1711,306,3675,369,3098,2403,3909,1035,3756,1230,3782,849,2515,4874,3148,5053,4262,3279,1321,1552,1915,1980,4870,672,435,4856,761,837,534,1378,2437,2530,4082,4391,25,1137,4597,2573,4979,4374,2130,911,410,2731,1104,2076,1986,5152,858,1393,25,4844,1099,71,4653,4533,2298,2591,4380,1546,4132,199,623,3013,1332,5518,1191,4295,3348,1108,1522,4100,4793,819,985,1837,1156,1641,4753,3843,3645,2230,476,3821,620,2138,1949,2884,4516,1922,1866,4453,3710,2025,564,2632,4746,2455,2871,723,1991,3982,3932,4501,2276,664,1762,2323,1358,968,2232,5063,4940,2290,4879,1931,188,841,3038,4790,2201,2624,2316,5057,744,4832,3169,5461,1912,3290,5071,2004,3504,2215,2880,4429,4543,1324,4485,1611,281,3234,5089,4191,1428,3600,1254,2992,4014,2642,3924,5291,3900,3843,2463,3204,4402,476,3776,1857,4305,3435,330,5369,2959,1228,4319,658,3671,4987,4212,558,1150,3349,1504,2175,2726,3592,312,1000,468,1604,252,634,2005,4374,3603,1448,3731,4922,1521,5021,1284,3647,2868,2920,4206,4622,2731,1285,1088,1413,3980,914,3964,4722,1499,2722,3078,5359,3325,3625,4656,3324,4640,5193,2400,1874,2299,3724,132,2233,747,4698,5408,3520,5523,2578,2822,3741,4778,877,5164,5095,5131,4055,1164,2422,1839,4061,3266,1626,2997,3572,2221,1167,3785,4600,103,3012,2502,4064,3086,4393,1915,2478,3032,2379,3768,929,751,238,1821,4822,1705,291,1370,2272,3565,4407,6,1663,2996,3353,1504,2870,3687,1454,4009,4363,2552,3954,2905,2599,4346,4766,503,1099,4418,5260,3651,1872,425,1710,938,5162,1796,1938,1606,1549,4810,4352,5336,3571,1192,188,549,140,3482,369,5135,5525,4355,3401,4623,1285,2755,2282,2209,4933,378,1972,5359,1881,2498,4517,1660,5212,2266,2333,1314,4786,3729,1902,3467,4090,5400,24,885,5520,2453,1327,4749,2610,1073,925,2137,4359,4995,1682,4535,548,1399,3302,1545,1521,1623,4163,2393,2966,4676,3996,1465,4467,3623,1161,1672,657,3579,2627,1183,1032,531,950,3132,2532,110,5461,2362,2619,5417,2666,1226,68,2438,2353,4226,161,2961,5150,1006,4560,599,2763,3922,1934,3260,5415,1746,3482,5128,1733,3039,5462,625,960,3696,871,722,3745,92,5277,3838,1932,1317,689,2570,5350,833,2861,3075,2042,1846,3816,908,1073,3398,2597,4715,194,3128,5334,3858,5410,1728,5189,599,1276,420,4088,128,5131,706,3751,4010,1940,987,2550,5144,2620,1122,1854,4698,4982,3679,3449,718,4480,4204,3719,5097,3310,724,2059,3041,518,4889,3411,2072,3612,5438,795,4644,3435,3232,920,1841,1081,962,2103,1464,5231,4995,873,3135,2128,94,3533,3399,2350,2504,889,19,2135,3919,2432,4401,2156,3500,4007,4049,3803,255,4243,3303,2490,2817,105,4643,1685,3324,1320,691,2282,4535,4369,209,1743,2872,3411,2393,135,1078,3415,5307,615,3535,4729,2873,1519,2769,490,2657,2753,760,3650,3092,5414,2075,4091,4611,1930,85,3595,3930,3227,4515,3240,3107,4170,3481,4652,1846,4989,4167,216,3609,5152,2181,3913,3758,4551,3302,4552,1026,1542,555,4504,2164,1540,3923,2889,717,1343,2147,4037,2292,3889,421,1969,132,3180,5002,2436,5351,1412,1609,5040,2825,1453,2488,531,3036,4789,517,2094,313,5382,3058,43,1608,238,4745,3969,1609,3302,2864,1723,5327,1024,15,3484,2768,1837,5207,3783,1500,3363,4208,2840,2185,4147,1864,415,3123,1470,4707,3591,1892,2151,4276,1397,4572,4332,972,4761,1414,2877,4265,274,5111,635,5222,5343,1650,167,4501,1026,2045,3158,4007,3512,3067,883,4481,799,5312,548,4223,3743,3083,4625,3578,157,1689,5397,2012,1467,2267,37,882,1801,1243,1209,1971,2570,102,292,2365,5040,5337,3998,2566,1982,4702,2988,4571,697,83,3700,3286,2127,1135,2382,1495,1643,1393,2192,1810,1186,4623,521,7,2954,1938,3334,3695,5298,2100,133,4844,4246,1018,2597,5397,861,257,4980,2678,103,2179,3501,4560,154,3141,4531,1127,2266,1457,871,1855,3255,2642,3381,86,5175,4083,3501,54,5490,3488,4764,2791,2074,407,3914,2819,4387,2981,505,482,1185,2227,463,3549,3776,2675,3211,5286,2368,3893,893,4057,5011,1333,820,4740,5206,3815,4186,379,5000,3964,5001,1645,4235,3224,1176,1651,4655,262,5126,1736,1767,1737,3536,3125,1012,4089,1429,2832,1798,4913,3839,5128,4762,3076,2435,4321,2145,4180,3983,1945,4649,4617,1956,3213,5171,274,3048,82,2380,2010,5125,3978,3409,3818,2297,785,2384,5206,1101,4435,2699,2825,713,3759,3497,3959,641,798,5113,873,2373,269,5184,785,1356,4428,2374,3992,4021,1170,986,278,5276,3036,2621,1221,1575,3527,2183,668,5339,3396,3603,2865,3527,2744,2410,1813,1014,5408,1414,2628,4701,258,1082,3211,4446,543,259,4438,4213,4475,1107,2847,822,3388,3954,4654,4349,3197,3463,610,29,4478,2647,366,2031,3829,5474,5299,2871,3176,4359,2312,4536,1072,4549,5137,2549,487,3194,4773,4453,4946,1199,1898,387,2021,3232,4777,4203,582,4195,2289,3896,1724,1407,4760,305,5163,1946,998,3214,3310,5213,2202,82,1253,3886,2117,1204,2759,2906,4554,4013,3097,1281,4283,2315,4893,5115,3100,3084,1892,5199,5057,3592,4687,3021,4892,280,4816,1392,4573,3537,1081,2545,1016,4158,3629,1772,2962,4250,2187,602,1267,1071,3774,668,1532,1825,2115,2904,4533,991,5483,819,1767,2639,352,3687,845,692,2760,4774,4506,4310,5469,3377,2657,2254,1023,3044,465,76,1089,1090,2192,2823,4967,4381,4198,3495,3835,894,5481,4425,2087,1967,4155,231,1538,3754,2872,5114,4245,4464,3519,4981,1566,2228,3933,3050,4279,15,1997,591,5161,3021,3418,248,2616,2002,3928,2461,1510,2246,716,1778,4462,3668,3017,3679,4561,1375,194,4214,5129,1595,3325,3614,450,5110,3024,1814,4414,5338,3074,1960,2509,4149,3407,5374,5049,1977,3661,5508,1528,634,1831,2190,5507,1037,3743,4546,4108,4656,2095,1839,3947,4733,4292,3321,1698,1225,146,4219,1705,1111,680,1183,191,288,869,3474,93,3372,415,489,3451,2386,1114,3418,5094,2575,788,3764,2762,1952,2572,1561,42,2380,4410,4119,3630,4162,4147,4343,14,751,3466,2648,2920,1570,4833,3070,2884,1227,4474,1422,5399,3924,2596,969,1630,4417,4021,3554,246,3994,1136,1389,609,314,2162,3561,610,5079,2643,4877,4298,5014,916,3956,5062,5463,858,4912,4588,124,2445,5179,3112,3835,186,380,3036,2658,1918,4796,2163,4172,1243,4394,4931,4137,1992,2794,4615,512,186,4611,994,3652,1034,1320,3193,4669,2534,3428,1588,5438,4215,4088,3216,5345,4953,85,1921,1033,997,1589,4971,784,2433,1092,4738,1615,3180,284,2968,1503,4161,2920,4442,2785,5058,2509,4172,317,1953,1984,5475,45,3745,3959,258,2407,2230,4363,3145,175,1951,4720,3499,1352,1404,3807,549,2636,1037,4212,2750,1082,375,2888,1431,3967,1221,1261,3214,3335,4626,1897,2666,2254,4895,2144,3980,2859,1653,1664,5352,731,5514,24,4062,4173,4588,4440,1457,5091,4790,284,4306,2093,1433,3296,2374,446,1518,2441,426,3563,1317,2895,1214,4922,1262,1046,2383,3547,5510,4810,82,3457,2459,2225,3733,1366,2910,2655,2224,3393,3271,3983,3526,1063,306,4051,3816,1065,1318,3860,3715,464,2928,5214,344,227,5000,594,3620,3957,3859,4704,2289,4032,3938,4467,335,5366,4405,1826,776,3617,172,4426,124,4013,4808,1629,3784,173,4342,1706,2977,2156,44,1140,4278,1446,88,2085,5465,3349,4475,989,1071,2260,4748,4126,3868,3369,777,2244,942,3579,1272,5379,3996,4708,4534,1163,4449,4704,133,3529,5211,1178,3590,2155,3948,620,4297,5264,698,1082,1965,3293,3976,447,5314,371,4314,3797,2334,5507,1296,4557,2041,4137,2224,2507,5464,1256,3674,1825,598,1955,134,1035,4709,279,1232,3132,3401,3333,171,1873,4740,4457,96,4665,2638,3865,1206,4652,2229,2684,916,1275,4063,5444,4783,3580,2780,2455,5043,5069,2425,4569,597,5280,3667,2753,5217,1889,2113,4047,2101,4944,419,255,4155,369,3427,3277,873,301,2987,3896,4473,3335,5016,1824,2785,249,138,5017,4723,734,4194,4828,322,109,581,3431,3836,393,342,4374,1982,4575,2873,5136,4772,1981,3309,2004,2823,1924,3937,231,1024,3241,2878,2919,3763,1944,1970,4238,852,4607,4116,5435,2874,4139,3644,3673,2342,2366,5057,3514,2959,4911,640,2892,2602,3938,559,1351,2469,5198,3902,4299,1407,5118,423,2453,5072,162,3250,1874,1682,1574,4124,161,4308,1033,1665,3618,1336,4012,3840,1513,160,1158,1717,375,2264,1446,4181,2540,1622,790,2032,5169,3935,2260,3813,463,5159,2321,742,1669,1439,4192,3234,1558,4903,479,2329,17,3862,713,712,3033,4551,1328,3644,5420,646,2782,488,1072,4056,2655,2800,992,4452,2902,2427,2118,4660,1479,3066,4755,1254,2403,1339,3041,785,4384,2127,3151,2280,2135,3523,3825,2380,5047,3865,1471,2270,975,3127,3351,4202,2978,513,3963,5188,4054,659,4524,3559,126,620,4997,3053,4279,3361,5022,4031,2601,4835,2673,5079,3953,3657,1632,5448,4770,1068,2863,2985,1167,3885,2799,5096,1756,4783,1664,2551,3419,3233,4005,1907,1859,3234,2104,2882,305,1940,841,5411,5507,1854,652,3561,2391,4680,2264,672,2252,3690,762,1663,3805,2843,2573,5258,778,3380,2874,1898,2035,2686,4709,604,1824,3561,1364,351,1548,4343,4199,3948,4680,4689,4912,3785,3097,827,326,1935,3486,4640,4106,4950,2464,4612,3941,3121,627,4140,5155,3346,3887,2343,4208,12,1810,4527,2126,4619,4299,192,1181,3276,4392,2470,5193,4401,2743,3132,231,2100,2970,2133,851,2059,4397,595,5223,3674,376,1452,2200,1007,267,1009,2723,3207,510,3899,5081,536,3595,1961,3870,1197,3454,1647,5319,3390,4202,1223,3521,4743,317,304,2460,5088,1635,2473,3771,971,4727,3264,1262,2680,5084,902,577,1842,1728,1187,3173,883,1105,3225,4960,5311,4117,4426,2180,4233,2571,2942,1617,1806,3131,3460,5373,5304,4912,5264,2862,2235,349,3878,3631,2441,523,3525,5088,3097,2254,462,436,3717,3669,2360,1780,5125,1198,967,3059,858,3423,3021,33,3876,495,4640,5236,4672,2130,2883,4145,145,604,5152,770,4729,3275,5295,1721,2402,304,3848,4319,457,5227,648,2014,402,2707,226,1882,4777,1061,1182,1569,4233,1709,2543,2828,1344,2445,2046,1292,1755,5338,5441,2113,1002,2241,2152,3630,1664,3745,3251,1427,3331,2792,1325,2250,4825,645,3920,243,1965,3109,801,2670,433,3420,1408,4346,363,3937,5221,3551,1264,2119,5068,703,4469,3369,5437,3865,631,3269,2197,3232,1346,2152,428,3454,4495,4641,564,4375,3289,2385,4042,2379,3883,848,4158,3794,306,4140,1037,3256,3654,267,3886,2777,5333,4525,80,766,3144,2652,307,2477,3973,730,4295,1894,82,3327,2634,4184,535,3147,1249,1777,1541,3729,824,1835,3929,528,3768,2331,852,3061,3801,2749,3103,836,3277,4331,2561,1260,1672,3222,5219,4315,1216,2846,1645,903,625,5400,3583,1728,53,4912,4450,267,1077,2050,2147,1087,1994,1935,4705,1472,1696,2391,217,1440,1396,1760,2471,4244,4445,3454,1304,2953,916,1674,841,4653,81,2915,3124,907,4999,2340,2870,2601,1811,2299,4326,4926,900,378,3438,5026,475,2426,4477,707,3522,116,5320,3380,2488,788,3376,2236,4468,2118,3323,2005,3797,1516,1223,4028,2095,2314,2782,5242,2401,4329,384,1925,4941,50,2301,5236,1135,2470,1723,383,4717,4852,3484,4106,2964,4810,4402,1197,2157,2036,5120,2953,2103,5426,4337,3847,4764,5129,1976,4336,2797,1342,500,3238,2342,808,423,4562,4165,680,1094,3777,2972,5519,4699,2774,999,1869,4147,1693,1713,3626,2904,5064,4268,2106,5106,1898,351,4065,2317,4361,568,4445,802,2044,3948                ]
        
ttt = [233,2000000001,234,2000000006,235,2000000003,236,2000000007,237,2000000002,2000000005,233,233,233,233,233,2000000004]

tt2 = [-185,143,-154,-338,-269,287,214,313,165,-364,-22,-5,9,-212,46,328,-432,-47,317,206,-112,-9,-224,-207,6,198,290,27,408,155,111,-230,-2,-266,84,-224,-317,39,-482,159,35,132,-151,70,-179,104,-156,450,-13,216,190,238,-138,354,171,-398,-36,417,26,-27,-142,478,-362,-91,-262,-11,469,248,-286,-269,-69,-221,-70,26,484,-31,-236,-173,-380,-8,312,-138,-96,23,-7,39,-345,269,156,349,200,52,193,152,168,159,181,272,-259,210,76,194,-31,139,392,-16,-151,50,166,45,9,44,-179,151,-8,75,-277,-18,49,314,-332,449,24,362,88,159,14,-279,232,211,-206,-192,27,238,-339,-79,30,-370,-29,81,251,-189,21,-202,-41,198,51,-6,172,108,26,-168,316,271,-76,-20,-249,-111,47,-86,303,35,127,113,-181,289,-105,-30,-16,-9,95,-144,-422,198,320,7,-227,-161,447,486,-406,-121,-280,-76,285,-453,42,15,-335,-189,-154,280,-206,68,-313,-375,-401,47,184,-320,369,-146,-60,150,378,87,102,138,-54,169,33,-339,-19,147,333,84,92,-57,104,76,-239,99,300,217,-140,153,-344,-103,-6,-37,399,323,-138,279,-259,217,172,-94,-55,29,462,-327,-177,-163,-444,-84,-281,-87,350,-180,20,0,46,331,-15,-244,-370,69,-194,-30,-85,-112,-235,-242,-188,231,123,-233,-29,113,-294,90,64,-3,-364,55,120,-48,-323,99,-76,-70,79,-351,300,-44,-30,25,334,-199,-68,-451,19,57,293,-188,-16,-46,-392,-162,50,-304,23,166,-130,-146,-35,-141,-25,124,-239,114,-104,285,-108,-137,177,-129,-443,341,-112,134,-293,-181,278,203,442,-206,-20,457,-267,171,-321,208,-4,8,-16,-474,-214,-18,-139,-129,-239,-152,45,443,160,-226,338,-384,198,-77,398,296,-405,-156,290,87,-423,-15,-374,127,259,-20,-62,426,-86,-44,184,-207,257,44,-106,-166,260,-181,-282,-68,-90,-39,-3,375,415,20,-207,391,-201,-143,60,242,-192,-74,426,-86,1,74,208,107,-92,114,-37,145,-216,99,319,-298,124,243,73,-127,-139,56,298,24,-354,30,-166,175,82,187,-24,112,-22,-392,-166,-376,470,139,284,-93,162,-160,89,-240,36,-380,-58,-249,104,-1,-172,198,-70,-381,29,20,305,-197,-253,-145,72,98,-375,-152,91,96,-64,170,142,66,398,97,-19,-298,-175,118,-77,-361,354,-29,-47,71,231,-174,-11,-347,-87,36,-318,50,-157,-182,-348,10,96,-241,-82,473,-50,-10,-75,-148,71,20,119,-37,-188,35,65,-346,50,256,-20,-80,-358,419,6,-341,24,-113,-169,108,-488,-334,249,234,-73,-208,19,-264,-89,-41,66,-3,17,-95,2,-143,-11,-348,-324,-366,-183,-148,-76,-197,201,57,-94,-1,0,43,-6,70,-183,71,-304,58,-35,359,103,238,93,331,59,24,-145,92,-34,3,147,-241,-54,-90,1,313,-116,436,162,258,468,-154,-31,111,207,-484,-19,440,201,9,-230,11,-355,246,-78,295,-84,97,43,317,158,-78,183,132,-265,360,-398,-284,-69,212,112,-236,-111,108,266,200,386,-355,36,-3,-3,304,205,-142,-250,8,-45,-35,-165,54,390,175,-44,-255,-207,-64,431,-186,-279,-126,-65,-211,42,246,27,-302,-342,-386,-193,-123,216,71,-391,-343,3,-15,-486,138,142,463,27,-126,-84,39,188,145,402,-260,41,423,6,-86,10,418,-4,-37,-256,-345,-47,49,314,-169,-81,-351,218,-163,0,-6,-432,189,245,-167,92,2,-83,-176,-312,222,108,-18,-119,193,-84,87,-299,220,2,-323,-61,-300,-142,142,223,90,211,107,326,-43,247,43,-27,-114,187,260,-25,-263,-69,-194,-316,-73,230,95,278,-176,37,134,290,-166,-78,135,259,146,-148,1,-210,-209,-59,-92,89,-216,-250,-411,-181,-78,419,21,-370,-9,-154,-24,-306,57,-27,-254,86,-364,71,-99,-70,-79,141,206,-187,227,-362,-293,81,313,-311,-208,-401,-206,-282,-123,86,3,-22,-324,72,-126,-84,216,-411,19,115,-393,-102,-300,275,-376,30,-403,449,465,-243,-168,-7,-43,-23,-219,149,-43,-14,-139,384,-23,15,-10,-263,-375,156,158,-76,27,-263,50,174,305,22,150,-94,-368,-142,61,119,154,-247,-52,-38,-81,-105,402,-21,-148,2,-28,-164,-387,358,216,168,148,200,4,-222,183,281,-428,-13,2,-289,-459,-188,117,193,140,463,-56,159,29,-250,216,143,12,151,48,174,-105,-83,247,324,-204,-181,71,-184,411,-52,-110,-220,168,46,383,-223,-56,24,322,50,-14,-206,-84,-2,-173,219,150,-356,331,-78,-123,468,-184,243,-160,-96,235,-70,214,253,113,313,-80,201,383,125,83,-124,33,223,-48,-55,-175,-364,-98,52,223,45,90,-23,18,141,71,258,-214,-142,-230,159,-319,-440,219,-217,-72,198,56,240,210,76,22,46,-264,159,-153,-189,-212,317,-420,-71,19,-46,64,-37,-15,-397,-27,-236,-135,268,-223,112,392,-300,371,-209,51,109,-465,-219,-155,-138,77,96,-10,33,77,-366,491,22,83,180,-70,-404,-312,-384,251,8,305,-316,157,-318,435,100,274,123,-180,499,-285,221,-135,-199,145,234,12,-13,-164,133,115,-160,315,149,-36,-164,107,-74,300,-34,246,219,-148,-182,26,-143,-321,73,-140,-395,-119,169,38,-148,290,5,319,-126,61,-289,13,86,98,170,-153,-326,-213,152,23,-19,253,154,-116,-3,191,-13,184,283,-71,-116,-315,278,160,173,-151,199,441,-208,-385,95,-338,179,466,37,-50,386,-343,16,162,88,187,-247,328,201,1,-127,274,-152,117,-50,71,59,-33,141,-245,321,-258,-112,82,40,184,310,359,-92,-176,-65,137,-9,-168,79,-66,106,56,3,-176,83,-379,451,64,-101,-65,-403,-193,-31,109,368,-454,119,-340,175,346,-28,275,-293,107,-262,-311,383,140,-7,70,122,-251,-2,-133,9,157,113,349,151,-94,-37,-24,-340,264,-286,92,23,36,-364,331,-419,-107,-342,63,-65,-364,-262,-19,-271,-259,-123,140,-32,29,-38,-401,-491,41,320,-67,-82,399,-294,176,152,-183,173,185,162,100,399,-255,66,194,178,44,-208,-354,-152,-336,-3,-23,-335,22,-71,-244,-246,166,272,227,-350,221,279,66,253,-493,199,-249,81,-189,102,-91,-197,-445,-206,67,-50,-384,-116,-295,-225,22,-350,-364,24,269,-285,34,-123,5,-207,-482,92,-418,25,280,-330,-351,-79,-87,4,-278,251,71,115,214,-141,128,-193,111,145,-215,-116,-216,114,72,-460,142,67,-171,-252,409,27,-173,152,176,-300,-288,11,270,115,-246,-323,192,18,272,-147,61,114,49,155,33,-160,-134,43,206,322,-96,-89,105,-60,181,-78,-249,123,-30,2,-304,166,72,31,145,-131,222,36,-108,142,69,149,16,167,-85,86,-282,311,57,306,-46,-98,28,107,405,-323,-427,116,-29,-156,99,408,-12,120,-57,79,-204,-162,19,-244,82,-221,178,371,139,309,-278,118,102,175,-429,249,82,182,-231,159,180,113,-128,183,-149,18,-126,-34,-319,24,-220,25,-223,24,136,-373,-58,61,-53,-189,402,-104,-42,43,90,69,174,-22,-197,-183,424,-111,42,210,152,-27,122,350,-358,259,283,-222,131,337,28,-259,108,289,-313,-178,-316,-433,-6,-31,-150,285,-56,6,261,5,484,-76,-77,85,178,-279,87,204,77,65,29,-138,-202,-80,48,-407,-285,-204,358,67,-86,75,55,27,217,-183,-225,280,-55,-74,126,279,-67,116,-297,7,-169,201,-147,314,-268,-469,81,-401,-155,47,314,-175,361,-314,-147,331,340,-121,-42,99,164,36,-158,-82,226,-97,-231,48,-83,-132,158,-147,44,-182,191,-320,268,145,-14,89,144,-213,141,346,-266,148,-286,-10,-97,129,17,-9,84,-141,-326,7,-197,321,-447,110,-80,376,367,-122,331,72,-190,-68,124,268,-44,-20,-120,131,168,151,5,8,-86,-72,-335,-255,-408,36,180,-407,169,213,-292,-223,-244,60,-271,-178,143,-274,25,-466,119,127,-470,-323,392,23,-291,-71,-123,-12,-186,-3,-51,-15,380,389,-204,59,-292,26,-4,83,-5,-19,6,-223,-228,259,122,375,60,21,297,212,240,-220,96,8,19,417,-44,-121,-214,-13,-252,-74,-9,-100,-126,198,19,425,-156,73,338,305,465,-9,-329,-79,-380,-167,-93,-151,-65,-299,122,161,-48,-72,-243,-134,-420,-61,228,-106,240,222,40,194,-248,240,-276,-273,-468,-149,-345,295,-433,60,-425,94,-239,-301,0,460,285,-281,40,-207,442,-89,-277,60,335,169,-472,115,39,-467,234,317,-175,192,-41,-438,-101,283,40,139,-178,1,-4,101,81,-178,-75,-204,27,18,-215,-97,-311,413,-230,-38,290,254,-173,-33,355,-145,-30,-89,-123,168,-118,-328,26,-99,-221,-13,69,60,273,475,18,-396,-134,-140,256,-256,-144,-195,141,-334,483,267,154,-234,134,-195,-277,151,-28,-37,339,-190,-208,185,-242,121,188,329,277,-99,-364,-136,-280,-45,-320,160,-32,182,212,42,237,85,-130,140,-233,187,-13,-171,-1,176,-45,-6,192,-350,-8,78,-241,174,121,-376,-127,-16,249,-335,-271,441,-32,-229,-109,80,-164,141,326,-137,-96,452,351,-322,186,458,-78,152,203,149,-493,-15,-154,-85,78,-240,309,181,37,-189,-178,1,-311,198,55,140,105,260,208,82,-281,32,335,-371,-46,129,-116,45,-225,-61,59,249,125,-193,162,10,25,172,-99,-134,-433,-73,141,-253,125,-260,-67,208,-421,-6,-306,473,306,12,-83,-339,-90,-179,-388,349,-166,165,-169,-37,-132,-43,33,375,-443,12,-377,-140,406,-15,26,45,-207,-173,57,-49,290,109,-254,7,86,-100,90,-15,-84,343,85,-184,359,199,345,-194,-246,397,-173,-281,-154,-2,-9,-50,91,-254,37,147,72,-56,93,121,-14,-52,-124,-148,194,-170,-51,-143,95,143,-97,-292,20,161,-121,-397,70,-28,42,41,-180,86,-12,298,304,-63,65,-43,208,-238,-473,259,16,-275,115,62,-103,-161,-383,-88,-53,349,-114,308,13,62,-263,-198,182,-48,-55,417,-17,-29,271,69,-175,120,-190,-154,-263,-138,5,296,-22,-98,-266,-80,-144,374,-236,-220,334,120,-168,-95,144,-291,-137,-116,59,-154,-87,-175,133,-109,-60,278,-209,102,-237,355,-79,230,45,-2,390,107,-152,29,160,73,-482,-234,445,-181,3,-243,-140,-163,48,-28,410,-98,-226,-148,112,236,-26,380,-246,-189,263,-151,422,-269,135,-238,-238,224,-269,112,-336,-322,-167,-61,-123,-267,-219,-86,-11,-164,70,-229,133,294,98,176,157,242,-98,218,311,-381,247,-223,341,-229,43,394,69,-79,-110,-99,-68,61,159,-98,-139,194,-130,275,-61,-182,-31,313,-75,-32,-252,145,-311,412,117,414,325,170,291,353,-105,354,254,-194,-239,-64,203,-93,161,-298,-274,440,33,-10,266,104,-256,-164,-15,-209,346,-194,176,463,365,379,-66,-46,7,90,236,-225,267,-18,403,43,89,299,146,-241,104,-47,80,-245,299,-38,-57,48,431,-194,91,33,210,-407,-152,244,272,117,-451,383,116,-146,-102,-310,-174,30,-275,-101,5,-52,-74,194,-38,17,-320,-138,-23,-86,-85,159,-74,-197,-57,-199,-30,-129,292,-142,-194,-31,-92,478,-225,-371,0,158,113,-294,-391,-207,146,-210,-404,-458,68,-33,-25,-25,355,53,8,129,54,69,84,-21,151,-72,161,-108,62,-103,18,-82,119,-88,-426,70,-355,-260,-315,70,-421,11,-218,-10,139,331,231,116,-233,74,165,-173,-85,12,-285,71,162,-39,116,-232,197,-71,89,-47,135,-340,82,63,79,203,-181,129,-318,-240,108,-76,-40,118,-276,-280,-209,122,367,245,45,-373,-148,189,-89,233,-101,-124,55,400,-98,-434,292,415,-117,97,302,-339,90,204,136,-123,271,92,55,-113,-422,429,-295,86,-452,-112,65,-146,-243,78,-140,-268,-74,360,13,81,-179,-108,-59,237,-212,-289,402,31,105,38,147,64,-180,226,-85,-337,-124,110,29,-123,-22,-449,7,379,118,-140,50,-285,44,74,-167,297,-114,-77,-220,101,184,208,-381,-98,-199,439,-304,-306,-132,315,-282,217,-317,6,223,106,-101,-29,-189,-450,-409,281,-77,-60,-17,60,-160,44,35,-30,-44,-266,-97,-291,127,-292,-130,304,154,334,357,-21,-206,65,-21,-117,98,42,213,-242,457,-143,-144,29,-10,-105,383,-11,-221,-452,-201,-186,196,-201,276,-67,-269,326,312,-53,285,210,355,70,121,-67,-66,-10,-102,-27,-480,147,-214,51,109,152,-370,-231,-296,-96,147,-215,-38,-61,178,-21,-438,106,3,98,-33,-16,-447,335,332,16,-269,-206,-174,132,-164,-51,-222,-456,99,278,-413,83,-62,45,-467,-187,102,-26,149,35,158,202,325,-416,6,155,104,339,419,-271,80,-238,355,372,-6,-37,55,-174,-68,282,-18,300,11,-97,244,-59,20,76,-67,187,-205,-62,-279,241,-69,-109,234,172,198,25,-89,-37,-139,272,115,197,133,-217,-152,230,-31,-186,257,-319,-77,-103,383,-112,360,-52,24,177,-59,387,-91,51,187,-176,-340,197,-138,-317,-185,209,-244,-81,366,341,-218,-199,-384,-179,129,245,299,114,0,-128,368,-316,-79,83,51,-161,-165,31,70,-297,-430,-194,111,-161,27,-111,-104,196,19,243,432,-19,-175,119,-146,108,-159,21,349,424,-154,-138,-22,130,-90,278,272,-39,78,-1,247,145,465,-145,-482,-54,-56,-155,251,180,-4,-231,-234,395,192,-74,-93,-109,267,-182,102,195,194,-376,64,-33,250,-134,342,25,-65,-217,-137,84,229,52,-28,284,-132,207,-162,-11,61,253,-184,-351,-266,-140,294,-120,-318,297,212,-346,80,237,-198,327,-266,-100,-183,-477,1,47,-8,175,239,-340,38,-30,63,49,-165,-141,98,-63,412,-24,-353,-64,338,293,216,159,132,145,-15,9,80,-11,92,243,98,280,251,-375,8,-203,397,-55,253,242,-27,8,-121,376,-95,189,-432,-70,-24,-53,27,28,-77,275,-31,55,319,-340,54,69,350,-488,-469,-324,5,-102,-290,10,135,158,76,-276,-26,-123,122,147,-140,-114,-32,-135,171,-372,142,-16,-127,369,-459,-302,101,102,163,-343,102,-19,-136,-283,-144,-393,-161,-209,26,200,-140,-88,60,131,-10,308,354,370,-256,23,-237,-177,-134,302,160,8,-392,136,152,-367,-329,-164,147,-63,-125,71,338,279,239,-283,-54,-186,1,-361,288,323,130,-103,216,348,273,66,348,-366,393,-11,-390,-391,46,149,278,-172,-55,-81,79,0,61,78,-176,-41,146,-164,-44,317,172,25,-313,301,114,-454,110,163,142,258,-195,46,61,366,47,287,-15,171,79,30,-74,-15,-329,59,25,226,-51,31,-437,-318,-160,-144,110,-32,366,188,-170,456,-159,46,270,-129,189,-7,-57,217,-13,-142,333,-280,88,189,67,170,-176,395,295,-198,173,76,171,196,36,-125,168,154,-227,-30,-407,278,-52,-122,127,-187,-32,188,-352,-179,84,432,-229,42,18,-73,-179,-155,247,285,99,357,23,10,-19,226,-8,-85,182,89,198,-284,-102,0,-37,-33,361,277,257,307,90,191,-150,-298,-117,-170,-252,-405,-192,-137,-204,-46,80,-467,-221,-26,-244,-335,-219,-265,-121,-356,113,454,-283,90,383,-42,-308,349,94,-232,153,148,119,-189,-39,407,6,-97,-234,90,438,399,-88,92,40,88,125,18,-3,441,93,-478,152,111,-219,-502,109,346,-113,-231,66,-135,-37,153,258,146,427,-128,-388,-241,-58,251,-211,-26,-401,-302,-28,-78,-86,-149,101,398,158,-256,-386,-23,-12,-259,183,-114,-147,296,-37,-318,-276,33,-343,-84,268,-200,25,-50,-205,63,-19,0,185,-154,270,-207,-372,324,-74,-28,192,-26,123,-348,-24,-263,276,201,21,76,-164,-231,-204,-316,73,-395,172,19,231,219,133,258,-74,-135,-95,-2,-291,-304,-324,226,-345,-125,-205,159,225,-477,275,199,114,-27,163,28,18,-321,-210,-56,-46,-133,70,-453,71,313,-225,-151,-99,-71,-16,0,360,-144,-154,49,-5,-272,244,152,142,-210,-250,12,-47,333,133,160,-155,-88,-240,34,-61,189,-336,-264,-119,-111,119,77,-439,42,-94,-41,-230,-222,103,197,-133,244,-81,237,-420,-52,-199,336,263,-178,68,-151,-149,-214,-110,-283,56,125,228,84,-475,160,112,115,6,-364,377,42,189,403,123,167,-119,-89,6,-258,74,-265,-57,214,-312,4,-194,-464,128,-220,198,-59,-65,258,53,-28,19,70,84,44,205,-55,-180,-339,246,251,364,50,-109,338,-238,129,-27,58,278,381,28,-80,22,296,191,90,-130,-160,70,132,-212,373,456,264,-54,197,-41,-61,187,-56,-159,367,-327,-374,-85,64,-276,101,-359,467,-128,-221,66,-460,-126,341,105,350,-59,-359,190,68,378,414,-171,-116,207,261,-499,-274,-5,-150,241,63,-28,-357,-86,49,-220,-328,14,207,-373,36,315,190,-126,-404,52,-47,-243,254,254,-206,82,45,-107,-109,-119,-383,-202,-307,20,-274,-283,225,-143,-133,-119,102,251,-306,20,-25,-378,-79,10,-350,-20,-27,81,-288,129,74,-437,-48,-95,465,204,-433,-108,-161,363,194,-166,-368,-344,229,280,-10,3,-159,265,-111,90,51,192,-370,-250,-195,-293,-59,-232,-459,-102,-133,353,0,88,-231,32,286,230,-323,-188,20,156,232,-20,155,236,-144,-434,292,392,-192,-21,-370,-30,3,2,-98,52,-193,92,432,64,145,-115,-41,-267,-276,264,419,-68,248,221,-100,290,54,233,-68,-116,147,-178,-67,-25,-113,-16,-121,121,-71,39,-40,206,-149,117,-150,-45,-267,-295,-139,245,99,-357,-12,376,-106,467,98,-20,-132,140,-275,-120,329,-58,-157,-249,-415,85,66,-50,-175,25,-74,342,324,335,-284,-184,-49,-62,-12,-231,-354,-150,-9,238,-103,126,-178,268,152,-95,-12,258,-211,-108,357,-258,-375,229,19,189,-192,30,-50,-11,-188,143,343,102,-34,-121,35,37,50,-58,206,-138,-34,-100,-368,-120,-97,-74,-100,131,-6,396,-125,166,-271,-59,-114,33,48,-82,196,-199,-190,-291,-13,-161,105,-211,-288,73,34,25,185,-246,-41,252,125,-320,-249,-255,-281,69,80,-88,196,315,101,359,92,301,-19,-28,-119,-387,-319,-129,467,379,120,-172,-76,281,29,-115,299,-317,269,-206,-42,-265,-219,-307,-147,1,-207,388,-86,-152,156,206,-331,-31,309,-207,200,149,-65,-154,-325,191,-159,308,-71,-116,-145,192,-126,25,-120,-346,-240,-2,-109,240,-318,-158,37,157,242,-362,-112,-245,-74,33,338,-244,-466,316,60,-269,306,-210,11,366,-258,248,-448,284,181,18,-315,-134,56,-284,194,47,365,-199,-69,-122,150,155,241,56,-52,369,-182,-331,59,75,-243,-154,179,-223,157,73,-56,-315,-337,272,246,-21,123,145,175,26,20,-49,355,-184,-71,315,25,37,-275,-169,75,225,181,421,-49,23,150,81,-134,481,79,-27,-121,-48,-398,248,-197,36,119,-88,-184,-137,27,237,17,-96,446,32,180,55,84,-97,-162,-22,-133,-101,172,127,-193,248,-276,-43,-391,-240,425,-390,135,-345,445,277,-223,-305,-314,-124,158,265,-34,-34,272,-45,-336,75,-378,-190,-148,102,160,-12,21,123,119,-19,-23,-106,-12,-134,498,310,-140,-109,-23,-229,-134,166,421,-107,296,-123,-63,280,154,289,149,-197,-266,-5,42,-102,-167,-135,93,-94,-27,0,-231,-296,61,51,-422,-3,143,-48,2,213,-125,-392,300,210,-225,-42,-201,59,-169,275,-146,52,37,-436,24,37,258,23,1,-14,15,119,422,-236,121,-132,-56,119,-84,-284,258,-186,-22,-397,-106,-245,243,350,151,-307,-307,80,-194,-92,-267,464,171,-166,-236,61,-376,-468,10,96,286,-192,-154,-144,-56,445,-364,293,31,174,-114,-267,293,171,254,-322,-56,-138,392,-65,-41,-98,-10,28,114,379,-23,59,382,-59,35,-9,34,108,-72,-58,235,-271,-162,-18,-202,-141,102,-161,-48,-56,325,223,-429,143,208,103,-402,-291,-218,-178,-129,-64,31,230,49,-356,205,-47,255,344,356,12,229,18,168,-134,-31,-107,-35,-161,218,284,-215,5,365,-81,40,-147,-52,333,-205,303,-216,-36,471,188,-195,288,-73,-85,-333,97,-240,-161,231,-10,186,46,63,59,-107,-415,-28,155,199,80,-300,-17,384,30,-289,108,-237,130,-136,114,-230,-102,-319,158,45,-13,-126,-2,-107,176,212,-492,-512,-247,187,84,233,-318,-61,-62,-45,128,-150,-398,342,-51,-395,83,-188,133,-22,63,183,-466,-132,251,-211,60,-53,297,-245,-28,424,91,5,-166,-444,11,285,-22,-24,-307,283,-379,65,-308,239,-90,-61,-104,265,2,7,-313,-300,-370,62,-158,81,-157,-191,159,131,370,75,-384,48,-26,-122,-312,-129,-76,87,206,107,-50,198,207,389,-37,244,-26,-339,337,178,-203,62,84,103,117,-188,34,-104,393,-421,-62,-226,-223,77,18,-124,-296,433,167,-268,-290,45,341,64,-138,-180,-90,109,-151,-191,-102,-201,90,-41,383,171,47,163,366,-189,196,-385,-2,81,-172,-19,5,253,-160,-82,-477,416,0,-16,-130,-23,-292,-205,-179,54,-40,232,132,-38,65,182,-334,186,-80,-372,-467,4,101,192,-193,-132,-38,-192,309,-78,-329,-306,61,2,-321,287,-297,-148,-32,-102,186,55,-210,75,-58,-1,43,-14,216,-72,-194,-180,342,-156,-133,-452,13,-377,364,152,-383,126,31,-99,278,33,241,-401,133,-164,-85,-264,-100,-45,365,-1,-34,167,-81,-424,-191,-152,29,-187,300,19,96,170,234,-233,-414,120,234,224,18,-78,-446,253,48,86,-194,-129,244,-235,20,-306,206,89,163,130,-49,78,325,-147,-200,-77,262,-182,94,377,24,199,-316,-301,42,114,235,166,293,146,107,148,-58,25,179,144,265,19,-375,217,-125,372,-269,211,-265,-22,59,-56,-214,-288,-286,-272,39,161,-236,-70,-90,-124,11,-14,-51,-136,-79,-266,290,313,-95,337,71,477,193,-61,78,-61,355,-397,-142,-99,-141,-209,292,10,-98,-21,301,151,25,469,-189,213,-95,51,-51,80,147,5,-300,-45,-55,20,-101,-46,-312,-377,-142,162,246,386,-6,287,42,-168,224,301,-221,-5,237,158,-98,105,71,-23,-101,-40,-282,324,-82,-13,343,176,125,-222,44,-33,-76,19,-20,98,35,-42,-189,213,-99,62,-524,274,149,-128,-185,-264,16,-120,-157,378,-52,-321,-181,6,-87,-234,-48,-477,179,-10,-215,7,23,-58,207,-437,51,-112,255,212,289,-90,5,267,283,105,-272,-27,362,-25,264,-218,-190,-363,-214,128,214,30,-104,98,5,249,-241,295,294,117,-382,-107,-31,256,-17,193,113,267,405,-79,14,-345,418,86,-108,-161,277,-55,-4,6,293,-8,-264,-224,352,-357,395,230,416,-343,238,-56,-112,-145,-166,141,331,-324,-449,100,-63,63,262,-50,171,-47,238,-98,429,-204,-129,-362,-256,37,-35,173,92,24,24,-482,-403,174,-192,-126,-28,-255,154,-37,237,-148,-286,172,256,27,132,286,-85,511,-67,-72,404,31,-192,49,79,133,137,-261,-69,75,-190,99,55,-327,372,240,171,-152,153,-146,29,17,233,-267,-94,-25,270,-154,-69,-37,136,403,-81,126,-154,-10,446,-84,-176,-84,-100,98,-75,-134,375,81,-196,217,-207,97,137,-138,253,-374,220,156,-39,84,22,-321,134,-15,121,-106,-208,-243,29,-159,-239,169,-391,207,-17,-255,-6,247,-274,251,-241,47,-314,-323,-276,-232,-183,173,-275,-94,283,58,119,-283,243,-244,-120,-88,66,98,-136,-152,-118,-82,-66,14,-136,114,291,-328,-142,-273,85,-200,-287,17,94,-190,-272,-296,-95,-332,183,21,9,61,374,-273,-137,180,223,-21,138,52,-95,70,-486,132,121,264,4,206,-212,224,-14,30,8,269,93,-83,159,-75,-120,-15,-360,40,374,-365,213,-31,-66,-175,-317,-88,439,370,120,-448,-202,212,9,-98,309,-291,-147,-43,-45,361,-162,-130,-23,-107,145,41,-327,-42,45,-282,-45,-287,437,-21,-467,69,-41,108,-124,-126,362,19,-102,394,124,-232,35,-273,-91,-39,131,36,80,130,-43,-109,-143,259,-75,263,-298,102,-356,-15,-59,126,-339,77,66,-60,472,137,-162,-292,202,-67,333,-189,-149,-212,-21,352,240,-337,-171,133,36,-237,-176,35,-104,-129,72,-230,-272,-49,-334,-288,-13,-9,-376,-356,52,237,-34,116,-35,-30,304,-39,-292,-2,115,-79,-55,-428,-184,-311,-309,-98,220,-53,-23,203,-335,275,339,126,-270,201,-22,347,39,-386,65,423,310,-382,305,268,-128,-405,245,-93,-137,-237,266,-325,86,160,140,198,192,38,-151,-100,-99,-234,-351,-78,322,-70,-8,158,-249,-316,-72,-139,150,-313,381,74,108,-76,111,-340,59,387,-118,-78,-55,25,71,49,47,271,-70,184,168,109,-337,428,128,-158,-43,-262,-53,-92,389,-390,-306,-413,-30,15,-222,179,-194,-6,-279,185,400,-200,55,276,387,-174,107,-299,-45,428,265,85,-213,141,332,-252,-151,215,-11,83,-93,-178,288,256,230,18,-2,99,-303,-21,-490,-166,-167,477,39,-138,-233,-266,-31,72,62,264,-114,-143,-22,-4,-206,-1,-1,190,284,396,-185,-252,-7,-135,-172,301,-31,-342,8,17,275,-308,200,-90,213,112,237,211,167,51,136,-220,-101,218,-206,-1,339,-367,-241,186,377,188,110,-115,-3,129,-397,6,-116,-383,6,-131,-182,-361,-339,-72,258,-85,292,84,-426,388,-434,61,-37,372,84,-63,31,-217,-193,132,8,-99,221,244,-36,193,-137,166,-9,-10,-490,160,111,49,59,71,436,-109,405,-44,-155,389,-196,-99,-174,117,-159,-341,150,-104,-59,-43,-102,-65,-170,-75,143,-247,262,-117,-39,-66,-174,-141,-143,-283,-145,-2,9,283,-103,123,444,68,106,-372,-339,351,-297,-125,-221,133,272,84,70,-84,-335,-205,152,-32,81,170,-364,-78,95,283,-283,248,116,-33,-128,77,-85,-196,194,68,24,111,-205,212,-95,-55,-172,-176,-164,-297,-39,74,39,339,0,-141,-58,-24,109,4,241,-65,-272,66,356,208,362,25,-133,-232,-29,-36,-310,-23,-247,90,130,138,-52,-173,155,-209,-130,-69,227,-115,170,-38,-375,121,-26,27,-194,-300,-43,61,96,506,-138,20,-80,114,-99,-250,-210,116,301,215,-158,-247,188,270,-417,-364,-210,-159,-9,22,-86,86,6,208,-45,-95,-174,136,5,110,44,-256,361,-296,69,7,108,437,62,103,-61,58,-26,-92,-260,425,435,185,-246,-42,207,199,-93,185,-81,375,8,260,91,84,74,-276,-204,-236,41,-268,-177,33,395,258,-21,-120,-402,268,49,470,-87,-1,113,-70,113,51,125,247,-271,397,-220,-213,-231,60,-389,145,286,181,88,-11,340,-11,41,237,458,241,66,193,99,218,15,-234,278,-54,229,-166,-30,-64,251,84,-129,91,-76,271,120,210,51,110,177,-222,203,-124,218,6,5,-149,-99,113,44,207,-246,114,-212,-8,-97,-450,171,462,-98,-485,188,-46,33,131,-168,27,-389,-406,153,130,163,-234,170,-280,-471,212,-325,202,364,-183,257,310,147,130,-84,125,113,-46,153,-224,-321,255,140,-32,-27,-209,-111,261,-169,125,-147,-49,-5,-68,-149,-79,-37,-62,-343,437,83,139,-119,130,-207,-58,-130,-165,50,247,-147,-37,255,269,156,-152,46,374,397,-193,-31,158,136,-39,-94,61,297,328,-410,63,51,-65,40,-280,191,-290,10,-41,-259,68,-215,-9,-72,-275,223,-20,241,219,-186,98,101,418,276,158,-169,227,-265,-371,-274,186,47,177,40,105,-1,-77,150,-232,-326,-227,-17,-128,289,-281,329,-182,190,-133,57,74,-50,-92,-234,345,-54,-357,384,135,-57,-438,291,84,102,-248,316,-340,-95,118,309,-88,269,260,-90,500,191,99,-341,-210,387,-170,-178,80,79,-345,-399,-227,-155,-32,-37,94,129,238,-127,204,-95,-5,-370,139,-39,228,31,-187,67,411,39,19,-150,-214,114,-427,-86,-17,297,-10,-297,-75,12,196,-52,361,-41,-101,-350,19,82,-167,33,42,102,3,-200,-249,363,-166,147,178,-227,-53,263,-49,369,-107,61,170,-8,33,89,-8,104,-95,-163,50,338,-28,-259,-70,112,-457,-161,169,-419,-9,-62,-218,197,190,371,-425,-120,292,-56,349,30,71,97,-140,-67,48,-123,-184,-337,481,219,302,317,-52,-65,-239,-113,456,85,-143,-349,-28,174,71,-92,-101,189,19,105,329,174,-208,124,-180,18,-4,-237,73,-9,-157,123,89,31,38,149,64,-19,10,-29,281,184,-205,-45,-54,203,-90,-109,279,395,429,11,106,261,-163,156,105,-40,-30,23,49,-81,197,18,229,-12,-320,370,276,-162,-122,124,395,-12,206,397,150,11,-317,-219,218,-175,-71,214,97,310,372,247,-34,316,8,118,-39,205,219,157,424,-45,30,-291,97,-433,-272,192,140,-292,344,109,101,113,-266,-251,-116,42,99,441,-72,266,-194,118,-435,93,9,201,12,128,-162,256,-370,-444,9,155,-65,96,-257,107,-186,-16,83,-137,63,219,138,74,75,-36,-318,13,-338,-69,-111,-286,118,25,-351,134,-247,42,-83,213,-106,-79,-31,-69,144,116,-135,-303,107,-124,-424,-280,-12,111,-62,207,198,-11,14,164,-233,121,-93,-122,201,-236,-87,-191,171,-156,-69,19,7,-117,73,-170,-160,-137,197,-290,24,-251,214,36,383,375,-206,488,333,-244,216,-193,-5,-313,7,-399,-162,238,175,-95,185,-256,-295,-125,505,256,-45,175,156,211,467,-32,-393,-364,219,67,65,29,-311,-75,165,56,68,368,286,-315,-341,-2,-344,-105,408,-94,378,115,-364,-62,81,-10,281,-159,-144,50,-75,77,-163,-123,347,-120,-380,-261,19,-75,431,-119,-30,221,-213,274,306,412,253,-255,150,-111,-247,407,-172,53,298,370,124,-393,-124,-46,209,419,-6,242,-226,237,102,-23,143,46,-55,-211,174,-113,496,16,-232,12,95,-405,49,-33,-284,-189,139,43,276,-245,236,-7,-371,272,-76,10,374,-61,-83,98,105,-378,-294,-18,-374,-26,1,386,230,143,-205,135,350,-145,-354,-49,14,-144,-101,109,-238,-174,216,-58,375,-93,215,-88,4,-95,310,108,114,-73,99,32,305,53,-232,259,26,388,229,-402,-134,274,-435,-71,350,-317,-71,270,-65,500,189,-456,-172,213,43,-18,-89,-126,-128,-24,37,-81,341,-373,-173,-348,21,-93,92,79,291,-456,-19,-43,-472,185,-161,380,-69,-170,27,-135,-162,-228,-233,357,-113,-38,-241,-111,22,16,245,15,-218,297,8,-383,367,257,30,320,415,136,234,-263,185,219,-114,-302,449,-36,41,-53,-99,40,-55,-166,127,187,273,280,-274,231,-231,224,-307,-56,-44,-81,54,-62,-35,56,225,371,-35,-10,-372,-260,114,-137,244,-106,214,-173,1,77,290,-161,300,-108,-307,-192,-299,-33,67,249,-339,-108,-27,-126,-212,-227,58,33,447,-245,9,0,321,74,419,50,372,-110,32,228,-148,-260,-99,277,229,-234,107,-257,-129,-237,-362,-268,331,142,304,-305,-225,-66,100,-118,-221,84,-44,-201,-146,-4,-235,-111,-53,189,261,47,159,-99,70,-210,78,-344,283,-172,92,-42,21,191,37,126,73,-400,-223,50,409,114,43,161,-288,146,-357,212,427,242,3,314,130,-102,-142,0,110,-342,216,24,438,-92,-130,-171,-336,138,251,19,-412,-75,85,159,-47,15,-22,22,-22,186,-24,-113,-202,-337,-370,-20,-62,-43,22,-415,-29,350,31,331,-445,-7,-180,-4,259,-44,41,278,77,377,107,-36,245,-14,38,233,78,-310,352,-112,-302,325,216,-207,-247,-111,-143,36,-123,-186,184,-73,-239,-13,39,48,49,-242,-64,16,-231,7,-32,-36,110,291,-71,140,218,-338,10,-136,6,171,-170,-126,-316,-2,-159,39,164,-159,43,-12,-41,-89,-47,294,-229,308,175,-183,-252,-152,-25,22,213,-109,-260,339,369,106,80,433,19,-145,173,-155,331,307,-192,77,-80,193,-350,-12,27,225,-71,251,306,100,161,-54,-236,112,96,-133,224,-502,354,-225,138,-113,28,476,64,-207,-23,281,143,-334,463,467,138,168,-320,9,-78,-279,-156,172,-179,-259,-293,-130,104,-260,-96,-75,75,36,197,-8,213,24,-46,-401,384,19,-81,-60,-289,-154,99,-162,116,74,385,125,50,-210,-42,-394,132,41,46,175,-118,-329,46,-131,313,245,171,215,-276,-167,-59,130,-237,346,50,-238,-350,429,-394,299,-16,-106,-41,352,32,-78,-11,-46,-114,71,85,-70,-229,229,-62,-428,-117,335,62,184,336,-7,151,-119,-193,-284,-197,263,-78,291,68,-182,-151,-50,-246,-114,-106,40,-169,137,-142,-72,107,-74,-171,-7,44,222,140,286,93,-4,212,114,102,102,223,176,201,-221,-263,91,281,61,463,-189,132,206,201,-94,-16,160,-395,398,-101,139,45,65,-3,-375,-378,61,66,1,-170,-185,-183,188,97,158,-19,-2,293,-31,-131,53,-95,38,447,105,343,-7,400,-178,386,368,-49,-57,96,-25,-291,207,96,-487,342,327,-396,-239,35,226,113,8,186,154,88,81,-82,89,-48,15,102,384,-265,-15,-69,-138,296,430,81,274,-245,313,-372,286,291,283,-227,174,-182,-10,13,-447,-55,26,109,242,-33,281,28,278,-125,-84,-179,90,348,197,-141,-96,329,158,252,-10,103,351,31,38,4,-255,388,207,110,181,117,301,2,-214,-68,140,-151,277,405,-212,19,123,-4,13,160,42,68,82,94,-154,195,85,120,-278,293,234,262,126,-318,114,-212,33,224,344,-193,-471,252,-61,-44,101,338,56,-14,132,78,-269,76,-122,25,77,51,-420,96,305,109,-110,-335,10,258,218,-77,-114,8,-131,-203,289,-4,216,-96,24,336,92,-64,47,78,213,-368,-41,134,435,51,-45,356,-42,141,-341,191,248,337,112,209,138,-18,-353,-110,-320,292,-28,165,228,460,-3,-169,28,-111,-8,-166,85,17,-437,55,256,207,-244,150,293,53,-366,349,-10,-266,198,-10,-21,-456,-77,-48,292,-246,-117,-165,-135,-146,51,-222,129,-51,-176,-195,-25,-181,-192,117,432,127,-93,217,130,309,-27,-116,262,282,-204,18,-123,305,80,174,-157,342,7,-259,-258,-82,-213,-38,143,-71,44,362,30,161,246,34,433,-408,346,-81,315,-51,195,31,160,-261,-361,-287,333,-46,-403,316,58,62,-4,-174,-117,146,9,-17,-271,-132,-254,119,185,16,-108,-20,-4,-102,-451,168,-396,63,113,296,-271,-58,-201,21,69,-70,128,461,-186,-36,-165,224,-53,-229,-156,20,67,35,309,195,240,-46,42,219,-153,-33,85,254,-123,-12,-46,235,441,145,-107,154,-251,-35,-75,-352,-78,112,193,-237,271,-89,-385,-205,-65,93,-129,192,-299,125,-16,246,-127,280,171,-41,-132,-40,-253,81,-361,88,63,400,-155,-137,-215,-192,408,137,381,185,-288,75,209,138,-288,-67,-176,-60,-305,336,52,119,-406,11,-68,-409,40,18,-78,60,25,-4,432,8,-408,254,-44,-355,172,-200,77,-216,217,205,322,-64,155,23,302,86,5,7,-53,155,-323,41,54,410,-379,-93,178,-336,147,-34,82,173,244,-356,-209,188,-332,-66,-164,123,36,181,-53,39,76,118,305,64,42,-7,-278,22,-348,-293,276,394,-265,238,68,37,369,365,-302,232,-328,103,236,-344,302,28,-29,513,-346,-121,-358,-73,1,-161,-89,246,77,109,316,317,-85,60,199,-5,218,-140,-159,-308,-47,-341,-11,-134,35,-30,15,50,154,-20,-93,119,236,-357,146,5,-137,-93,-1,260,262,-301,-325,155,-26,-179,-330,7,142,-40,-201,149,124,58,28,191,-227,188,-269,-169,153,405,263,42,85,-317,-230,42,134,384,257,322,14,336,-77,167,154,373,422,-143,44,-161,54,-129,-86,-108,74,-478,-31,-36,-48,20,32,-82,-86,-178,176,199,-50,26,151,216,270,418,185,-213,-226,-56,-228,15,17,46,268,-77,-194,-173,54,163,216,14,59,-243,40,6,-364,412,11,-162,32,167,-225,-58,-54,-132,239,-311,175,116,249,187,-5,-65,-138,85,-346,-128,407,-20,17,-128,14,-42,50,199,-231,280,-438,-216,95,-300,-204,-348,92,-262,130,-327,-149,201,-331,-102,-450,-199,-305,-55,54,-62,252,-175,230,-30,117,51,53,19,109,11,-438,-281,148,-235,-48,1,-170,310,-231,124,67,-154,-74,143,56,25,11,-325,-429,465,-95,51,233,-233,4,142,-200,-459,121,34,-187,-352,42,322,183,260,-261,340,260,338,299,374,250,-10,-58,-312,6,-108,61,86,290,37,-7,-108,-324,102,-71,-5,-318,-62,-318,-40,324,-478,26,12,-292,443,-89,408,50,-154,374,440,350,92,261,80,395,-232,85,153,-369,80,-134,-306,250,107,38,-292,-367,118,353,-229,-59,253,-70,-135,-135,242,56,111,-130,-180,-307,223,-241,-167,-288,8,17,-72,114,200,-79,-123,299,-216,-99,163,-201,256,49,-127,-61,59,-137,-149,-478,-283,348,369,72,-253,-248,-342,-174,-16,386,112,-10,-48,251,-96,-254,126,397,-21,-4,229,-209,25,101,140,121,147,63,-119,-211,179,-336,-68,-326,-199,332,71,81,-242,356,455,71,222,-310,80,87,337,28,-237,-116,415,-96,-516,-46,255,33,-78,442,-23,3,-378,-70,-12,10,-431,-98,68,148,-55,220,-476,-418,133,-48,-283,-106,167,-131,-240,123,19,-100,-89,-1,-158,-81,204,298,391,-389,308,-62,-96,175,-38,-5,42,-97,262,-152,-269,-52,-33,-32,-297,103,-31,-173,58,-69,305,217,-316,-199,-54,-71,-375,-19,-215,448,-167,-94,-270,119,-126,244,70,-310,-444,-208,-360,-292,50,177,63,-111,45,312,-57,288,31,-355,-80,-106,-421,-8,388,-196,81,-342,28,-227,-22,35,167,-99,320,-35,189,186,39,-79,-87,58,8,-295,-93,-311,-314,-194,-30,-222,-294,-8,-47,-21,-122,-143,442,92,-469,131,-291,294,-133,-293,-261,517,97,114,-253,33,-236,-17,-137,-221,-39,-331,157,-120,-28,-103,-327,-329,294,-27,36,26,281,226,93,29,34,270,-180,463,73,-77,230,-163,54,304,258,-20,166,-99,-217,435,-425,-13,-19,-239,65,208,-122,-255,270,224,-158,45,-341,-151,114,-224,-105,-185,154,-276,20,-319,89,-58,-197,-257,379,-391,-107,-191,76,354,137,103,-17,243,-237,-77,230,92,58,130,-133,231,419,21,28,25,127,25,25,100,165,268,253,-21,188,62,-211,-425,154,-211,381,275,337,-260,-153,-459,-128,34,-30,-489,159,290,58,-208,-246,-296,-1,-423,167,47,282,87,50,124,-351,27,438,-30,-201,8,267,-271,386,56,-124,479,-74,-37,46,70,-286,-94,116,185,426,-141,156,-122,7,-417,-256,180,-118,-266,-441,-136,289,-145,175,279,276,-380,479,-379,-31,-277,402,-2,-10,-15,231,346,-141,443,-295,273,-121,-126,26,-207,-26,-246,-12,355,333,-103,-38,212,183,230,321,215,-396,173,183,-101,-170,-173,-319,-236,117,-19,-113,-160,-312,115,110,-41,-290,115,264,-171,58,-136,197,306,262,34,-10,-11,61,188,29,-139,118,187,54,-315,169,-6,65,-218,-220,199,-159,292,72,54,-329,-263,-134,-158,261,442,-203,-138,120,-109,-89,-356,329,288,9,157,293,-367,-53,-229,-68,90,266,-163,-91,230,-55,84,-268,-408,3,-40,-163,-329,-330,191,377,115,339,74,84,206,245,-191,-311,183,-352,-98,94,124,-160,-108,361,-29,112,110,125,300,25,-68,119,293,41,-29,429,-21,165,-214,-165,188,-417,-390,-221,25,-309,-22,-32,-161,448,-108,310,-59,-126,-485,-52,284,-113,-141,-141,-187,-93,208,64,188,-212,-60,392,-408,-25,43,160,-246,318,-435,-80,-103,-252,-10,288,29,-340,110,100,-46,-437,96,222,330,-90,438,-291,-427,-53,115,121,-252,358,115,198,-48,489,-96,13,54,-274,28,-149,24,214,-188,-167,142,-87,216,-66,-365,376,272,322,20,227,117,-315,229,-43,163,102,243,245,182,82,454,-13,262,58,-66,250,-17,189,-241,357,-8,-203,-129,-132,-329,72,-66,-109,149,264,-229,188,132,-170,-220,-150,-477,143,-233,-255,-253,-95,-312,-162,-394,66,260,107,299,-495,-291,-147,-299,69,137,-14,-278,-55,-451,-173,264,93,96,333,-446,-318,-371,-20,-29,-55,389,115,49,290,-295,131,-272,298,-238,14,122,-505,164,94,-437,64,-88,24,-62,78,84,-232,-263,-98,-153,50,162,-230,-336,-102,274,67,20,-62,116,244,-175,-75,-131,-228,28,341,155,11,-325,195,-114,-6,-21,149,-68,-32,246,475,-88,492,292,189,-240,62,-236,-160,248,53,86,-117,-35,-408,136,107,-269,456,-22,-2,471,-102,-270,-388,451,-386,16,-322,66,-28,-500,-252,391,96,183,-283,-136,-163,-114,-279,56,474,238,436,95,-98,145,-362,-441,-49,151,-102,22,-215,-209,-255,-14,409,287,-252,340,-311,-114,169,80,-18,207,262,-82,-116,-130,-91,15,101,182,338,-357,44,-190,-267,159,-245,430,254,-63,-50,167,76,-239,-184,-308,-227,294,-146,150,351,-359,-220,-330,-192,144,248,-458,400,314,-419,198,-463,272,-354,-71,116,-151,-166,463,363,-122,-258,-174,250,211,280,-100,-153,77,253,172,-140,-185,191,-412,137,-295,-73,371,337,73,162,223,1,-28,295,98,-216,-107,125,173,117,332,289,386,-108,-19,-2,-2,146,153,-315,-12,100,56,-41,-230,281,-31,107,171,-178,-171,188,51,-492,88,24,20,-56,-95,125,282,136,93,-124,-6,-358,99,-410,-2,-330,-327,-209,210,356,-355,201,-205,167,-213,-259,-232,101,-270,73,-58,-59,-77,227,245,140,277,-95,87,429,-54,-453,83,122,-75,400,60,-247,248,408,305,224,-3,116,120,-187,97,70,-233,-384,4,-122,135,310,-113,14,221,295,-54,-432,168,223,396,50,-85,-182,363,21,53,-68,402,136,-127,-240,169,-262,327,303,-124,-124,57,67,121,-156,342,-152,347,-7,-328,195,-246,179,207,-48,-59,211,206,12,236,204,84,108,57,-306,-36,86,-275,-90,319,103,-90,-198,21,-275,28,-70,-275,-406,240,-213,-109,267,-246,314,-100,35,-273,-71,-120,47,-119,394,4,44,4,43,-66,-126,457,38,-431,-7,-366,130,130,-295,-160,-95,-208,28,366,115,-187,32,-66,328,-41,-443,-132,301,239,-305,387,-235,-230,-5,170,-37,-112,253,-192,238,89,-175,-210,73,-11,-183,-188,-481,274,-154,181,-78,14,134,20,263,-109,36,153,313,-228,350,-299,281,185,103,-42,273,-266,58,42,15,3,-182,-286,216,-404,-452,397,320,175,150,6,-210,-27,129,-226,-141,281,506,-396,-151,-126,79,314,164,-328,-263,-18,262,210,241,261,16,288,266,-230,-231,33,4,-374,213,314,-177,39,-333,133,404,-158,417,-112,-330,-58,-89,-60,160,142,20,166,180,320,-229,-85,398,-223,15,106,-45,252,187,195,206,99,140,-297,243,26,355,84,298,11,101,-309,-150,-192,173,-30,393,83,-153,10,100,57,211,-70,414,-74,-90,-336,270,-63,-102,57,128,255,208,90,116,70,-210,244,-133,50,340,-51,15,-37,-299,1,-49,-298,212,99,425,226,295,222,185,46,492,-18,315,-394,-310,166,-74,3,-83,-30,-192,329,-163,-375,445,-265,-357,197,300,175,-110,-232,91,403,377,-52,-184,-76,-69,-362,-258,-16,113,-31,-250,163,208,-227,-246,145,138,-190,-23,-5,-60,24,368,-446,-185,228,249,88,329,143,-98,450,135,21,-168,202,-370,-88,184,72,-269,-46,314,-204,-115,238,-241,195,95,16,-89,280,154,236,-115,-120,28,238,37,-208,-364,102,-33,-323,-64,409,288,-249,-114,-262,152,28,-388,-95,24,-32,297,-250,107,-182,-34,253,-136,-237,39,-418,53,-47,96,17,-42,-110,338,125,33,121,60,15,181,-9,-49,-163,-41,380,127,60,385,-258,-131,-118,120,127,-374,184,-141,85,44,-92,-330,38,46,238,-244,-222,151,133,-413,379,131,54,55,296,-233,5,-487,58,-430,83,71,402,-26,-241,-149,114,-149,116,22,442,27,-238,-81,-123,-231,218,187,-24,132,20,189,53,217,122,284,-20,10,-6,179,-315,183,236,-51,-74,-147,-266,-21,206,259,74,40,-96,-311,-464,297,42,-39,-263,113,-70,392,125,-93,-223,399,-293,-183,122,108,212,286,96,60,-124,3,-259,-235,-64,-93,-153,127,-90,-351,135,25,219,171,19,12,109,-73,148,-238,252,-64,82,-229,-159,63,140,246,-107,-64,-299,-54,-149,-326,103,121,-228,416,-181,433,-322,25,167,72,190,211,-254,168,-218,41,356,-176,-358,-6,45,-342,-71,-43,-16,268,-177,-385,-43,178,140,-125,-344,-28,40,-133,-74,4,68,-82,157,-115,71,455,10,-38,11,103,-327,-86,49,-295,-42,273,15,-51,11,93,-260,208,85,97,309,44,-125,-398,113,4,-142,-271,237,52,-245,-144,171,86,-13,-56,-44,140,55,396,-53,-256,-5,-46,-373,1,190,-75,5,-60,187,35,-292,-80,13,157,-210,-280,140,519,335,-178,-362,231,156,-86,238,-198,-95,331,-307,49,298,110,23,213,211,-167,-292,-364,69,200,-93,-279,-184,146,6,-126,-327,-13,-106,170,432,-179,-225,70,-223,-200,8,-374,120,335,-203,-197,479,371,68,-88,364,-104,8,-307,46,-380,291,28,-10,3,155,-30,144,354,192,-227,-236,296,137,372,129,-30,130,164,233,24,180,162,-279,-193,416,-118,-343,163,-351,-103,496,111,165,2,-295,-235,-11,-346,221,-49,185,150,14,-116,-197,-5,19,-267,-11,-26,-410,-353,163,-151,89,-281,-240,237,-153,-102,7,11,-131,340,61,-9,133,-63,-201,162,286,65,-106,125,-275,-54,-384,-215,-100,229,-208,-22,-489,-383,-194,-52,2,34,-390,179,-445,1,157,-68,-8,-160,-40,77,-29,22,-462,-44,-139,162,266,87,-138,-270,-162,425,59,-65,377,-234,-300,-424,-58,33,277,22,290,-402,-65,210,-44,198,97,-207,-241,54,7,-226,132,178,-81,-118,44,-110,-474,298,68,426,2,96,-27,86,322,-53,17,-56,472,79,347,345,-1,30,50,-211,24,-8,-150,259,-260,-128,11,124,-106,177,-5,314,-61,166,46,2,-182,-65,-195,-137,-206,18,29,142,81,408,-258,22,8,29,-106,185,149,-81,-238,-236,209,273,102,-35,-282,430,391,35,285,-133,-132,423,291,-425,341,-53,-211,-315,-49,440,-11,93,-80,-140,-206,-486,-121,92,-127,-149,104,-148,314,-73,137,184,189,-204,-52,249,-34,-38,-211,-175,291,-140,-23,-39,-339,-180,-84,177,28,188,-134,269,-353,-64,220,337,-138,187,-450,125,-238,-144,114,-321,112,-59,-84,2,-167,-153,110,152,-47,36,239,75,-269,-44,-261,-133,-91,50,241,106,262,-188,121,473,-30,-167,-69,355,-218,462,13,42,-4,-115,-62,394,353,-164,-393,-400,450,-295,29,-280,-40,105,166,137,-139,-259,16,-322,56,194,-148,-199,3,58,-177,-367,218,-287,-97,84,-66,-6,155,152,321,356,-413,-222,326,35,278,-63,86,-116,-81,-398,-76,-38,-32,122,189,-118,-45,154,246,250,-181,-182,4,-442,2,68,-126,123,-377,72,20,-117,-367,134,15,236,192,-84,-121,-44,-39,14,-77,346,-178,-31,-73,25,87,-309,202,139,-188,468,-224,-121,-5,-370,-378,-187,135,-349,98,352,-35,370,428,-149,-302,162,-148,9,343,166,-5,-410,400,-213,-302,235,190,391,-380,-188,441,55,-50,-40,53,27,112,-206,-191,-97,-322,53,65,43,-70,-141,-70,133,-78,294,-24,-315,7,-91,120,56,-378,91,-75,94,62,156,-73,-519,-323,-38,205,-77,-7,73,76,210,-259,-32,-237,-324,44,-68,-280,233,226,-7,-362,-150,-56,-246,206,-178,-222,-93,-25,203,324,278,101,349,-298,20,232,-447,278,138,-149,83,66,-118,260,352,280,-404,-320,-232,-218,419,134,-40,-138,-184,167,-79,-14,-217,197,271,-208,-23,-352,-122,24,102,243,-101,-252,-54,-69,11,69,-409,-7,-156,23,183,180,222,143,-57,55,77,-48,-110,272,-476,156,-53,340,-136,142,-52,-79,225,-28,-222,-133,-54,-95,166,-27,-85,-345,192,10,93,63,501,-100,-157,-117,362,220,-206,97,-53,-122,159,-30,-125,93,-316,198,144,157,293,218,-291,-303,79,160,-176,-102,-178,-133,-451,-61,-291,123,-20,144,-104,-235,-340,123,-214,384,-8,-91,-13,49,-136,-316,-146,179,-249,339,-260,-295,195,-207,419,-144,-52,-448,-75,381,188,90,-296,-184,-341,60,-32,19,-183,-14,-156,-141,269,-424,-45,-225,244,-372,-130,-172,128,87,-393,188,195,188,4,125,58,126,188,-374,-140,-37,121,221,-47,23,140,-125,312,-198,-70,-171,84,-26,320,-335,-76,183,-194,16,-6,303,70,32,28,478,-91,-47,415,-177,138,-12,135,-28,444,-218,-23,247,391,387,-260,-97,155,71,247,157,-1,-100,5,-454,-206,80,87,196,-39,-32,-151,-300,251,-30,-411,-69,-100,-320,-98,244,165,164,-6,-210,-30,392,54,254,378,-48,236,25,-19,132,134,-19,-54,-254,409,-277,159,-158,49,397,-361,161,-315,252,345,336,69,266,-231,396,-23,-7,291,70,-282,-88,-27,-39,-205,335,283,8,-12,148,-314,-242,-236,-245,173,453,398,350,-91,-254,-416,-255,-66,-264,4,283,396,-86,-111,-124,66,5,-514,410,176,-190,85,-93,-89,-284,-3,-91,169,140,-205,171,281,-6,102,146,272,189,-58,-214,171,-64,89,-24,329,-41,3,264,-275,-357,206,43,136,43,-313,198,-275,-171,-448,381,-167,-185,-283,37,-41,154,23,-34,-184,33,280,355,-52,310,-239,-88,-87,-176,383,-292,134,368,65,322,-353,-136,-233,-177,-359,-8,129,81,-223,-426,-2,-153,129,-331,-93,216,-241,473,-283,155,92,78,-288,41,448,-321,-221,-2,180,-221,-117,-19,-451,-93,102,8,-211,-75,423,-142,-230,-136,119,186,-236,-85,250,101,284,317,-78,97,-53,-69,489,-174,101,-135,-197,-38,-268,-103,197,-216,-303,-94,-164,10,230,-77,384,83,-203,138,382,248,-396,-71,187,143,407,-178,-7,127,-2,188,325,193,179,-144,-205,-310,28,-293,-207,-190,91,-448,485,-446,191,-41,-112,57,-164,98,422,-148,391,-134,136,-155,-294,363,-251,-161,-136,175,300,-73,-119,-241,41,283,223,67,-84,-438,336,422,85,89,49,-422,-29,-228,-28,-350,-23,-216,-141,-39,-368,-312,83,-136,-37,118,28,-19,305,-83,6,-255,440,-180,-48,-111,-458,-6,-470,311,349,-223,102,-27,120,36,12,-484,277,74,105,312,109,-158,-445,240,-347,305,-338,-414,-104,-45,-214,-232,-306,111,408,99,163,-310,263,-96,-257,186,114,-9,246,314,320,-362,-80,77,306,76,-343,-375,11,242,28,-288,125,156,134,43,207,102,-183,11,1,401,218,88,-184,-24,419,62,-29,-76,-165,-345,-295,-23,-193,-157,273,167,-232,-80,-73,-169,-36,-76,-271,20,-297,54,-52,-383,39,-214,22,-375,222,30,-80,-58,-99,115,-152,-99,-269,-59,75,-442,-233,98,-188,-136,-241,-288,-50,214,79,399,-36,-170,224,-212,284,87,43,-424,363,-1,92,-323,37,-108,178,-76,13,65,-234,291,-243,-228,-108,-290,-135,367,215,-348,101,-487,-272,-64,236,463,352,126,-178,-232,-146,-191,-102,-275,-131,216,21,106,19,-226,282,170,162,-222,227,-128,-63,28,66,49,-241,-215,-334,-389,-99,43,295,19,-255,126,100,-118,-44,-212,233,-272,193,-298,146,62,120,343,41,-156,256,-144,230,117,4,59,-356,282,-413,-85,-352,-174,-48,82,-115,-80,75,-416,165,-312,401,-125,18,20,-7,177,-317,-111,-98,-159,50,-352,-49,-254,61,292,-37,-57,40,-105,-65,128,-104,-167,85,275,-325,325,96,-28,91,19,-376,123,22,-51,-439,-243,-386,15,8,-115,-154,290,-209,92,127,-185,82,279,246,-5,140,-130,-138,119,94,-37,81,-169,100,399,-180,118,-129,-445,-96,-59,-319,286,287,407,65,-284,362,96,255,175,101,251,392,-115,186,47,-182,218,183,-75,-21,330,280,187,19,112,373,-454,-164,274,215,94,308,206,154,-69,40,222,41,-314,454,39,443,-211,230,48,-368,51,-128,56,-41,22,0,13,274,9,72,387,247,-235,32,77,-13,120,95,298,-141,-226,-11,-37,-33,23,220,-328,-264,355,-179,36,-46,-19,-108,-15,35,97,19,-148,-176,-395,412,-481,-150,-62,-86,144,-81,-261,59,-321,350,-333,35,-281,364,511,-201,-140,189,-239,18,266,-58,-146,130,22,-106,238,-112,128,-72,206,-94,-235,457,1,-369,163,-317,-404,-36,176,-306,-146,-60,115,-234,291,-40,-32,-20,-23,-44,162,-109,-300,204,35,-190,-138,83,307,-87,108,63,-138,8,-63,-225,-41,-371,-296,79,-101,199,-90,145,-160,-260,186,54,-377,91,-98,71,-110,-280,30,-15,373,-77,272,-7,368,93,41,-77,-132,-2,-91,189,48,-217,0,331,-185,-420,-152,309,301,182,209,-128,223,-438,-98,-187,41,-14,196,-210,-19,29,-87,111,118,-35,221,-75,-15,-195,-276,141,-410,-379,-19,258,252,-85,-22,-238,-125,-148,-33,-42,273,-42,227,-151,20,211,187,-112,348,-162,-77,-36,-97,-365,226,86,23,-523,169,43,455,-179,-289,278,-268,229,481,264,26,-52,18,254,-223,-54,-379,-63,-299,84,-238,36,11,-432,3,14,445,-346,110,142,-50,146,234,-183,209,-121,111,355,-106,-164,-25,-38,-8,-76,135,-45,415,47,125,-151,134,-304,197,-28,-78,-30,-1,-49,152,-303,-197,441,177,-4,-122,28,122,-98,-168,60,-436,201,118,34,182,348,-220,-262,-165,36,131,-243,-198,-177,-20,161,26,-395,-171,156,194,264,296,335,-126,38,41,2,212,162,-287,138,142,-274,202,-218,90,248,-146,-90,1,-34,417,97,-474,142,262,338,-55,237,-490,169,-300,-168,-70,-231,164,468,-28,336,-52,160,384,-231,-233,-475,108,279,112,94,-27,338,-251,406,-49,359,20,312,206,101,361,325,-19,-381,8,-496,219,404,-39,69,283,-397,-4,238,-15,-25,50,274,73,509,360,429,-208,1,45,7,77,138,36,174,286,17,335,97,-197,-90,187,97,94,-8,-285,428,0,-36,69,308,-364,-42,-231,307,64,37,266,-191,-127,22,102,187,-24,202,386,39,251,448,137,352,98,256,-266,250,-7,329,-35,-97,-46,322,-171,486,-108,-281,-78,187,366,-226,34,-250,-271,-280,98,105,-364,123,-39,-94,337,-72,-90,57,-23,260,-366,-47,-33,-310,53,71,429,-142,-55,404,-4,35,-148,187,-253,48,-138,-39,-36,22,-30,79,80,276,26,17,-91,-194,-31,-93,-42,-42,416,171,-356,-169,-300,-189,18,-136,-426,15,100,156,-46,-111,-74,5,77,183,-202,-144,169,339,33,34,289,106,218,435,-389,-323,206,82,-17,370,-292,109,207,-304,279,-132,126,-399,-368,-34,-448,-226,259,171,-140,18,88,9,200,2,462,-78,-368,5,-65,-38,37,-216,467,8,-10,-365,161,-353,-336,-240,-383,-352,112,-107,-14,-91,331,63,125,103,72,-27,478,-78,203,216,-474,380,-376,-164,-103,-123,-123,19,356,44,-447,138,196,22,213,271,255,0,-66,-100,-109,37,-196,465,-119,394,-176,206,-117,54,255,-288,-175,234,318,-212,-186,163,-166,-21,-155,360,-396,50,-250,-42,-264,-73,-354,-1,65,123,-275,370,-20,428,-62,260,74,-152,-254,24,164,20,43,256,-40,41,27,38,-65,-450,-154,-13,-488,-41,-68,59,361,-315,433,-212,187,-289,142,37,-101,94,-255,414,-249,-142,-239,16,-31,-25,-114,268,-112,-169,-402,21,-119,-213,-120,14,78,-289,206,-165,24,-410,-82,-68,-488,-59,295,-381,-138,-177,-43,99,386,-253,184,-169,49,-391,8,92,343,-136,344,6,-38,246,409,298,-362,-76,-401,-242,-385,-182,-336,114,63,-213,58,340,-41,-222,-19,-292,422,466,214,-227,-28,203,407,272,-106,88,125,-94,442,-231,40,338,12,71,-102,-99,11,52,354,412,-61,-356,428,-86,-27,-359,-304,330,-52,-142,-186,341,74,-77,6,-359,193,-119,135,-351,80,2,155,12,-268,172,-139,175,-226,-441,206,-347,194,388,-139,-21,-401,141,141,-99,163,-236,-40,-138,-49,213,35,-45,-272,88,-267,43,328,-257,122,366,138,355,327,315,259,-210,384,-255,215,-204,-5,-161,-379,81,137,-59,-280,374,294,-244,-11,-125,249,-39,-25,-288,-47,-180,-234,-10,295,392,22,-18,176,36,-45,-257,-210,-406,286,511,442,-65,-53,393,-23,139,-94,409,-77,35,466,-310,-30,352,67,-108,-256,127,-159,-192,235,-72,-178,-332,-242,16,-384,-217,110,349,443,268,236,267,160,59,262,-27,165,183,165,-172,-230,-93,-96,505,-437,-82,-228,342,-61,-268,-450,-44,52,-171,260,137,328,-324,72,291,-421,31,221,-338,-79,1,-35,-431,298,333,55,73,-10,-56,401,375,91,304,-149,26,493,-25,1,-72,-382,-48,-49,-18,-258,362,-189,-327,444,88,-497,-182,-29,-226,-272,41,262,47,188,-68,125,-180,135,-231,-320,-155,165,-114,148,219,156,308,-252,-215,302,446,-279,127,274,-168,105,122,-144,259,-204,28,-38,126,439,296,93,95,-32,-137,-198,214,278,-436,-35,50,-166,-338,-59,149,415,121,204,-175,-75,85,201,-94,-28,360,-149,-341,220,25,53,248,-238,315,16,227,349,-120,104,221,84,21,300,202,12,-242,27,-323,337,-368,405,330,142,-58,-80,67,-293,165,-265,215,26,-15,-42,-1,-406,227,-93,36,-117,-323,-258,222,247,325,-63,171,334,-254,304,-145,-14,103,-344,-63,233,28,231,-107,134,-172,-90,74,133,105,-409,349,-198,-435,-395,18,99,192,-144,192,-421,98,217,-383,-84,-97,-38,-128,2,16,92,45,-98,-256,-77,390,46,-224,281,-315,161,-159,181,-226,28,-131,-82,293,-34,-72,10,134,8,-209,45,-32,-161,128,86,-97,-164,-164,-69,262,-30,-29,-13,267,-115,423,-137,7,197,-63,326,-171,-187,60,236,90,-207,-345,-385,-119,-2,79,31,344,-146,234,207,-225,137,80,-85,469,-1,-393,-207,26,-248,-19,-341,-18,11,-40,-267,-169,-325,-13,93,81,-129,-193,83,212,-212,-199,373,-1,-297,-276,103,-172,-341,-346,-36,197,128,-68,-346,132,436,4,-65,102,-169,130,290,-155,485,-92,-354,219,80,45,221,-193,-59,-157,-88,-15,143,114,406,153,113,-67,-32,279,-434,14,-93,-161,115,292,-72,-14,-278,181,-31,351,121,7,117,-258,449,49,-16,113,-32,-117,-139,-65,108,-341,35,6,407,338,-404,24,340,60,-83,371,-389,48,-328,-120,-292,-273,121,-27,120,-142,94,-274,139,-199,-69,-316,311,-95,-226,247,17,-41,-205,246,-273,-99,-484,-168,-123,395,-263,53,-243,282,267,380,394,-73,-290,-29,-356,196,256,-350,-141,352,-265,-178,187,-58,222,358,-356,-296,455,32,-150,-87,-40,24,62,438,134,63,137,-461,-22,-144,81,-243,226,37,-184,138,3,-77,-133,-360,136,-134,383,-14,60,-75,-16,188,25,86,205,-104,378,131,0,108,71,-100,-245,-122,181,359,-98,84,47,114,-179,85,-125,210,50,-11,14,375,-109,-84,42,-224,88,-107,127,-392,14,61,330,277,-219,323,-198,-363,55,85,159,127,42,312,-146,268,-185,-181,435,309,-392,-323,-288,-169,-338,-7,-171,209,84,-75,-94,86,218,-167,-125,117,244,47,-30,200,63,252,-316,60,-268,79,-130,235,254,439,38,-107,149,-245,-314,107,22,238,-103,251,-168,-242,-16,5,81,377,215,55,-291,57,30,-256,20,367,23,187,264,-201,224,-99,-30,124,294,263,159,-292,54,143,-105,66,-154,166,-18,158,30,25,110,127,-134,121,-291,-79,150,30,-74,-75,-69,-77,5,239,-148,-1,5,18,516,-126,-6,56,320,378,-450,99,232,92,-289,99,56,41,21,-126,173,-101,189,2,263,-174,362,95,-48,17,108,-79,54,178,166,120,-298,358,-177,423,102,-318,-72,-59,-143,189,84,169,328,-374,242,53,-21,31,23,406,104,101,90,-59,-319,-6,170,214,-24,-87,370,184,-264,-299,-157,167,-149,-214,150,-242,166,-213,127,-43,-265,34,-138,-137,-137,-6,-231,365,307,-100,-262,17,177,23,208,-162,129,-114,-75,-188,323,31,58,-5,250,64,207,219,-75,278,272,-113,-45,182,-111,328,-73,-274,-323,27,86,36,-269,135,39,61,154,-210,-73,16,39,-170,-56,-145,-400,-44,21,-141,254,-142,-124,6,140,360,95,110,-140,-40,-17,-179,-50,105,-28,378,53,16,239,-291,228,360,24,335,19,-163,-158,26,17,-241,-118,81,-67,-81,-171,49,63,4,104,-398,104,352,-361,303,-230,201,34,-319,-192,441,233,-261,-189,-33,290,-364,-423,-268,244,295,-127,409,149,-9,-443,-139,158,-290,-285,188,253,251,47,-353,160,32,28,-78,-192,498,-240,45,26,163,-452,58,-294,397,170,24,-372,182,334,352,32,-174,194,-318,475,220,44,-421,480,112,-234,-96,-401,-3,19,84,5,420,74,113,178,-5,-218,-205,-189,-219,-136,-80,87,-294,346,-68,32,-28,-207,-61,194,296,138,-14,99,109,-280,-137,-97,-63,33,271,-85,59,-385,39,-4,-369,-77,-301,69,435,-25,93,-245,19,-31,-37,248,240,-90,-138,373,79,-255,364,268,-339,-5,322,-342,181,131,-11,44,-241,107,185,-216,348,172,204,61,-230,-192,-191,-113,-409,96,11,-69,-76,181,43,-109,-204,-68,-95,-373,-144,-243,176,141,208,-415,122,190,128,-66,-156,-179,204,-497,-116,-150,321,-5,-234,-39,196,15,347,-34,-371,294,31,-189,-250,131,119,22,39,-109,437,-177,-98,448,32,-97,8,71,-163,88,-252,28,191,-364,-46,262,-14,-267,-208,-252,52,-90,30,-387,-331,-226,-235,-133,-47,-51,-284,421,-445,-3,-150,253,107,43,440,-108,-85,-70,88,142,93,-58,250,80,32,347,35,330,293,370,-482,-78,-198,122,406,-23,-63,-75,424,319,195,-184,-104,104,-13,-82,459,-59,67,16,259,426,89,-13,325,-27,55,-434,276,-33,43,22,-68,84,373,334,20,46,291,-230,-274,189,152,-4,-24,-34,37,91,219,-28,-161,-471,-97,-346,-371,125,18,-77,76,-124,39,164,149,2,-145,-19,-324,430,-511,-88,-26,50,-291,-94,125,-22,11,281,72,-63,-89,223,-226,148,-11,110,302,-385,152,196,434,-76,266,-324,-62,447,238,357,-26,344,-220,38,-208,291,122,249,-454,49,335,-38,-15,-139,-22,41,238,-190,112,138,90,82,-258,177,358,92,176,345,273,435,223,-324,-231,74,-348,-1,376,-326,166,226,266,157,289,231,-50,-196,-43,255,-112,174,-121,324,103,276,311,224,169,-329,-356,-298,379,-428,-193,-451,-253,-367,133,33,-142,-138,161,-357,-54,-237,281,109,67,-312,-85,-369,-188,175,-19,40,215,-137,199,-8,-314,153,-308,-243,300,-23,5,187,417,-66,319,-128,-209,-237,-362,140,477,121,-26,50,-125,359,-47,418,56,-132,-233,-212,163,144,-47,342,163,-86,140,-118,109,-222,115,338,-122,32,179,-61,-457,-113,-263,165,249,-348,-324,-99,362,63,231,-387,-325,249,-241,267,-90,-226,-168,-81,-307,210,-193,-210,-118,-243,-168,-105,237,-55,-187,-107,58,52,423,-61,-270,-165,-13,102,-171,-46,-421,52,-168,289,-203,253,74,162,-349,43,-318,9,-38,-208,94,-260,-54,-59,80,-51,34,-29,204,-28,-369,264,245,-52,-411,185,-223,385,-443,-78,-54,-280,128,87,64,-18,-175,195,151,240,358,-3,134,136,-87,97,38,245,-54,271,-49,116,258,-479,84,380,-76,393,343,300,-151,-303,321,-28,57,262,67,-119,-240,-55,-178,193,-111,276,224,279,-82,-157,-165,7,58,285,-44,194,76,-219,48,-159,-33,401,170,103,99,31,269,-85,7,-7,86,157,81,-100,-155,-283,-151,365,-141,404,167,0,-165,124,157,216,-157,127,77,36,-30,201,182,417,196,65,114,341,197,-3,-185,-74,57,-230,-303,69,-23,-121,142,-79,253,-24,172,6,-217,323,-259,68,-15,258,111,-339,272,288,-371,54,-273,170,-6,-303,-129,10,-79,-92,10,201,-189,228,-80,315,-197,9,120,-10,-189,319,-126,-206,-336,35,460,52,-404,155,395,-63,-104,-164,-334,-322,-219,140,161,-323,-300,152,-100,308,16,-133,-30,46,76,55,-130,252,-331,2,163,42,148,44,175,181,278,-147,-333,104,-244,-290,106,-254,255,-85,-416,-20,313,-325,-74,216,-24,-214,233,-267,117,332,-197,49,-79,-217,254,-13,-266,-116,59,-169,-201,-313,201,-103,44,457,389,346,314,154,-96,190,-214,-331,-183,-125,99,-475,259,34,-100,-83,-433,-83,-304,114,215,-308,-223,55,133,97,-237,327,167,244,58,-162,305,271,4,-69,-304,333,-69,-414,-255,170,263,-18,418,358,400,-87,-142,114,-390,213,-225,-304,-155,-164,202,-30,-36,-369,-201,-266,-1,-171,125,189,-108,-63,92,-172,-183,32,268,-299,-182,-393,-7,-146,-79,-326,2,226,210,482,-293,235,-189,220,38,480,215,380,-306,132,138,-82,250,265,35,174,131,-222,-253,-56,-66,252,-290,109,-80,-6,-48,-50,299,-84,-230,4,371,-366,37,-47,-166,321,403,-174,103,-303,91,412,197,-99,-314,-127,35,-131,-303,7,317,-80,-147,-96,393,186,33,-340,123,-204,209,186,242,242,-328,121,45,49,2,189,-193,-226,-304,240,-216,-283,-28,114,-365,143,404,179,-450,285,-414,339,-301,-244,262,-91,64,-207,146,-86,29,-24,-300,-24,-211,295,20,99,-64,-302,-324,231,404,54,27,-94,15,292,282,-133,-179,-51,-37,360,134,82,44,-226,48,-7,90,45,116,142,-333,8,-22,-56,19,-328,43,-226,406,-41,-67,34,21,-122,6,503,-13,-20,31,-19,76,-87,-368,-254,-230,100,37,-369,182,168,122,-106,-298,-133,12,41,-16,-170,375,425,-331,68,391,-9,-213,-206,-99,-152,-239,-147,261,-214,-208,-297,-283,20,-235,-221,-493,-336,-99,242,-214,262,-125,-134,166,27,113,-133,-32,81,142,-168,-388,313,-382,305,49,-177,-338,-177,-97,317,243,50,85,101,-202,-229,-11,-23,52,-370,53,-86,282,13,-187,-133,156,-257,-152,-334,-169,299,-321,-83,88,-261,-81,307,-377,-124,426,-284,14,116,234,-224,-53,-16,114,-154,34,242,-262,4,-66,-88,-513,-274,-161,328,-159,163,-240,-132,143,88,55,-427,-54,361,-269,328,40,330,-177,247,24,39,367,-369,102,228,-188,-231,387,-419,-121,198,61,321,-10,237,-338,-94,186,-47,-198,-191,177,-79,-270,325,-298,-268,26,63,-241,-65,-41,-178,139,7,-120,-142,-451,28,41,284,489,239,77,-38,-13,-458,-9,110,234,-100,168,-150,-46,-257,128,-9,98,-95,498,120,72,187,-21,287,-223,-313,-78,178,-86,-169,121,-199,36,452,284,154,-334,-52,67,-75,-321,1,-141,-259,-404,142,-252,322,176,-150,5,216,-10,-29,-9,96,316,-115,-99,-309,16,47,487,150,-292,148,211,-292,-108,-94,-283,63,-333,335,1,-27,27,262,-30,-265,-81,-55,-211,96,406,177,279,-78,110,-20,436,30,413,-240,-448,-228,445,13,22,273,-290,135,-110,-375,108,264,432,453,-216,-260,-46,-262,304,-50,-405,445,-92,63,-88,61,57,187,-265,19,358,24,-179,8,53,322,8,-102,182,141,-121,-402,228,61,300,264,117,-133,-174,-349,-261,-239,96,131,30,414,118,30,79,453,314,83,-199,80,-308,251,-213,281,232,133,-401,-176,-70,-96,-476,-149,-7,-156,97,183,416,206,-72,177,373,-159,-357,364,-30,-410,361,-24,-179,-365,29,-186,-105,283,58,-191,264,29,-53,440,-400,-322,-315,-250,-192,70,279,194,269,-76,-7,-93,-146,-306,-154,-9,-26,104,-80,63,91,-29,379,116,-324,-32,-145,258,-234,28,22,156,102,59,8,-268,169,-123,-163,-80,-79,339,25,35,-72,288,227,-213,417,-315,-94,-82,114,-10,4,-237,239,127,-240,4,149,118,-340,306,64,-1,-253,-303,-254,-366,388,-102,41,-371,-10,-50,309,165,-114,-267,-1,-306,-26,181,-249,-69,358,-141,79,-20,181,-373,10,186,-282,-338,-155,396,-95,-90,68,182,89,105,126,-366,-133,-298,-379,-456,-250,449,-306,-256,-266,13,-15,-58,-196,-359,58,-286,-234,87,-263,13,356,-314,237,219,-1,326,-151,-269,74,237,-7,194,448,-361,222,-181,-39,280,-414,-132,158,-31,-123,20,-192,-295,-171,-136,-112,94,431,240,140,-42,-256,296,438,144,114,-229,150,-317,113,144,-41,203,-7,-156,220,-256,-237,-81,-108,7,-293,177,-217,-174,121,-68,-22,272,-26,300,213,37,292,-249,84,-209,363,15,-192,-157,-493,227,-81,52,136,-66,-273,73,113,5,-364,47,-11,61,-334,251,65,249,263,54,-86,-195,242,24,-248,-45,-89,213,369,329,85,414,244,89,98,44,22,-23,-271,37,-204,150,83,-245,60,-50,107,-243,-95,-190,86,-377,-35,-104,-85,-159,45,-212,-166,-176,-167,-53,-194,-115,173,-213,333,118,316,-262,-353,159,-58,1,82,29,-285,86,180,-152,-90,-26,300,-225,-487,136,141,-74,117,-181,-108,268,337,257,87,-99,-114,-45,13,-446,89,55,-204,12,-45,84,60,-379,85,-135,44,301,-317,-83,-241,410,54,-203,-287,-101,-215,-199,4,162,-178,279,-275,287,256,94,-124,61,-177,440,157,358,237,-163,277,-260,-242,408,176,166,-55,284,-74,-286,-52,-385,-114,-164,86,-191,155,-151,70,-49,-417,-184,300,-9,-302,-303,-46,121,159,-174,-80,67,-403,-279,11,341,70,146,-52,157,-91,199,112,-206,-282,387,-265,-43,194,-280,-157,56,226,64,193,-84,-377,497,24,179,321,-220,260,-151,264,-50,33,-257,-214,197,-174,-26,-213,-109,457,-97,-174,-150,-326,-149,69,270,339,45,124,-185,31,-138,-7,-124,136,252,-359,293,-57,83,-35,-392,90,30,-352,76,-61,242,69,-214,384,178,-253,-267,241,422,289,186,121,230,71,81,405,330,10,-10,210,312,138,17,-225,346,93,77,-1,450,-92,59,276,177,82,146,-61,-40,-140,218,218,-67,-20,-185,166,71,313,130,-144,-72,-68,110,-6,308,172,100,-16,9,-109,-119,-64,-132,-216,129,208,-120,-56,-194,184,58,-208,-276,175,248,-341,297,249,-305,130,120,-3,236,5,-211,-296,-96,-176,132,-12,178,-236,126,127,183,163,318,90,-211,-313,-201,4,90,-106,-116,-255,166,-69,-80,-477,37,320,111,-229,138,-361,19,-23,-157,137,354,109,-2,330,98,130,413,-399,-88,-26,85,-20,172,-334,311,76,48,197,-366,51,-144,-28,-157,348,241,-49,393,-367,209,152,2,-73,-30,65,66,49,57,-176,103,360,299,-48,360,236,-406,10,-50,240,-224,140,-377,41,-3,-97,144,-220,-346,296,-263,-315,-1,82,127,50,-218,-164,-239,-247,231,317,238,-62,418,257,-291,-193,-47,-259,-187,-12,-357,-217,-106,-242,175,-73,-33,-218,236,-453,-260,264,388,-213,442,-157,-13,-383,313,33,-84,134,47,208,217,-48,-141,-423,100,116,-80,274,-2,484,470,-156,63,399,93,59,209,121,53,491,96,61,214,83,152,-244,69,-42,-289,135,77,-213,277,-412,-205,-363,-138,357,-366,-330,303,-263,110,161,-198,89,-211,121,-35,-386,57,-31,-114,-449,325,283,-58,-210,48,73,172,-131,-196,163,126,3,-317,-64,-107,76,-218,441,-295,-254,33,-51,168,-76,156,171,-21,127,-384,-236,36,-397,58,63,117,-152,47,-3,265,115,-84,50,-121,244,-300,37,51,-309,-79,-174,177,105,-69,-37,33,299,-150,30,23,-191,-136,-19,-23,232,23,63,150,-155,-207,167,178,113,-68,65,169,-38,153,-52,-215,-89,351,115,38,367,329,287,-56,-139,144,174,-337,186,265,386,143,-431,72,214,-37,18,21,178,250,275,-235,-1,115,77,115,87,153,-1,-213,-146,-101,-125,249,260,-65,-65,44,-431,-195,-17,-123,10,-363,302,-70,73,25,214,-354,-161,262,261,-318,198,52,-69,213,-316,57,-102,239,-46,337,206,9,-389,38,23,-248,270,104,48,-329,216,361,85,-14,308,49,411,-170,-41,177,-284,-106,422,268,-288,-83,98,-161,-373,-273,-4,-66,88,-285,-44,228,102,-2,84,436,-101,-132,239,42,-71,305,-97,299,-58,279,-237,-157,-212,-30,69,-519,-464,-43,229,76,-324,-429,134,5,84,128,162,-454,46,307,-322,158,-136,148,320,34,172,154,248,-280,-106,89,-206,-25,-222,-10,195,10,135,-232,-223,199,-74,-30,-143,-58,112,-145,-59,40,408,380,129,-55,337,54,-16,-254,304,6,352,344,-306,-117,-30,-31,-270,144,-266,122,-62,345,-266,-427,197,32,-151,127,-74,-279,-104,116,14,-121,-272,129,-17,-212,-112,-273,-180,-132,-307,-47,226,-252,-3,-103,64,182,-425,283,-1,-145,431,-371,-19,-34,195,-89,427,95,25,-232,206,-405,240,-348,87,-62,-123,-236,-345,202,-127,-12,13,-101,-208,181,393,-89,-228,350,-22,-132,118,201,-173,245,250,-58,22,56,297,-391,112,94,39,369,-311,125,-55,-130,-437,74,-75,-121,144,150,-115,241,-500,389,90,122,74,118,-351,-297,39,-356,-90,-184,382,-218,-48,-202,340,-225,436,161,45,139,-222,-191,51,357,-147,-197,91,-103,-448,381,-472,260,343,-324,-145,-215,-20,319,115,-148,-48,-122,-82,-124,-203,298,117,214,-14,-73,432,169,367,273,-216,227,78,-35,-76,249,46,273,-278,189,36,254,120,469,-166,162,-378,200,443,288,-213,-186,-193,126,-65,357,83,59,23,-90,-335,-261,42,0,11,59,-169,114,-108,-288,-142,217,183,133,-280,227,-245,-47,159,440,-222,8,-220,181,295,-44,-121,86,464,-113,-84,146,256,-328,-165,143,-231,205,-185,219,229,-180,-24,-432,-54,129,-256,351,116,-258,206,144,140,216,-146,-383,-449,-40,-200,175,127,37,-22,358,-165,321,-318,150,-102,-287,314,4,337,-176,454,32,156,59,396,15,-261,-18,-381,92,-28,-45,77,-186,23,-111,152,-56,-115,138,221,-7,-60,82,-127,-86,-237,-369,325,12,-86,-108,-224,184,-139,155,-74,-409,274,196,-495,-61,493,319,281,59,118,323,-211,297,-217,-140,-45,114,-317,278,-296,-72,123,372,163,68,143,-56,246,-290,150,-221,399,-63,117,152,69,-253,81,111,-153,-24,-185,-275,339,269,-384,437,-165,19,-65,381,66,98,-233,158,-244,-137,182,356,150,428,-78,73,140,134,142,4,-123,238,142,-33,-38,-223,-247,-5,234,-1,405,21,-56,-337,-407,96,312,147,-268,-72,235,75,34,204,8,-56,180,-76,64,-291,-251,-12,149,-5,-149,217,-110,161,-39,-226,429,-156,-195,-312,222,-256,46,-195,-107,-65,-18,215,0,131,-418,-177,435,58,249,50,157,82,49,-174,225,-135,-387,106,-37,-124,-24,-161,84,-288,-264,3,-123,-373,73,-307,151,241,47,196,83,17,-149,-313,246,81,223,313,-117,95,-199,-260,-141,-131,114,380,-218,138,306,40,112,26,137,123,-293,-30,23,-183,483,-31,-38,26,-30,113,219,106,160,-350,44,-19,-170,270,63,240,-69,138,-152,215,-216,115,140,168,61,-120,146,-27,-186,-332,128,-132,-256,-125,-155,31,-111,124,52,-3,119,48,34,-41,67,256,281,-136,174,-224,231,89,-289,-29,-266,336,319,35,223,347,-36,-85,-64,-20,-36,-76,93,-62,121,20,-22,-18,66,-146,17,228,-15,266,67,329,-105,304,-203,-53,23,287,-304,261,-99,-144,-397,-97,-3,-307,168,-130,-91,-19,157,-26,24,44,207,-31,349,-144,-277,-277,-226,113,-98,-471,166,-144,173,127,261,-59,-295,26,93,-4,-155,-155,223,-83,58,286,-174,-224,-135,57,-396,246,192,67,-21,-146,190,-268,-5,-31,-14,281,-86,-112,114,192,46,-323,100,-314,165,396,80,155,27,14,75,-127,300,-145,147,-40,-322,-9,191,423,-350,369,282,82,110,114,-375,56,-204,-9,60,-13,191,-51,-134,333,-161,-171,-79,-86,-4,-202,-318,-112,163,-46,-105,-146,-85,-202,29,59,38,-198,-451,-270,57,87,33,128,-445,-57,23,292,196,45,-29,35,-46,-389,-21,-324,257,-95,109,-341,-202,8,159,-366,52,-88,100,233,-161,-437,156,187,64,-446,-18,-13,0,-53,-164,115,-187,411,-173,157,133,-102,-52,230,-3,-266,44,-176,-119,38,32,166,148,253,62,268,97,-105,-165,-183,-33,-35,43,-195,-111,-25,-265,-179,185,-147,59,-70,207,23,82,-278,-50,-322,101,87,-346,235,75,-104,335,224,-12,242,-24,-255,283,-91,41,25,264,70,153,47,-260,147,229,-157,113,-13,296,281,-272,-293,-213,-83,-219,-96,69,37,162,-64,4,-214,46,-271,1,334,-166,-263,-69,-164,265,204,180,491,76,-115,-6,-183,-109,20,-303,11,-33,41,36,-143,57,-1,-14,431,162,53,44,122,78,94,-135,59,90,48,58,248,74,-349,151,96,-30,-240,258,-372,23,40,357,83,115,-145,173,111,-160,316,-113,64,-7,423,102,196,-68,56,194,261,-373,-279,-7,-332,237,-167,-182,-293,34,-89,350,-408,44,-87,-6,83,99,-40,-40,-356,90,71,453,354,-74,154,252,-188,-129,370,-105,141,73,-82,46,236,231,-41,-282,-21,178,-11,-87,-32,77,-141,-430,184,-221,-308,99,-335,-256,5,-306,212,240,-239,-429,367,-380,53,-390,-134,196,57,158,275,-80,-278,91,226,201,-268,200,210,408,-117,109,96,5,60,-75,-365,-189,244,202,41,-256,72,40,-238,458,219,-77,-192,144,-397,112,-10,-31,-210,-250,125,389,96,53,-198,-167,-172,-269,161,-186,-21,28,112,106,127,74,-235,354,-236,-106,-71,-196,-45,17,287,122,118,182,10,-121,135,68,-466,338,-65,-199,202,-356,-31,85,326,195,330,-197,343,-55,-143,206,-242,-324,-390,-97,234,-429,65,-189,192,104,368,-200,-262,-31,216,-29,-505,298,133,-22,156,-40,79,-389,-356,280,-197,-13,137,-49,95,-235,-304,-252,-104,-193,-260,110,277,304,-255,-227,-433,-60,-382,256,-360,-73,-105,-426,-172,94,134,-280,261,-269,319,388,36,-195,107,-456,-88,-42,-188,-200,-97,-220,-158,-145,366,78,-130,-10,-123,89,-115,-106,34,-70,-22,-362,296,-328,249,298,476,239,419,-325,173,108,-204,-2,-113,173,-42,-151,-168,-93,-261,10,-510,-209,-367,105,89,130,-69,-415,177,-87,-90,258,348,218,-164,-202,-291,159,127,-354,167,-75,321,-243,77,-9,169,128,203,-191,101,63,113,-130,114,-20,-259,-27,155,121,-380,-446,171,73,3,382,-90,250,289,-262,126,-192,248,-45,181,410,-305,76,114,43,-158,49,-130,-159,-12,-88,-119,-254,-60,287,199,227,-242,-17,-384,-272,-226,263,-305,-176,113,-475,-140,-382,226,259,-230,186,-222,-78,251,122,331,55,132,240,35,-422,-165,-226,443,298,-502,113,424,48,-67,474,325,-143,-157,-381,-432,394,-313,131,-292,152,7,-124,503,-58,15,71,-46,47,202,-183,-292,-185,-14,102,-1,-438,149,223,353,263,-93,173,398,112,-321,-173,-166,-170,-275,317,288,42,-42,28,-214,-381,314,-154,188,433,43,-109,178,71,13,12,498,-150,69,-370,83,-167,6,1,140,277,-57,179,7,-191,-117,84,-124,220,333,146,281,-100,417,-516,-28,-114,101,33,266,-304,-74,30,-48,-273,-10,265,-116,16,205,-326,119,147,-262,-163,146,162,-3,-411,189,-265,0,-54,346,-309,26,-105,-172,-57,251,-277,-424,105,165,139,0,28,64,-321,-107,-67,-61,158,-418,-191,-299,375,442,275,170,173,435,158,186,304,307,-495,-422,-370,-18,133,99,-99,355,-69,-459,-166,-412,109,79,55,-386,-19,-24,-335,-206,-15,-173,143,162,-165,-363,-166,-3,194,89,-47,-34,-404,237,24,512,29,340,-124,-231,85,256,0,25,-34,221,-24,-255,213,111,-7,-138,120,-272,261,-11,84,-292,111,247,289,-276,357,325,-355,266,-155,-4,53,-110,297,-77,-168,277,-82,234,91,54,220,-15,-288,21,-18,316,-64,-269,-90,-77,-202,-10,14,-389,176,-479,-343,-409,8,-56,503,-126,-38,-153,249,-190,406,-13,376,157,-96,376,-56,108,241,150,26,255,-164,215,57,283,290,-73,-424,65,59,120,-289,167,-15,-134,382,-341,-177,-91,-256,-297,-54,-24,-222,-409,109,-52,-195,115,182,-49,-162,-170,204,280,-136,111,-110,335,424,327,-181,-172,-110,-213,22,-172,321,-86,6,204,-258,-352,-71,-306,-26,157,58,-99,162,214,18,-235,87,-316,92,-144,114,-167,-342,-266,82,-208,102,130,168,-342,202,-31,317,-285,-116,269,-126,69,-214,-69,82,-167,89,-214,54,-44,-30,38,-216,-30,312,53,83,-214,-186,433,268,151,64,-92,-71,-106,-363,-392,-304,377,-131,-271,13,116,18,-98,-352,-60,78,-299,162,333,-137,-76,-108,256,269,-153,241,49,119,-62,-203,267,236,-186,-14,-243,-34,-140,34,107,-9,333,102,-120,-111,307,45,67,-181,-186,-140,-399,-34,-49,292,-110,-441,-137,-52,-231,-18,41,-270,339,-278,-47,173,115,-347,107,207,44,195,165,397,-138,140,-363,-6,11,-61,-120,131,457,-172,-121,-156,74,11,-245,-84,416,32,-102,281,-86,3,70,198,158,-248,31,-220,135,-7,-94,-141,25,-79,-162,-309,21,89,78,-467,-81,-234,-199,155,-369,-360,-203,302,147,2,11,-228,66,439,1,-173,55,147,-170,138,-453,479,123,309,279,-330,-430,285,300,-126,-153,19,96,-115,-57,-284,413,-166,294,219,93,-412,-127,205,92,341,3,436,-361,251,437,308,-89,216,-51,39,188,-309,298,285,468,41,25,480,174,87,-17,51,18,126,211,29,-190,-130,36,-93,-153,130,-85,-80,207,232,178,-153,-215,-398,-35,83,119,436,-112,-49,238,-153,244,181,-187,45,174,-404,-42,-318,133,328,-287,-78,162,44,-442,-260,-432,-303,-39,-269,190,-24,108,132,234,172,-430,-341,81,270,346,-185,-340,83,-141,276,354,404,-187,382,-123,-27,168,-17,49,110,-89,-261,-136,-405,-36,-78,193,246,251,-46,-98,60,-52,-70,181,-82,85,-57,-43,151,462,-303,-2,24,-160,199,15,-7,-420,9,177,-267,99,-160,-161,-184,8,-111,258,263,-96,-144,-206,273,256,358,-221,359,-402,204,-313,211,14,-469,-43,192,141,-26,-249,293,46,292,-5,-46,253,146,-58,-215,293,169,189,-66,20,248,231,-97,232,281,-242,73,-58,-8,278,-3,-214,-70,-374,-313,-307,-150,-163,-23,-48,122,-201,43,26,148,-14,-306,-125,146,-105,-281,-283,121,-16,18,-81,-182,-482,95,14,-402,329,-322,-216,284,-146,-243,-175,-99,135,-63,462,14,-175,-75,-326,75,119,225,-148,267,-196,-3,-82,5,-200,-187,11,-256,-161,29,91,-371,-133,-33,-379,-47,37,28,133,-3,305,-124,316,308,183,137,-21,-223,-320,-425,376,-23,-480,86,-125,-201,-40,15,-447,-283,-129,122,243,69,-39,191,-41,438,-485,337,-285,359,-14,74,77,-78,269,-189,88,351,98,-187,210,325,80,239,145,409,80,11,11,66,-279,-30,-263,-388,-220,224,10,258,-324,-229,325,81,-16,228,-190,-68,246,364,-363,-55,-45,-169,-134,-253,236,403,-9,-36,400,-166,128,-217,-161,101,109,-13,-476,328,207,80,67,177,141,132,93,-121,21,-104,27,-89,390,-226,354,-155,-179,-27,-333,153,-69,-439,476,179,147,446,56,54,36,-110,-301,345,434,-372,290,-67,-254,-198,-326,448,-80,-70,-335,-143,-187,-116,321,-155,-360,-85,70,-38,-126,-294,-206,-393,-75,-329,425,-168,82,-254,-395,-154,-224,-172,-213,-128,-29,94,16,241,36,-294,-105,-4,-106,140,-220,40,359,267,197,-285,-41,-139,450,-461,143,132,-39,-10,-251,-323,479,89,43,-26,36,-363,-174,334,193,102,240,-178,293,-415,-195,-272,391,17,-457,372,-105,387,-61,49,-188,-71,182,-234,-117,141,-68,-225,364,-310,-133,404,319,-17,174,-94,-419,138,-36,357,-223,-190,-55,-25,109,-116,340,-113,-211,179,329,170,389,-165,-169,181,319,-66,-63,232,189,-236,96,213,42,-126,285,166,-29,184,-424,-4,-171,250,-293,-154,5,130,-228,437,-326,95,68,81,471,199,-295,269,-18,52,328,-260,-290,-36,-68,46,29,263,279,-216,-119,-98,-178,145,78,134,388,-5,-335,51,-387,-111,135,-111,-7,20,-279,208,-14,-223,216,54,-204,451,-208,207,-88,-13,150,-112,338,58,390,-239,-455,13,-52,-131,5,-28,7,-40,270,-418,-266,14,-449,-81,-143,-169,-315,-70,172,46,-236,288,316,-376,-360,-63,35,-114,135,-126,-97,160,31,217,-385,-325,-100,507,156,126,335,2,135,480,-233,203,138,30,-318,331,-51,-211,28,-311,-280,389,-248,-43,25,-43,178,41,-128,252,-49,-100,60,92,-58,-486,69,268,-273,-240,-106,160,-144,-54,-22,-164,-48,324,82,-228,52,341,266,120,-306,-167,99,-13,-10,59,-464,-158,-404,155,17,75,289,-250,35,194,-289,-187,-42,-244,-4,35,-392,-293,-201,-160,277,85,383,30,-450,-125,81,-161,-93,-270,46,-110,101,250,340,-119,328,-100,91,342,35,139,-195,315,319,407,-235,-146,302,110,-252,-359,-336,-243,123,331,282,-380,-23,65,-29,-146,5,59,23,164,-18,73,4,132,77,-324,301,-233,178,-355,-148,-435,-272,133,-161,-142,241,322,258,109,160,-293,175,375,-120,156,-163,-425,130,150,-8,101,330,158,246,-335,-47,-189,141,188,-110,-79,67,-220,152,-99,211,61,65,141,173,213,-73,-45,-89,-196,292,176,-16,144,-47,189,-219,33,244,114,-121,-190,-170,85,-155,-163,-10,47,211,237,-194,114,30,353,97,239,-70,142,159,21,-431,392,253,358,83,153,72,-101,186,-196,141,118,-294,-109,98,-289,-136,-116,-145,414,238,123,79,328,-7,35,268,-73,36,-31,273,-125,-232,19,50,18,-328,23,461,261,114,-64,105,-193,47,-177,113,199,-137,-291,428,45,104,124,468,331,-186,-281,20,478,-210,-291,17,275,210,150,101,146,479,-83,359,65,-49,-501,-262,102,58,-30,-124,-63,-232,-15,-215,-383,-67,443,8,-184,-51,74,38,-190,-454,5,128,72,-294,-3,198,222,-15,-212,102,244,164,-199,411,-3,41,216,-23,-139,-102,-76,-211,110,-216,-72,-249,-285,-287,113,-151,143,-200,259,53,-94,-273,-244,-61,-42,299,-368,178,148,-44,153,63,-140,-111,-319,-126,81,-216,281,-268,-427,-184,-339,-433,292,-310,218,-279,-410,-212,-148,-70,-4,-78,380,-383,-4,-27,144,-16,-53,348,157,-68,-45,-425,160,-132,89,-71,-82,329,209,-149,-144,-110,-1,227,-363,139,-143,-76,428,-478,-98,56,-294,129,-24,358,270,-224,-93,175,212,187,-34,64,-51,360,415,81,-185,180,380,-166,193,19,-266,-76,188,144,-112,-161,-219,94,181,161,-49,283,89,-173,-17,146,21,-33,88,101,22,424,-156,39,171,-276,-323,79,-122,-238,331,-129,403,-50,-294,-220,-4,-242,112,283,198,-256,497,-111,-34,-247,-103,-404,-14,-275,48,-183,-219,-21,25,-22,12,-65,-51,-249,227,45,-358,304,218,134,94,-309,292,125,-396,58,-13,111,-203,-422,57,152,-71,76,236,-63,227,39,183,42,170,103,150,-474,243,-196,207,-332,198,-57,86,-234,-132,-6,263,-88,90,-358,317,189,-375,-70,-297,163,-194,329,-174,17,-63,-64,238,59,-64,-159,-178,-187,-245,-113,-118,-104,-154,-61,-444,192,66,-58,251,-33,114,17,110,-42,245,-306,-105,27,158,-32,-32,302,266,257,37,-111,33,45,127,100,-469,21,-237,-334,168,-174,79,145,168,9,-404,-202,116,42,-326,-324,-142,121,-368,-191,219,-124,52,-79,73,127,92,31,272,-226,56,94,32,306,100,13,-197,-65,243,13,220,-229,-237,29,-19,127,195,272,4,-85,96,-82,58,-208,-74,-10,-14,-99,-23,-159,-295,-163,-134,183,302,278,95,180,98,-48,-224,89,-33,130,-51,-106,412,-164,-206,-166,23,-109,51,-284,19,221,-282,-121,199,-155,469,-250,-387,171,-187,-312,223,279,-157,-93,-235,-247,319,-80,-1,-35,-7,-18,167,142,359,73,-170,12,8,-110,-331,386,-162,-17,210,-191,91,80,-4,61,-167,37,-376,-210,107,194,60,224,-260,247,-25,87,441,-147,-346,59,-33,-18,-162,-300,-151,-23,-73,73,-81,-326,20,-54,439,182,-414,341,-272,138,29,-32,64,99,84,-42,-285,224,-47,5,129,-36,-266,2,132,9,188,11,284,82,43,-110,130,-230,109,0,-93,-4,-5,-33,197,234,303,-60,-164,-141,246,-35,116,-315,-162,-153,242,-135,79,-103,75,18,62,76,-206,-422,-179,-149,16,195,139,-122,119,-267,169,-231,12,222,6,-96,143,147,-155,473,77,-11,-352,-333,-5,152,216,189,310,-75,123,-158,145,419,364,-71,107,-97,4,3,331,23,-53,124,-410,449,-144,-30,-73,95,-95,224,32,-26,237,-73,112,-83,21,92,-26,292,304,247,-280,-322,-97,89,26,-90,175,-9,245,106,-404,144,-69,-84,206,434,-186,-232,47,138,50,152,-89,-219,265,-243,-64,272,71,-176,209,104,-129,55,-154,469,-339,-433,63,-3,-89,24,-126,-300,170,-38,234,-190,-315,-171,-60,296,-31,101,-241,-118,-51,249,249,291,-43,81,44,-24,-75,-156,245,123,-55,40,30,-398,204,-150,9,-187,-125,-204,-454,-143,-57,220,263,39,319,46,366,-116,5,-85,426,-229,-57,-29,79,161,114,308,-292,-80,54,203,-299,-242,23,16,207,-362,-291,-304,183,-130,143,-228,190,-276,-42,67,-25,142,-113,36,100,-287,203,95,117,-229,88,9,35,68,-204,-151,-247,-58,213,-43,-377,328,-31,-194,-114,-182,-374,-260,-316,-191,-230,122,-179,-146,192,165,148,-165,-50,119,-262,141,-165,160,-117,68,-21,213,72,-389,108,-380,108,-441,-20,-49,209,37,-377,32,-80,36,15,-247,201,11,12,64,-239,-63,-117,17,-127,130,339,-170,-48,-302,130,-242,-147,-231,222,356,68,22,246,-174,262,-67,-25,345,-4,5,-443,183,-255,173,367,462,-150,200,-123,46,252,94,-252,252,-361,173,87,207,371,39,186,181,-107,-55,-272,-66,-152,188,-27,-62,433,371,61,340,0,377,309,34,-257,106,-148,-228,271,95,-20,182,-35,-161,156,-177,45,42,-69,-461,-54,-189,-190,271,56,-95,1,-159,-125,2,78,75,-42,251,64,-246,135,-66,-38,103,268,221,92,-221,58,2,-440,-282,-65,-399,193,43,-129,71,-183,-416,188,425,-247,159,-104,178,-221,132,-2,113,22,130,145,-219,-437,-107,-65,-51,272,-151,-247,148,117,161,250,158,322,88,201,90,97,-31,376,241,-182,5,-80,-156,343,421,-414,46,318,-10,-148,7,-31,-81,-283,-12,15,130,24,1,-188,-354,168,-2,-64,30,-351,98,-24,166,265,-147,-77,-321,-228,-7,41,154,233,-313,108,-289,-84,51,-297,-24,-451,-138,-209,227,-109,-459,372,-99,190,267,354,133,-158,255,-454,-30,168,116,3,77,-79,-111,-245,123,-213,158,-97,-167,210,-120,287,380,-123,-353,179,423,-321,80,-198,71,329,75,-90,-225,407,-350,247,-168,-47,49,-387,-326,-51,-289,-26,186,3,122,-352,-17,36,-90,117,-17,247,-200,-147,-35,-129,122,-24,5,-196,-152,-47,447,-276,274,-219,-22,176,35,-60,-417,-34,-89,307,-286,-260,-267,56,55,463,403,-51,-418,-457,-349,-181,-19,-131,404,193,70,435,-26,314,21,146,14,284,125,313,159,84,-441,-105,103,-145,77,405,382,256,83,80,-224,9,75,-111,22,-270,-177,-222,38,-284,-12,34,-24,224,85,154,251,-461,6,100,-237,-288,246,268,360,92,32,-339,-95,277,-281,-64,-324,164,-356,-207,-259,-245,35,-26,345,75,29,108,-206,422,83,-138,79,-370,397,-3,124,-492,-155,359,-129,211,24,191,-118,-172,-85,4,77,174,-137,-410,-422,305,352,211,-185,-200,-176,127,-328,-70,54,116,176,444,-130,-358,96,134,129,30,282,-88,41,110,7,279,453,-180,47,263,330,74,-241,226,-259,-13,-171,-281,92,36,56,-268,239,64,-111,330,103,-347,-474,-15,-404,9,-198,364,-66,-189,200,72,-134,38,-67,264,306,-144,192,-64,166,-138,40,92,-145,-185,411,450,-476,22,-235,6,1,-69,303,69,-188,184,-75,-483,-179,-164,423,-302,65,-143,-100,-180,25,202,-198,158,261,367,-261,18,-252,-289,249,-109,-204,-337,121,-16,83,-65,227,314,96,267,334,-31,-174,-87,82,-174,142,-214,-28,53,-118,-177,-129,-478,-105,204,322,396,308,-228,-78,-242,154,30,4,-332,482,5,75,-157,74,-234,44,-148,-256,57,30,-192,47,-154,132,171,-332,-273,43,-37,-83,-384,-58,-116,-114,-186,-374,-87,422,473,25,-46,-72,-297,84,169,-360,22,100,113,-153,102,-440,21,-269,409,-70,-353,-219,-2,176,-381,-63,-5,-28,-6,265,-222,141,6,17,-35,201,151,-343,131,-354,-352,300,167,306,113,179,-120,53,-204,-28,-31,-19,200,148,-4,-377,7,8,344,-77,174,483,-174,-422,-213,163,-328,259,-172,-123,110,176,67,-11,91,-324,319,103,-81,-363,-226,-63,-219,-320,61,240,-248,-186,-181,138,17,69,81,-279,-174,-276,-27,150,247,-329,-115,-25,-39,-60,336,120,114,-145,-16,325,276,140,202,72,-95,-145,137,218,104,198,371,-2,169,-81,-168,250,50,216,277,222,98,6,-113,-358,220,-6,75,-30,377,48,-52,384,83,48,-153,296,244,-302,144,-27,-257,-292,-250,24,-130,-88,-12,-234,-33,91,-304,-44,42,88,380,-174,-110,-6,176,18,-315,-15,-17,43,-254,-78,-84,-205,-279,-193,-181,-159,260,297,-130,-264,198,56,351,-271,324,-225,-96,-32,267,158,72,-242,97,10,-91,186,-31,40,9,93,-34,-336,-251,-110,-404,183,243,127,-12,-304,-11,275,65,-50,33,281,-76,163,-154,-188,85,53,-384,-275,-380,-346,-46,-26,52,329,131,14,-4,-195,-98,211,-313,-390,-205,-428,74,187,-330,34,265,134,164,349,40,-195,-123,247,-226,272,-88,-344,-79,-180,-248,-139,-416,-179,11,-288,279,8,75,375,-110,-356,-31,-306,137,-304,-1,287,275,206,10,121,-167,-337,41,465,139,-156,71,278,-9,-22,230,-428,-60,243,-347,-396,65,46,422,201,-268,13,-389,32,-219,122,254,155,-100,-40,413,292,93,171,63,-146,-484,110,4,-82,-230,254,-146,235,-183,-99,-228,105,127,189,181,-83,-271,-170,-77,-98,84,-234,345,-46,-320,-315,-199,-117,12,-16,-104,-174,237,132,-184,38,83,-85,-136,119,-94,434,54,50,289,-274,-8,-313,-52,244,-289,232,-21,-353,288,-10,30,-202,208,-27,275,399,162,157,74,419,-262,-26,-52,192,104,208,-231,-133,82,-133,-279,-56,427,94,247,42,-393,-410,51,78,-73,-229,-119,-138,4,35,129,15,-320,294,-278,264,-94,345,-152,-116,-251,-124,-74,14,-57,378,-176,465,106,-151,-161,-121,389,9,-151,118,14,-54,-201,-2,-269,54,7,191,-151,-276,146,510,-394,-392,372,-240,-96,-93,33,53,223,-43,-19,35,183,432,-23,-187,-105,-92,247,-34,287,21,192,352,339,-324,-84,495,-309,230,197,-41,327,-201,-376,-101,-278,175,280,-29,320,138,151,85,-24,-293,-192,-242,-253,106,41,-218,-406,-75,147,-112,252,442,-113,132,-325,470,10,127,-188,402,90,81,143,-229,460,-40,-159,-135,-42,-235,-39,174,-440,14,-329,-41,102,-355,143,7,57,148,-65,-61,-77,208,283,-59,20,-90,230,-36,-288,-103,22,-9,190,59,-39,304,-106,-269,-122,-166,87,-221,-426,-42,129,195,-394,-111,277,-135,-376,-287,486,-6,-249,-20,-328,198,509,-420,68,7,359,126,-221,24,216,50,374,-241,-15,170,191,287,-74,-83,138,-153,406,1,-43,-430,-280,365,-149,-332,-228,210,123,-70,-27,299,70,-173,103,-375,-283,151,366,178,324,199,-146,-83,-142,161,192,-7,-34,186,288,-107,10,60,20,356,-324,242,157,211,0,-414,-165,21,99,333,86,-23,292,-202,225,-20,-148,333,-268,-233,415,-348,45,179,230,305,16,-80,-21,-279,303,62,-321,77,251,-26,98,-379,-60,-36,417,-96,-130,-371,-306,-256,6,87,-141,421,-261,268,-201,-373,317,48,413,31,-37,343,-128,216,344,-307,-87,63,391,20,-298,-3,41,-231,-15,3,-193,-168,55,-25,427,370,272,-1,-420,-334,-32,184,-124,-286,-428,-71,-187,405,19,-127,423,-29,252,-173,-165,453,56,242,-104,-10,-256,-262,-386,-297,100,-214,-10,401,24,177,-181,90,-68,18,20,127,-27,-100,-138,-380,-258,205,-74,48,172,342,65,201,207,-192,-14,197,311,40,395,-344,413,185,107,166,342,-65,-19,-122,307,62,250,-240,7,66,91,273,-101,287,-116,2,87,345,232,-127,-456,-12,-49,-177,180,217,56,189,249,124,97,26,-98,214,-4,-197,-234,110,244,146,49,-19,227,-165,296,84,-19,256,-51,425,-199,-116,152,46,-255,-64,180,-72,274,-239,171,122,-416,40,-52,-82,-459,-91,246,52,-155,119,320,-219,168,-303,-61,-72,-33,9,-184,-217,150,-7,81,-315,-28,38,-412,70,-358,149,184,-238,344,123,400,83,-320,-354,261,-264,-153,89,92,212,221,-206,356,199,93,-315,-3,-108,247,-420,-7,143,-181,426,-43,-434,130,19,-381,153,299,-12,169,-186,93,0,155,-33,29,-203,55,-181,-27,-236,-67,-2,-264,-170,38,-101,-400,227,-93,-54,-221,232,-26,-292,-216,-337,-43,49,464,224,-222,14,330,49,178,-320,71,27,-12,97,379,115,-133,-7,-292,392,-287,6,221,-223,-157,201,344,-69,107,-46,-309,-422,238,150,413,-381,362,-228,-104,120,-213,246,262,247,-185,-319,113,36,18,-52,152,3,34,340,-20,-187,-307,-385,-370,110,-33,-367,327,-35,115,23,-79,-148,-147,-245,-30,424,-126,-189,-213,-2,282,98,195,209,322,14,498,-410,-343,314,316,190,496,15,253,58,115,148,-311,-26,75,278,187,97,107,108,417,-205,281,-4,-45,-115,-236,-2,-21,-326,-432,89,-49,69,73,-240,-133,-155,-172,-150,-289,-283,318,303,-132,-189,-476,-280,-61,-143,-246,238,-108,2,-45,63,-281,-275,-388,-183,-74,179,507,174,147,292,281,88,40,-191,148,41,179,-21,16,252,-218,-120,91,429,167,119,347,-387,80,-44,-33,38,-280,65,17,190,71,-45,-167,222,137,221,229,318,28,-222,-207,306,40,-12,161,-216,-247,247,108,405,-35,222,-379,346,64,98,-215,-88,302,42,-76,456,-474,86,54,160,218,-204,-109,8,-49,80,-120,-345,-213,-364,192,-235,433,-137,-102,-256,-45,-227,50,358,10,-422,110,117,-88,57,38,16,-438,-203,-143,2,245,435,-197,144,-116,176,175,-63,96,297,-382,23,160,12,351,385,-208,-238,-446,-8,168,-88,215,193,73,-408,191,-179,104,238,55,94,-20,-33,-337,48,-316,49,97,392,306,269,166,410,-355,-290,39,-190,-159,77,145,-506,136,31,171,-279,44,-158,-2,355,50,221,174,200,201,116,219,-159,70,-50,303,-347,210,-46,213,-28,-77,196,425,-273,-57,-119,395,-128,133,32,64,-109,297,-102,255,311,120,-5,313,137,-252,-61,156,139,-458,164,-130,440,-39,-99,184,109,-185,-126,-261,-161,-17,89,-85,189,-7,-224,-31,308,52,265,98,-100,158,-72,147,-34,507,214,-106,156,96,140,-172,335,1,61,-107,24,133,360,169,-119,25,-72,161,-101,-340,-35,284,253,-87,42,-77,-366,159,-42,140,140,38,-362,-231,-440,254,123,-146,114,331,154,-235,-455,-295,283,-134,-329,-78,199,440,-62,-231,-246,176,-386,-239,-54,-236,77,413,-36,234,348,84,-228,-48,-87,307,165,7,-343,14,-27,34,231,-31,46,-435,117,94,-20,148,112,-141,22,-267,43,280,-250,283,-86,345,375,15,421,-349,150,339,-321,266,259,37,177,384,125,136,-472,-86,68,-267,59,131,-237,155,361,239,13,-18,-384,-187,171,-76,-373,53,-145,476,-237,217,42,135,172,-180,-44,-174,388,-150,-302,269,-91,173,-362,91,-126,40,241,-40,-216,113,93,-104,243,117,-118,30,-27,-19,203,323,-16,-233,194,69,108,-67,-442,327,-214,82,-2,282,-63,39,121,368,-379,-189,-103,4,72,-10,-223,321,5,5,-10,440,145,167,349,-299,188,-7,28,446,361,-52,9,153,-399,131,-256,-180,233,-32,152,20,313,-94,-208,1,-144,74,221,94,-173,392,-68,-197,112,-287,355,-127,127,-16,-352,343,186,-246,187,108,-244,365,-64,-34,90,-228,-27,-57,-94,-315,-439,149,223,-98,111,221,17,-53,-88,99,-196,-253,111,-144,179,-250,-139,225,-197,102,172,65,-440,-315,362,234,115,66,-157,323,341,33,-406,-160,171,-125,-401,66,31,-493,466,-290,-218,-213,-109,359,107,-325,-385,26,-241,221,227,247,145,288,-38,-305,-104,-442,247,-372,-19,-135,-320,-20,-86,163,14,-377,50,190,231,-247,-33,-201,290,-340,114,-315,192,406,19,90,-313,-143,-423,63,180,-456,114,210,-55,63,142,15,263,-45,365,210,-365,119,119,-322,200,-106,-19,-464,328,177,-269,29,157,106,-275,38,62,140,81,361,198,-151,-25,269,198,-211,-183,-290,299,-135,-175,93,-200,-317,451,101,-64,55,281,-71,44,158,-202,-351,452,-188,172,263,232,-293,-88,-121,114,34,109,-28,130,331,62,-63,-272,77,-94,-24,260,-11,-169,203,-148,261,-269,-321,-26,63,-235,-225,101,-7,-184,152,-42,265,50,-409,334,-303,181,198,-422,-114,35,-393,-373,19,-3,-333,-71,59,235,-51,30,-195,160,-78,330,-138,-33,-27,-110,-226,267,191,65,-11,377,77,-431,88,376,-60,28,-211,-235,-202,-82,198,-336,-165,236,192,-195,-333,195,123,-24,384,90,141,-88,204,328,-52,-282,237,285,-327,119,-232,228,-38,-369,9,-354,174,80,357,-220,9,33,-52,-146,486,-392,170,55,153,-258,39,-191,18,-270,-255,-121,399,255,476,208,-156,287,-5,230,-165,89,-190,276,-263,-194,135,69,55,235,-325,-469,-422,-178,242,-175,-146,264,39,-354,385,-141,68,-264,-48,32,346,-244,0,142,-226,-279,193,-27,-337,-246,-258,433,-276,-414,-47,-82,91,-215,-330,-263,218,-350,-257,59,259,-9,59,59,-233,205,-298,-334,437,360,-48,215,-235,-73,-128,-458,26,350,237,169,-44,-15,174,-202,278,-56,-232,3,45,39,232,-140,-16,25,7,-145,39,182,295,211,-223,-336,33,236,8,11,-278,296,162,63,-11,259,-9,-67,65,374,124,-310,288,-31,-236,-84,-272,-152,120,-265,-159,-323,-71,-21,-213,395,215,136,281,-10,-15,-2,-74,456,199,330,62,73,461,-219,-325,117,-186,-201,346,-218,421,257,122,127,-8,-6,444,104,-253,-286,367,70,128,-338,-32,-235,44,-57,222,-93,31,-285,221,240,154,-129,-334,231,191,-199,-54,-19,58,-68,-49,33,76,107,-63,-36,-138,-159,-307,-211,-294,-298,88,-166,-314,21,-515,23,-388,5,-102,262,454,152,353,146,-300,-372,151,-45,124,404,116,-10,328,-36,-10,-170,-31,-80,-221,-123,-239,-62,115,-332,163,-46,-45,-424,146,10,-429,137,364,-19,142,335,-35,122,13,174,206,-47,-326,136,221,413,-213,-85,173,420,-198,-229,39,-149,105,122,342,297,71,-372,-108,-370,108,147,-298,129,-66,-318,-225,-91,333,322,273,-85,208,161,-41,179,64,50,126,8,188,-149,11,426,64,146,-292,311,88,-202,-299,344,194,-377,415,126,-385,93,74,-10,-337,417,-107,-120,-136,-336,364,91,146,-289,32,432,-244,32,22,487,218,-280,-115,159,-395,-23,169,5,-379,-398,-241,-324,-140,-445,-66,19,252,283,-78,-294,-16,20,1,-52,-397,134,70,36,370,-113,214,245,-252,82,-48,137,-100,-100,-386,98,230,-99,-210,-119,-52,-150,-242,110,-206,146,418,-50,-24,204,-146,-246,204,-361,84,-132,250,171,-308,287,115,-342,-246,199,61,130,145,-478,120,-230,221,-196,345,-212,-277,506,39,-86,177,-25,270,-207,-2,-271,201,167,-203,66,114,268,100,155,-84,-65,8,181,-37,-98,115,27,-195,8,-221,23,44,81,272,65,-281,-229,-115,328,103,-384,58,200,-218,-25,388,275,-500,-40,150,73,-91,-236,82,29,102,281,-12,-225,389,212,-372,270,204,296,-157,-5,-54,204,438,-302,-159,-78,76,148,329,-362,83,146,-190,-124,-92,261,206,-182,382,-277,-167,370,-227,148,192,-191,255,17,179,310,146,9,-164,26,361,-269,94,186,96,319,380,-460,-311,-117,8,282,39,-433,131,62,-385,205,-84,-130,-75,124,363,158,22,432,30,41,-384,171,181,-147,-134,-73,-17,-283,311,262,-336,143,210,185,-83,-141,199,-89,135,48,345,84,128,-44,-12,-195,-420,302,-71,254,2,56,190,454,-157,-208,134,-339,-250,69,386,121,230,-64,-173,208,-110,-243,143,310,-1,-34,-44,112,-60,-199,-38,-384,165,130,360,-100,83,466,278,54,73,55,7,54,94,-21,-189,114,-145,-263,-329,-65,359,191,-102,-258,163,-332,-331,-134,77,-115,-103,1,11,33,493,-185,-125,216,-41,115,-73,50,-71,-23,-55,-87,27,-56,-330,124,240,168,-142,261,383,120,-244,193,-227,149,277,386,150,-398,-259,-146,-18,-119,414,262,-55,-66,-370,-491,14,123,-41,-145,76,-109,331,-191,-280,179,-351,75,-148,235,-262,329,92,-63,-251,216,-26,-316,-345,-89,63,-93,45,-26,165,-232,190,72,-106,178,-40,-7,70,37,-2,138,-4,-381,53,31,-55,-24,12,3,80,383,353,-308,-85,-15,116,433,105,40,-37,246,-340,-130,100,-266,130,-2,-301,-321,-357,287,-43,43,168,161,219,-73,-55,60,206,-3,305,-280,117,-56,-118,-244,-115,-316,287,-381,3,-46,-313,-392,-7,359,480,-69,31,-140,210,-213,76,247,265,-224,-162,-156,-146,-71,413,170,28,69,224,208,251,-94,-393,93,-76,23,-36,-291,101,-159,189,-95,-111,23,-11,142,47,120,-338,20,-246,114,-298,-298,12,226,88,-121,396,-215,-218,438,-126,249,-173,-349,239,131,101,-297,329,-309,28,-316,-404,-446,199,-278,156,184,156,123,-278,-171,334,-204,-64,-119,-191,-89,-37,326,6,202,51,-394,-216,-425,384,-331,-375,-338,-396,-28,-77,16,-184,170,-61,-182,-52,277,69,-130,211,21,241,13,181,8,-168,89,283,304,-61,-354,-403,191,-494,-244,-126,-19,-112,-24,199,-15,-326,-338,191,254,66,-307,-34,-150,77,465,86,209,152,-157,-249,449,203,-151,281,140,15,-105,-55,-46,57,185,-482,-47,-210,-290,-120,113,-324,-104,-180,373,-71,-207,270,124,-8,221,-40,66,171,150,-68,20,-238,-45,37,134,191,71,-245,100,21,168,312,210,224,-140,-90,101,-317,-150,-251,143,259,299,-198,125,-209,-307,-14,320,-256,45,-172,189,63,170,261,498,-52,-319,-223,101,28,-381,267,5,-263,348,-300,17,357,245,459,-96,-180,314,-370,159,59,-149,-436,39,98,54,154,69,97,-356,-237,-53,156,-88,-229,186,480,416,379,398,157,-20,281,-254,-221,104,-104,3,-227,201,40,-267,281,-93,110,-252,17,-5,-307,-46,188,-59,-165,-318,-289,188,124,193,282,-441,-210,214,247,-234,28,-136,-111,-104,365,-98,21,218,89,113,276,-191,277,99,262,224,-107,-5,314,-47,-112,182,-271,-38,296,-246,-244,-305,-6,-340,32,-26,-252,48,-4,200,259,4,204,145,-5,-78,16,-355,100,118,20,249,-329,62,157,99,-463,-203,350,107,-153,172,-195,96,-173,31,-340,-410,92,-56,74,-260,-39,-27,-147,-221,-197,-194,-256,39,22,-109,-175,-202,130,-32,468,-367,96,53,-19,440,-196,193,235,-211,-94,13,27,183,-9,165,190,-127,-168,-88,-76,-148,-65,-93,2,-185,80,295,125,-173,236,-269,-66,-444,299,-26,227,206,59,107,-179,336,31,-119,-164,164,222,-82,-37,433,180,213,-121,-32,280,99,-115,251,300,-138,267,56,-382,185,226,-474,-98,-28,-380,285,-394,85,-62,16,24,-5,161,-244,-149,-315,-231,-79,-149,-167,284,121,55,-425,-159,150,137,240,-466,-87,-230,-178,-300,345,-396,105,-176,-130,-146,17,-41,185,-4,181,37,-61,-81,-150,-194,-99,437,-189,8,-7,69,-251,331,149,126,66,149,-84,-13,222,113,33,441,-320,-81,-238,102,-68,-225,-289,-48,298,-91,283,23,-3,-302,-120,64,316,-240,-19,21,-15,29,156,-181,397,-176,-344,-109,460,230,-316,141,59,83,58,-85,-189,-63,53,23,-125,470,385,-16,-22,346,-202,-354,141,375,301,-211,462,-85,40,141,267,-358,-145,199,174,302,115,-154,152,188,-78,382,211,-45,318,-144,253,-331,316,-1,171,-331,110,-177,-86,42,179,-183,199,59,-11,134,-427,-38,-130,120,-135,409,-57,-66,-57,-142,268,344,226,156,346,207,153,-142,-157,75,138,139,-170,-175,7,209,-367,429,248,-222,-224,-353,-45,80,-37,-351,75,14,-152,254,-19,-312,-256,-61,-54,234,128,418,149,-151,269,-67,31,-152,-123,-347,444,-65,-453,-358,-124,123,42,-269,164,3,-115,-71,109,-101,111,448,-117,-181,90,-34,399,419,-485,-174,218,92,49,-325,290,194,293,-144,69,122,254,-281,-67,-226,148,76,331,-136,26,-474,-305,-62,-5,158,50,-364,253,-145,87,208,193,-400,-218,311,-63,294,-280,121,64,400,24,-170,-48,38,270,-74,-76,148,-238,236,209,-6,238,165,-118,56,273,-11,-57,-374,-243,132,228,-193,247,300,-343,294,237,-61,77,335,-10,-130,-397,90,-36,305,-137,456,15,84,490,187,312,48,-306,-55,464,-161,-142,-218,68,15,-154,15,-21,247,111,202,235,-98,142,-106,493,-46,194,40,204,-64,216,-51,-37,382,267,86,-45,269,226,86,163,-182,219,-2,-234,-69,274,-116,-291,-291,46,-184,279,9,76,-117,213,-211,-77,177,-66,141,154,-43,-100,-450,277,-228,102,353,135,-103,-39,148,124,-11,188,-139,-206,-302,-6,-75,35,-108,419,151,-302,235,-146,-301,56,-155,164,75,-19,372,91,306,106,-237,36,-366,112,-138,-146,319,145,258,2,241,-137,-200,444,-397,87,316,98,-239,270,-305,-365,77,283,-319,25,356,-285,-199,299,88,149,-24,24,-95,305,-177,12,-287,-156,115,-383,-157,-117,-69,158,-342,-481,-108,411,-332,-19,295,149,108,-122,14,78,118,-213,104,425,104,82,-77,267,153,-74,-85,312,240,-66,-102,-211,-79,366,-371,-170,-42,-139,241,32,70,180,-136,-88,37,60,169,-318,-208,85,4,185,485,275,-40,215,211,299,-86,-105,103,-13,-106,115,-484,153,-30,-152,-19,-125,47,292,283,-123,41,-37,-362,-147,-81,47,150,252,-55,-323,-165,-112,-40,4,190,229,-447,-90,-262,-366,404,-15,-283,158,-366,-51,222,-148,58,-278,-96,-186,-219,138,-230,229,-205,-279,-17,150,-164,-352,77,-156,-101,-283,304,294,-34,-365,178,-431,65,139,-98,-71,-72,10,-94,-144,276,374,-85,56,-498,17,12,-83,-14,-9,-26,157,-276,182,-349,-145,293,54,113,-48,259,199,290,-46,177,12,-235,-106,76,302,-182,159,-272,144,120,43,26,-108,34,73,449,-12,34,57,328,167,85,31,-167,-321,42,332,298,106,-287,324,123,289,-108,-48,-228,-175,-85,-259,434,80,317,268,-111,-454,-168,166,-88,-79,298,-272,-17,231,97,370,-357,83,42,-392,-81,232,36,-56,-280,30,-4,-70,328,9,69,-110,-74,-386,0,-98,62,154,-76,252,52,36,34,362,-319,-149,-167,-274,170,149,33,143,50,-16,-62,-48,470,-117,36,17,212,-90,-179,8,11,128,-236,150,259,-120,362,-137,95,-351,6,376,45,-247,79,98,-228,-138,264,82,-119,-100,-12,389,-153,-194,-2,199,-277,179,-58,-143,166,-476,-204,-171,109,-220,267,122,95,61,-38,44,86,101,-138,-318,394,148,240,-373,138,3,338,-29,-15,-137,-254,5,252,-447,-345,236,266,170,14,41,-263,171,94,55,-255,-389,-237,119,423,-24,-225,348,89,31,-325,-149,-338,-228,125,-397,440,-199,0,-53,221,235,-234,243,-471,57,-347,-102,206,110,-56,-486,151,-83,-128,-71,116,-133,-179,103,-116,-32,-166,-117,-39,-59,-202,-32,-132,-31,315,21,200,-284,300,224,179,-212,225,335,217,195,192,-152,228,-200,-60,33,181,283,-326,75,-456,-150,-95,141,-80,-230,301,-403,-152,-56,25,194,11,38,-217,-404,-107,-170,210,121,196,-349,107,342,-15,232,-242,19,-246,60,253,-75,300,-13,-121,-303,181,301,254,-341,342,111,369,86,-211,24,224,219,23,103,301,-323,348,-510,-236,7,254,172,14,58,-141,60,-356,-358,-78,-211,82,-193,218,-45,-105,-328,-16,-368,-63,60,432,302,83,-159,302,464,-27,-166,-5,369,-77,-487,-165,153,-341,188,-26,-236,193,311,-256,36,-34,14,-129,349,-249,-321,-76,-29,137,-297,-20,39,57,65,106,-55,-14,-206,185,-91,12,-273,-97,78,-272,-171,124,241,34,246,212,-53,-369,186,71,66,115,7,221,87,67,85,-54,-377,-59,306,-118,-13,20,203,-3,7,77,-23,-137,-77,-245,46,189,410,95,-166,-270,-35,-94,145,402,321,199,111,444,282,-83,-254,-45,372,-194,289,66,80,387,64,22,-157,-18,-385,286,190,-114,318,75,-24,-486,-221,47,383,-110,-337,69,94,161,45,-195,-108,48,-286,-241,104,-113,13,-293,116,-37,-329,9,-78,-195,-144,71,-274,198,-357,-274,-211,-71,214,250,-350,147,148,19,-198,49,-323,-80,-24,-41,184,337,-248,94,147,-23,-135,-300,266,57,384,-69,-482,-56,-164,332,13,45,242,490,-97,-14,377,208,-24,290,502,-108,185,107,-111,240,180,342,-273,161,-1,60,19,155,47,-31,-45,-253,157,-192,-126,55,46,138,305,-87,194,-351,-384,73,262,-332,264,255,-300,96,264,-133,-41,56,-339,-391,-316,145,251,-198,3,-50,-71,-450,459,-146,-471,-326,-343,-166,30,-194,41,437,-372,38,-277,-357,274,-241,118,-231,80,110,104,13,228,-53,189,-291,29,133,-240,271,-214,-36,-158,65,-41,374,226,135,-339,-173,11,214,217,-126,-371,-90,120,-229,18,297,97,-257,-211,-68,-32,-112,-146,-1,136,222,119,301,-107,-102,-244,1,-106,-1,-379,-407,494,44,213,-59,105,103,132,-38,-68,-77,-118,183,-163,133,223,129,-97,68,260,91,262,-65,142,208,-250,-409,38,114,238,434,68,-44,289,-113,-268,345,-71,-187,191,-56,67,-38,-178,381,9,507,-77,338,-93,266,-220,196,-38,370,38,195,-358,158,-44,-125,120,-347,204,391,-256,-6,90,370,326,349,-301,488,339,-14,-80,141,24,69,-146,-378,53,121,-152,361,213,14,0,153,-87,0,213,-125,-133,-442,110,90,-28,185,266,92,-281,-118,308,288,-146,278,3,60,-99,-15,-266,61,45,-209,-51,253,-124,120,-135,243,-280,-57,257,203,170,129,41,381,-232,30,225,94,-78,-81,-8,-194,-381,-10,1,-51,-343,-451,-120,193,-258,-338,-29,295,-154,120,-358,-243,224,262,126,-6,213,-213,80,8,254,334,-50,430,-101,-55,240,116,-98,93,-110,267,-14,-19,370,300,-244,63,-146,215,-192,-76,22,-25,-158,-170,-113,259,63,303,34,-405,-164,16,-38,245,283,-107,0,120,-265,-301,-250,-441,-390,-113,250,91,344,159,358,-84,233,-197,-189,-82,475,231,-176,274,17,230,298,-128,91,251,-226,35,-192,117,190,-17,140,124,-63,53,-137,-215,364,35,347,46,349,150,41,362,324,143,261,-203,-124,-42,96,188,-281,224,-60,220,19,-276,167,15,241,103,-93,290,287,324,37,382,29,69,-67,-391,-46,223,219,93,-341,-193,135,-35,144,-147,365,29,377,-60,-51,111,-18,334,-273,380,-211,-139,-309,-430,-83,-30,110,61,314,-1,309,47,407,175,18,504,-188,-196,-118,-167,-126,234,-200,127,115,169,282,68,300,-397,43,-70,-411,327,-25,308,-51,-21,238,-63,-301,215,-115,-345,-340,-288,-355,-238,-190,25,-10,98,240,-12,-204,222,-14,216,193,-143,153,-365,40,13,-431,259,-261,264,298,-249,297,375,24,389,24,218,-325,-184,-141,164,143,-141,31,-26,370,-33,250,127,328,45,239,63,-70,-220,-349,34,-200,-87,50,-63,-458,87,161,48,149,-87,-376,-301,-10,185,284,260,-262,139,298,-195,-175,201,185,-22,-145,58,251,-62,-213,105,83,261,-205,261,178,-233,132,-93,116,-278,-77,-123,127,-272,-275,116,124,119,-389,-314,-176,133,-150,156,-30,-76,0,-8,261,38,200,76,29,7,358,-108,-37,-40,-269,351,-237,148,259,281,-77,-215,172,-243,383,-107,68,-10,354,231,227,-124,-157,-361,60,-144,-132,-184,-200,-64,-235,145,-98,39,-69,-41,191,137,165,13,-163,-273,-33,109,-89,201,99,-301,47,-192,152,483,-444,208,23,-230,-83,-51,171,-69,-335,-92,100,-197,110,429,-291,21,-10,4,-217,47,128,15,52,39,-138,-187,354,88,-141,-177,-86,-117,-152,-333,-198,80,-204,440,-233,21,31,-436,-271,380,239,-354,-230,197,-278,188,34,-173,21,51,-64,-416,241,-130,-167,206,-484,28,103,167,29,-271,424,179,311,-317,-50,-147,42,-292,267,-288,195,281,-16,93,136,65,-35,-310,-177,-428,141,-186,181,-167,9,-466,-123,-487,-4,222,40,122,213,403,332,402,407,65,-117,-56,294,22,219,281,282,-191,-9,-73,-355,269,92,165,-71,86,-397,178,120,139,-10,-28,-38,-274,1,-58,-162,-411,12,48,257,-343,285,-327,206,-110,-452,-21,-150,-58,-191,-171,-36,383,-13,-319,200,-11,191,328,247,-122,-304,-158,241,377,-54,44,164,-24,-172,101,286,-150,-53,-79,73,-7,90,364,-346,-6,194,337,-52,230,-346,-44,-97,-62,-113,229,201,372,-167,-67,113,-250,154,495,53,-388,244,-256,150,326,11,68,185,-187,449,-126,-175,-195,52,98,331,157,-348,8,-296,-59,97,68,-13,-95,-307,41,119,-184,-9,-264,-191,-58,-175,212,40,283,20,-270,60,248,35,-274,-36,-276,246,-446,158,-93,-144,-142,-296,0,14,99,-76,75,60,-197,-262,184,-82,448,239,167,134,-131,116,-60,-308,-252,180,236,-393,-369,-288,364,-141,64,144,-5,371,-253,68,-429,131,479,-33,-6,327,168,-370,159,140,4,-328,30,-145,-73,-259,28,304,-213,28,-61,-83,158,-318,-93,-193,-344,-176,-104,-93,-97,123,-432,190,39,178,20,-378,479,-2,-508,55,21,38,-83,70,-73,6,29,475,-128,-258,60,-355,179,-125,-196,388,132,-22,187,-120,350,479,230,258,468,-3,8,-195,328,-62,183,-173,-112,-112,-262,351,-23,-273,-132,255,-61,-165,39,-188,402,415,-69,-468,-67,322,174,267,16,-173,95,-125,-8,-23,-476,362,-138,26,-238,390,-140,134,345,-336,-87,59,-49,-242,-167,-355,-330,311,323,-127,-288,132,53,-118,216,211,-67,-90,-59,26,-32,158,444,88,-113,82,-184,-315,-103,-338,230,59,50,-138,-148,-131,116,-66,126,-356,-209,179,132,124,-57,-342,20,341,-34,-14,-107,-1,-68,-135,-29,-253,-158,138,-183,326,-340,47,289,-96,-178,-161,155,350,-213,352,-417,111,-264,-8,-283,281,310,207,440,47,-238,6,-394,361,349,-172,-258,-286,196,-269,98,20,338,-202,192,-303,-321,218,389,14,230,-19,-219,155,-66,101,-322,-244,200,348,168,-67,-243,24,-33,454,71,41,150,-63,-24,22,271,-61,136,34,270,115,147,-204,-228,203,22,-171,-235,-188,-273,-195,105,-74,-424,-290,109,11,451,-299,-349,-96,-24,287,373,-310,-474,-323,-34,68,-100,251,-41,-383,40,-177,10,-196,152,-222,508,-73,481,16,192,-229,250,247,350,-30,128,-87,7,-336,191,10,-182,-130,116,-7,-88,-283,50,86,-14,-242,131,-395,59,-132,-333,-142,97,-18,289,-77,-72,4,230,119,-121,-71,-392,70,345,-376,66,-186,0,-60,225,200,266,285,-142,8,-216,-51,91,-71,-149,-57,56,46,-199,10,224,31,-396,29,0,40,-5,159,204,-216,122,65,165,-8,132,107,178,-465,-31,-196,-473,363,-276,-201,-98,-339,-255,253,-144,-279,285,36,280,-117,-50,293,-134,149,-247,346,-159,79,53,-285,-205,57,292,378,-162,117,-89,25,63,-16,253,-114,42,87,-414,-12,273,296,59,286,-21,-322,226,328,-214,-178,47,-261,91,-175,150,98,-257,-182,-59,-349,191,340,-88,347,369,-138,-143,-466,104,484,77,-63,366,297,228,102,-73,55,-61,2,-99,217,188,223,63,-262,-30,215,90,244,159,111,-123,-147,141,102,-361,128,51,-196,273,-101,-262,-206,370,381,-176,-82,161,-198,200,-413,-157,-282,-281,-237,77,-50,59,295,342,125,296,-311,88,-99,-85,438,279,64,-44,234,37,-227,207,-483,171,50,-303,289,152,89,-103,-213,-46,-214,156,243,-463,12,390,344,187,-280,-131,-188,174,-146,-175,-353,38,184,-64,-246,42,278,191,239,56,18,328,250,-272,-224,136,-193,-443,-184,225,28,148,-392,304,137,-412,-241,8,73,-175,-301,158,203,-154,46,-60,8,-19,-3,-161,464,-126,411,-50,-195,184,-140,-134,-54,-393,38,269,159,-87,415,243,-23,-521,154,375,-430,-389,240,-226,11,67,217,-261,132,-217,205,-274,-113,-117,-139,203,29,-39,-52,-217,-52,93,314,146,-209,15,323,-80,438,106,-409,-20,-27,-97,361,-136,65,-74,-280,17,13,236,342,288,-352,423,348,-17,-58,-318,-80,31,-23,-28,255,-211,-80,-40,-170,-90,-334,33,-249,-78,-107,141,-265,-34,-203,13,-70,331,-52,-77,276,421,-150,133,35,257,324,-189,-73,-188,-25,-153,481,44,336,-349,74,200,241,255,216,327,94,49,79,-243,14,412,45,0,-172,43,133,95,121,162,47,-151,341,183,-485,-134,279,-128,43,-194,214,313,-248,-78,80,128,101,-72,87,-36,-52,-73,-91,348,108,-270,-312,-10,-233,142,73,70,-40,-273,335,-47,-124,240,-168,344,-36,338,-165,-393,279,-281,118,-257,101,91,150,225,289,-223,-120,-38,-156,-132,-29,-209,-86,-268,97,-6,-342,-148,-60,-232,-125,124,186,39,114,178,-4,-91,50,-124,-76,160,65,71,165,-42,-176,13,212,227,-159,-221,-334,318,0,-29,-176,-47,148,-94,-194,-213,54,125,-74,14,-1,-122,377,-227,-181,174,-311,-104,240,-118,34,-301,35,-412,-153,-217,-370,-319,51,-158,-344,-179,12,87,-159,-214,249,-286,-45,-108,0,-125,185,-318,386,169,-124,129,212,-24,-405,-35,390,-48,83,-262,224,97,307,-79,86,-133,33,-144,-11,-382,-58,-408,181,-217,274,157,49,53,-428,295,-82,74,-21,174,58,-35,-10,334,-227,13,306,-417,75,71,265,8,-367,-193,-186,6,-313,64,73,-294,-15,218,425,245,198,-338,24,202,-229,-186,155,-155,-180,35,167,91,270,271,24,-125,-87,89,147,196,83,-85,-92,-60,251,69,1,-15,357,-111,-298,-88,-429,-63,159,-324,-208,374,479,-12,-21,147,240,223,216,81,147,-176,-86,-269,335,167,245,429,-256,-85,-120,-340,-144,-18,107,-221,106,-103,-75,416,229,162,162,183,422,-310,208,218,22,7,297,105,61,-159,104,-115,-303,169,131,366,-184,177,-248,231,-87,-210,-10,-147,401,-383,449,114,221,-189,-81,-300,336,-284,88,-318,142,333,104,56,-3,-49,31,-145,23,-128,-154,21,34,216,-255,-335,-74,38,55,6,-359,131,-32,-45,-330,8,3,128,-360,-331,107,-32,-296,259,-27,-131,-308,203,-85,98,-56,-75,-332,337,253,61,195,337,202,-95,-158,1,-198,234,16,-366,155,189,-131,-207,-281,59,41,115,-203,100,126,256,-353,126,243,114,-454,-18,57,137,300,-135,89,96,27,-98,316,368,-157,10,37,-74,-58,254,163,183,505,-235,-104,-113,-23,-315,243,-51,88,-302,219,-137,131,116,123,-89,-329,-390,4,328,152,130,-343,-224,-196,-115,13,-153,-86,-279,301,12,20,-318,23,-109,117,-229,122,146,-49,110,-282,-42,133,252,358,8,45,-458,123,-8,-155,-140,-440,-246,105,-482,-206,121,58,381,313,206,-98,0,66,247,-176,188,-368,-148,159,46,145,10,27,-102,-221,416,374,129,-86,284,-486,-56,-54,326,-20,-200,268,211,-198,-183,-368,-307,-204,-8,-138,317,250,494,-283,-76,-108,108,-180,-194,-208,-193,288,-159,194,108,-289,131,374,-158,-57,-96,139,91,-184,315,-326,111,-46,355,208,-86,-250,140,-261,15,-9,273,-60,211,-175,107,-48,400,245,274,-156,147,-52,-75,188,-5,-425,88,-144,147,-33,398,-90,-25,224,22,51,-321,-60,371,-271,83,93,-73,6,340,-11,129,-423,-329,-252,16,140,-241,19,83,204,14,-32,93,32,-176,-150,-319,-116,43,371,12,-372,165,-172,-347,129,19,-227,-196,47,-98,362,-270,-63,-232,-70,-167,-385,32,-426,85,100,-205,-22,-60,-10,-54,98,409,-182,50,160,-53,-91,-279,71,-25,-295,159,-97,-385,200,-132,328,-343,253,-13,1,251,376,-31,343,35,92,-271,48,-116,37,5,3,-16,94,0,-339,132,-79,391,-36,290,-271,264,96,-119,68,133,-98,-211,16,316,-192,422,43,220,-446,250,-116,-361,188,71,-165,140,14,-31,150,140,-39,229,315,-34,-99,-45,354,-8,-102,88,-12,52,45,-84,-215,369,-68,-231,260,124,-54,-165,-217,58,397,97,-54,-36,427,43,105,39,-450,-44,154,129,164,105,400,85,299,-117,-148,-144,-66,-289,160,304,100,2,-319,-342,166,-385,-58,130,-222,306,-177,174,189,199,119,-64,227,89,15,446,77,-296,-61,11,146,-380,341,-36,-119,-456,203,210,34,384,-163,30,-122,-502,-186,455,166,169,307,-234,98,226,70,191,371,-264,79,-64,3,-83,173,57,-129,72,262,297,40,3,-417,312,-9,-50,17,-100,-170,-155,265,58,114,-24,41,-63,-2,202,-55,40,164,-421,-31,98,252,-334,-198,78,447,107,-330,-310,90,401,368,57,-23,4,173,-38,-253,101,196,-93,151,-64,54,-167,178,-52,283,281,383,-303,-85,296,193,-149,-203,346,413,163,348,-129,450,184,216,-2,-414,-201,-12,-412,-185,112,-214,459,-241,-174,45,14,-56,22,-100,14,37,325,17,-31,-329,-221,400,161,265,-48,22,-356,-211,-213,-413,318,-4,-397,-30,120,12,-229,-473,122,510,241,407,96,-378,317,-53,-123,4,417,310,117,82,-261,-148,-244,-386,-47,252,-121,-217,-393,6,232,10,-33,147,407,-220,-77,146,15,81,39,-441,-348,-396,227,-32,-235,2,364,43,-96,7,132,-62,21,-108,356,-120,362,-438,73,349,104,3,56,76,169,-60,-128,-1,-253,282,-152,89,80,269,24,124,-210,-159,223,200,5,29,-16,-97,14,-259,-80,313,-249,13,310,-130,-224,-451,131,-140,286,411,-75,-38,147,490,179,32,-357,-76,240,149,58,-4,437,-210,30,-85,260,-405,6,-96,-3,37,316,-109,-57,234,-89,88,318,285,7,353,233,32,-44,138,204,-215,258,295,-3,129,-28,-458,-226,155,218,133,57,53,-249,-29,74,-239,-157,-252,107,73,242,95,172,324,-131,105,142,4,-146,244,75,170,-292,223,67,72,71,327,-77,-2,22,-208,-112,-32,178,47,228,73,64,-142,-287,-379,-5,155,-63,-220,211,328,141,116,-95,142,-107,-177,-325,109,-279,-250,-251,64,-238,-104,-28,394,212,168,15,-216,-320,-422,124,60,-193,-89,-108,277,105,-52,-113,-25,240,-250,-426,64,-36,333,88,-214,-166,-349,-330,-115,-198,139,383,184,-191,104,-325,349,232,-309,-124,100,-43,-131,-95,-78,-52,-261,23,-146,217,-7,-34,-92,-453,116,155,180,-271,-212,-27,-67,352,-150,-173,-311,45,79,-370,9,107,195,408,403,-165,231,128,357,-147,-325,340,-236,-418,-199,290,-317,83,-17,63,-176,-251,146,45,117,322,77,-149,-196,-90,-123,-80,201,149,376,44,-383,233,237,30,270,52,-388,194,-241,71,47,-6,-30,-118,-242,152,283,-3,32,64,244,279,-329,-95,-381,88,81,-354,-16,204,-190,109,-82,65,-40,176,394,62,472,-196,-40,-128,291,-343,-360,-235,65,6,-61,-369,281,-275,361,-133,128,483,137,-223,-226,-444,-25,238,-22,-119,200,175,-156,-108,25,-72,284,98,7,130,418,-15,183,152,24,-268,-349,251,-298,-90,-375,3,-114,2,-117,-158,-16,-316,247,-147,39,205,167,47,-178,-132,-174,389,85,188,140,351,191,-7,-374,-449,-256,-90,-323,-251,-54,-8,-409,183,356,107,18,-270,-50,-150,-459,262,-142,-168,172,-458,-333,-309,66,-35,304,-239,-222,-97,56,11,253,-314,-238,368,-22,-359,-104,-388,188,-282,-102,-195,35,320,469,-384,21,76,-84,-209,-10,445,47,384,504,-409,-74,-277,-81,-100,238,117,-6,-249,34,-90,-142,449,-249,-79,-181,74,0,35,-134,206,170,-257,-106,257,63,-405,410,-179,-109,155,-211,199,254,-11,-6,160,8,-362,-132,31,368,-98,-33,15,-185,158,-228,63,204,187,60,102,-104,-253,109,200,8,85,-215,-230,1,202,-231,417,-140,-201,-45,-64,-199,-353,89,-29,-251,-58,-179,34,61,148,-94,377,64,229,14,-433,-324,38,-149,96,157,-60,-103,59,315,-81,265,107,-94,236,93,391,415,-146,-5,207,-15,-174,-22,30,-229,116,381,-269,-237,-69,-201,315,-21,90,179,2,276,-108,2,-47,17,-37,-210,-309,55,103,-20,-270,-152,119,269,-230,-318,-212,171,15,77,-222,-2,30,315,46,-108,176,-438,265,91,-225,-343,421,-6,-21,-54,261,-150,-206,-55,-140,-198,24,23,246,-142,181,-417,-61,-66,17,173,-49,-141,-200,183,-25,138,21,-11,-107,-192,143,-39,-49,60,32,207,142,-176,-124,-72,-225,168,21,-376,406,-323,-9,-368,240,44,-159,-86,-63,-119,-300,2,63,-171,-386,225,169,59,356,-327,-308,167,259,246,265,-418,-14,-119,131,122,-64,327,232,-139,70,-67,50,-34,-358,21,-262,-194,-470,158,-24,209,444,27,240,-327,-72,-186,-30,320,-193,360,-266,-252,197,24,-321,43,-259,-237,31,-98,-143,514,238,-232,262,68,108,179,-49,119,408,47,101,-177,397,49,288,-267,-457,61,124,-118,54,-66,319,182,167,-122,264,168,45,-66,-251,-97,246,8,-81,-140,192,-12,21,-293,-243,-56,-196,426,-436,-394,47,-115,-420,-174,-314,214,1,-45,-94,-62,-179,313,-56,125,-467,-402,200,320,-118,379,-392,69,51,13,102,-96,166,52,107,-235,-34,-137,-228,20,-121,100,142,-86,-204,-117,-64,-511,-31,-82,324,-50,205,301,-333,-447,-264,-369,-39,-225,76,-5,15,-290,9,-109,188,-7,-116,210,128,26,-133,319,-111,175,-77,-234,126,184,-75,15,-35,356,43,24,-133,214,-285,24,-394,-275,-346,254,-93,-45,289,295,290,75,93,258,46,-428,-259,-1,-321,-187,246,107,111,359,-53,-381,344,36,-175,-27,-304,178,139,-38,-324,228,-158,-17,11,-56,82,-3,-332,324,-255,419,201,19,-158,255,57,350,-137,412,-97,-399,383,-48,-108,45,-213,307,357,97,133,-185,-314,-10,-314,73,-47,-198,439,-72,225,185,219,130,-211,-7,-48,-434,162,109,-313,298,33,488,58,-162,-60,-121,-264,18,-143,89,-440,218,-99,-149,39,66,260,-154,-210,79,69,-93,-147,-174,-130,208,43,-138,-260,113,-96,-281,-134,321,-138,27,-478,219,-131,-235,-224,-132,-306,-75,-399,159,-151,-327,108,-163,51,72,-346,-128,269,-226,51,262,-170,-164,49,-206,-153,-107,-144,-210,39,24,-101,-354,-210,129,-18,20,-21,351,45,179,179,-2,-207,-283,397,-288,130,304,76,51,137,-275,-174,-54,-95,96,-124,-128,246,323,97,-434,109,83,-90,144,307,-42,-24,-180,-232,266,49,-421,-87,-182,362,-148,-7,226,224,83,-77,401,50,-359,410,238,395,235,-381,-333,81,-68,-173,-110,-88,397,-114,-18,-10,-10,-124,-288,-400,-344,501,169,144,148,-289,-4,26,110,361,260,-197,-433,113,-215,-120,80,152,-389,-319,-143,-309,-308,370,-36,188,-178,-357,97,160,268,364,-20,-296,-486,-85,122,92,-172,-312,81,-180,238,-351,331,-402,-417,-106,47,162,-135,-369,-284,-97,-38,227,291,127,94,-165,-453,23,123,-169,-91,13,-245,115,222,-468,230,-432,-219,83,-336,208,-160,199,109,-105,-422,215,238,79,234,-247,150,-9,-361,-60,-22,224,81,145,-283,-228,171,260,-484,37,129,-479,-37,78,-23,-432,48,314,49,-335,17,415,-116,-180,-17,-262,25,126,-415,-357,273,-26,-294,325,150,-172,-25,0,-262,-174,89,-92,47,23,303,-189,-177,286,-86,292,353,111,72,-360,-302,355,385,-114,172,-15,218,158,-470,-104,44,-188,-55,431,460,378,7,-51,93,58,172,16,-26,-478,222,414,130,-13,367,-181,-139,-250,23,139,-164,-110,17,-376,-39,-353,-97,51,54,284,64,-8,150,-252,0,170,350,54,49,135,169,48,-329,461,311,180,154,-271,-107,-20,145,-212,-375,-52,25,-146,-396,185,-202,130,-100,1,276,69,-114,-230,161,115,-154,-197,-276,114,-67,11,-355,-10,-258,38,0,-62,-86,387,285,-368,-6,-154,-116,-57,421,242,156,-168,-206,-231,168,-13,-4,75,450,387,5,-78,-42,-12,-259,-355,431,311,77,20,208,-84,-114,-75,-174,-248,-362,121,-206,460,142,239,-51,109,-146,-73,216,47,78,-452,-351,-127,-95,-220,8,52,25,189,29,-59,297,-269,-35,-173,-182,-46,-11,-303,-11,-294,-327,-51,266,235,-337,-366,299,-70,133,414,344,334,36,502,158,-206,-136,59,300,-61,-378,306,-99,222,-208,25,-64,212,-56,-148,-209,342,-345,18,210,120,-174,15,163,202,218,-121,160,-314,341,-62,454,-50,81,194,31,6,150,-214,74,32,191,12,64,256,104,178,215,-51,21,520,357,-99,-263,-47,84,1,-3,-106,-19,-248,5,74,-154,-258,120,149,-17,241,-14,225,-30,516,-220,-158,-267,147,-268,-222,-117,120,54,-346,-7,8,268,-265,-411,0,-286,167,-326,95,-442,-217,131,135,-124,162,-290,-335,5,-216,-51,-30,254,-8,-183,-118,210,-287,-28,277,362,-144,107,201,-49,0,107,366,-153,-99,342,-25,392,214,-496,233,-426,-55,258,-268,-390,58,247,-225,-240,-8,28,79,-236,-12,115,88,-315,106,190,-63,62,62,183,238,-178,-28,-61,-41,-95,326,35,-14,-163,63,115,-260,466,-70,-55,77,91,185,-225,-379,51,-352,-456,-91,-361,-65,-35,9,404,-156,-288,-286,248,-188,245,84,29,271,-412,374,38,403,-106,325,21,-36,-179,-156,-139,-449,284,97,6,341,44,-102,-25,183,27,-257,164,92,158,-211,-63,-307,-308,412,-10,334,-244,-64,2,-13,-89,70,133,168,-97,-352,23,165,386,-190,54,-139,-104,-147,-153,106,-16,-218,-75,138,22,80,-250,-423,93,-225,196,179,144,72,-48,-371,-23,-200,51,-139,-1,85,-174,40,-82,-90,242,42,136,249,306,-233,-376,77,-39,23,76,397,111,387,-115,322,20,193,-8,-311,-91,373,472,-360,3,198,-72,43,-75,-271,-36,90,-24,-190,-102,66,243,156,-299,79,-94,150,-166,-46,235,152,-48,-103,184,9,148,-19,253,-356,-88,-76,-292,-7,-59,-60,345,110,-46,-327,29,407,-168,-19,14,299,114,-421,-266,126,5,-342,289,-179,-190,-33,5,16,-252,-20,-186,151,396,143,211,-130,199,-106,239,-142,492,-74,-175,151,62,295,-24,-354,166,-326,203,287,-277,-301,4,376,196,-38,-233,9,93,47,-283,133,460,-246,-52,434,150,45,-44,162,158,128,-11,-418,267,30,-103,-161,297,-69,14,290,228,-434,85,287,-32,378,-31,349,195,174,-165,-2,263,66,-59,-260,-3,73,379,-80,25,354,-114,248,31,139,-179,231,-341,279,-210,241,47,209,-17,-196,104,161,-204,-278,394,-41,-18,261,-129,-82,-25,-204,58,-377,-253,-483,-324,64,221,9,230,161,460,11,-195,-290,238,210,358,176,-24,14,374,-367,-26,-305,69,-105,-51,206,46,-358,275,190,141,-17,365,156,-207,-26,437,394,-275,-106,244,297,44,121,90,396,12,-258,100,102,-162,34,-6,109,-39,-36,110,-492,46,43,-16,213,213,451,-282,-74,-433,380,200,215,-41,466,-185,55,-137,-19,-78,-114,277,94,82,-357,-153,-16,314,-324,178,297,-437,-72,158,279,128,378,-148,41,223,335,308,-165,-133,296,90,39,329,-430,-140,153,350,134,340,-178,-177,63,-299,-104,210,149,-62,96,13,258,-76,340,333,-156,-198,287,91,-239,374,-350,-116,488,-122,-211,156,202,501,-107,13,27,-334,68,242,-89,-74,-9,175,-65,-351,64,-45,-320,-321,-91,-270,-41,40,-132,51,0,83,131,179,-307,244,-384,-122,33,157,-52,172,-17,152,-323,152,341,153,-166,226,-30,-373,262,206,228,-186,-102,13,337,-230,260,143,-181,-38,-266,-96,-140,-192,249,457,304,220,-75,175,158,-320,233,-193,10,-172,1,-231,-450,-407,111,211,-241,-93,-444,-162,101,276,0,-24,197,-37,18,82,107,-19,-6,46,71,-67,275,333,115,-412,339,268,-63,-35,362,-49,237,300,210,180,-45,161,120,453,-192,378,276,-74,91,-408,341,136,-2,-206,-82,-155,-336,57,243,-312,-437,373,-29,87,-78,268,-202,155,19,-117,-418,276,-72,371,-434,-125,-43,-66,-43,69,-27,-213,-165,221,18,188,368,-177,13,-136,-155,190,-76,51,-380,-14,-308,321,118,-63,-169,301,67,97,-120,203,6,-26,-90,272,-43,273,413,-237,-180,-213,-14,181,-337,-108,-170,-370,-258,-5,385,-27,-444,-272,168,184,-265,-66,131,143,82,11,-55,-423,-1,225,-410,-113,94,182,-128,-215,314,252,-30,78,-53,129,125,268,-213,51,-189,51,132,159,-8,37,-332,243,-261,-264,-6,430,-340,-128,-267,215,-91,121,-69,215,-66,75,195,20,181,-77,-287,149,-164,-306,-314,20,294,-15,35,-173,-383,459,-208,-61,82,310,341,319,294,-254,-271,292,-473,-125,76,120,237,-396,-47,-196,-285,116,-58,-248,-19,-459,44,-159,75,-125,77,-316,297,250,-211,244,-167,80,213,-140,46,376,366,25,-273,77,161,170,14,65,-67,-464,-9,-198,196,47,90,192,-101,105,157,-154,103,15,-345,-97,-21,-436,129,255,-139,57,4,-226,-348,-246,-64,-234,390,-137,-17,-181,227,-46,-270,-182,296,246,-210,-267,-68,-183,234,-264,-74,17,-229,-160,-377,-138,344,51,315,112,-19,248,155,130,-295,160,-33,-171,-305,-100,-232,-207,-270,15,310,306,-26,77,322,-388,60,115,-55,-217,209,-72,-230,-291,93,-134,140,-406,133,-153,-50,273,222,-94,21,150,43,135,-318,-232,-177,193,-270,372,-197,178,270,136,182,367,-295,-328,13,101,321,129,137,-154,-169,-51,-123,275,-214,102,-179,129,-5,-338,114,410,16,90,-335,-146,346,169,-151,196,159,-182,26,-22,-333,-30,218,-111,-42,-5,63,-60,-215,-18,431,84,-445,-76,-152,-320,-17,401,197,-73,-15,131,-101,-309,178,388,-154,94,-21,-301,152,-74,150,-297,147,-91,-157,390,-164,-369,293,73,-115,-312,-84,106,-73,323,260,-409,-144,238,-238,-67,161,-120,-150,204,-24,-3,-244,248,26,-394,210,-130,-185,-103,-334,61,37,-444,310,-266,-299,189,73,-285,39,406,-476,-124,40,120,248,-43,-155,76,97,328,-82,-66,-220,-100,-277,288,271,-255,251,93,-416,-112,-336,56,391,-172,28,-212,-512,-32,-155,208,-133,-192,-160,227,-112,468,-33,-245,119,220,81,76,-51,40,330,260,-25,-15,-3,87,40,-179,-230,177,319,-125,12,53,-16,-220,81,264,419,13,2,-403,162,-36,-180,-287,12,-40,147,-30,10,-88,40,-324,169,1,69,-56,5,151,177,-172,183,242,166,-283,-355,-150,162,386,233,354,116,-116,0,347,-164,-132,-237,82,41,-152,31,33,412,81,-45,279,-273,37,-227,-112,328,199,-71,432,40,-179,-60,-247,-179,199,-116,-150,-341,118,244,172,-38,35,-60,134,121,70,-185,273,109,-257,197,25,27,-145,64,48,-161,-76,-209,-24,-180,-334,-111,-238,232,-372,-218,-331,44,-273,-275,176,62,-34,-335,98,240,160,25,-462,305,-261,-143,42,-98,-313,325,-11,-118,-268,-155,161,-189,-36,-239,-35,112,110,-363,182,-44,165,373,228,285,181,-98,-207,189,-246,271,-59,-169,204,355,1,237,202,-67,47,-50,176,182,-21,304,110,284,247,90,-58,-28,48,-44,-296,92,121,-2,413,-108,-128,-424,-174,-16,-144,-112,-252,-132,-214,-369,26,-23,-239,194,-482,111,-53,-31,53,347,-192,-265,351,-164,28,-234,-256,232,42,446,-62,-305,360,-2,259,98,-168,-120,157,-299,-82,115,-46,-459,40,-255,-101,0,-233,-4,329,-125,-87,155,130,381,-72,-6,424,292,-193,225,-22,-155,178,-254,-27,-26,230,-459,192,151,110,-125,-326,195,271,168,-26,368,-143,285,92,266,-320,-26,205,-241,-434,-3,192,155,-13,58,-13,20,427,-87,-139,211,46,137,-217,-195,231,343,7,107,63,-27,72,479,15,359,-93,-223,44,-370,-83,-50,263,-210,307,-64,-358,-136,-27,-106,60,37,-34,175,-223,-237,404,112,32,472,54,-316,-23,316,-59,-127,55,-287,10,26,-250,103,146,385,336,-225,122,-12,-119,-84,282,-403,347,-352,-259,-124,101,-153,83,-113,61,-21,316,247,118,307,373,377,299,196,-259,251,-8,-351,403,254,-110,63,-169,232,-239,-124,-371,-176,226,-91,202,70,178,223,43,44,-160,29,52,-381,-211,-256,-367,-285,-209,-106,235,415,-4,250,99,299,-22,92,55,-11,-333,-124,394,-130,-153,-54,483,-93,-25,-112,394,-187,-36,-235,-107,-85,-206,-33,-107,70,36,-134,504,-357,-126,-59,-186,148,-129,-112,-103,43,-236,110,-74,-285,-190,-90,-275,-239,-29,-167,-71,-108,320,-147,55,129,-155,282,160,375,-470,278,-141,194,105,-131,305,50,-114,414,79,27,96,446,153,129,319,306,-464,-196,38,341,-43,-348,-33,-92,124,-374,-127,-186,23,-374,-142,52,-24,-206,98,-305,374,177,129,8,-150,-215,-473,-299,302,61,108,-80,387,-74,192,221,313,-207,-105,21,-395,300,-494,203,281,-5,27,104,167,100,211,351,230,441,415,-43,14,165,-324,-45,-194,222,192,300,-19,6,-206,210,395,340,84,-79,26,-28,-346,-157,84,-221,-305,124,294,-39,-390,52,237,220,-30,-90,77,103,-423,379,257,91,-13,-413,22,-162,350,-67,-295,-333,-145,-8,-265,32,-213,170,410,-105,-89,-38,-145,-5,65,-409,204,96,36,-287,241,-28,378,38,246,-253,-29,-227,36,-162,-216,409,137,-356,200,68,63,284,-21,-316,-515,-54,498,192,-59,106,-281,-211,-88,-71,-110,157,-107,-291,-449,-36,-49,402,207,-121,-77,-4,116,41,178,-291,93,84,-144,270,-100,-105,-68,294,-302,288,61,7,132,5,-45,-37,7,-177,-323,237,467,279,-54,117,-102,-40,156,95,46,-1,56,305,-10,231,252,-257,-37,285,59,207,-123,-38,130,27,488,210,-44,421,-51,-57,5,-11,-299,31,-68,449,308,37,-192,120,-10,201,186,228,164,-34,-46,6,63,202,379,229,157,56,-228,181,-135,139,64,91,-7,41,-372,-339,223,-93,36,309,116,-75,263,173,-224,143,199,-454,105,-300,-154,-213,206,-187,-279,-217,-78,237,-198,311,-9,173,-108,-97,-434,178,-63,-273,-335,235,266,19,-55,240,-110,38,208,84,-220,-364,127,23,-430,-77,136,191,-258,-191,134,-118,76,58,-412,-227,179,139,-338,-267,-176,345,-32,-56,-34,112,-82,-101,6,144,-385,358,177,363,-274,-184,-441,-278,-395,226,7,42,28,137,-128,1,-154,-64,-58,-215,224,-382,-146,-53,152,-277,-179,104,-78,-39,195,288,46,-260,374,261,-115,55,98,-32,-58,-262,44,71,-199,-393,-262,-82,53,-168,108,244,286,-16,108,111,57,486,62,-444,9,303,-413,240,-3,-34,-41,-110,25,113,6,23,-323,125,138,116,22,70,0,-161,-341,-261,-54,355,-171,-32,1,247,133,-251,-92,291,412,164,130,-198,342,-14,72,-13,-107,57,22,290,248,71,-66,274,-485,171,407,45,-302,164,74,-80,161,-40,25,-103,-65,141,243,129,193,325,-132,210,14,111,308,-340,256,-105,-96,111,-31,208,-45,-365,-128,-187,32,384,85,-169,-266,233,-114,-29,125,-122,234,63,27,87,152,13,-80,-355,62,-88,119,106,176,-345,-368,-21,335,419,34,324,481,-255,182,-230,-264,-10,-148,217,-454,211,-220,145,9,54,-162,168,416,-370,-127,-257,159,225,-275,109,-121,-155,89,65,-91,-229,145,-113,-206,-387,164,-440,-149,149,30,108,-304,115,-311,-113,-104,30,290,-156,141,102,-53,17,38,225,26,129,-217,-109,-383,115,-135,-29,-52,-158,322,-142,-92,190,196,-276,-230,-34,-159,10,-16,-114,39,-485,-121,122,-170,420,-312,259,427,130,-340,20,-39,-253,-435,-68,-240,-146,-137,398,67,-219,75,-77,260,257,-53,-200,61,59,69,173,-21,27,40,-191,-230,-82,230,-20,57,-50,-56,231,11,0,43,-196,-45,-183,74,-48,-72,89,203,199,-347,66,-63,-322,344,123,-314,-261,403,28,-91,-220,91,-20,108,-210,210,-144,-136,86,-190,-186,28,112,150,381,130,423,233,-19,-396,-242,57,300,151,469,383,264,71,-168,-35,-67,101,-76,-469,19,-215,271,115,-122,-203,-214,108,151,4,272,127,-171,-169,-29,-57,-101,173,-468,-173,131,-332,-63,133,-191,-45,393,-185,-44,194,-126,323,-52,-191,82,-237,-17,136,-98,-301,170,-176,183,-287,27,344,201,-435,-172,34,280,-83,61,-207,216,-279,28,25,29,-98,-147,227,-59,426,130,-434,218,-28,-1,26,-167,-79,-367,-216,-116,-56,-191,150,-74,-184,-77,204,493,-287,-179,-292,-462,-313,16,79,-158,-171,-81,167,88,125,-310,-471,212,-72,142,-253,60,-235,65,-70,-346,227,176,-212,26,228,187,-305,285,-162,-101,-291,-75,69,378,137,-104,-320,-153,-82,15,-8,-249,231,231,103,-302,-202,88,-49,-305,-374,335,-226,303,-31,-99,193,-3,-233,84,56,54,226,-61,-279,-167,51,-127,325,89,203,-22,-131,25,394,-95,104,174,285,13,44,-202,-196,-207,-432,162,-385,45,-99,-286,-105,112,9,-156,317,49,137,123,7,-145,197,-434,376,211,102,-7,364,-42,-335,-103,-51,-176,307,14,-17,211,78,434,-303,-11,-225,148,-121,-4,-108,-71,206,98,-220,-101,-72,137,127,-256,-37,-112,-281,-111,-207,210,-7,240,-271,-261,87,1,-99,-186,-131,-103,-131,67,-315,-108,-75,35,-178,-69,-80,-173,210,137,232,281,0,-87,-182,70,198,-58,24,-298,-232,9,-154,-221,129,-156,-233,-99,-112,140,84,81,207,-34,-34,-277,-203,-44,-63,148,-213,138,107,-189,-28,-170,54,130,137,-77,-250,-362,-79,300,-131,-399,-66,108,-186,388,-94,-145,-87,-86,-43,15,224,-301,-74,-68,171,350,-53,-88,-50,184,73,10,-17,100,129,157,391,-232,-315,146,-25,-152,-251,-49,-214,150,-94,124,159,81,-137,22,183,154,-18,37,-347,23,-226,81,100,461,-26,-338,81,-34,-119,-365,143,-298,-238,-179,346,129,-136,-172,-168,-144,-262,68,21,352,387,-221,-258,-15,-283,196,-194,74,249,-41,-438,277,-236,-128,57,-432,170,-303,-104,29,-350,-167,326,148,32,-38,-36,83,513,378,199,-12,-12,69,-214,202,-142,-21,30,234,-228,-70,319,11,-188,-35,115,-199,-92,205,282,301,76,4,7,-113,22,97,-290,157,-179,-156,170,-232,-112,28,276,307,253,60,268,-56,-372,355,132,111,192,-57,153,191,145,94,79,248,-178,-171,137,170,-205,228,141,-362,192,-119,161,-258,208,-79,91,189,64,80,311,53,166,-125,-118,-13,-125,-163,-113,-54,-168,-206,9,279,266,19,86,-172,-295,148,-15,127,243,278,-52,102,-259,47,-187,-460,214,-22,135,-2,-87,-397,-139,-350,11,-99,332,-24,107,-62,-44,-312,-119,-223,32,64,-71,186,-158,106,286,35,-58,-118,243,292,-403,-95,44,190,-102,108,-238,-50,-123,390,286,16,-95,103,276,-136,324,-242,-8,-60,-111,-336,129,193,57,-405,232,-186,221,203,-168,-140,-300,67,-128,242,-54,-274,87,135,227,185,-46,-332,-47,29,364,253,3,71,8,6,154,9,163,-179,218,252,72,-10,-46,-348,-1,-162,-155,50,-7,392,-81,385,-237,341,8,2,424,-88,-231,370,488,221,-202,30,204,255,382,13,-203,443,53,-77,-262,-4,88,67,123,-223,-118,55,294,-46,-151,29,-113,-14,-124,234,136,361,-123,131,-31,-282,-13,453,419,54,-314,182,359,94,-78,-105,200,-187,49,-109,-431,-133,-327,-187,-386,157,4,136,-463,15,263,16,445,126,99,-384,17,138,-99,448,-14,-215,21,172,-8,421,-69,9,-87,-79,275,215,204,-71,-23,-432,60,-285,370,396,-387,143,48,-242,-160,-11,319,13,-275,-297,27,-389,-40,93,-119,116,-387,-130,-374,103,270,22,-412,-255,-162,38,473,172,285,57,-33,308,311,-119,-223,-232,168,173,113,-202,-27,313,163,-8,87,-153,-334,142,225,206,-23,22,-222,334,-86,-92,43,190,-53,329,-12,-49,82,185,-97,-56,-43,-97,123,-95,-333,295,-443,478,65,129,-504,-33,-432,-31,-112,-175,-46,259,339,131,180,340,-34,333,30,36,51,171,-5,347,-169,98,245,273,-6,-176,14,-138,242,-228,-83,-165,217,-118,226,-169,-137,37,129,-176,227,38,69,79,142,-224,397,44,-393,115,-22,-35,-268,291,257,103,24,451,-333,-266,241,307,-201,308,110,136,-104,22,-259,-45,198,-73,-282,55,-216,-48,242,119,289,173,112,-10,17,180,20,213,267,-48,-144,190,396,6,-487,-6,-270,76,25,289,207,-486,206,-296,-230,359,66,138,-307,-374,-103,426,-114,-72,-220,7,-42,-175,216,-79,-239,-338,-234,-60,259,-53,74,212,44,247,191,-133,260,-314,-160,180,261,-211,278,214,276,207,-42,-290,-72,-20,-163,9,-98,-20,56,-12,-50,73,83,-134,-355,172,-82,-37,-268,309,-30,95,-322,-15,-151,-302,183,202,-301,-17,180,431,-177,6,-345,-84,193,293,-147,-177,106,416,99,97,-34,-1,154,-3,30,367,297,-203,93,284,-251,19,-418,52,-221,327,142,412,-345,-309,249,-262,-20,52,-214,110,86,-218,185,335,75,-2,2,223,7,-200,-121,279,-236,-398,-23,235,-97,-27,-238,-139,-255,314,-115,-228,117,65,-8,-64,457,-147,-75,-334,-80,-248,23,318,-206,164,-214,146,-2,-240,374,86,95,-183,410,-174,-403,192,-366,-69,-305,-161,-378,273,-377,-164,-203,-265,26,233,28,147,74,-25,424,55,163,43,15,-262,-436,-3,-224,-176,-321,166,-317,139,35,114,-70,-299,222,-89,99,190,-349,318,-205,355,70,-5,-65,340,280,122,396,-117,276,-142,157,-157,17,-28,39,-267,-266,424,299,-296,18,-390,154,4,-383,-327,409,-3,-231,151,25,237,-151,264,-151,-120,-69,234,-4,-165,269,-26,112,-394,267,297,-120,-287,-222,91,66,-143,-47,-332,-279,-97,-25,-191,-43,249,14,-182,159,12,-111,-367,113,372,178,-108,261,166,369,-277,-280,-124,-123,-121,96,61,320,140,-256,-116,253,-441,-1,223,-224,-270,254,128,-67,415,-2,374,205,-341,311,234,-142,-262,15,-187,-325,-87,89,-122,33,222,-52,-71,409,255,-226,7,67,44,-275,-17,254,234,-386,-110,258,17,117,92,131,125,1,-36,-111,203,-36,-145,-397,-158,31,112,-64,306,-50,-95,199,90,-65,185,8,-127,180,0,-149,10,-19,72,351,90,-385,-293,12,316,89,56,14,-114,345,3,-337,352,381,-341,-111,-62,188,-158,-47,-330,-14,-279,-184,436,-156,38,-111,-74,215,-259,-40,-208,-118,-131,-314,-54,390,-134,-61,73,-19,-365,-469,-321,338,240,-176,69,171,391,-261,47,-237,311,93,12,-262,219,131,-329,55,18,263,-378,-44,-36,-74,349,44,348,4,-71,-201,-337,198,-48,87,239,-291,-61,-1,254,-13,19,4,241,65,-155,-17,-19,391,53,84,289,-83,-3,-202,30,-164,-220,-48,378,-151,-296,-488,-253,-215,-301,-338,59,197,140,107,134,-184,-346,206,32,30,110,160,-183,78,214,-42,-186,10,475,445,-245,-138,429,-229,-301,-125,186,-43,73,334,242,98,-63,1,-227,-296,305,-47,193,3,-37,28,-254,-76,132,-179,39,313,-138,-220,35,-249,-262,143,-339,77,229,144,-74,227,128,-67,170,-111,-176,273,-32,-253,-276,-61,212,49,178,151,26,-488,155,370,73,-108,-98,134,-200,-38,13,83,-192,-217,53,-65,-333,16,33,188,29,-144,32,92,-91,28,-50,22,-354,16,-25,-25,-32,-136,-87,310,-519,271,189,292,51,332,-15,94,165,-281,-130,-227,-164,-175,-234,43,321,-368,-249,301,-64,13,14,-26,-63,-103,171,260,-451,212,247,82,-88,181,139,68,152,206,-27,148,-277,152,256,-288,-212,69,226,-321,181,280,-52,-131,429,-142,100,-47,-103,-200,372,288,-203,-248,-315,445,207,-102,-20,-89,-105,11,34,-149,82,-22,-170,16,324,-251,-234,-137,178,32,-266,217,-165,504,-445,103,204,75,58,110,-32,200,-81,135,33,-248,-87,44,-139,430,-423,41,-269,402,-34,114,105,92,-125,-84,-402,138,-52,4,-233,102,-65,175,110,-3,103,-80,135,-354,-14,-108,66,-266,230,382,-18,72,-473,352,-45,-380,-292,24,26,336,-109,5,-149,-40,279,-100,187,-2,-30,-76,-278,403,221,-128,63,158,-155,44,163,-172,189,-43,120,-361,-389,-138,251,494,21,146,-489,185,-81,-59,-37,209,-134,-193,1,-100,114,-125,-166,390,-130,-77,141,-59,-47,-41,-69,83,254,-255,209,294,355,-68,-243,332,87,-113,325,-12,-417,44,-154,105,-245,54,103,-153,4,-177,-270,-87,376,-45,276,-49,15,-110,217,81,-134,350,-188,159,-193,-45,89,-138,-35,206,139,-222,418,-285,-93,316,311,-258,-61,41,-458,-39,-458,377,319,56,-41,65,415,378,-32,-53,93,216,343,205,-151,71,-427,-12,246,-225,-381,-356,-302,206,-268,-269,214,-73,397,47,-74,82,194,305,-14,-9,-201,-114,11,-279,-155,182,-197,62,-93,97,54,-110,-27,-133,-88,-403,28,62,102,-271,-153,-124,-69,-30,414,-200,-201,71,-57,78,142,206,128,-256,-454,98,113,-45,-34,69,355,140,-413,206,93,-179,25,107,-169,-15,-385,219,-102,7,-21,116,-166,-239,149,-30,323,82,381,240,395,97,-164,54,-201,183,220,81,402,226,-238,202,-117,-67,-470,67,297,-14,2,325,-44,-27,-251,75,-321,-50,-97,-49,-147,255,-17,-16,-36,-186,233,-254,-239,-121,-149,-74,-38,328,-427,172,-130,-222,-50,-222,-32,22,-305,372,45,43,-358,-154,283,-4,-53,-151,172,180,-168,101,220,-176,75,133,-163,99,-157,234,-41,78,-374,466,7,-432,-434,3,-7,-338,-133,-26,236,-316,-207,55,384,195,177,171,-175,361,-468,90,-122,347,165,-209,248,451,182,123,456,81,-120,-440,-203,-200,336,265,3,-19,83,-13,-92,-109,244,-378,-101,-187,-347,381,-152,308,209,113,-304,65,-304,17,32,172,-201,67,260,-180,-180,-189,-2,3,-20,-169,318,-14,43,107,157,-249,24,265,473,208,-232,-270,-74,-200,30,280,466,-308,153,62,-40,-238,246,-286,-136,402,-108,242,-202,-52,-61,-231,-8,79,-1,-149,323,372,349,117,-422,-387,166,70,116,-293,-140,-109,-195,68,31,363,-387,-478,-212,-445,140,-188,-513,-129,325,39,-17,-172,-337,-71,-12,31,-220,128,-336,166,154,-8,326,163,-40,-284,-72,92,37,139,-175,-90,240,415,-347,44,-329,-366,-202,59,241,135,189,-98,-221,259,21,82,191,38,-25,-178,-69,-150,53,-220,-78,-87,119,299,321,71,-34,-41,140,-494,-75,159,102,463,-72,18,-471,-432,-6,232,-38,73,-36,-131,-207,-186,-238,97,42,66,-99,-339,137,-90,162,-363,-117,-90,-25,-273,-329,44,254,173,204,-479,144,244,48,93,32,122,228,91,-286,24,-46,23,253,237,-269,87,446,-33,194,24,-272,-50,17,-174,-75,-55,413,-201,257,-97,-204,-376,244,12,201,15,-45,88,-132,149,-292,77,117,72,23,-241,-445,99,14,5,208,223,-58,254,-13,383,-169,-79,224,242,-330,-162,65,-65,-44,380,-293,-262,-418,-309,74,-228,14,-296,33,-3,-12,-113,-71,-19,-64,182,-367,-93,152,76,23,-58,141,-19,-25,-15,16,22,-214,211,-265,124,165,61,-420,-30,1,-157,52,-298,-29,220,-66,404,-244,-280,-158,4,-159,-95,-401,57,-94,-299,-15,276,-313,55,88,197,-55,-161,-389,141,65,282,227,-60,42,-132,-230,-322,133,245,54,211,41,207,-157,-115,-144,-344,-180,-25,133,-300,88,-130,140,-292,-175,272,-364,18,158,-67,72,-39,77,-126,-24,-195,160,123,-338,241,-253,148,113,104,-47,122,-274,-86,-63,249,-252,171,254,133,-25,104,112,310,39,-177,-34,-31,-160,58,35,-177,-215,104,-146,-250,-438,-210,16,54,56,351,-206,-201,-191,-43,-263,-184,254,297,-320,32,136,97,-128,-214,-244,-432,-235,-142,342,-2,-234,158,315,-55,40,-81,51,-44,97,-96,107,14,-268,337,-113,410,-210,-243,398,57,-209,-453,23,168,-23,-332,240,-405,91,-331,-100,105,-69,36,222,-149,314,-25,-152,-39,209,-69,342,183,-33,-78,-416,266,-325,416,-41,-498,-48,-261,132,-101,10,-118,-94,-49,-379,-351,-21,137,-79,-188,275,-495,-173,-84,-15,-119,164,259,196,328,175,61,-49,-146,131,-232,-471,112,-172,-224,-138,-115,-238,-109,12,196,186,89,28,76,66,53,-181,329,297,67,56,149,18,25,-88,-73,-42,308,132,-136,367,89,-39,56,-237,-350,44,-56,-38,99,-359,-92,-67,-133,30,31,-132,-129,-82,-97,-8,5,-135,224,-153,-342,-171,-6,192,118,-49,-58,-26,-49,-43,19,-195,174,69,113,-505,-236,400,98,287,-138,159,331,-285,388,-286,16,-201,84,372,205,-217,-374,-25,-295,-61,302,-18,-310,236,-130,-218,-121,256,-218,-52,-302,-154,176,354,274,-139,250,-28,-157,-198,75,35,22,-402,-191,65,-111,20,295,-131,151,-8,-215,248,175,171,295,-339,39,118,268,207,-32,-284,-329,63,-99,46,375,197,64,-16,-233,-211,65,28,362,-124,-1,437,-71,-242,-176,113,-88,-27,-127,57,109,-346,167,159,-151,-53,-410,-215,24,-139,402,78,-281,-154,-188,357,106,194,105,-201,418,427,-140,69,200,-217,-331,-317,-91,129,-346,1,-230,382,-191,348,65,-216,-189,-258,-357,-392,438,-143,-97,-109,-130,73,98,-218,187,166,-189,217,-96,292,259,137,-45,90,-38,44,-265,-186,-73,-145,-199,227,284,-343,-18,-373,-143,-7,-184,131,-162,4,-246,-257,-114,-170,-163,-106,-255,215,54,33,298,-152,176,-53,-351,-423,-20,-422,299,280,-152,-257,-181,-57,171,27,-19,-24,37,152,-396,-120,-184,-2,194,134,106,-28,23,169,74,-41,-112,-85,23,73,439,-89,88,-42,375,244,186,-182,44,-239,167,324,189,-58,425,-100,-375,-19,-413,-161,135,-30,145,55,-260,100,304,16,279,-31,11,-192,53,55,75,169,-277,282,-154,315,-68,197,49,-8,-216,29,122,334,-348,342,-14,-493,-166,295,19,222,307,-10,-167,-76,-150,133,64,-36,252,-239,-120,-149,-25,82,-370,-479,5,-394,203,-89,4,-117,175,50,-429,-29,-175,-416,353,21,-119,133,-222,-11,91,17,-25,-157,-118,318,154,310,349,65,-124,258,150,-124,221,-318,-76,43,-232,184,16,-147,37,22,-127,-60,51,419,65,-89,369,-324,-56,-259,142,-463,-18,-353,142,59,111,28,167,319,56,309,326,119,168,68,384,50,-76,68,145,-30,-33,129,275,-32,24,-338,310,203,-89,-130,4,-437,53,102,134,158,-176,-300,258,-65,193,-132,-192,8,-193,-37,65,-310,260,-1,-244,-141,7,49,273,393,15,61,-190,32,188,149,30,-121,142,-228,136,385,59,-477,75,367,-117,134,157,-57,-165,-170,39,135,480,2,-174,215,339,4,238,-22,-317,-150,-104,-113,352,-213,67,-269,-104,-492,-177,437,77,355,-32,-133,-70,118,235,-110,13,-475,-220,-220,294,-370,127,-293,143,-213,65,-75,-256,156,40,-45,386,12,49,-23,-264,-41,-11,-97,194,-2,-188,-46,449,94,86,323,-147,-195,57,-17,284,-311,90,-174,-138,-290,36,-331,-121,220,-126,-92,-210,-265,285,139,-472,-215,109,-80,-37,-115,477,276,234,-3,-292,9,30,-162,-63,9,-88,-110,334,-85,348,10,172,264,91,-198,199,-324,-124,52,35,-18,73,-416,-157,-47,-96,310,328,73,-157,-489,212,-280,13,280,142,-293,332,285,-153,-152,0,250,-233,-345,170,-155,62,-136,57,218,-222,196,5,-199,11,25,-96,246,-244,80,8,-305,18,314,-66,-33,-202,35,160,-274,-253,-219,-257,14,284,-161,-197,-142,-153,314,414,-193,73,28,-225,-267,153,395,415,29,46,45,-261,-10,-132,235,484,213,-16,-295,-118,142,94,-243,21,-329,332,83,-393,280,-13,-116,-32,-313,86,-112,-9,318,26,401,89,-104,167,412,109,316,55,-48,107,207,269,-131,155,-26,-7,202,-7,-396,-166,201,-1,60,19,-404,46,-88,-228,76,-468,241,-228,-157,268,305,232,-231,122,72,0,-181,-145,340,-344,34,-427,-118,-394,191,-12,-129,-149,-58,270,316,-78,96,-337,41,-290,394,75,-181,78,430,111,8,-171,373,-198,59,-52,-24,261,317,34,198,243,46,178,89,8,50,-322,228,172,244,412,-232,12,-2,211,-391,-2,123,-189,-178,257,267,-412,-336,320,-434,59,142,-376,132,327,-182,-193,86,208,30,379,57,-54,491,62,132,196,298,-43,96,-28,-420,267,-191,233,134,-380,106,-262,-104,-13,417,120,129,268,-265,233,-208,177,141,-7,-326,-105,-9,119,-314,-67,242,-85,-20,-492,56,10,-285,-83,-84,234,-390,68,-159,-307,-154,-90,445,-65,123,-366,224,-357,157,0,141,34,-1,-119,111,94,-53,201,333,139,384,100,269,-66,-199,59,201,-131,435,-120,-207,-19,-270,-146,301,-60,-25,-18,-62,-135,-467,-265,-65,-217,-335,-460,-317,63,-321,350,-384,-34,-29,311,14,95,-248,-214,-90,-285,289,-31,228,155,283,41,111,174,-235,45,398,-369,-196,16,-31,73,-336,-12,36,333,-50,-116,19,282,9,-224,376,-132,-107,-188,-91,444,-102,273,-34,-84,458,-312,-430,-230,-61,153,280,-68,45,-29,129,170,441,174,-59,-174,16,138,464,-282,-24,-305,-172,353,-47,69,242,-32,36,-358,-426,479,45,-67,300,130,-419,-202,141,-67,-234,-174,53,117,-19,170,-409,-58,167,-457,10,-188,50,-375,156,270,-442,411,327,-103,118,-147,55,335,-129,-210,203,45,-70,-396,-216,-389,-116,366,-31,-120,-77,-34,179,-65,168,39,-34,-258,405,173,-121,-84,338,-166,140,389,-258,158,-327,-426,109,-45,8,296,-13,203,-100,-15,-3,165,367,184,172,119,65,1,142,-214,-249,290,169,-437,222,113,-152,300,106,173,-1,42,112,264,-109,-165,102,210,138,-24,81,-250,-9,-78,236,-410,68,-100,-85,-213,100,-45,212,182,47,-323,-241,-207,-328,353,207,-221,-262,-406,155,17,-219,314,-395,96,-382,373,-165,-58,-65,-261,-179,-93,-14,-107,-223,-169,-132,-362,316,389,-393,327,131,-113,354,130,171,163,23,311,-58,224,443,-104,-40,306,174,-168,8,288,-319,-249,268,119,-280,9,-47,374,-224,58,35,-41,386,94,356,-272,-37,-211,-169,-76,197,-117,363,-78,-18,193,-282,22,19,-247,215,131,202,127,7,-189,328,-171,179,-96,-472,341,164,-19,297,-100,-58,-145,-54,-37,158,222,-486,-109,-207,215,338,235,170,71,7,-34,-109,-58,-256,-23,-97,-27,405,-81,-90,51,74,-407,365,-388,56,-336,511,-9,217,24,-27,-108,-118,-99,231,84,280,-118,400,-215,265,-55,228,-63,45,-83,-129,270,-22,123,-2,220,317,-220,237,95,331,0,112,-14,369,-71,-263,-8,58,-418,-46,-201,-237,402,236,61,-88,51,29,-17,363,-16,-151,-206,-371,436,120,104,278,43,53,-3,302,58,158,363,-387,213,-194,-442,25,137,111,53,-48,330,-56,-98,281,-150,-229,-309,284,-113,-63,-329,-172,-37,-132,-53,273,-141,-24,-178,-149,-21,-52,215,199,136,-289,-116,103,111,29,-11,212,-96,138,-150,-99,92,120,401,-344,-405,167,-141,312,-136,-424,-396,145,-232,300,284,-62,155,138,-149,271,117,-58,-44,256,-144,-57,11,71,23,65,392,177,435,-142,-108,361,415,345,-93,-278,-53,250,-21,342,-57,50,-394,234,219,-182,-169,-77,-9,246,145,-195,-125,-27,190,2,99,83,125,-151,-405,-156,172,-138,-65,-240,246,-93,45,268,-358,-9,-273,-200,85,-65,-319,169,394,-98,144,59,120,60,-23,-22,255,-202,74,-12,-195,-510,-125,-440,-307,-430,-321,-386,-378,321,-10,151,337,-19,243,99,-264,-169,-55,-470,-148,-227,343,-6,-127,366,37,-139,-6,264,-27,443,-166,-323,150,-17,-106,-261,-209,394,-83,-86,319,-122,-430,415,28,177,-68,-203,164,-193,43,256,247,283,52,-84,67,-446,-111,-293,-23,91,321,29,204,184,24,226,-70,-245,-178,105,26,-76,-37,-258,-178,-58,11,38,-208,-108,-279,127,-102,20,-123,-379,278,94,436,-367,-70,-16,9,16,369,-368,51,-45,-449,-334,-68,42,98,-32,38,-421,-227,-60,86,-317,379,-129,-125,89,-220,34,-49,-249,271,-89,98,140,-133,-2,208,54,-40,-261,-306,309,179,-278,-3,-298,121,29,-170,-35,45,-110,-73,156,27,28,-261,77,-236,12,-387,-144,-268,-4,-141,18,-331,-10,-2,82,-57,-113,-79,-233,-65,116,-267,-131,-203,-312,30,-6,-253,120,-85,323,-129,-139,-104,-116,-173,341,76,-158,-255,-9,38,-258,203,-256,-200,104,103,12,130,507,-42,94,-101,39,-295,-137,-336,77,338,31,83,-142,113,-95,226,425,-136,-234,-246,-418,195,123,26,281,-113,10,-261,-295,42,213,-402,63,116,135,-7,412,-351,-115,97,5,23,-60,-420,81,-122,227,9,23,90,-284,-500,227,-136,76,21,-28,294,140,-184,334,73,145,-345,-364,175,-27,-381,-274,135,-118,-90,20,10,174,-121,330,85,223,-46,-26,-17,-227,399,-185,-66,-64,-55,24,-370,197,-174,-127,305,288,104,478,-327,-151,337,-180,-323,-31,161,155,262,-57,-254,-224,-382,377,-3,-21,-101,188,-108,-196,306,356,347,93,-99,28,145,-31,6,103,-93,402,-231,-290,329,258,131,-275,138,153,192,-293,152,-211,137,259,-395,-55,-205,104,-101,-416,-245,40,70,97,43,424,-250,80,-61,-277,239,-40,-84,-10,-113,146,-268,-200,32,286,-54,360,94,-22,-406,177,-106,-29,24,72,-151,-200,-404,-83,187,-236,-104,72,-72,222,248,410,-11,106,245,31,143,-7,-128,-293,-79,-61,-51,-188,-207,-218,-334,-59,221,119,2,165,473,27,23,228,128,29,42,2,98,-65,-262,129,-319,-359,213,93,361,-254,445,-128,119,117,-3,-3,-74,127,-138,-223,-235,236,144,-177,174,-277,180,16,-152,-218,240,88,-31,-292,-205,-131,135,373,264,0,-61,491,101,255,-9,18,137,-100,181,143,262,-292,87,82,447,161,-249,364,-51,358,29,-254,288,-138,-157,-283,-260,205,-88,25,68,-221,140,-321,304,-191,181,-344,-212,79,23,-3,-22,-13,-173,166,230,-151,-66,-15,345,-130,337,253,286,368,-237,-298,244,-128,-172,279,59,-7,241,-139,78,-63,-200,5,-244,11,-32,-113,-277,22,-322,304,13,-148,-420,-324,117,172,0,199,356,-141,106,-371,-73,-96,-107,483,181,127,-9,-276,1,-72,208,10,90,-226,-351,149,-117,213,264,-74,-179,-7,-202,-361,68,-225,-267,-59,154,177,-229,-225,243,-33,-354,-196,30,235,59,-46,-117,254,463,122,167,63,424,296,-197,266,-284,53,110,-207,140,-217,239,17,-224,2,49,-73,-219,408,261,-142,-8,-219,-114,-498,-312,-300,106,150,206,-130,-222,-246,-148,-66,147,4,-291,-126,105,257,-505,-43,-107,0,116,416,-53,-377,10,-95,-82,52,402,67,217,207,121,289,-29,160,12,74,-268,395,-11,10,-145,-4,3,243,250,219,153,-235,156,-54,-203,182,165,396,-15,252,231,-75,-341,117,-35,51,-348,-194,-148,405,-436,-433,75,-382,196,-284,168,-25,-117,86,363,-111,-32,84,-220,220,38,-220,-132,-254,-115,174,-116,61,-340,19,-280,85,-71,80,-187,-3,-87,-176,-222,-25,-209,304,146,104,334,-54,-41,-105,60,-203,3,-67,-133,60,273,-107,-65,484,-220,-419,-478,-396,364,222,279,-176,438,62,129,-158,229,-163,-27,116,-438,-196,193,195,97,-222,-196,-1,112,139,-141,407,323,-152,-232,-302,118,148,44,186,166,231,-167,-226,383,134,60,182,345,-235,-143,207,-244,58,-369,30,142,-178,-168,65,402,-152,147,-281,-100,-300,-46,37,-11,128,368,15,-165,-299,-112,190,233,16,121,247,-243,261,-287,347,-199,3,364,298,-128,-25,-33,-362,370,412,189,-132,-216,513,-118,363,-164,-163,30,-217,-205,-242,-277,-103,368,246,-115,114,-89,429,-383,7,-257,-85,155,259,-225,-158,-51,130,-170,55,370,-210,507,24,10,121,-315,156,-150,33,-117,450,-33,4,267,38,-308,-185,-107,214,-114,-304,296,-132,122,124,-208,429,84,-324,272,-191,38,-121,-195,14,36,-258,296,118,-143,-26,-240,-66,-36,169,219,-355,18,432,84,26,-274,380,118,-174,-167,-4,2,82,26,-203,346,86,-82,44,120,319,292,430,-81,71,287,246,124,244,251,128,357,-226,-46,-51,-84,-31,1,122,-165,94,-116,60,171,-234,-114,94,-118,91,52,-13,7,-115,2,-45,173,-100,-158,393,108,37,-90,-12,-159,88,211,167,21,-11,-223,374,-287,75,-8,-305,285,-108,189,-130,184,54,-282,-360,-250,129,-21,-184,308,176,382,22,30,-315,-165,107,-36,191,-210,167,-187,65,18,7,182,-108,38,111,175,-139,31,62,193,48,131,-296,422,-84,0,108,118,177,3,29,51,-488,-27,-93,-22,60,281,114,180,-136,450,52,166,-185,14,312,436,138,58,-125,-32,312,4,-13,-148,144,198,102,-206,-45,-37,-30,-173,262,270,268,404,6,-362,-265,-153,-357,60,162,-1,214,6,-41,141,-370,88,436,-252,157,-86,-149,123,174,5,-39,-198,-279,429,27,-143,-48,-176,-171,105,-123,-46,85,-172,-105,283,296,249,83,-359,62,49,202,-207,221,-9,387,353,206,-345,-486,36,-188,-110,10,-373,-461,45,37,-198,37,372,65,-242,25,278,-195,-99,36,287,67,312,215,-181,166,-4,227,-280,-161,-91,-444,-68,-177,-64,274,-276,72,-318,-43,190,316,-57,-358,-43,-93,-188,-283,-358,27,-143,-263,303,52,-121,522,-199,-24,-162,-103,-169,133,-54,177,231,6,-410,300,94,204,-213,-20,351,-111,-480,-268,370,45,118,300,128,22,-54,236,-151,310,373,-79,310,-108,-351,-143,150,128,152,-108,-16,-339,186,385,-355,208,0,171,258,-114,-159,-115,139,-382,103,53,-281,-282,86,-162,-295,-290,142,-61,301,-333,-220,4,275,220,-108,-147,-114,-59,361,164,-281,-38,-296,-83,261,-69,232,-96,201,78,153,396,-190,316,-13,-179,99,-240,379,370,-36,57,-63,192,99,240,48,309,368,-87,-50,193,-32,-1,-292,86,428,4,-209,-379,-311,-353,-268,-222,42,74,278,47,-29,-94,228,71,-192,157,-37,0,65,-135,127,-92,-86,-155,-470,86,-179,214,430,-124,-22,29,-250,139,245,40,-51,-359,-29,-183,255,410,232,391,-191,9,-420,370,54,-362,340,18,108,-93,-168,208,-33,-180,143,-97,275,164,30,-358,-266,109,-175,101,-166,19,-106,-32,-159,-71,220,64,347,42,220,-173,-153,-138,-129,-406,29,-101,70,151,-123,112,100,31,141,299,176,14,121,49,269,-78,-460,-53,-186,81,270,-204,-59,-165,-109,-64,-16,120,59,118,146,101,-133,256,-302,158,88,-329,115,35,189,-265,53,259,-231,-264,286,5,-152,-122,-348,458,-316,442,79,-444,-82,-28,183,149,38,477,197,-211,38,263,-278,-404,27,-4,182,23,303,45,193,-147,186,153,-29,141,198,-196,-5,-85,266,29,32,310,-121,85,-358,-134,252,204,-22,370,319,329,151,-73,-101,72,202,263,47,-273,453,170,-71,-301,-297,219,-156,-71,162,262,313,160,185,-166,61,17,-162,218,-49,266,-178,247,-53,110,-5,-64,430,126,-359,147,-138,-2,-296,-442,-182,78,284,-253,-271,-19,26,-166,-106,-177,46,-372,-10,135,51,56,18,-73,-100,57,44,242,-96,-62,-35,-102,-234,-71,-77,-247,-130,-294,-101,81,-180,-122,270,-19,29,32,97,99,376,-24,-340,181,-43,13,-232,32,86,102,-237,199,214,-204,285,1,-224,-250,189,-68,-39,100,268,-71,138,-150,-312,164,111,-30,39,379,418,-260,32,-328,-5,205,-71,-83,86,102,168,81,-309,-296,127,3,-445,46,-15,30,-236,203,213,435,-246,45,13,-199,440,262,-306,-186,-442,-261,-85,-456,-445,304,388,-180,-416,392,165,-110,-388,361,36,123,-212,51,-152,324,-54,335,46,140,128,-216,-38,-67,200,-37,-69,229,-306,-324,220,-331,124,197,-384,432,-408,-191,-161,90,-208,358,36,45,-345,102,40,126,0,64,-115,89,-323,-215,41,-261,104,196,104,392,-279,-54,-109,334,-218,285,-242,199,-234,-96,-413,214,-354,-195,-194,68,105,-19,35,380,-119,138,-137,-32,317,-17,13,157,-400,-352,-275,216,1,-378,179,140,100,5,7,56,-23,56,-271,17,-106,-215,16,-190,350,143,126,216,-287,-227,209,180,78,48,285,270,-117,-417,370,124,442,24,202,-81,343,-30,-36,112,-105,13,-183,28,-66,-60,235,339,14,-110,148,44,-106,-27,-81,35,53,198,61,-12,308,-92,25,-8,-67,323,309,-123,15,-237,84,1,-273,46,140,-156,49,71,416,185,83,-456,41,-30,213,-336,-261,-280,-284,-112,-27,111,-453,191,343,148,290,-242,186,14,-18,-27,-495,-420,-113,253,-228,370,-4,148,-28,106,-88,-1,-145,-254,-37,-208,-438,192,-250,-197,-46,298,420,-84,301,283,-225,65,-387,-263,47,10,-270,367,-226,-2,14,291,197,-135,86,-28,278,-88,137,177,-81,342,-415,-95,173,293,242,215,-99,-55,303,291,-397,-43,82,-350,-303,49,-211,-7,-103,420,197,-422,-171,-75,-44,228,1,-120,-202,389,-359,-193,-466,-134,-137,-92,-36,328,271,69,-66,-80,330,97,-15,-67,-37,-42,-405,358,80,-329,258,132,-293,79,58,89,112,-54,60,239,-398,-64,-318,145,-171,109,278,264,80,198,-28,53,120,410,-326,-276,-12,-313,46,245,-43,25,161,-161,42,166,232,-124,-15,43,104,57,-19,54,296,25,269,-11,153,-56,278,47,-109,422,-21,94,-368,-23,151,51,-345,94,84,-171,-151,191,-220,-228,455,-92,275,118,329,135,128,92,386,-297,159,-401,393,-138,-166,156,262,483,-116,5,309,194,-271,-374,-186,176,-316,395,54,160,-147,437,-72,-135,234,267,411,-38,-219,-68,-310,-310,-242,-189,-296,239,-122,-140,235,123,415,-293,115,-133,-424,-54,367,233,188,-482,289,-82,-234,-59,42,227,307,-65,-77,-445,-132,311,-28,5,-146,172,-89,62,-225,-102,129,-231,-340,-189,-92,-200,183,6,-97,-103,379,59,-318,-40,233,246,369,-249,-252,256,292,-17,-68,-473,-237,-321,-287,306,87,189,-377,433,-112,52,-147,-139,68,-146,425,240,-128,-172,-118,314,-158,-195,79,70,-288,231,-275,41,99,59,-65,-106,188,55,-437,-22,-301,-220,-153,21,23,-176,3,-81,248,-28,192,176,66,225,278,-46,42,-431,472,291,324,-174,-57,-18,-123,-33,455,-12,368,-4,67,-137,-291,356,-96,116,-61,173,-39,-412,115,-10,-26,-183,-270,-250,100,55,286,478,-323,-72,-144,163,140,184,37,-222,136,-107,-87,400,139,-49,-268,38,-69,266,-8,-138,192,-73,-86,-68,-39,-163,-275,-67,-251,-191,36,219,-53,294,206,-40,218,112,22,235,115,193,8,215,-113,-382,-49,-187,-148,-373,-89,-20,-136,89,-50,400,41,251,-263,-113,-157,217,208,183,-252,-116,266,209,-484,-200,186,-364,217,-136,89,-322,38,139,-195,120,-132,164,309,-301,35,-32,27,-94,280,38,154,-215,50,-358,55,-94,-79,68,82,-108,-56,-80,-88,292,177,55,181,-231,-332,14,-30,-104,-244,124,14,-82,126,162,-191,-243,-390,437,215,-49,-44,-384,-117,-322,15,145,205,358,54,204,-87,-115,4,-64,113,-68,-120,287,-186,97,-3,-136,41,384,-133,276,-219,88,38,-462,183,-214,58,-24,-90,288,49,256,50,-102,35,-255,-5,72,-106,-33,98,67,131,35,126,15,253,-229,79,-152,59,407,153,297,-301,-79,35,368,82,251,-295,283,-132,-161,27,54,187,-319,43,136,-189,-322,-304,98,24,-149,-59,-269,172,-35,45,-405,179,-267,-357,182,486,135,-153,96,-333,-222,325,-107,-185,-290,-96,4,-55,-70,79,-13,151,-69,-246,-141,-129,-204,-348,-205,0,22,84,61,187,-232,-69,-67,63,186,70,37,-175,474,361,273,237,-89,-14,64,237,-54,57,-161,-282,223,-196,2,161,-72,-134,-236,-10,-177,-127,120,-14,52,7,15,-325,458,231,-355,288,15,0,-148,-9,424,-255,174,130,68,319,-361,166,25,-141,-228,-45,420,-190,-195,-141,-39,-270,-182,245,30,-241,163,72,402,124,89,-309,-5,-196,113,-233,-33,295,340,101,130,-149,423,-177,-265,341,-357,224,57,83,-26,47,158,413,71,2,-20,215,-127,-265,-427,-21,256,-408,-373,395,-257,251,-276,-43,151,292,266,-84,90,362,-491,-29,-72,-312,8,312,-123,-289,-264,366,-256,188,-285,271,-113,284,210,184,-46,70,-323,-199,122,159,156,229,-175,175,109,-65,-259,-93,263,133,376,-77,317,3,-105,432,-422,-253,-300,-202,-127,-181,-100,297,273,-463,-50,373,-431,190,48,-113,19,428,-272,262,-24,36,383,201,206,143,-203,-124,108,15,400,7,-210,-48,-12,233,422,275,-261,313,143,-41,-133,-93,-189,-4,356,105,-6,260,271,140,-112,-29,-5,108,-222,25,82,-55,-434,128,90,229,-307,135,331,12,54,8,-362,341,-147,-68,-58,166,-15,-304,87,-80,-114,13,-162,139,204,-151,-34,14,156,-44,-117,-212,-58,137,-171,-39,-129,-182,-64,82,342,-26,-144,-142,-53,41,203,381,-343,68,-201,85,30,138,-207,164,-478,-383,-229,-180,-17,66,188,141,375,-171,-54,343,-205,202,-358,-114,424,-156,-122,-142,-8,-88,52,322,165,-270,320,437,-195,81,-27,-16,196,438,-81,103,229,-275,-26,16,-41,-29,-152,-70,-80,-10,24,222,-108,-322,18,-240,249,47,174,-418,151,-225,-275,3,73,8,137,-191,31,-235,-425,450,-91,-72,185,-62,139,31,290,-136,55,50,-363,-27,84,188,289,-48,-314,-64,32,27,80,-158,-68,404,-18,-10,60,-204,109,-38,431,19,-14,106,240,414,300,30,-38,-8,1,-112,-298,12,339,-78,296,49,-212,103,-54,292,203,296,32,259,53,156,-45,-97,42,-307,409,276,40,48,-43,202,-330,413,388,-8,22,139,-27,-64,233,-62,-124,-3,294,96,216,88,49,-63,-97,-399,102,-326,-100,-179,129,-93,181,-195,-331,83,-148,-38,-208,174,329,-72,133,29,-61,-19,-15,144,-241,-345,-81,186,93,-54,-15,271,66,521,-403,368,101,-211,-70,-406,43,-38,-34,-217,96,9,144,237,115,-358,-132,281,-98,73,66,-196,56,23,57,-332,418,286,-452,-174,-99,76,58,-153,-246,247,-447,-148,259,-89,-34,226,281,-291,-252,348,-224,-133,65,-117,441,-273,76,69,-81,192,-64,262,23,47,111,-155,-251,-204,-287,-99,-418,-225,90,329,474,-177,92,-255,144,-230,-392,-79,13,-426,1,263,189,119,-40,-336,144,-82,-150,68,-144,142,72,-15,71,-78,64,228,-64,212,371,82,150,-127,119,78,162,1,117,-15,188,396,126,-222,69,-319,253,183,-249,499,191,265,38,197,61,-33,58,-197,-180,-115,-78,293,-85,168,-311,-251,-62,-184,360,-359,-188,-182,32,287,48,2,44,-222,109,-338,-162,-295,-20,84,-216,305,107,405,-1,306,119,-280,0,-115,78,22,-34,73,407,123,8,-391,330,-155,152,-200,332,12,-27,145,113,-81,58,139,78,-245,-244,-348,298,58,-49,339,-164,140,77,3,-180,44,-224,52,2,-239,111,120,-160,-106,247,148,108,66,-94,-61,-93,56,-188,95,-34,4,118,387,-103,221,-297,-11,-354,-215,-226,342,-292,209,262,184,-322,0,-104,164,147,258,209,-8,-250,-66,349,-182,117,8,-136,37,31,-135,209,37,-294,-304,-4,-27,-3,177,-4,-26,-22,-29,95,-29,0,-308,-232,-32,-91,-186,68,219,66,56,-206,-102,20,31,-294,24,39,221,238,509,-74,241,27,-63,-194,186,-107,174,20,-230,102,-336,-144,-330,108,146,-124,-175,-190,-88,128,-44,-62,82,-165,168,282,8,-172,34,-34,-220,336,-96,280,236,180,-47,53,285,-377,247,-252,33,-139,-21,172,84,-148,-158,-83,16,-165,112,219,-360,-159,113,200,-59,227,-96,-12,-87,-56,225,156,-7,-148,129,-180,62,146,27,117,70,214,73,-214,-45,-61,-126,-287,45,-436,-55,-85,276,-5,198,-174,-123,-164,199,-120,82,-287,-3,-298,253,341,-107,-80,252,101,153,492,-367,292,7,163,24,-3,-24,195,55,292,10,119,-63,253,-171,127,-264,-338,299,-175,323,236,-301,51,124,-451,-58,104,128,-153,-463,87,326,172,94,346,-215,141,232,448,133,-138,125,68,-10,44,428,-115,409,-318,461,144,-168,38,-51,80,359,235,293,23,-261,62,162,31,509,14,-86,-231,-155,-196,-19,392,158,-51,177,-382,-240,-17,127,143,-87,-276,400,-374,-364,-16,277,-269,260,-159,-11,-177,315,-27,-215,-370,19,-103,-141,-35,96,-129,-210,-280,-169,160,-103,-240,-137,-265,125,406,-226,-73,-34,-165,-363,86,-5,238,-270,-209,-31,401,296,-262,-16,265,-191,-24,-80,199,-10,18,-122,115,-246,5,-171,33,10,318,13,483,-109,-215,51,-102,-211,-113,67,-172,54,441,110,-287,43,-159,495,274,-139,38,368,39,196,24,366,143,313,122,188,173,-437,-38,-192,233,38,264,76,349,-232,-147,-35,-40,184,375,160,-231,-3,122,0,-41,42,150,5,357,113,341,-126,-130,-187,41,-276,93,-170,229,0,369,-277,131,-325,156,-248,163,-155,-36,157,-238,163,5,309,224,112,-39,-455,302,113,91,-4,7,63,-271,3,311,-369,-152,364,182,-207,37,27,177,-237,12,69,-149,217,277,366,-58,-262,-138,422,258,-17,93,-326,377,238,-10,-267,-110,-95,-141,154,-35,-240,-445,-15,-15,-21,-397,325,129,307,-292,-275,147,-90,307,-314,-299,-122,-179,116,217,-419,160,-225,42,16,-175,44,101,70,96,110,308,466,103,231,-285,103,-364,54,-144,-53,107,33,-177,-236,-381,-283,-113,-84,95,339,-162,48,111,433,236,260,256,147,238,-111,-49,109,-365,-42,433,115,-146,-189,12,184,269,-5,-159,17,-203,-63,-85,304,500,104,45,-35,93,300,-85,-383,240,106,-270,5,265,49,454,136,121,-276,67,81,-28,-245,-289,102,-65,273,169,-5,-33,271,104,128,190,26,-75,-110,50,249,278,121,-430,76,141,-53,-23,-151,179,-209,-276,314,47,-105,-191,-70,19,333,228,179,2,14,-275,115,-334,-132,52,287,-226,-71,118,-106,-100,-40,14,391,-227,-11,202,-294,92,-94,-57,-38,210,167,36,-125,-118,286,-139,-455,21,306,144,-282,-77,182,226,-57,303,-157,267,14,-201,-33,-356,-313,192,-193,165,-456,-224,-7,-9,-11,211,-142,94,100,186,-215,-323,9,-97,-15,204,-78,-62,-183,-37,117,-317,-184,-431,-47,-364,-176,-127,-3,285,156,-118,-128,-21,285,225,84,255,418,180,93,-288,-319,151,364,-305,299,20,254,433,-196,-6,198,49,-58,316,51,-77,-193,-58,-291,-122,-146,66,-221,272,-29,-89,-67,223,226,324,49,-134,295,141,224,-13,-269,-59,26,-75,-67,257,5,27,277,-86,-254,-351,40,75,-70,227,-286,-407,-443,-190,21,483,-109,-108,-250,-122,174,66,-382,-233,-56,112,-32,-344,426,-69,46,-188,73,-369,51,-104,245,136,-12,-299,-70,76,-254,-73,175,162,-173,516,-75,351,17,89,214,-13,328,344,-294,124,-322,-90,-328,-44,-138,104,-313,267,-362,-182,188,-161,44,465,-63,-252,395,47,-153,-270,245,123,283,94,-361,-76,-159,-33,-157,-267,195,-155,-177,356,-262,6,91,-266,-222,-17,34,-182,113,458,52,52,32,-44,10,-165,205,-124,132,65,-394,77,184,-252,104,-219,79,-361,-233,-331,205,-252,117,-314,-232,153,-210,130,-110,314,-396,50,-91,-305,-190,107,146,204,251,-292,-90,278,78,-95,197,-424,-138,-71,110,180,-194,42,278,405,-38,134,-34,-41,-280,-140,-52,-152,-10,-196,-325,-236,-123,55,-481,-199,279,357,64,-370,18,-432,-22,-267,-222,-104,-153,76,17,194,-63,131,-140,-475,135,79,165,-106,66,54,96,223,-268,76,238,-49,131,-248,-64,34,-15,16,-11,31,124,-75,14,88,6,-495,-26,426,167,43,-47,207,-441,-295,-124,202,287,78,-232,-87,-94,82,-128,-324,132,-49,-231,-31,250,-64,74,269,-64,-79,-408,-42,248,-440,149,-272,183,-65,133,-99,64,-76,-24,-87,97,84,-129,0,206,-42,-60,184,491,-177,-179,-208,485,7,15,413,-278,32,511,38,32,-29,55,-329,238,-161,-5,172,121,-67,-141,-273,18,8,-56,-14,311,-166,-200,-80,-247,-396,-340,-154,116,-265,-182,382,33,408,-341,-12,126,-22,152,186,271,360,-27,137,-8,70,359,0,-48,163,-308,117,141,258,7,162,-86,181,77,176,186,-14,-254,113,-202,40,-5,32,204,330,344,-258,-131,161,182,-36,229,283,427,-398,-17,-369,89,-82,-92,323,152,-120,464,-102,-127,-308,49,-198,-149,35,367,15,144,413,137,-135,172,264,313,2,406,-91,8,402,186,167,308,-295,47,214,139,184,320,-76,-289,409,-53,-142,-163,-105,-96,354,-43,-109,-10,121,87,222,-30,-185,302,-28,-126,-328,-350,-226,64,195,-67,-282,-302,-212,63,98,63,-190,-318,-126,-346,-194,124,-347,97,17,-200,289,110,-171,-69,-64,-137,124,86,-44,139,-117,211,-252,152,34,-104,-183,-365,160,314,-150,284,52,-219,88,-48,-37,107,-148,-15,-227,59,-15,-231,-83,-152,-176,92,78,3,-44,291,-332,89,145,147,173,447,-91,233,-190,-148,-346,325,4,-116,-303,502,-320,181,69,-373,-123,275,-236,-369,-148,-147,-133,352,130,190,130,61,-142,82,17,-204,381,-218,-393,110,317,190,-237,94,96,-196,12,-213,-252,221,-269,-31,73,112,13,-125,465,89,-350,-249,77,70,395,182,-373,-24,184,-175,137,-20,-336,11,-365,116,-126,-160,33,30,-38,-304,-70,-424,-33,33,5,-251,491,105,-86,-130,97,150,156,-130,-34,-83,39,-273,36,3,-263,256,-131,326,-118,-13,-78,30,288,-101,443,201,-60,-122,355,259,-210,-43,197,275,170,-222,-246,-389,91,-103,-465,-169,-27,-259,-71,100,318,177,369,463,-368,-132,262,225,-373,-60,312,147,164,-90,120,67,7,4,262,177,-263,-47,-62,399,-489,-2,-104,257,36,-183,46,-143,-163,-294,6,69,-392,4,-151,117,-8,49,-86,120,493,-43,137,-272,-99,-288,-99,-276,65,320,99,-212,-81,274,-124,-11,-225,-14,-75,229,-361,-309,-28,9,-181,-163,339,-388,-289,-117,204,50,-184,442,122,-36,154,-78,-78,135,203,155,-100,48,-101,30,-253,51,171,-91,245,-250,-258,492,-253,-269,-94,-305,156,83,146,-85,381,-200,-193,-43,-409,36,-100,-83,-447,88,-426,-110,388,-77,-91,-355,241,5,-52,295,-396,-127,437,167,-68,253,-346,-80,-137,-298,26,254,-256,-19,-121,195,-315,218,-171,103,291,49,-224,-125,-495,-198,432,180,44,-34,-189,17,15,-268,207,-259,-148,158,-353,-325,-207,-294,354,33,90,350,-142,-503,-342,81,-22,434,-151,-102,-2,-137,-214,-66,-256,154,-253,304,107,120,210,-418,-218,155,-236,339,157,183,-150,-173,480,-262,-22,-151,232,367,-203,368,110,-346,123,39,58,157,88,-79,388,-106,24,109,-414,27,-450,-82,106,-369,294,497,219,139,271,322,-162,31,-182,208,58,98,-50,-139,-332,-382,259,3,-168,-202,-464,-124,-160,-404,-45,-126,-129,-100,-61,-325,56,-111,122,-34,-242,-128,78,46,-169,-216,73,328,-258,372,60,-54,71,75,-16,-226,-5,281,-293,-21,388,-106,164,128,-462,30,-437,395,-338,-187,-107,-36,451,188,-114,-398,50,-51,195,-395,118,19,286,-407,-262,12,31,-25,-42,403,19,-40,-465,-251,-312,-34,-176,-7,136,-169,311,-142,-219,241,-185,-130,221,-8,-353,-167,184,-24,-400,-171,-30,-179,117,81,-291,40,-133,237,354,212,-6,217,-155,20,357,249,289,15,83,-15,140,-6,-54,-221,5,-161,149,183,-384,19,-237,-111,1,-203,121,75,-327,12,55,-157,89,-97,-228,-63,-406,-70,-78,79,-294,-141,-251,-379,396,-315,-23,81,-305,134,231,223,75,-41,124,-273,-67,224,141,-7,-82,26,72,436,455,-173,-164,-54,-23,-45,-106,124,211,196,-134,-4,210,160,19,57,56,427,309,59,-33,219,54,112,-259,-406,306,-139,-164,-312,-188,-38,-7,-161,-31,-287,-47,-279,-213,207,139,336,-47,-64,-416,-455,294,52,-118,-77,-295,181,-171,113,68,-171,436,94,-81,-9,-6,-479,-79,170,-103,-381,226,-1,-129,391,-28,-478,14,-425,-181,79,299,222,-348,-8,148,-265,-439,-156,-181,-49,85,-421,-14,-22,-91,-80,-521,-59,-252,-216,-129,-123,-160,164,155,78,19,-35,-236,-305,-485,138,10,-77,133,216,-312,114,-389,-333,-160,-304,160,-37,-216,295,-153,65,354,-303,-233,36,346,119,327,-293,260,-15,51,211,232,358,-443,-40,-105,272,5,474,159,70,-14,-352,-91,333,77,-299,284,259,-291,188,-47,304,31,129,-183,462,-67,48,32,-4,315,137,-51,-276,-304,259,400,227,153,-215,-344,87,148,-171,218,34,37,296,-144,258,75,31,48,-49,-109,-319,-25,-115,-11,-266,39,95,-17,-143,-20,-143,-362,199,-62,-107,-195,14,280,-384,223,-117,-361,68,397,82,12,-64,-262,-140,-95,-235,-87,411,175,-293,-32,-206,-94,-61,83,105,245,171,27,-78,163,189,-31,-218,343,23,-483,88,47,-223,-487,296,-71,406,-288,-91,400,206,-81,-262,40,326,-94,240,-90,158,158,-96,88,7,212,-224,-296,131,-192,337,123,-153,234,-285,12,-167,-313,189,264,-45,291,243,116,-172,-12,-137,-287,77,-180,-290,-196,-57,-129,-94,-214,-338,-231,-447,76,85,-263,-330,10,-199,-12,-316,-424,-238,98,407,-26,-56,51,-13,-4,378,37,-85,157,-82,-53,-140,-466,246,44,-277,45,-122,58,-136,26,227,-225,-225,-101,-337,-149,178,149,-249,-235,-38,93,-209,-189,-246,113,49,260,279,56,52,-277,282,-97,-324,-504,-115,47,-28,-247,143,84,55,184,373,6,-224,173,-40,230,-54,339,-162,8,-184,435,380,-157,-233,419,-367,-98,146,100,-16,446,64,63,-198,-142,-159,213,-235,-166,-362,-59,68,435,-111,36,-81,437,96,-22,53,194,-366,71,-95,-112,-16,-226,-107,144,-307,199,-130,-24,-276,141,-54,473,-5,-317,132,-177,322,233,-289,-17,-130,-335,418,13,23,189,147,361,-133,84,-199,7,175,-383,-366,246,-121,-27,-9,94,17,-221,453,-145,-93,-203,116,-50,8,-103,-44,209,-71,45,-108,122,-71,-427,-34,8,-229,-6,-185,13,-328,-315,1,-196,-254,290,-159,291,94,-429,37,-40,317,-215,49,171,-137,299,312,-24,300,225,-20,-164,-110,239,376,103,307,-245,-477,-113,117,-177,-252,215,-222,231,-65,117,295,-121,45,24,140,-419,8,3,283,142,140,241,48,-189,87,-13,186,171,49,78,307,-97,268,41,-333,104,-94,-388,-99,-39,197,123,-40,-29,-98,-165,-430,-178,-28,184,398,128,-112,167,125,-279,-178,161,96,428,-37,-92,215,-105,-282,-22,-15,-311,-82,152,-414,-209,-386,-62,-70,-159,2,-48,-96,188,-150,58,-113,-95,51,62,103,135,4,-17,102,18,-262,50,-17,-318,153,31,326,197,375,136,-112,-373,-215,98,-308,331,-188,-361,383,-38,54,-399,-51,-83,161,88,106,-253,-46,357,242,-210,180,-186,-79,245,199,-33,-83,24,274,200,-217,-447,263,287,-93,40,34,173,-319,306,154,428,-102,-403,40,61,149,-417,-246,151,351,-252,161,211,-406,153,-488,102,-168,234,-29,80,216,-262,-45,-138,84,427,102,287,399,-210,39,-1,-188,-190,220,101,374,259,27,169,-82,-393,197,-172,379,-56,-256,34,-236,10,24,-73,-10,-301,-76,-140,461,277,183,210,-102,-133,17,131,189,158,-1,-135,-109,-85,-298,-113,59,456,-212,-19,-272,-193,174,75,-213,-153,-306,-174,319,89,146,-432,112,-386,2,141,-80,423,-452,46,-55,-198,55,-5,-147,-328,-424,-44,264,-430,211,-17,-363,-6,61,-134,-194,-181,1,162,98,-84,167,114,-318,22,-89,-272,427,407,-59,198,-91,62,61,128,264,31,-505,-317,-95,-222,-162,-319,107,-190,-235,-92,186,-3,18,146,-297,12,285,-33,-77,200,-7,308,227,-83,-76,208,-116,329,-264,-213,237,141,20,-83,144,-311,-301,-460,-94,75,7,-306,168,369,-248,156,93,-82,417,248,-245,-419,-3,104,417,-24,35,-152,364,74,-348,-316,-122,-67,-253,392,-21,-124,6,334,-15,-217,-12,27,-192,-80,-53,53,3,-263,179,-193,-76,-307,-22,344,377,72,120,260,131,173,24,233,-372,253,-207,-108,0,346,349,-187,426,-96,128,230,203,-55,-454,-316,59,65,-14,124,-168,-285,-63,138,151,373,-207,81,-67,86,283,210,-270,-24,-150,26,-342,-28,-25,-114,-171,51,-16,108,201,-21,135,24,-391,44,-205,-4,-18,389,49,182,-178,-3,93,-13,16,174,-58,237,80,-142,-87,-46,33,198,-89,44,146,-348,-126,-369,25,246,-49,-209,31,-354,-120,-209,-201,-8,-315,196,-304,-350,-276,-273,202,-47,113,328,71,300,-30,44,132,-61,311,-395,-185,221,348,-65,366,-220,101,-147,-60,-157,156,159,-371,72,-427,40,-73,-31,153,-101,-426,-172,-104,308,176,319,49,334,-16,-225,-12,61,60,267,165,-194,-76,-316,-57,114,-171,-73,-129,192,-47,-275,-157,441,-31,-348,334,-43,-228,13,-305,-361,225,340,98,-205,-91,-1,-78,-39,498,237,-310,-363,400,-340,-205,103,203,217,-455,-127,68,85,-215,113,-210,-291,124,-164,234,191,16,-259,-65,-211,-145,377,-168,170,94,-140,34,261,88,-216,263,-244,-419,31,57,118,153,12,18,-121,71,286,-43,27,-173,-488,29,-218,-307,160,-19,-77,21,-73,-156,79,-73,160,221,405,-60,-197,-8,-457,152,-435,66,-95,103,210,253,-377,-305,475,-85,-105,-84,-44,96,212,337,68,-75,83,-77,333,27,225,-13,-89,-118,222,378,-173,159,210,149,-60,63,-178,45,105,-129,92,330,-332,-90,277,-87,240,200,248,-523,5,-212,-125,312,154,-63,-188,-289,352,406,-161,136,-187,-34,-81,-353,16,296,-247,93,-4,370,-210,383,-119,17,36,302,-368,120,265,-136,-68,51,-485,-90,377,203,-151,-173,-146,-34,-267,168,169,-97,205,-165,32,-5,-177,-156,-270,254,46,-45,73,59,-468,-108,234,-50,198,-202,-137,200,70,-37,147,22,-2,313,-28,144,-127,62,374,-347,76,435,-108,-215,273,-128,-37,113,-160,-2,289,-102,311,223,-354,339,55,14,-77,88,13,131,-108,-300,11,-2,-86,131,231,98,-102,-241,254,87,-369,-21,-230,55,255,-209,134,94,-121,23,157,-173,1,-355,-253,165,170,111,-47,101,-213,-41,136,-223,-103,-412,-154,-123,182,84,343,41,195,-199,204,19,-172,29,-120,56,-83,380,170,-94,-46,-425,192,-199,-160,272,-36,-168,222,160,213,-12,19,2,-57,5,150,-32,27,-235,-237,-171,396,273,-34,-129,342,58,-97,-74,23,-120,-66,302,-320,91,-334,-149,-87,63,1,61,180,267,-91,-71,214,-416,-217,-132,-116,-172,418,365,-268,444,30,9,-60,124,-56,-45,-212,-261,320,-39,-209,-376,-112,122,-288,88,142,115,192,154,-22,-399,357,506,198,-18,34,-41,236,-120,-140,-72,-303,172,223,221,308,-126,-129,-52,171,17,-89,206,-19,505,-417,282,-50,-108,-143,370,372,342,-7,219,-262,59,170,-36,167,40,75,-269,-218,-152,151,-17,313,-79,123,480,-56,-152,140,15,-4,465,116,-187,84,367,-231,278,286,-66,67,-156,-63,-161,-103,125,159,263,316,-51,-67,-19,167,12,-502,268,29,-235,78,-133,336,-49,-360,-347,248,-41,49,6,-12,-96,-236,-184,22,-195,-234,176,-21,176,324,-169,200,188,-132,309,-140,432,51,-60,67,132,-201,-385,-133,-169,195,407,91,-17,3,15,249,161,135,-251,-71,-130,-270,185,-159,-81,167,-284,-65,163,-124,40,206,416,16,-342,235,-18,-94,115,-332,142,138,-392,197,203,-266,-326,396,68,-383,224,302,-449,243,250,320,-161,-172,278,-206,38,260,46,296,-162,-326,-146,-50,71,390,13,78,-108,115,-328,-271,-246,402,73,338,-155,-128,-15,-35,-212,298,-497,-363,-126,77,178,-65,-89,-151,159,-252,-50,-420,182,101,-216,-127,268,-280,292,-15,-121,244,-1,-248,-214,-233,59,-141,187,447,-31,-70,335,-251,19,-1,218,-26,-135,120,263,99,-241,-254,28,127,155,35,-370,319,185,-224,-289,170,-20,-8,229,-338,-178,383,230,-378,104,-46,-222,-270,-458,-278,1,-270,0,-106,-201,423,366,-477,21,109,142,-380,128,181,260,127,94,135,309,122,253,-108,211,17,200,25,-149,-55,243,-61,-88,362,-46,-427,-310,84,-7,-120,-54,-368,-291,-68,-490,-165,62,179,-45,335,-441,-20,152,0,-178,70,-127,-76,-139,-227,6,-8,-136,-275,-59,349,-262,-109,-378,-48,115,-149,-262,-47,-29,-330,-103,286,-238,370,-104,40,228,-115,64,14,48,339,0,-4,-5,176,-321,425,-429,-95,-286,489,-309,-133,18,23,109,-370,-106,-93,-153,228,-451,218,-97,6,-177,73,-98,136,231,-22,187,401,-178,0,176,-5,284,-288,-263,163,-24,-152,-326,268,-90,74,-134,-388,-119,-165,85,-175,435,-222,95,126,149,99,73,432,-19,117,399,193,-270,-434,-162,197,-254,470,-208,29,350,21,166,-79,102,-245,421,-36,60,259,-140,-117,121,-304,-162,316,234,42,350,119,-46,7,-348,149,76,-258,-66,-107,-327,-403,-101,-217,101,-53,259,-83,-156,-421,-368,34,-215,-45,-256,305,-118,-164,-79,26,116,-122,-148,27,-403,261,89,380,-308,-316,137,-68,-47,-97,-81,467,192,-354,-464,178,150,431,-36,-231,200,3,305,24,60,-103,-7,-331,-488,-136,109,-54,201,-92,119,293,28,-157,22,-409,72,420,31,15,-364,-259,238,148,44,6,13,268,-51,-54,-32,-189,199,-120,-175,44,222,79,-34,167,-454,-239,-139,368,130,-235,38,373,-265,-163,8,8,121,55,-385,256,-73,-7,-315,-189,63,-85,116,-377,-349,80,-127,49,-55,-324,-190,254,-140,100,110,-197,112,430,459,-2,38,24,-178,254,-395,-288,149,441,-363,179,-182,157,348,421,232,38,79,33,243,-180,-401,-65,-78,-518,-116,-311,-78,72,-181,-285,-334,108,-311,-151,62,-135,118,-165,339,34,298,83,44,211,-367,309,75,-84,215,338,146,-165,158,-8,433,461,258,-21,301,-106,-344,370,-330,39,295,335,-317,288,-499,19,151,268,-78,-236,-83,-50,-472,22,-137,235,249,383,-157,-288,27,27,-171,265,174,52,-30,46,16,-325,-78,-164,295,161,-408,-120,-340,178,-269,-347,2,-26,-148,199,-302,102,-152,93,-336,-92,-180,52,-46,73,277,-25,141,-362,111,104,152,-293,110,19,306,95,118,-394,-70,-19,-70,43,38,-146,208,48,248,-16,-187,-230,-410,-158,-428,-226,39,338,16,18,40,34,253,-345,-339,-290,-318,104,-35,211,-333,-395,-28,197,137,220,-151,-312,22,-124,-108,-275,54,-254,143,-33,230,63,-53,95,-36,263,-153,-333,-69,-222,-245,157,-197,72,-214,517,95,116,-292,117,-112,-37,185,-93,81,48,-185,175,-213,-18,156,-280,-325,-9,85,-116,45,265,-220,5,-27,107,-131,-97,-33,96,-149,103,-71,344,-179,260,363,-15,157,241,371,-129,197,-163,141,63,-270,256,-101,77,256,-20,-30,-11,376,166,94,384,213,49,-217,-315,88,88,427,154,-215,87,339,357,-239,-34,-19,82,-186,-174,34,-460,32,-35,-86,185,-70,353,184,-338,147,50,-21,81,68,108,-170,43,-365,25,90,-149,-247,26,-200,91,-302,433,-349,39,-46,460,-294,81,-66,64,77,-39,393,17,287,-151,-496,285,-387,-90,23,-67,-42,286,349,-10,181,149,67,-176,201,-261,396,-67,127,-299,-326,-120,-173,102,-385,157,-335,54,323,133,356,1,220,-225,241,313,74,-343,85,-269,270,17,231,368,-315,-42,-344,170,-73,82,258,-52,290,-431,-295,-259,302,30,-75,297,282,6,103,117,-2,-4,45,-176,-277,-245,-57,212,78,-154,-45,207,-49,-230,134,317,-220,313,-231,393,397,-356,-37,-370,-204,-411,4,217,47,-75,128,350,219,-218,-209,-153,-68,-184,-445,375,116,91,299,-417,322,30,-18,185,146,-254,335,-292,163,33,284,-7,110,5,16,400,-56,-104,155,309,-242,110,1,-14,-38,29,-197,297,4,104,259,276,112,-23,-255,-70,-326,-53,-354,-238,-50,290,-72,316,185,63,7,-450,166,-156,-182,-71,-51,-29,-357,-264,-248,273,-189,167,126,-180,-207,-126,-115,13,-502,122,-88,246,268,-129,275,32,-258,-5,231,239,254,421,-268,-180,-131,270,-314,131,100,132,185,-220,19,9,150,241,127,-23,-1,-62,334,-35,-251,122,136,295,-123,-125,-195,161,-28,195,-7,140,-243,-174,-56,-280,386,-30,-258,271,55,-340,-118,193,150,-255,-511,-357,391,330,279,48,-96,-203,-244,-300,-436,-5,424,95,-323,-62,361,120,400,153,-132,-45,348,327,-163,31,-126,-149,-7,45,-74,-40,166,-47,125,240,166,-120,245,22,201,-75,104,175,166,71,-35,-52,432,-268,-334,-29,101,-280,-44,4,-19,-207,9,-308,-307,117,180,120,-317,-162,86,291,77,110,-47,-321,2,232,234,78,461,-164,-60,-374,-21,83,-354,-126,27,-224,88,420,212,58,-152,-325,343,370,-415,70,13,30,231,32,55,19,-136,311,-224,-4,127,145,6,320,206,424,330,118,121,-120,319,113,37,-285,323,-47,37,-144,-393,-276,425,155,-140,170,-244,-38,-77,-26,338,-77,-9,-410,263,-307,-406,1,-287,-220,-66,-227,-296,58,10,354,147,394,200,-5,-246,387,-184,318,-265,74,282,-83,-282,143,19,-26,-10,-356,216,-258,-132,-28,-302,-138,390,49,-84,479,180,118,-29,169,-341,-160,-228,-55,220,298,-129,-9,113,-493,-252,58,-30,-345,42,-188,47,-254,-81,167,277,423,53,-123,-149,-68,204,-160,-85,-428,324,-345,436,-77,2,-169,299,5,-51,-399,122,-69,-267,-42,-67,4,46,177,335,-102,381,318,109,22,-240,-112,181,-116,-56,-282,-307,177,115,-208,-135,223,149,223,342,-102,-47,-218,-26,-253,-25,126,103,-170,1,-196,330,399,313,3,187,267,106,85,-400,33,307,320,93,-348,-398,-127,-174,205,-320,-8,-412,-117,-248,-67,-390,-188,130,-104,-160,-141,420,-54,34,-428,399,-123,-101,11,-77,293,-319,81,234,73,117,-154,36,158,20,-110,217,175,-23,-35,236,-137,-38,132,-133,-203,89,251,-277,-40,392,-114,-317,-100,353,264,307,-131,-247,-284,-314,-110,19,-221,358,-56,249,-37,-78,132,-245,181,-350,18,424,37,-134,336,-31,-375,316,-80,16,-21,31,-380,-354,175,-468,278,98,-109,-41,-1,-151,-73,69,470,241,121,294,199,15,209,262,-114,-94,137,230,233,346,-4,-325,45,37,227,-272,134,124,380,-7,-313,-99,63,298,-125,45,254,-407,353,-401,118,-156,-213,270,-391,164,-323,-110,6,10,-458,321,-25,-47,-136,-29,75,-155,203,-11,-88,292,-255,183,174,-282,2,122,-106,-271,-260,132,57,320,177,-26,51,-62,96,-191,-161,437,-373,499,30,-354,55,349,-129,115,22,15,227,-390,-352,-313,158,56,-200,138,158,129,81,-272,134,151,-19,-160,-143,-335,-234,-77,-41,-81,-383,-138,-160,-178,-22,29,299,87,-444,-95,-231,-346,451,-364,-60,276,-95,382,157,-123,27,224,-1,-118,-440,112,-231,-100,-31,104,20,-251,-183,-55,-126,-78,3,175,-139,-119,-173,138,55,233,47,59,78,-26,278,-58,-169,58,48,12,54,-16,164,-14,-188,-196,-167,172,-61,418,102,-61,-195,-263,-73,151,-428,-31,156,133,-140,318,-12,198,-162,-63,-328,-232,123,237,14,-41,-285,-160,-275,-306,30,203,203,-120,233,21,426,-301,48,-96,-349,-139,-243,262,249,89,-164,89,154,112,-167,172,44,-28,40,100,17,-139,-320,41,327,268,-183,-88,7,-333,4,81,173,259,42,-366,168,155,421,380,-143,190,-78,-44,-207,265,36,54,240,-156,-225,31,-82,-438,46,428,-261,-90,237,-34,68,-105,14,-321,58,115,57,-357,129,329,-251,-4,175,292,-74,90,294,-469,-446,-17,-146,79,364,-25,-212,103,208,-57,-48,-432,219,231,123,120,-213,184,478,166,71,73,-162,58,-242,-101,35,26,41,-208,3,-425,-154,-158,186,432,209,-278,0,78,-395,-330,-205,185,-53,116,-6,188,-87,-146,-268,98,-140,232,63,-149,421,-78,-258,142,-5,-213,208,80,-200,505,196,-75,38,89,89,-408,-252,89,-13,4,-258,119,-7,294,41,0,102,87,-479,-423,243,-200,166,71,56,-216,-120,-301,294,-374,-16,411,-32,149,48,218,5,182,-162,-209,-97,-71,137,8,174,335,-343,275,101,315,131,-277,40,-31,245,218,-156,161,-263,233,445,-199,-96,420,-490,0,-93,-133,-324,-24,66,-104,-367,-2,248,24,-238,-36,264,-44,-152,313,222,-155,-22,-219,-52,252,-186,406,217,-109,-123,-314,122,251,-16,-179,-290,147,211,-93,-229,281,93,279,-292,-295,327,-453,-47,186,-115,70,261,350,-30,-61,316,252,-270,398,-235,63,272,-52,-339,67,-24,-55,-277,322,-144,-293,434,172,247,318,-144,396,84,-42,-151,44,-180,452,-328,-202,112,220,138,-282,101,383,139,59,173,30,40,69,84,-79,-252,-38,95,-315,307,425,142,86,-107,283,-406,-72,-239,-70,360,215,-198,177,-71,112,-138,32,129,-79,120,237,102,-395,-164,93,305,-15,-153,62,165,-23,-128,-409,382,-99,48,104,-243,279,-187,25,-48,288,356,-41,-338,-353,113,-117,-107,16,61,43,283,-2,-243,-29,82,2,112,-203,114,-127,-73,48,-61,139,-9,44,-92,-114,-234,128,-276,-118,-322,278,-137,-232,7,-458,136,-64,-225,-236,136,-8,-142,-352,194,272,105,-377,-252,139,256,330,442,-268,54,177,-269,-228,274,240,-370,-40,-219,-119,-131,360,-59,-322,-272,-44,188,-25,56,260,-128,-378,-6,183,-132,76,6,407,-403,-299,206,122,-17,118,90,58,428,104,113,-401,-76,-280,415,-103,-39,21,-440,-118,4,-474,24,191,-279,-111,-55,-16,-33,232,191,-85,175,-298,90,-413,-63,253,11,-309,108,209,-332,170,456,371,192,227,-87,50,54,-14,245,138,213,249,4,-144,256,209,-21,-67,98,449,179,39,-42,237,97,-175,-283,228,-172,-197,-171,69,-117,189,-218,128,-260,182,212,-75,126,187,-361,-69,-79,330,-119,-291,-271,-385,-6,105,395,42,-160,-136,-219,-52,-242,-65,-318,-146,428,103,-101,-71,87,361,-348,50,359,190,63,-491,282,-163,32,67,359,350,-194,199,231,281,-265,42,25,208,-64,30,-158,247,-203,-46,-129,215,21,20,-69,-228,-67,-152,41,-282,233,-162,-271,-214,-126,82,83,260,234,134,164,417,24,-246,135,238,176,-52,-138,134,119,-288,99,-194,-145,-74,496,-53,40,-156,41,302,31,59,46,-308,-223,-193,-185,-386,-32,-40,-16,-190,-371,319,-94,225,-130,-167,416,263,-49,80,-289,275,412,-30,321,104,291,97,-106,401,190,140,254,-115,173,-23,-297,164,-121,-3,72,-433,329,-19,208,-293,-219,388,-169,-321,73,-316,270,2,17,298,210,398,103,405,-267,440,109,87,384,-128,-108,-28,-139,-65,-239,233,10,-394,-228,148,197,442,153,-75,3,103,-316,-51,-53,-205,87,-232,-59,-23,32,-144,-350,151,122,320,308,214,-151,-18,198,65,239,289,71,-446,55,-134,-83,401,-81,39,-249,-308,319,-219,72,-214,-167,262,-63,128,149,256,313,-11,-40,-355,-433,320,-437,224,-273,-133,49,-336,-45,265,-204,346,-157,-189,-94,-30,-79,83,137,75,-316,39,220,4,278,303,151,-211,83,259,-106,110,51,-393,-113,-218,-203,316,-266,135,-18,-112,-323,-99,44,315,-257,112,77,26,45,-169,210,61,83,1,-114,331,-190,392,-139,145,-56,78,-114,-29,27,-71,139,-97,-31,-72,-76,201,-123,-26,235,-289,137,-463,185,-375,-307,75,43,357,-178,-180,-172,184,-213,-461,343,89,52,45,-90,-343,-110,-95,322,-339,-18,199,-73,59,320,409,-166,-165,-105,126,-168,-379,198,-95,-57,-203,146,-57,76,277,-172,325,-241,-19,-79,137,-81,230,131,-162,-37,-198,-137,192,356,339,-187,-166,201,-138,-150,-79,225,-23,284,253,273,71,397,20,-457,19,-75,-107,-25,-16,-432,82,112,-279,171,161,220,-109,25,-282,-107,54,-219,32,175,-32,-208,-406,153,-169,209,308,-60,-123,456,425,-119,91,-233,75,-31,-42,37,-222,-129,-94,-6,-173,-198,106,201,-199,-325,222,-76,-406,53,391,-214,397,60,-25,141,417,-334,37,72,60,72,-262,55,-20,-136,-59,87,85,251,-129,-173,-264,-157,2,196,-215,63,129,33,31,-453,28,-288,210,-188,180,-89,-4,-297,114,258,-191,452,-35,254,-268,-349,66,-92,73,62,-264,-80,40,112,136,-145,107,-55,26,-125,79,123,-52,-128,194,-99,15,-270,-71,-19,-113,146,-94,161,377,17,105,221,-393,346,-479,-132,69,347,-250,-175,-242,246,79,-274,-144,-124,245,216,107,54,354,110,-303,-32,-12,-164,-40,63,72,176,-426,22,60,314,-246,225,206,433,116,-30,-362,240,-155,-285,-203,11,134,-111,-134,298,-251,230,-132,-358,271,166,230,236,-134,-74,170,261,-191,-140,90,25,-269,-162,-104,212,112,59,54,404,89,425,-44,-103,-140,-123,229,308,34,32,-89,80,416,48,63,5,33,-230,11,-390,218,-225,-215,-22,-40,-112,214,-319,423,9,-123,312,-126,202,428,258,-366,217,0,180,-173,78,101,45,-159,-100,-256,26,-200,-135,102,-188,438,113,25,-405,-88,-147,-335,-138,-28,-313,-384,129,-319,-19,-53,-233,152,-159,251,-340,-70,105,-99,14,11,-101,-124,25,58,153,81,-152,-343,125,-129,166,-83,-130,247,196,278,7,67,138,94,-263,95,-111,55,102,-223,-32,334,51,262,-314,379,-192,259,-110,5,103,148,4,8,402,380,21,-195,-247,-30,-270,-80,484,292,-13,61,234,-127,197,265,269,-26,-188,-227,-243,62,-446,-269,-113,295,215,135,37,148,-283,-61,-353,108,-13,21,217,-395,133,-40,-244,-89,24,174,40,-339,451,-77,-173,155,-221,-116,-76,-64,130,-21,154,-181,151,12,61,-1,-73,314,-273,261,16,389,-183,-251,96,-196,-100,-133,-100,-104,-214,101,46,-423,258,75,412,95,-429,288,-18,173,-341,-11,-269,-279,-460,-21,102,121,-49,123,-240,51,-47,-280,296,-276,71,177,-331,-103,128,-96,14,-228,75,-172,174,-101,-27,-94,-183,-175,191,234,369,89,-10,-48,-282,38,-131,61,171,61,-207,-64,215,84,127,435,193,-105,-70,3,231,244,18,115,320,220,-31,-122,-100,-338,194,52,-439,-442,153,149,221,15,-247,-145,176,-80,-73,-366,-380,-47,242,-14,-143,82,80,-199,-193,113,55,-218,205,229,-3,10,-267,-156,106,60,0,26,-136,177,194,70,-453,153,331,147,-119,-293,-97,-89,383,76,59,-245,-233,360,-175,100,-219,83,-191,-367,-142,386,80,-283,345,-134,383,187,-182,-215,-289,243,-101,29,60,3,141,-19,48,420,64,301,65,58,-75,-30,84,103,-33,10,-24,390,194,23,-54,81,-224,-342,188,-86,-6,-50,-455,293,-271,-371,-101,358,26,292,-153,245,17,-117,-128,-88,112,-124,164,-218,322,-396,56,-137,-197,390,323,157,159,229,-109,-27,-355,-428,419,-59,-249,-114,-209,-255,220,-514,320,396,196,-305,262,326,-189,116,15,-68,282,-346,214,-134,59,-111,-119,-194,286,-234,145,9,48,183,-37,164,-70,-159,163,-226,237,81,-301,-316,160,108,-186,273,-124,28,281,-72,-434,-40,-2,281,-116,-282,-199,215,147,33,79,-114,-105,-258,157,196,242,382,268,363,-139,-65,-43,48,136,-119,16,-159,-51,341,-313,-363,43,-264,-313,168,-286,-223,124,-53,124,-260,-344,-192,-89,455,211,-67,249,246,139,292,99,168,-389,143,-401,169,-375,127,-425,-27,66,27,197,-149,454,-376,185,-169,-321,-38,93,-8,-370,-383,126,126,65,-364,-234,166,75,-214,340,279,302,-183,-188,-135,-45,483,3,-41,435,-481,100,160,221,-428,282,-14,-448,107,96,287,-104,-302,127,101,331,-78,210,-428,-77,366,-339,-103,-105,89,-118,13,-32,-312,-67,-34,-392,237,87,-95,205,-462,-183,-77,-32,316,310,-295,-79,259,-207,-30,-54,-113,243,468,-136,99,252,172,-411,37,23,-212,-298,170,371,-136,105,-297,-71,24,56,26,-44,234,48,-400,-70,-61,-237,29,-238,-80,-183,260,290,-210,-263,-212,304,-95,168,-318,268,-335,147,-114,-388,399,135,380,-205,-154,54,-102,80,-16,114,241,-14,103,-179,-229,373,26,-24,67,71,17,-2,-185,195,374,-93,164,262,-247,-294,-8,270,-175,-33,163,-226,300,84,284,235,-207,269,7,64,206,170,-79,-37,-294,-353,415,163,7,-3,149,-206,-46,48,96,39,384,-257,-107,102,258,123,-164,128,46,142,8,-192,117,-249,5,242,-64,120,-238,-3,26,-121,-23,411,-100,-224,-34,-116,149,-271,-83,-2,-208,408,-110,26,242,-420,-445,235,17,-348,352,458,-75,-159,-66,-430,-340,320,154,35,-190,103,-224,91,348,157,363,-175,248,-136,39,35,90,6,130,-101,328,294,-147,-121,308,-9,268,256,447,270,-201,10,170,94,-204,-244,-192,382,-243,-145,-146,-39,324,177,-2,-102,65,-17,151,256,-248,47,-272,-256,-33,54,-85,-86,-265,34,173,-62,-91,-33,240,161,-156,93,-103,-450,-326,-151,330,-204,369,82,75,-202,145,-52,171,-92,306,-10,-63,-83,57,75,149,-268,-196,215,-423,123,-222,341,159,-127,-285,-255,-102,352,-7,-328,-202,-60,-357,206,-59,-280,22,-83,192,-12,-9,-141,255,-346,-114,162,-194,153,190,-106,-361,148,-254,-437,-263,177,268,-102,-20,-21,-398,94,-160,13,-176,27,-318,37,239,-304,180,191,-21,-172,169,-218,170,62,-225,6,-51,41,-194,279,-275,-230,460,-51,-239,-113,-199,-72,213,59,2,198,-249,278,-99,-414,-288,-209,-81,-310,-376,321,-366,-131,-122,-448,-127,-416,249,145,-490,-236,79,-53,337,-112,-94,-138,282,196,193,-353,250,-39,-165,-257,-199,205,-187,290,104,13,-308,-77,-211,-202,261,-397,-83,-242,214,6,-65,249,147,-223,-211,-5,464,190,33,116,324,97,396,-236,-102,-389,133,-305,-85,-267,-356,-49,-37,-409,102,162,-179,120,225,83,-296,270,9,60,68,-136,105,-254,224,91,51,145,255,91,-216,-12,219,-55,-3,95,-127,-233,-106,-123,-351,-173,-91,188,-235,491,-279,85,-43,-44,25,291,298,105,287,191,-31,63,162,-44,-191,174,-303,63,-448,278,189,-234,-416,-6,174,305,-34,-44,73,-54,-115,246,86,-150,40,-274,113,-70,-286,-189,-460,-57,141,31,195,391,-200,259,239,70,-280,76,-320,32,-384,-185,-65,-96,-351,-324,-127,32,127,227,-115,36,148,-10,39,44,41,268,-157,-303,47,-126,231,-204,155,42,-398,-128,60,219,-86,35,-258,-271,-115,-84,152,6,27,-117,-68,-309,67,288,11,-190,-142,-3,-141,-219,-342,-240,471,-239,-286,261,83,-161,274,156,231,-45,170,-29,-128,-142,151,206,-229,-386,-252,18,-6,11,104,432,237,147,49,44,114,40,-133,-93,-415,403,-190,215,384,-404,-165,-20,119,-53,-53,-8,-58,265,96,133,-406,-345,-42,247,-117,-208,-162,-106,355,-467,111,260,182,-246,80,-273,-125,68,-174,47,78,12,385,79,300,-260,-492,362,-23,-172,400,-367,33,-223,-219,-4,-334,-92,-85,-130,11,-272,-261,205,-15,-196,-7,428,-119,-37,52,410,-110,10,-136,-46,-87,-331,165,80,22,11,-35,42,-4,-408,-238,-83,-56,4,-429,-270,66,-469,6,252,-205,-228,-314,295,66,103,134,113,154,20,344,21,53,1,-326,407,-337,-78,330,-85,97,-151,223,-340,250,99,-317,278,-20,98,161,209,-356,-216,388,143,124,-346,110,51,-114,-137,411,-151,-67,-199,191,321,-21,-379,22,497,-75,30,29,28,-148,-71,-279,-169,38,-24,-147,229,-235,268,118,203,-184,-49,244,-23,-248,279,-141,-185,-476,-82,272,355,-180,-142,-154,-492,228,-205,-197,140,261,-421,-294,-406,-49,-221,361,-393,-276,-358,83,-55,-163,27,339,-87,139,-104,100,-200,122,-297,228,75,-237,136,190,15,-310,12,-212,-13,121,92,98,-201,-202,492,391,3,240,33,-213,111,-10,223,-29,255,-187,370,-22,245,-107,24,237,-91,172,93,147,-179,-183,76,267,-36,-494,-169,345,-109,360,31,161,-117,271,50,-228,66,-363,148,426,-42,17,-91,-257,-6,-99,47,-477,-404,-72,-274,390,-32,-343,76,99,200,122,-153,155,313,-80,450,-151,147,-7,120,-180,59,-254,-241,202,-374,237,-159,273,-135,139,171,74,99,-277,162,67,-14,-252,24,96,164,22,244,-210,161,-230,292,-1,-51,35,-117,45,216,-408,348,87,-257,424,2,32,82,-242,265,246,-203,65,-269,-197,-313,-3,-158,-232,127,-43,-196,111,133,-100,25,-92,-301,-144,132,-32,48,-348,-38,111,116,281,-426,-53,420,-216,159,-387,22,-88,-48,233,-200,-101,23,206,-83,-232,-105,316,-10,-183,267,-256,264,159,292,-92,-43,-34,-4,33,55,-144,30,-339,202,-296,-55,-28,15,119,-63,376,55,256,-171,263,-160,127,-278,99,204,372,246,-264,-54,-9,-194,-258,-463,147,-134,101,-330,-286,54,-196,37,-252,504,-104,128,163,105,273,-176,-181,-249,9,-149,77,-188,186,216,-299,-199,159,-174,328,-58,-74,-31,15,-297,25,91,49,-395,93,184,-10,-383,62,0,508,-190,-191,272,-98,-101,-292,-74,-104,174,4,-38,-194,-363,137,81,262,-213,-280,-245,-45,30,-136,171,243,113,26,-263,43,-172,-99,-264,-13,-47,122,303,104,26,175,-158,334,17,179,-95,148,-111,405,250,-21,-432,-24,-268,202,21,245,129,-73,-88,-409,-214,-220,-165,210,267,69,121,-294,66,-43,-125,-99,-17,-41,90,110,-231,-82,-54,-299,4,439,-16,-97,118,-381,281,-152,-56,-234,118,-118,95,44,145,-473,84,-136,-59,-7,285,37,-297,28,227,112,-340,-150,386,-69,27,305,155,-204,142,-410,-190,-194,-14,-75,97,-512,-277,134,-283,226,399,-212,83,-39,404,72,349,323,-486,73,108,-378,-392,-231,10,145,434,148,-55,-208,-102,-45,-118,-60,-210,-9,102,-269,30,-264,-104,21,-192,369,329,-163,-91,-19,-346,119,29,-219,235,401,151,87,-25,-134,18,262,-136,-207,-297,355,-297,191,56,-266,62,227,152,-26,21,200,-56,-20,-4,-29,-473,43,6,13,173,203,-231,-73,95,183,146,85,-361,51,346,4,285,-120,-142,33,-17,111,16,-501,358,129,31,124,-122,81,271,70,-211,-496,98,-283,286,-230,225,-235,27,-38,-93,160,247,-44,-288,-16,72,-48,-430,-66,362,198,-104,447,39,-74,15,209,352,43,152,260,94,-94,-362,212,-112,-75,-30,72,-25,-364,253,-471,237,-282,-105,49,-268,-153,224,-482,-80,-45,180,-160,-124,59,-150,-379,21,-23,-150,360,-259,-451,238,-4,-18,-247,93,-21,-170,-98,127,-90,339,-341,-28,225,-12,462,479,-273,140,17,28,-422,-214,-63,173,-220,-258,-234,-254,144,294,110,169,-33,273,-144,-363,-286,-374,195,44,436,90,-409,64,344,-106,48,-111,369,17,-244,18,-53,-178,143,364,-434,-63,-29,-85,-184,308,218,154,184,-110,9,90,-404,-23,-168,201,15,-251,265,-6,118,-159,289,-175,-179,-93,48,-72,-188,9,-202,-382,-258,171,353,-149,-69,-318,423,136,31,-273,-213,437,38,133,111,5,416,-264,35,64,335,43,-407,239,194,436,2,-418,156,103,-232,31,94,-76,301,-150,-96,-8,-370,178,229,92,-278,-323,62,-250,160,-132,-148,33,-187,404,27,-116,-26,-279,222,96,24,-70,180,-135,126,245,375,270,-269,57,131,-95,27,161,-146,159,-423,-24,-348,-184,333,343,-473,285,147,-377,-230,58,-400,236,132,272,110,198,112,-31,312,301,32,-51,19,-58,139,-268,-25,-79,-68,-286,-256,-325,393,-123,29,198,264,-11,-47,231,-440,-78,17,-135,-412,18,98,-13,-206,-196,225,233,101,-173,-429,-33,265,-145,-318,72,180,-285,349,-113,140,-87,-100,199,289,15,283,176,-43,221,356,-106,5,89,136,-203,100,-26,235,-182,73,-377,-240,-202,-291,-59,133,-204,189,-61,-185,253,-75,-317,16,-147,182,2,22,8,73,-235,76,169,-109,264,60,286,-281,201,-13,-116,-242,-409,-193,-278,-198,-254,20,-333,-218,269,64,241,112,494,151,-201,-106,38,20,-220,-156,228,151,-59,-109,220,427,44,-160,22,17,425,191,104,149,-310,-189,138,52,-447,331,-144,442,123,-262,400,144,123,32,-457,297,36,94,381,-424,90,-298,65,64,53,-304,333,116,101,-114,-307,-89,461,-363,-289,-158,166,184,145,-101,-80,-2,181,255,105,-342,193,-207,-371,232,-39,178,191,72,-155,62,-64,0,-22,-214,207,-391,17,-321,211,-410,16,140,216,-223,163,273,156,-152,112,176,261,-97,49,-83,-57,-208,63,-1,185,173,137,236,-168,-378,295,-184,-163,17,8,229,-74,-320,51,41,-270,329,-68,-166,-32,-39,-179,279,-26,308,91,407,-102,-228,361,267,24,330,-320,68,85,251,39,409,15,222,-33,50,-439,-371,49,-10,-93,-65,-370,217,95,-95,-371,-245,242,-209,112,-36,-8,-198,35,314,355,-447,199,100,391,-179,-224,-12,83,172,-93,162,-136,30,74,-398,-3,-200,378,-434,-266,-114,-320,-109,33,3,-192,87,-218,180,36,-318,0,-321,-27,-1,242,-318,-37,217,-82,200,16,252,121,-159,-247,33,43,271,-291,-34,256,34,151,-449,68,265,98,-57,-350,-134,76,-158,-27,92,381,331,108,50,-108,-272,287,410,-233,-167,66,441,-283,125,133,132,207,-413,260,-280,-1,-36,-156,61,-121,-20,-79,-6,-341,-62,-24,-127,46,479,-48,-387,206,400,19,-436,-135,-254,-206,320,-124,-85,199,-73,-21,-110,408,-89,-180,215,-40,-16,-124,110,172,-116,54,240,-44,-180,56,323,207,212,4,338,281,299,-167,212,-86,-170,-275,-174,-6,-208,-21,263,-73,238,-95,-274,-33,203,66,264,-414,-443,75,-4,-45,-144,132,152,-158,-41,-354,117,148,2,109,348,178,79,-26,-105,258,7,-75,-306,-156,-400,-248,-364,87,187,415,-95,266,-342,107,-413,244,-172,21,147,-304,31,214,-121,-112,-243,-460,-160,162,0,-336,94,243,-154,65,-324,-258,-271,-45,158,-385,52,-268,202,90,-82,230,-341,-54,78,91,-67,72,-135,319,7,109,30,169,-125,43,-85,30,-220,101,-311,287,96,-106,36,351,-58,-400,-183,343,-130,305,-12,-366,-47,-395,-174,-19,102,-324,-250,-127,-410,88,-147,124,-273,72,12,-109,262,108,50,-302,412,-38,62,415,-136,32,-268,-165,74,331,-98,-277,-122,203,84,-443,167,334,102,254,-74,121,-50,55,-319,-24,52,218,-277,-222,241,-10,369,-147,-42,-149,39,-215,-276,-68,-218,307,313,22,-289,-147,138,-445,182,-291,171,283,1,193,205,304,-48,264,-156,-51,-68,-162,-322,-206,-185,-119,-53,-356,-229,258,19,91,-246,121,-428,-237,78,35,-48,-232,-306,373,-157,14,-77,-200,371,-408,-98,-256,39,201,162,-97,-24,164,-187,-174,-136,279,109,-59,67,13,-145,180,-98,79,-309,295,-192,-120,56,403,115,22,-163,59,-81,-83,-9,66,250,330,-175,331,389,-62,-76,-126,30,-40,125,392,-109,333,-210,-108,-303,-141,134,220,-128,321,182,67,304,-179,-392,-303,347,-317,-170,489,257,-144,75,-317,-233,-254,92,-141,266,-24,-154,240,-287,-13,212,-415,-204,137,136,-120,-122,-307,-71,-197,-95,88,-18,187,-312,373,199,-62,-175,-283,147,-90,276,-141,-12,216,82,-47,-169,143,-105,-148,-362,-439,184,232,-108,247,-473,-329,38,212,-350,25,25,285,206,-142,342,-33,122,-255,-440,-193,283,171,157,-424,222,145,-109,166,-178,-292,-73,-115,-12,23,217,256,325,38,-42,-197,-353,-148,-15,-50,-103,316,145,141,-79,-199,-255,-20,250,-221,201,-311,105,-181,408,171,-341,-67,228,-73,-74,-321,89,-198,396,-183,84,-305,289,241,72,-379,282,75,44,15,-423,-327,71,63,-377,-313,-268,-56,429,202,-318,155,-231,89,-52,-237,-9,-234,88,155,91,-260,480,-25,130,-72,166,-437,-125,-10,316,192,294,196,-161,11,-72,212,74,-302,-133,-97,219,15,272,84,-324,177,69,379,-22,-5,223,14,-210,-323,-262,-103,4,89,196,-346,120,354,-270,171,-381,136,274,56,-259,-213,382,393,158,-137,-268,146,-204,104,-369,-208,-205,-22,-329,266,42,-47,35,-192,-389,292,11,492,53,36,-382,246,111,-343,432,-98,-176,-466,-22,-328,182,-66,-58,136,-25,56,370,-1,-15,-2,81,167,239,377,223,-86,57,-78,-131,52,-167,-119,-150,-12,27,-413,-449,213,155,57,20,136,59,356,85,-6,392,56,-97,70,-24,37,385,-335,127,145,111,-41,88,-338,13,-106,-57,2,-106,-58,58,156,-445,-376,307,174,20,-19,336,135,-26,108,-442,3,203,-38,88,-68,146,346,459,-193,-144,-272,-305,233,-282,96,-88,197,248,265,-44,-57,171,36,72,-394,369,-137,-298,263,-401,-325,52,12,-28,-92,-20,-87,102,182,-245,231,3,3,242,-210,-349,83,224,-414,36,-169,-221,-257,-44,-191,-286,-156,152,404,238,80,219,-306,-155,-5,304,-177,-395,-49,72,-60,9,-347,-355,-314,-242,356,-135,-390,-96,3,-88,12,-127,170,163,107,-153,288,95,-78,-4,-43,4,-126,-77,-128,264,432,38,31,397,339,12,118,-344,-114,-56,-98,97,309,-50,115,-342,-199,-72,-56,-261,-36,-96,124,449,109,-279,-372,137,39,-98,-346,31,-180,201,-339,-213,-95,-273,-144,-355,29,-74,-191,-3,223,-134,6,37,-424,134,-15,-278,136,96,96,-400,-12,45,-22,-521,258,-366,178,323,-135,-9,-213,33,-90,24,285,11,209,-283,-127,182,-203,-267,-156,15,89,-26,-15,-36,-102,-52,205,22,365,131,231,-170,22,212,463,35,239,-111,-164,-185,-375,-464,-352,-201,-69,-291,-117,21,-136,151,-339,144,228,-147,27,164,344,369,415,143,-257,64,360,-318,-141,256,140,-463,-393,238,-111,-245,276,-149,-125,-9,-370,-409,7,-171,175,-124,34,-94,-234,-181,-68,224,392,-170,-97,-61,7,378,110,111,-241,-157,-124,-35,-193,-13,4,146,-369,334,218,130,-138,0,465,71,-199,-317,156,-398,290,86,3,293,-143,-67,-113,511,-270,-85,-268,-407,-322,68,-157,13,20,-77,-21,-448,-212,-98,-176,-370,-129,149,123,-493,-248,-58,67,421,-23,-26,325,221,26,-14,92,75,363,-95,-325,140,-55,-28,316,110,123,69,-53,-138,167,-22,-7,136,-189,-158,45,-259,102,8,-161,447,-71,163,-242,164,55,-106,-324,-254,478,244,-73,61,-101,155,286,189,-105,68,-134,186,222,-31,183,-226,-279,-78,-66,-103,205,258,164,308,92,277,-23,-275,-156,154,58,-162,26,-270,-371,-39,237,-371,-462,-76,-234,8,33,-49,-2,-280,-234,80,143,-128,184,239,-319,-272,-302,66,311,-266,-356,-109,237,-45,-369,-305,238,-412,-189,-29,25,104,26,488,310,212,-416,-229,-270,36,255,45,40,241,323,100,-74,-384,-21,-175,-57,-109,-52,-248,306,252,-159,-336,-324,276,-27,265,41,394,-71,-124,-231,-320,-152,-250,-377,-402,219,-50,-25,-388,13,190,-241,-351,274,119,-206,6,-176,308,-45,-171,289,-114,-200,9,-132,285,-382,120,-13,-6,159,114,-58,-3,-423,-65,181,-13,259,-119,253,79,-63,382,26,62,184,96,-142,-272,-90,-308,-121,-479,-67,-50,-181,-66,-266,-39,287,-6,-133,-73,313,-51,-87,-128,454,324,57,49,69,219,279,-282,266,-18,3,211,-158,198,-168,-361,-11,-37,-103,67,-26,-81,-150,207,-137,-183,413,-81,-148,-76,144,-219,-322,-119,236,-92,-24,-43,3,-300,-125,-3,192,-82,-83,155,102,477,251,-390,-415,-19,54,3,-53,283,-179,57,301,34,-388,-27,-102,-152,127,-83,-227,-73,-71,8,-477,-254,-109,-173,-232,159,115,-233,136,-17,-25,6,29,426,-89,74,29,206,181,-44,-167,-23,446,42,204,149,309,390,209,195,210,129,-230,-203,-49,355,-117,38,221,90,-174,130,-239,438,121,-223,-71,271,58,-251,119,181,-196,-26,-161,-220,-191,423,-424,93,400,-30,30,99,-240,202,-364,-227,-167,207,62,311,-16,-82,184,-92,-483,-158,-277,67,-103,44,-51,-58,-48,-22,-219,98,-10,-16,140,31,198,-95,-104,16,41,-264,-330,-86,101,33,-146,-91,56,43,-182,130,-192,-332,384,-137,7,217,-96,16,364,177,489,319,47,-175,293,-235,18,40,-26,-359,50,-259,42,19,-353,-9,-266,-5,411,1,62,-403,-13,335,-129,130,96,213,1,-264,129,-15,272,-188,-39,219,-178,89,-49,105,115,-340,166,-309,-164,-193,285,280,-102,-72,221,-244,-186,-127,-85,-282,73,-128,164,-132,284,-153,-271,135,347,128,178,298,-261,155,30,149,340,2,37,212,262,253,-108,326,-63,-409,-37,-223,149,120,99,-211,120,-243,-97,44,341,-85,360,-180,312,-50,102,114,-288,-445,356,43,165,-49,147,38,84,115,-70,-252,109,-12,-249,-36,72,-186,-3,274,-75,-37,226,240,116,203,68,-25,62,213,385,-376,-67,-388,-15,67,96,501,-327,118,-71,-367,103,208,-282,-38,-304,-105,33,71,50,74,-27,-128,7,-27,-95,46,303,227,-167,444,-72,63,-297,-370,236,441,1,-54,-27,26,-389,374,-291,146,477,66,-40,13,-302,-163,-17,148,-6,-127,-69,-73,371,82,52,189,-71,-242,302,108,-158,-201,-258,-62,-400,-63,-260,443,226,-103,-8,-198,-172,-164,61,-54,193,45,-96,-183,-201,182,-403,-238,130,50,241,192,-16,171,98,156,-116,491,-307,-65,-439,128,120,-64,-170,187,191,116,115,-141,-35,108,-94,118,243,-23,135,-308,33,32,-226,-373,-270,-126,-87,227,118,-403,-202,20,-246,123,76,489,-148,477,-13,127,33,231,-318,-460,-335,13,214,-424,-183,-2,-255,-48,-219,-73,-166,148,279,-135,-246,196,40,6,208,-14,90,-267,-54,-516,-269,247,-332,-288,51,4,-225,394,105,-120,-230,-259,-96,-126,312,-32,245,-185,-251,-9,157,-500,-18,298,-351,-429,-191,-230,-55,-207,473,100,228,-74,-99,-196,228,38,222,-137,76,-276,-9,117,37,178,-175,-148,-145,-443,303,-160,78,-33,130,410,-370,-34,411,-413,310,174,110,358,-27,17,-355,-200,-116,229,-32,416,448,-94,-267,-106,327,-237,-109,127,-22,-129,126,-83,-397,141,283,-110,394,-169,57,-126,-451,373,307,-331,-151,0,-5,-170,62,78,-61,69,152,73,-43,-93,-292,-330,-43,443,-43,73,-24,-94,-349,-311,363,-72,204,-355,120,147,87,75,184,217,-50,40,2,249,-380,10,-28,302,-96,69,135,280,-94,191,63,-229,26,26,310,-253,-74,339,-242,-40,-41,46,285,-291,150,-126,-323,138,144,396,-242,-43,141,-193,269,421,-174,136,134,383,-195,-336,18,307,0,511,-41,31,-188,170,-185,324,250,180,344,-382,235,272,-432,-50,-60,-138,160,269,41,-80,-23,-13,266,-424,348,-152,228,174,-12,333,381,9,-164,41,-132,90,-34,-76,-360,-21,44,-63,-419,205,-28,-1,115,-316,409,209,145,286,330,-334,137,288,365,-382,-30,-417,10,405,251,-10,-163,-182,-397,198,-253,-85,308,177,135,211,144,89,221,169,168,95,99,291,298,-406,239,-297,-112,-343,169,-371,58,125,-209,76,31,-332,416,-76,-60,-381,-61,38,-264,429,129,245,-446,-12,-336,60,240,148,75,-385,249,-218,386,13,153,57,-333,52,14,-165,118,204,-236,-80,-157,-264,-115,104,-123,-228,-185,-19,126,71,336,-216,-297,341,230,8,111,355,382,-46,-489,18,-29,138,214,71,161,-389,-260,-261,49,127,50,-247,-241,-27,-45,246,253,-346,152,-11,97,-350,-355,-295,-177,-177,25,18,-92,-13,58,207,-192,-432,1,203,41,-348,-255,351,380,311,-63,177,53,144,-263,367,-487,100,-268,206,-217,-128,121,-374,114,244,-135,170,-199,-12,238,11,-236,28,-290,187,74,-54,121,457,214,-2,193,78,-89,152,-151,-87,401,-57,-233,259,-91,-309,127,133,-50,-121,476,17,168,-38,321,144,379,-114,113,-394,342,212,407,-100,40,67,-18,59,-219,182,-91,44,189,-137,-410,-246,-173,-21,-49,-28,294,296,-132,-358,331,-136,38,-26,-170,71,305,253,214,-117,-128,234,126,-108,217,173,362,34,248,-133,129,198,-220,-3,-110,430,-231,70,-299,255,342,181,-170,47,112,204,188,159,-61,-166,-385,56,-106,-147,302,-119,227,-149,-147,188,-346,189,-295,52,189,147,-60,-307,279,-52,400,128,222,-205,-147,64,18,121,146,240,202,223,-110,205,160,327,160,-223,-143,169,322,-50,-12,223,-231,108,220,71,-211,109,32,-478,56,40,-313,-252,423,-60,129,-191,-218,-47,156,384,87,51,45,27,107,-217,-164,120,-196,-47,210,108,120,-235,10,-295,-382,-87,83,-418,-14,-243,242,-121,-113,4,-270,223,231,18,515,-220,-4,213,39,367,75,-21,187,-238,-415,-332,-1,50,-356,94,457,41,-387,258,-99,85,145,-57,222,-397,208,58,37,-43,-37,177,4,133,-58,-375,-280,178,-410,192,135,-131,60,418,355,373,117,-139,92,32,-307,68,-132,175,105,83,-188,-14,223,-192,375,-58,167,-165,45,-330,-106,311,65,27,-128,-115,-100,-20,-350,24,216,-146,-301,388,68,-236,-79,-390,-25,-340,-221,-156,-45,-289,-158,-281,-89,-30,-145,212,270,61,-388,127,-173,248,275,349,173,31,-87,-110,-78,-142,79,28,-271,79,-168,8,-34,12,-53,209,22,-206,-269,483,100,280,113,250,-68,-121,-242,430,-346,-256,-121,151,-164,-271,407,-191,442,-135,-247,-262,-227,67,138,271,-336,-278,128,-78,-288,353,237,323,103,-33,139,-171,100,-58,456,-87,292,98,234,316,417,404,158,35,-47,297,274,238,269,169,-38,17,36,-52,-16,309,163,-69,81,123,-333,-344,-329,208,-52,-262,72,-335,235,213,11,-69,-430,-124,102,-409,323,182,-133,127,-19,-69,397,-254,-136,358,455,-138,65,151,-199,-133,-173,242,145,314,54,-51,49,-267,414,220,-421,-351,91,-401,-64,274,43,149,-257,-146,333,-239,2,89,-96,-35,119,214,228,104,-161,-289,-30,-173,161,-117,-403,115,312,-115,-124,-3,56,206,-232,145,285,14,-229,-48,1,294,252,-110,66,-120,-312,21,24,-278,53,-365,122,227,-391,134,56,346,-40,131,-105,222,-10,-90,236,-127,388,233,450,380,-310,153,285,-127,33,205,-306,203,154,163,-114,-318,-237,-20,47,156,169,-148,66,140,123,123,326,-329,253,-236,371,-358,-206,-5,82,84,-40,-190,204,455,-167,33,188,-40,353,321,-318,188,440,-295,179,436,-36,114,65,15,159,-286,372,-255,-105,-110,51,80,-170,136,-183,88,-64,-62,-72,-61,14,185,332,-270,314,58,292,8,-9,501,-82,-48,-73,-58,156,12,183,425,337,-189,140,-53,8,-29,-225,-287,200,121,-216,132,217,87,227,226,-149,396,-24,-94,359,360,-230,269,-161,10,172,-234,-16,213,123,231,-123,77,-355,342,-73,-113,-16,-252,-202,-256,277,264,71,186,-151,-196,399,297,-89,86,-64,-52,-350,334,-313,-34,-148,259,-225,-313,-208,-129,-310,-10,-149,114,163,-37,82,-109,-1,39,167,113,-12,-220,-47,-88,288,-179,70,-350,463,-65,-185,14,-131,452,416,35,-309,183,-67,-108,26,-102,-11,79,406,-77,145,392,78,198,-173,-363,191,-64,27,-226,15,228,47,437,33,270,-157,-187,-278,59,37,-33,53,0,-217,40,-117,-504,42,-402,336,-29,-418,-194,71,-104,-219,-112,-151,203,-397,-352,-8,217,128,85,-5,-15,-177,198,135,337,473,-177,-108,189,-66,-158,-43,401,-202,225,132,141,247,-288,8,148,141,142,268,-196,104,312,-14,127,208,402,10,43,-413,101,-112,-349,-257,-83,-129,82,38,-29,-345,42,101,99,-334,-325,171,82,45,239,-164,-124,-263,-276,99,-75,-139,-58,160,-83,9,-330,-280,326,215,123,-211,144,-318,-40,-133,-290,212,-82,136,341,-95,138,-19,370,247,-86,501,36,57,367,404,-206,-69,-4,-232,81,293,-85,-14,0,462,243,169,-252,-87,396,-219,0,218,-71,-168,-203,268,-307,-235,-11,72,-132,-64,25,110,26,240,-203,-323,-200,95,113,10,-314,202,2,123,157,-257,-327,20,-248,-357,59,377,-321,-309,-317,-214,212,89,285,-20,-333,-359,-388,-378,456,-261,-64,248,-114,46,41,-119,-261,253,-390,238,-51,-4,-159,109,58,-178,187,445,-357,79,40,64,-56,-103,59,-84,-196,117,119,357,-93,-254,-189,-26,173,-88,64,11,29,-278,-413,90,-356,91,-296,-106,92,123,52,-131,-2,331,-195,172,-85,-8,382,100,-122,-89,453,105,-60,14,293,-220,249,338,205,294,-54,-304,-182,92,-445,157,17,-106,-198,-59,175,-83,103,-143,-146,-43,396,-315,-342,-72,41,-30,-109,59,-145,-288,-275,16,217,156,-373,-7,-329,-50,16,-220,-402,-173,-281,148,-439,346,-261,-264,-34,-352,-335,-143,31,179,-82,84,-27,135,-224,-203,-222,-87,-482,-97,-289,93,-218,156,-78,-148,-239,-74,-57,-91,-127,-258,326,208,-316,-219,-222,-1,191,309,-151,127,-107,335,158,-53,-270,-249,124,-10,176,230,-128,-153,99,277,-114,-147,-250,-400,-155,-272,69,169,228,22,378,-206,-12,-310,-413,118,-491,-356,-207,-80,-53,445,315,-413,-311,375,-147,-408,22,201,-350,-78,-38,-319,-13,291,-253,-322,113,-15,-38,-226,-118,-456,-497,239,-11,179,30,12,22,207,-25,92,-328,-357,456,139,-180,-109,31,66,266,-296,-340,-309,-144,-196,-395,340,-235,-39,25,-235,242,68,-174,-220,-356,-12,451,-5,-61,-33,-59,471,148,-175,-20,7,197,400,200,-369,219,116,297,86,-107,-98,89,-117,35,-148,169,194,-84,198,55,-230,-25,-228,134,-134,-47,-27,172,120,-169,-393,154,-110,300,-200,152,-50,-7,-8,48,-258,-373,-84,-42,-122,-133,-205,349,215,-367,-96,-128,-299,421,275,-121,13,186,-293,114,-25,464,23,74,45,-256,160,76,-233,-80,-228,275,358,345,34,-392,353,130,-136,-398,-157,-295,-233,-70,-405,-124,-202,-314,-164,-81,150,23,-505,-350,20,-378,-143,-51,446,219,466,-92,-281,-113,-218,492,83,-79,87,156,322,-24,-245,360,-307,47,-45,80,425,-105,357,-450,-42,378,-185,-166,-383,-46,402,68,49,205,171,-28,-53,107,248,222,22,-394,237,110,-396,10,-363,-97,-31,-233,-279,-179,-194,7,61,45,-119,-226,256,170,-418,49,379,226,-182,-120,4,202,-30,-303,156,-228,-239,-386,-152,-82,-285,273,45,-370,71,147,188,-465,139,-260,-19,30,5,-274,30,-405,-168,-191,-2,203,255,-124,-248,339,-113,433,-96,33,380,363,-73,7,128,173,103,165,77,178,-174,-146,-338,171,-170,-19,215,190,312,130,-101,-329,-199,-228,22,331,9,17,-75,426,-220,-107,188,145,-340,-12,-185,-31,55,81,-61,501,-94,-247,-72,-6,-39,84,-292,-102,39,-267,309,120,23,184,297,17,164,117,423,18,-369,-275,-49,-429,4,-163,-134,-37,290,-295,296,-250,-303,49,-144,172,177,-161,-249,230,240,-194,271,-3,-114,-189,44,198,-175,-81,25,12,402,228,-451,-98,-69,209,-196,257,-319,140,-308,-29,-117,46,418,-52,168,-43,313,381,-237,-102,453,-182,-230,235,79,-159,127,-102,43,-116,-83,-50,352,107,-243,143,29,359,-328,33,386,-251,456,-150,94,407,-51,-172,9,258,-43,88,307,59,-2,161,-79,264,-82,214,-213,-68,-117,-186,-67,-188,-132,-243,371,200,-237,-407,-251,-35,189,189,-130,147,-7,-180,284,12,-123,-369,-319,-33,346,-134,322,-224,-352,67,-416,229,-234,-401,-149,118,-27,233,50,43,218,-130,-410,200,16,-43,-77,-216,409,384,-69,243,-129,-24,-37,417,-211,-98,-53,-25,-91,99,199,289,46,-16,141,445,100,-406,-3,-135,266,220,225,-357,403,-328,37,2,281,24,-244,77,82,-25,-284,197,168,-207,-359,-32,312,-214,-235,-76,-77,-68,-425,22,-434,403,53,26,-82,-62,-321,-135,-125,-56,-5,-326,-514,-22,6,-435,151,172,-4,-421,-68,246,-376,274,136,-298,186,-151,21,1,-102,210,-25,102,-420,293,508,88,-329,25,-81,234,-2,6,-64,-38,212,-181,59,-336,141,-85,109,197,75,49,238,113,-217,285,160,53,398,-8,-282,163,-81,54,-195,-285,300,-305,2,141,371,-284,26,177,-86,306,-352,-108,-218,-64,212,127,327,168,38,-433,-371,338,361,102,259,266,-324,-319,160,218,65,365,315,463,-361,-288,141,-156,263,-34,435,-342,140,117,-119,296,132,-138,-148,434,-374,-82,167,-389,287,-296,2,-268,65,115,-193,-1,133,-286,-228,169,-118,460,-366,-2,183,-81,-199,-58,-15,-160,353,149,-166,-3,-384,-322,-152,332,-139,194,340,-136,-44,-49,-40,247,397,-35,1,-326,215,67,-164,206,-337,-30,-9,85,242,-101,131,254,-312,417,-88,120,-453,-254,281,107,-332,-191,-363,125,-293,93,-68,-52,249,9,-159,260,40,191,346,12,10,-163,-133,86,-75,154,121,-51,16,214,92,37,-114,88,-95,139,365,8,-182,224,-313,-63,188,-43,64,-416,319,-296,-19,-276,-1,-255,-118,-268,-65,196,234,-359,78,-64,284,-383,-60,324,-18,-119,66,115,-161,6,-153,171,-150,-381,-42,-2,-116,212,-129,239,-380,108,378,202,-362,-223,-223,220,-87,-397,194,-371,-2,398,165,216,248,109,290,147,-134,309,9,-208,-206,95,241,107,-447,-263,61,-335,439,49,5,12,306,164,-342,243,-36,-4,138,461,-383,133,-13,5,-41,-404,179,290,-153,-252,453,-16,43,-157,51,-284,93,343,303,62,-190,-219,-340,-24,-260,136,223,130,-317,-324,178,86,477,324,-426,346,-200,405,269,-400,-279,-220,-115,-348,-59,-34,-23,-248,147,106,32,152,-106,258,-266,103,-419,73,38,-272,-113,-70,96,25,450,190,-397,477,398,40,-97,-107,-298,263,64,243,222,-114,-360,280,145,-334,228,-326,35,-96,311,34,-95,-72,136,106,-303,-53,70,-139,157,289,-53,-397,172,178,-256,-65,-92,-141,-124,316,126,272,222,10,-388,-276,-110,-59,198,-244,200,-59,-155,-183,102,94,168,-134,-293,-162,26,4,-8,-4,198,141,200,44,-179,253,252,228,124,6,271,-208,43,-257,-15,41,-324,55,-68,45,51,59,119,7,53,-155,-111,-126,144,45,-225,112,54,-420,-210,210,16,-29,355,-36,-198,-55,-144,100,67,40,-86,157,398,107,-164,265,-331,-193,20,-115,87,-76,-19,-99,-140,-63,61,7,-190,488,-137,159,272,-243,32,399,-234,368,118,0,-26,319,71,405,-55,-347,244,46,233,-114,134,256,360,-330,-111,213,151,193,256,123,-183,148,-58,-202,62,211,-54,-356,-47,-116,-269,25,47,145,-160,307,172,-275,75,-46,167,384,-491,-404,327,-129,167,-43,-371,-94,-174,-321,-102,-113,-31,240,336,68,20,321,-128,-52,-117,272,-275,-190,-87,-148,-40,-89,163,-64,-288,203,-149,-254,-44,-100,163,228,-7,211,73,61,-41,-150,-168,28,-239,103,242,214,-321,-73,-324,66,-33,-255,-80,248,-377,-488,-5,-117,-431,114,245,-144,321,47,-90,128,-24,287,-26,118,-173,256,288,69,-250,-321,101,-116,37,-133,-199,-60,360,-141,264,105,122,114,-211,160,63,8,202,132,-468,348,-324,-188,-291,448,-13,347,49,78,-111,203,31,-24,227,-114,-135,-494,-108,13,-311,39,-24,-274,185,103,-141,153,-64,-197,23,-150,285,240,-198,248,174,227,-69,38,-62,-233,301,-104,-145,51,-138,77,171,-395,-436,122,-90,145,-149,-162,-194,-102,92,-276,437,168,-289,226,302,61,20,-115,-270,262,-405,158,29,-98,130,271,47,154,49,130,-35,188,-87,339,73,-78,-149,140,-420,-466,367,-108,-118,-316,391,-262,259,-46,-374,91,143,8,14,109,237,-10,-264,81,-61,114,-55,262,352,227,-63,-69,-21,-424,-319,135,29,-25,108,-253,211,-151,26,-102,209,294,-169,-136,-214,130,64,-331,-171,390,39,-78,221,278,-327,-15,-202,-39,45,-35,-57,-87,-115,35,-132,53,-109,-35,143,-145,-214,80,77,89,18,46,-358,-140,104,-20,64,157,-37,38,-387,304,-366,0,77,-31,-33,414,72,-219,-173,57,82,193,-256,78,-401,293,-92,133,-101,303,171,-206,88,232,-514,268,328,-313,-172,381,-95,-50,126,408,0,202,33,259,-381,-27,198,271,-25,274,-237,385,377,-378,67,188,323,354,257,294,231,-39,147,-374,-99,-250,142,-120,-425,6,136,190,471,213,-7,454,-292,109,-42,128,-44,316,276,-306,-47,-81,9,-417,226,-277,-100,433,228,-6,-66,332,120,203,126,133,-65,134,10,112,17,-109,-188,-28,-218,-165,-21,-75,-261,-378,-64,-454,-63,-155,286,384,10,111,-273,132,-410,-44,121,-392,-217,-89,29,117,213,-116,448,86,-238,-114,-397,-50,392,237,404,20,238,-210,282,113,198,168,231,-160,-122,111,-119,32,22,81,-115,6,361,159,386,-88,93,-49,2,65,3,298,207,1,-95,167,-176,223,279,56,-84,63,-142,327,251,-18,25,472,251,57,-169,-150,267,250,148,53,203,217,-401,-169,253,215,20,-199,276,136,-104,159,298,69,-463,265,-2,-321,-147,-226,74,-315,-14,-65,343,119,-285,-418,-336,11,-339,-375,317,22,-18,-441,-6,-235,452,-328,44,-48,-344,-314,145,-308,15,150,-101,-77,153,371,6,52,-58,-217,208,38,5,97,38,-10,99,-183,-104,-240,24,34,-97,75,160,169,170,212,-145,114,96,212,207,-188,278,357,-371,-153,-227,-65,83,-106,157,102,196,-187,69,176,-78,-87,-90,-89,116,-370,66,-23,495,-425,-154,472,139,-193,-25,297,191,-191,125,-159,303,-221,-78,-253,228,297,181,-475,4,-136,-72,-60,329,119,-104,335,213,406,-90,-388,178,-202,-356,-186,-271,130,5,-348,285,131,-205,-190,-61,-159,-297,200,-365,-346,-214,0,12,28,-210,384,74,233,-79,27,203,-126,8,-285,-96,170,36,12,368,86,10,-90,-98,-253,400,325,253,39,-179,233,-113,-33,-117,-40,129,250,75,232,268,210,-131,-299,-418,-47,-392,82,423,193,-162,177,417,27,400,226,-23,95,-45,407,413,-35,347,-117,14,-55,465,35,-90,88,95,-245,174,-65,-291,-100,274,-402,453,-321,193,-87,219,-137,349,23,209,-42,-212,-116,85,134,239,258,-196,173,-222,-84,130,-36,300,210,90,-149,-89,352,218,397,-5,296,-196,221,-209,-185,295,63,19,440,114,-251,56,0,-108,-192,-145,-319,6,270,-497,-48,91,159,95,99,-314,-106,110,343,442,316,-60,-22,-203,223,29,235,368,191,45,63,-223,-153,88,263,254,-256,64,-117,-245,-317,-290,3,256,-217,245,-190,224,-177,-141,-37,-248,261,-80,27,81,-310,13,-134,165,120,-4,132,-280,173,-64,37,73,169,38,294,261,-57,363,365,419,387,26,-184,-64,-121,-267,-27,-281,333,29,385,148,205,-105,-312,306,250,-207,-213,-344,-198,-227,284,-256,38,-256,288,266,-100,-412,8,172,178,-159,-24,-327,26,61,-311,59,-290,131,134,-32,-388,-39,-154,-96,-80,-8,230,-264,-47,13,170,-16,-216,407,-198,425,-314,-152,-217,-97,-84,211,-209,141,-133,-41,297,-68,420,-88,8,-241,-265,-391,-106,316,-71,139,-59,37,135,-161,154,192,31,343,261,-80,-340,-236,296,-68,266,-396,415,-381,-84,68,17,469,285,-117,-118,114,-129,-424,45,155,-352,335,-388,461,52,421,153,-57,-175,-363,206,48,-361,-28,-251,157,-40,-49,173,233,178,-112,-87,-165,26,-276,267,165,-326,-120,-44,66,-234,21,85,5,-175,-79,-14,27,-240,-223,321,-114,-378,348,-284,35,-189,31,234,-127,-308,193,38,84,-43,-17,96,-144,90,205,-81,279,295,-265,-240,228,-15,-243,379,-104,274,-238,22,268,-171,-109,-267,246,-39,303,-292,-216,46,197,-20,291,-237,-123,-32,67,-100,-223,81,-51,280,81,49,-38,94,-88,-329,65,-49,-342,224,318,33,133,246,382,-387,-10,-267,-82,284,240,-15,-301,100,-117,99,-108,-310,-128,-242,-172,63,-175,80,62,-161,297,-440,-34,91,-112,68,-20,171,321,13,-134,37,285,-7,-166,-268,-218,-5,202,1,87,140,31,375,370,-6,-13,200,439,-254,53,-99,312,-419,204,247,128,-160,-144,396,44,393,32,-43,-142,-196,268,263,-6,402,302,38,198,285,-6,49,129,218,-452,266,-6,166,-124,-346,-50,175,138,-82,-277,-446,102,115,262,-263,246,42,81,-130,143,-374,109,31,169,-151,344,70,-93,-20,161,132,-130,30,-18,160,68,-309,-41,167,473,-37,-199,48,-109,83,-127,-69,-198,-281,203,408,-110,327,-48,-272,-211,-15,-371,252,80,191,-171,-20,-19,331,150,-321,-162,172,-176,-31,-119,-492,14,285,-137,66,316,78,-152,-130,37,128,-27,47,-245,265,-374,67,-135,263,162,49,211,-11,-106,-183,-48,-511,318,-310,-265,-170,-315,3,116,-335,296,148,51,-362,-116,-373,126,26,124,141,21,63,-274,185,266,158,-85,276,-358,225,-429,-160,-428,75,-195,142,138,-246,385,129,-506,-425,-286,338,-165,-287,-137,-190,224,-67,-60,64,71,-275,152,-100,-178,-199,245,-29,-398,46,62,-216,10,55,166,170,312,-110,-144,19,236,138,-127,360,-186,57,114,-5,-176,247,248,-422,227,-65,136,193,169,-74,223,-190,211,-369,-78,276,-299,302,104,-370,-13,-144,45,-35,-448,17,-66,-95,27,118,87,-383,50,111,-388,-88,420,-187,-26,198,411,30,39,104,248,-108,323,362,-60,-42,33,-160,238,89,-183,49,-126,-22,-85,-103,-183,394,-33,196,337,35,-230,116,310,122,92,-174,-76,-110,-94,-252,-117,-42,41,352,365,356,-129,354,123,133,-334,-85,120,-46,256,37,-138,18,-104,492,0,-75,436,449,157,257,-122,-375,-136,41,315,-134,124,11,-156,70,-199,-34,-360,-110,-120,-154,152,131,334,-12,37,18,72,136,31,-276,192,-230,-362,-287,-475,192,302,-233,-143,250,300,-129,-351,395,342,-337,276,61,-445,-83,149,-47,-374,-259,481,85,210,228,-147,-47,-192,-16,-57,-20,198,151,80,146,188,-225,-232,227,-170,-38,-120,-248,-375,156,-9,45,171,-117,55,-25,42,-73,-358,426,-40,99,-239,-142,-61,-98,495,-227,-222,142,-237,-152,-50,194,4,390,-29,-205,377,-140,-29,83,-222,-11,-317,-150,-336,-33,-94,34,287,-172,-128,-308,-255,-173,-2,-53,133,-102,188,-185,-3,-243,-192,254,-28,181,-184,-223,124,-262,86,218,53,112,-372,201,-261,57,6,-23,50,-60,-278,-76,262,138,152,-43,205,107,-168,246,-216,-97,59,255,-290,22,137,-6,322,-292,235,-431,31,298,428,-151,11,150,-318,-337,-61,-271,-89,91,-239,-128,62,-99,133,372,401,384,-109,-240,-138,-189,-304,72,-101,472,-102,352,49,310,332,-233,-441,259,-222,-137,-248,-14,136,-217,208,-111,-70,-199,250,223,46,101,-232,-106,165,229,278,193,-36,77,-96,-23,-357,-477,26,-404,-169,-177,-97,186,71,50,16,179,-264,-304,29,-259,-85,-44,417,-347,119,220,-88,-112,-43,-420,223,-202,136,135,25,-53,464,209,26,-389,-187,-179,45,-4,166,-111,465,130,-73,89,-210,-435,-220,-287,160,302,-19,-67,86,-106,-314,271,-72,-110,-302,-456,-40,173,11,-85,8,256,206,-467,-224,3,-85,297,-173,-115,100,269,164,169,-24,-260,254,50,392,-311,-439,-233,184,160,316,-215,-2,77,-309,228,-182,30,-395,-147,282,321,-128,112,-21,-317,-68,75,242,7,-118,-195,-66,9,264,-166,298,110,41,-398,-87,315,259,-71,-232,123,-90,91,0,-84,59,24,110,427,14,-348,-314,25,36,47,134,-343,225,225,7,289,184,27,-140,236,231,-30,-223,-39,-312,240,28,-390,236,27,42,122,-81,89,-183,377,-22,60,93,-323,390,314,188,239,150,-384,-259,359,94,100,76,-127,-55,-4,132,60,-373,115,286,-173,-306,-133,-271,175,207,-254,-391,-151,-40,381,-35,49,154,230,467,-271,411,-214,-413,-4,192,263,-317,-80,364,135,-425,-321,-124,9,254,362,-442,-232,-121,327,-147,-221,59,-231,85,-215,280,341,324,136,-136,50,108,-96,72,156,83,-245,-298,-255,-210,-46,54,-168,-77,-86,32,54,309,-79,-70,-49,-248,-211,122,61,-290,-122,-301,-89,-179,-288,140,14,86,5,-102,-314,336,50,-151,53,-55,14,-19,133,29,269,-156,319,-83,-62,50,-35,58,181,-58,-161,330,198,326,263,396,252,-31,-248,-241,-4,-315,188]
        
Output:
685353319
Expected:
625284395      





[25000,24999,24998,24997,24996,24995,24994,24993,24992,24991,24990,24989,24988,24987,24986,24985,24984,24983,24982,24981,24980,24979,24978,24977,24976,24975,24974,24973,24972,24971,24970,24969,24968,24967,24966,24965,24964,24963,24962,24961,24960,24959,24958,24957,24956,24955,24954,24953,24952,24951,24950,24949,24948,24947,24946,24945,24944,24943,24942,24941,24940,24939,24938,24937,24936,24935,24934,24933,24932,24931,24930,24929,24928,24927,24926,24925,24924,24923,24922,24921,24920,24919,24918,24917,24916,24915,24914,24913,24912,24911,24910,24909,24908,24907,24906,24905,24904,24903,24902,24901,24900,24899,24898,24897,24896,24895,24894,24893,24892,24891,24890,24889,24888,24887,24886,24885,24884,24883,24882,24881,24880,24879,24878,24877,24876,24875,24874,24873,24872,24871,24870,24869,24868,24867,24866,24865,24864,24863,24862,24861,24860,24859,24858,24857,24856,24855,24854,24853,24852,24851,24850,24849,24848,24847,24846,24845,24844,24843,24842,24841,24840,24839,24838,24837,24836,24835,24834,24833,24832,24831,24830,24829,24828,24827,24826,24825,24824,24823,24822,24821,24820,24819,24818,24817,24816,24815,24814,24813,24812,24811,24810,24809,24808,24807,24806,24805,24804,24803,24802,24801,24800,24799,24798,24797,24796,24795,24794,24793,24792,24791,24790,24789,24788,24787,24786,24785,24784,24783,24782,24781,24780,24779,24778,24777,24776,24775,24774,24773,24772,24771,24770,24769,24768,24767,24766,24765,24764,24763,24762,24761,24760,24759,24758,24757,24756,24755,24754,24753,24752,24751,24750,24749,24748,24747,24746,24745,24744,24743,24742,24741,24740,24739,24738,24737,24736,24735,24734,24733,24732,24731,24730,24729,24728,24727,24726,24725,24724,24723,24722,24721,24720,24719,24718,24717,24716,24715,24714,24713,24712,24711,24710,24709,24708,24707,24706,24705,24704,24703,24702,24701,24700,24699,24698,24697,24696,24695,24694,24693,24692,24691,24690,24689,24688,24687,24686,24685,24684,24683,24682,24681,24680,24679,24678,24677,24676,24675,24674,24673,24672,24671,24670,24669,24668,24667,24666,24665,24664,24663,24662,24661,24660,24659,24658,24657,24656,24655,24654,24653,24652,24651,24650,24649,24648,24647,24646,24645,24644,24643,24642,24641,24640,24639,24638,24637,24636,24635,24634,24633,24632,24631,24630,24629,24628,24627,24626,24625,24624,24623,24622,24621,24620,24619,24618,24617,24616,24615,24614,24613,24612,24611,24610,24609,24608,24607,24606,24605,24604,24603,24602,24601,24600,24599,24598,24597,24596,24595,24594,24593,24592,24591,24590,24589,24588,24587,24586,24585,24584,24583,24582,24581,24580,24579,24578,24577,24576,24575,24574,24573,24572,24571,24570,24569,24568,24567,24566,24565,24564,24563,24562,24561,24560,24559,24558,24557,24556,24555,24554,24553,24552,24551,24550,24549,24548,24547,24546,24545,24544,24543,24542,24541,24540,24539,24538,24537,24536,24535,24534,24533,24532,24531,24530,24529,24528,24527,24526,24525,24524,24523,24522,24521,24520,24519,24518,24517,24516,24515,24514,24513,24512,24511,24510,24509,24508,24507,24506,24505,24504,24503,24502,24501,24500,24499,24498,24497,24496,24495,24494,24493,24492,24491,24490,24489,24488,24487,24486,24485,24484,24483,24482,24481,24480,24479,24478,24477,24476,24475,24474,24473,24472,24471,24470,24469,24468,24467,24466,24465,24464,24463,24462,24461,24460,24459,24458,24457,24456,24455,24454,24453,24452,24451,24450,24449,24448,24447,24446,24445,24444,24443,24442,24441,24440,24439,24438,24437,24436,24435,24434,24433,24432,24431,24430,24429,24428,24427,24426,24425,24424,24423,24422,24421,24420,24419,24418,24417,24416,24415,24414,24413,24412,24411,24410,24409,24408,24407,24406,24405,24404,24403,24402,24401,24400,24399,24398,24397,24396,24395,24394,24393,24392,24391,24390,24389,24388,24387,24386,24385,24384,24383,24382,24381,24380,24379,24378,24377,24376,24375,24374,24373,24372,24371,24370,24369,24368,24367,24366,24365,24364,24363,24362,24361,24360,24359,24358,24357,24356,24355,24354,24353,24352,24351,24350,24349,24348,24347,24346,24345,24344,24343,24342,24341,24340,24339,24338,24337,24336,24335,24334,24333,24332,24331,24330,24329,24328,24327,24326,24325,24324,24323,24322,24321,24320,24319,24318,24317,24316,24315,24314,24313,24312,24311,24310,24309,24308,24307,24306,24305,24304,24303,24302,24301,24300,24299,24298,24297,24296,24295,24294,24293,24292,24291,24290,24289,24288,24287,24286,24285,24284,24283,24282,24281,24280,24279,24278,24277,24276,24275,24274,24273,24272,24271,24270,24269,24268,24267,24266,24265,24264,24263,24262,24261,24260,24259,24258,24257,24256,24255,24254,24253,24252,24251,24250,24249,24248,24247,24246,24245,24244,24243,24242,24241,24240,24239,24238,24237,24236,24235,24234,24233,24232,24231,24230,24229,24228,24227,24226,24225,24224,24223,24222,24221,24220,24219,24218,24217,24216,24215,24214,24213,24212,24211,24210,24209,24208,24207,24206,24205,24204,24203,24202,24201,24200,24199,24198,24197,24196,24195,24194,24193,24192,24191,24190,24189,24188,24187,24186,24185,24184,24183,24182,24181,24180,24179,24178,24177,24176,24175,24174,24173,24172,24171,24170,24169,24168,24167,24166,24165,24164,24163,24162,24161,24160,24159,24158,24157,24156,24155,24154,24153,24152,24151,24150,24149,24148,24147,24146,24145,24144,24143,24142,24141,24140,24139,24138,24137,24136,24135,24134,24133,24132,24131,24130,24129,24128,24127,24126,24125,24124,24123,24122,24121,24120,24119,24118,24117,24116,24115,24114,24113,24112,24111,24110,24109,24108,24107,24106,24105,24104,24103,24102,24101,24100,24099,24098,24097,24096,24095,24094,24093,24092,24091,24090,24089,24088,24087,24086,24085,24084,24083,24082,24081,24080,24079,24078,24077,24076,24075,24074,24073,24072,24071,24070,24069,24068,24067,24066,24065,24064,24063,24062,24061,24060,24059,24058,24057,24056,24055,24054,24053,24052,24051,24050,24049,24048,24047,24046,24045,24044,24043,24042,24041,24040,24039,24038,24037,24036,24035,24034,24033,24032,24031,24030,24029,24028,24027,24026,24025,24024,24023,24022,24021,24020,24019,24018,24017,24016,24015,24014,24013,24012,24011,24010,24009,24008,24007,24006,24005,24004,24003,24002,24001,24000,23999,23998,23997,23996,23995,23994,23993,23992,23991,23990,23989,23988,23987,23986,23985,23984,23983,23982,23981,23980,23979,23978,23977,23976,23975,23974,23973,23972,23971,23970,23969,23968,23967,23966,23965,23964,23963,23962,23961,23960,23959,23958,23957,23956,23955,23954,23953,23952,23951,23950,23949,23948,23947,23946,23945,23944,23943,23942,23941,23940,23939,23938,23937,23936,23935,23934,23933,23932,23931,23930,23929,23928,23927,23926,23925,23924,23923,23922,23921,23920,23919,23918,23917,23916,23915,23914,23913,23912,23911,23910,23909,23908,23907,23906,23905,23904,23903,23902,23901,23900,23899,23898,23897,23896,23895,23894,23893,23892,23891,23890,23889,23888,23887,23886,23885,23884,23883,23882,23881,23880,23879,23878,23877,23876,23875,23874,23873,23872,23871,23870,23869,23868,23867,23866,23865,23864,23863,23862,23861,23860,23859,23858,23857,23856,23855,23854,23853,23852,23851,23850,23849,23848,23847,23846,23845,23844,23843,23842,23841,23840,23839,23838,23837,23836,23835,23834,23833,23832,23831,23830,23829,23828,23827,23826,23825,23824,23823,23822,23821,23820,23819,23818,23817,23816,23815,23814,23813,23812,23811,23810,23809,23808,23807,23806,23805,23804,23803,23802,23801,23800,23799,23798,23797,23796,23795,23794,23793,23792,23791,23790,23789,23788,23787,23786,23785,23784,23783,23782,23781,23780,23779,23778,23777,23776,23775,23774,23773,23772,23771,23770,23769,23768,23767,23766,23765,23764,23763,23762,23761,23760,23759,23758,23757,23756,23755,23754,23753,23752,23751,23750,23749,23748,23747,23746,23745,23744,23743,23742,23741,23740,23739,23738,23737,23736,23735,23734,23733,23732,23731,23730,23729,23728,23727,23726,23725,23724,23723,23722,23721,23720,23719,23718,23717,23716,23715,23714,23713,23712,23711,23710,23709,23708,23707,23706,23705,23704,23703,23702,23701,23700,23699,23698,23697,23696,23695,23694,23693,23692,23691,23690,23689,23688,23687,23686,23685,23684,23683,23682,23681,23680,23679,23678,23677,23676,23675,23674,23673,23672,23671,23670,23669,23668,23667,23666,23665,23664,23663,23662,23661,23660,23659,23658,23657,23656,23655,23654,23653,23652,23651,23650,23649,23648,23647,23646,23645,23644,23643,23642,23641,23640,23639,23638,23637,23636,23635,23634,23633,23632,23631,23630,23629,23628,23627,23626,23625,23624,23623,23622,23621,23620,23619,23618,23617,23616,23615,23614,23613,23612,23611,23610,23609,23608,23607,23606,23605,23604,23603,23602,23601,23600,23599,23598,23597,23596,23595,23594,23593,23592,23591,23590,23589,23588,23587,23586,23585,23584,23583,23582,23581,23580,23579,23578,23577,23576,23575,23574,23573,23572,23571,23570,23569,23568,23567,23566,23565,23564,23563,23562,23561,23560,23559,23558,23557,23556,23555,23554,23553,23552,23551,23550,23549,23548,23547,23546,23545,23544,23543,23542,23541,23540,23539,23538,23537,23536,23535,23534,23533,23532,23531,23530,23529,23528,23527,23526,23525,23524,23523,23522,23521,23520,23519,23518,23517,23516,23515,23514,23513,23512,23511,23510,23509,23508,23507,23506,23505,23504,23503,23502,23501,23500,23499,23498,23497,23496,23495,23494,23493,23492,23491,23490,23489,23488,23487,23486,23485,23484,23483,23482,23481,23480,23479,23478,23477,23476,23475,23474,23473,23472,23471,23470,23469,23468,23467,23466,23465,23464,23463,23462,23461,23460,23459,23458,23457,23456,23455,23454,23453,23452,23451,23450,23449,23448,23447,23446,23445,23444,23443,23442,23441,23440,23439,23438,23437,23436,23435,23434,23433,23432,23431,23430,23429,23428,23427,23426,23425,23424,23423,23422,23421,23420,23419,23418,23417,23416,23415,23414,23413,23412,23411,23410,23409,23408,23407,23406,23405,23404,23403,23402,23401,23400,23399,23398,23397,23396,23395,23394,23393,23392,23391,23390,23389,23388,23387,23386,23385,23384,23383,23382,23381,23380,23379,23378,23377,23376,23375,23374,23373,23372,23371,23370,23369,23368,23367,23366,23365,23364,23363,23362,23361,23360,23359,23358,23357,23356,23355,23354,23353,23352,23351,23350,23349,23348,23347,23346,23345,23344,23343,23342,23341,23340,23339,23338,23337,23336,23335,23334,23333,23332,23331,23330,23329,23328,23327,23326,23325,23324,23323,23322,23321,23320,23319,23318,23317,23316,23315,23314,23313,23312,23311,23310,23309,23308,23307,23306,23305,23304,23303,23302,23301,23300,23299,23298,23297,23296,23295,23294,23293,23292,23291,23290,23289,23288,23287,23286,23285,23284,23283,23282,23281,23280,23279,23278,23277,23276,23275,23274,23273,23272,23271,23270,23269,23268,23267,23266,23265,23264,23263,23262,23261,23260,23259,23258,23257,23256,23255,23254,23253,23252,23251,23250,23249,23248,23247,23246,23245,23244,23243,23242,23241,23240,23239,23238,23237,23236,23235,23234,23233,23232,23231,23230,23229,23228,23227,23226,23225,23224,23223,23222,23221,23220,23219,23218,23217,23216,23215,23214,23213,23212,23211,23210,23209,23208,23207,23206,23205,23204,23203,23202,23201,23200,23199,23198,23197,23196,23195,23194,23193,23192,23191,23190,23189,23188,23187,23186,23185,23184,23183,23182,23181,23180,23179,23178,23177,23176,23175,23174,23173,23172,23171,23170,23169,23168,23167,23166,23165,23164,23163,23162,23161,23160,23159,23158,23157,23156,23155,23154,23153,23152,23151,23150,23149,23148,23147,23146,23145,23144,23143,23142,23141,23140,23139,23138,23137,23136,23135,23134,23133,23132,23131,23130,23129,23128,23127,23126,23125,23124,23123,23122,23121,23120,23119,23118,23117,23116,23115,23114,23113,23112,23111,23110,23109,23108,23107,23106,23105,23104,23103,23102,23101,23100,23099,23098,23097,23096,23095,23094,23093,23092,23091,23090,23089,23088,23087,23086,23085,23084,23083,23082,23081,23080,23079,23078,23077,23076,23075,23074,23073,23072,23071,23070,23069,23068,23067,23066,23065,23064,23063,23062,23061,23060,23059,23058,23057,23056,23055,23054,23053,23052,23051,23050,23049,23048,23047,23046,23045,23044,23043,23042,23041,23040,23039,23038,23037,23036,23035,23034,23033,23032,23031,23030,23029,23028,23027,23026,23025,23024,23023,23022,23021,23020,23019,23018,23017,23016,23015,23014,23013,23012,23011,23010,23009,23008,23007,23006,23005,23004,23003,23002,23001,23000,22999,22998,22997,22996,22995,22994,22993,22992,22991,22990,22989,22988,22987,22986,22985,22984,22983,22982,22981,22980,22979,22978,22977,22976,22975,22974,22973,22972,22971,22970,22969,22968,22967,22966,22965,22964,22963,22962,22961,22960,22959,22958,22957,22956,22955,22954,22953,22952,22951,22950,22949,22948,22947,22946,22945,22944,22943,22942,22941,22940,22939,22938,22937,22936,22935,22934,22933,22932,22931,22930,22929,22928,22927,22926,22925,22924,22923,22922,22921,22920,22919,22918,22917,22916,22915,22914,22913,22912,22911,22910,22909,22908,22907,22906,22905,22904,22903,22902,22901,22900,22899,22898,22897,22896,22895,22894,22893,22892,22891,22890,22889,22888,22887,22886,22885,22884,22883,22882,22881,22880,22879,22878,22877,22876,22875,22874,22873,22872,22871,22870,22869,22868,22867,22866,22865,22864,22863,22862,22861,22860,22859,22858,22857,22856,22855,22854,22853,22852,22851,22850,22849,22848,22847,22846,22845,22844,22843,22842,22841,22840,22839,22838,22837,22836,22835,22834,22833,22832,22831,22830,22829,22828,22827,22826,22825,22824,22823,22822,22821,22820,22819,22818,22817,22816,22815,22814,22813,22812,22811,22810,22809,22808,22807,22806,22805,22804,22803,22802,22801,22800,22799,22798,22797,22796,22795,22794,22793,22792,22791,22790,22789,22788,22787,22786,22785,22784,22783,22782,22781,22780,22779,22778,22777,22776,22775,22774,22773,22772,22771,22770,22769,22768,22767,22766,22765,22764,22763,22762,22761,22760,22759,22758,22757,22756,22755,22754,22753,22752,22751,22750,22749,22748,22747,22746,22745,22744,22743,22742,22741,22740,22739,22738,22737,22736,22735,22734,22733,22732,22731,22730,22729,22728,22727,22726,22725,22724,22723,22722,22721,22720,22719,22718,22717,22716,22715,22714,22713,22712,22711,22710,22709,22708,22707,22706,22705,22704,22703,22702,22701,22700,22699,22698,22697,22696,22695,22694,22693,22692,22691,22690,22689,22688,22687,22686,22685,22684,22683,22682,22681,22680,22679,22678,22677,22676,22675,22674,22673,22672,22671,22670,22669,22668,22667,22666,22665,22664,22663,22662,22661,22660,22659,22658,22657,22656,22655,22654,22653,22652,22651,22650,22649,22648,22647,22646,22645,22644,22643,22642,22641,22640,22639,22638,22637,22636,22635,22634,22633,22632,22631,22630,22629,22628,22627,22626,22625,22624,22623,22622,22621,22620,22619,22618,22617,22616,22615,22614,22613,22612,22611,22610,22609,22608,22607,22606,22605,22604,22603,22602,22601,22600,22599,22598,22597,22596,22595,22594,22593,22592,22591,22590,22589,22588,22587,22586,22585,22584,22583,22582,22581,22580,22579,22578,22577,22576,22575,22574,22573,22572,22571,22570,22569,22568,22567,22566,22565,22564,22563,22562,22561,22560,22559,22558,22557,22556,22555,22554,22553,22552,22551,22550,22549,22548,22547,22546,22545,22544,22543,22542,22541,22540,22539,22538,22537,22536,22535,22534,22533,22532,22531,22530,22529,22528,22527,22526,22525,22524,22523,22522,22521,22520,22519,22518,22517,22516,22515,22514,22513,22512,22511,22510,22509,22508,22507,22506,22505,22504,22503,22502,22501,22500,22499,22498,22497,22496,22495,22494,22493,22492,22491,22490,22489,22488,22487,22486,22485,22484,22483,22482,22481,22480,22479,22478,22477,22476,22475,22474,22473,22472,22471,22470,22469,22468,22467,22466,22465,22464,22463,22462,22461,22460,22459,22458,22457,22456,22455,22454,22453,22452,22451,22450,22449,22448,22447,22446,22445,22444,22443,22442,22441,22440,22439,22438,22437,22436,22435,22434,22433,22432,22431,22430,22429,22428,22427,22426,22425,22424,22423,22422,22421,22420,22419,22418,22417,22416,22415,22414,22413,22412,22411,22410,22409,22408,22407,22406,22405,22404,22403,22402,22401,22400,22399,22398,22397,22396,22395,22394,22393,22392,22391,22390,22389,22388,22387,22386,22385,22384,22383,22382,22381,22380,22379,22378,22377,22376,22375,22374,22373,22372,22371,22370,22369,22368,22367,22366,22365,22364,22363,22362,22361,22360,22359,22358,22357,22356,22355,22354,22353,22352,22351,22350,22349,22348,22347,22346,22345,22344,22343,22342,22341,22340,22339,22338,22337,22336,22335,22334,22333,22332,22331,22330,22329,22328,22327,22326,22325,22324,22323,22322,22321,22320,22319,22318,22317,22316,22315,22314,22313,22312,22311,22310,22309,22308,22307,22306,22305,22304,22303,22302,22301,22300,22299,22298,22297,22296,22295,22294,22293,22292,22291,22290,22289,22288,22287,22286,22285,22284,22283,22282,22281,22280,22279,22278,22277,22276,22275,22274,22273,22272,22271,22270,22269,22268,22267,22266,22265,22264,22263,22262,22261,22260,22259,22258,22257,22256,22255,22254,22253,22252,22251,22250,22249,22248,22247,22246,22245,22244,22243,22242,22241,22240,22239,22238,22237,22236,22235,22234,22233,22232,22231,22230,22229,22228,22227,22226,22225,22224,22223,22222,22221,22220,22219,22218,22217,22216,22215,22214,22213,22212,22211,22210,22209,22208,22207,22206,22205,22204,22203,22202,22201,22200,22199,22198,22197,22196,22195,22194,22193,22192,22191,22190,22189,22188,22187,22186,22185,22184,22183,22182,22181,22180,22179,22178,22177,22176,22175,22174,22173,22172,22171,22170,22169,22168,22167,22166,22165,22164,22163,22162,22161,22160,22159,22158,22157,22156,22155,22154,22153,22152,22151,22150,22149,22148,22147,22146,22145,22144,22143,22142,22141,22140,22139,22138,22137,22136,22135,22134,22133,22132,22131,22130,22129,22128,22127,22126,22125,22124,22123,22122,22121,22120,22119,22118,22117,22116,22115,22114,22113,22112,22111,22110,22109,22108,22107,22106,22105,22104,22103,22102,22101,22100,22099,22098,22097,22096,22095,22094,22093,22092,22091,22090,22089,22088,22087,22086,22085,22084,22083,22082,22081,22080,22079,22078,22077,22076,22075,22074,22073,22072,22071,22070,22069,22068,22067,22066,22065,22064,22063,22062,22061,22060,22059,22058,22057,22056,22055,22054,22053,22052,22051,22050,22049,22048,22047,22046,22045,22044,22043,22042,22041,22040,22039,22038,22037,22036,22035,22034,22033,22032,22031,22030,22029,22028,22027,22026,22025,22024,22023,22022,22021,22020,22019,22018,22017,22016,22015,22014,22013,22012,22011,22010,22009,22008,22007,22006,22005,22004,22003,22002,22001,22000,21999,21998,21997,21996,21995,21994,21993,21992,21991,21990,21989,21988,21987,21986,21985,21984,21983,21982,21981,21980,21979,21978,21977,21976,21975,21974,21973,21972,21971,21970,21969,21968,21967,21966,21965,21964,21963,21962,21961,21960,21959,21958,21957,21956,21955,21954,21953,21952,21951,21950,21949,21948,21947,21946,21945,21944,21943,21942,21941,21940,21939,21938,21937,21936,21935,21934,21933,21932,21931,21930,21929,21928,21927,21926,21925,21924,21923,21922,21921,21920,21919,21918,21917,21916,21915,21914,21913,21912,21911,21910,21909,21908,21907,21906,21905,21904,21903,21902,21901,21900,21899,21898,21897,21896,21895,21894,21893,21892,21891,21890,21889,21888,21887,21886,21885,21884,21883,21882,21881,21880,21879,21878,21877,21876,21875,21874,21873,21872,21871,21870,21869,21868,21867,21866,21865,21864,21863,21862,21861,21860,21859,21858,21857,21856,21855,21854,21853,21852,21851,21850,21849,21848,21847,21846,21845,21844,21843,21842,21841,21840,21839,21838,21837,21836,21835,21834,21833,21832,21831,21830,21829,21828,21827,21826,21825,21824,21823,21822,21821,21820,21819,21818,21817,21816,21815,21814,21813,21812,21811,21810,21809,21808,21807,21806,21805,21804,21803,21802,21801,21800,21799,21798,21797,21796,21795,21794,21793,21792,21791,21790,21789,21788,21787,21786,21785,21784,21783,21782,21781,21780,21779,21778,21777,21776,21775,21774,21773,21772,21771,21770,21769,21768,21767,21766,21765,21764,21763,21762,21761,21760,21759,21758,21757,21756,21755,21754,21753,21752,21751,21750,21749,21748,21747,21746,21745,21744,21743,21742,21741,21740,21739,21738,21737,21736,21735,21734,21733,21732,21731,21730,21729,21728,21727,21726,21725,21724,21723,21722,21721,21720,21719,21718,21717,21716,21715,21714,21713,21712,21711,21710,21709,21708,21707,21706,21705,21704,21703,21702,21701,21700,21699,21698,21697,21696,21695,21694,21693,21692,21691,21690,21689,21688,21687,21686,21685,21684,21683,21682,21681,21680,21679,21678,21677,21676,21675,21674,21673,21672,21671,21670,21669,21668,21667,21666,21665,21664,21663,21662,21661,21660,21659,21658,21657,21656,21655,21654,21653,21652,21651,21650,21649,21648,21647,21646,21645,21644,21643,21642,21641,21640,21639,21638,21637,21636,21635,21634,21633,21632,21631,21630,21629,21628,21627,21626,21625,21624,21623,21622,21621,21620,21619,21618,21617,21616,21615,21614,21613,21612,21611,21610,21609,21608,21607,21606,21605,21604,21603,21602,21601,21600,21599,21598,21597,21596,21595,21594,21593,21592,21591,21590,21589,21588,21587,21586,21585,21584,21583,21582,21581,21580,21579,21578,21577,21576,21575,21574,21573,21572,21571,21570,21569,21568,21567,21566,21565,21564,21563,21562,21561,21560,21559,21558,21557,21556,21555,21554,21553,21552,21551,21550,21549,21548,21547,21546,21545,21544,21543,21542,21541,21540,21539,21538,21537,21536,21535,21534,21533,21532,21531,21530,21529,21528,21527,21526,21525,21524,21523,21522,21521,21520,21519,21518,21517,21516,21515,21514,21513,21512,21511,21510,21509,21508,21507,21506,21505,21504,21503,21502,21501,21500,21499,21498,21497,21496,21495,21494,21493,21492,21491,21490,21489,21488,21487,21486,21485,21484,21483,21482,21481,21480,21479,21478,21477,21476,21475,21474,21473,21472,21471,21470,21469,21468,21467,21466,21465,21464,21463,21462,21461,21460,21459,21458,21457,21456,21455,21454,21453,21452,21451,21450,21449,21448,21447,21446,21445,21444,21443,21442,21441,21440,21439,21438,21437,21436,21435,21434,21433,21432,21431,21430,21429,21428,21427,21426,21425,21424,21423,21422,21421,21420,21419,21418,21417,21416,21415,21414,21413,21412,21411,21410,21409,21408,21407,21406,21405,21404,21403,21402,21401,21400,21399,21398,21397,21396,21395,21394,21393,21392,21391,21390,21389,21388,21387,21386,21385,21384,21383,21382,21381,21380,21379,21378,21377,21376,21375,21374,21373,21372,21371,21370,21369,21368,21367,21366,21365,21364,21363,21362,21361,21360,21359,21358,21357,21356,21355,21354,21353,21352,21351,21350,21349,21348,21347,21346,21345,21344,21343,21342,21341,21340,21339,21338,21337,21336,21335,21334,21333,21332,21331,21330,21329,21328,21327,21326,21325,21324,21323,21322,21321,21320,21319,21318,21317,21316,21315,21314,21313,21312,21311,21310,21309,21308,21307,21306,21305,21304,21303,21302,21301,21300,21299,21298,21297,21296,21295,21294,21293,21292,21291,21290,21289,21288,21287,21286,21285,21284,21283,21282,21281,21280,21279,21278,21277,21276,21275,21274,21273,21272,21271,21270,21269,21268,21267,21266,21265,21264,21263,21262,21261,21260,21259,21258,21257,21256,21255,21254,21253,21252,21251,21250,21249,21248,21247,21246,21245,21244,21243,21242,21241,21240,21239,21238,21237,21236,21235,21234,21233,21232,21231,21230,21229,21228,21227,21226,21225,21224,21223,21222,21221,21220,21219,21218,21217,21216,21215,21214,21213,21212,21211,21210,21209,21208,21207,21206,21205,21204,21203,21202,21201,21200,21199,21198,21197,21196,21195,21194,21193,21192,21191,21190,21189,21188,21187,21186,21185,21184,21183,21182,21181,21180,21179,21178,21177,21176,21175,21174,21173,21172,21171,21170,21169,21168,21167,21166,21165,21164,21163,21162,21161,21160,21159,21158,21157,21156,21155,21154,21153,21152,21151,21150,21149,21148,21147,21146,21145,21144,21143,21142,21141,21140,21139,21138,21137,21136,21135,21134,21133,21132,21131,21130,21129,21128,21127,21126,21125,21124,21123,21122,21121,21120,21119,21118,21117,21116,21115,21114,21113,21112,21111,21110,21109,21108,21107,21106,21105,21104,21103,21102,21101,21100,21099,21098,21097,21096,21095,21094,21093,21092,21091,21090,21089,21088,21087,21086,21085,21084,21083,21082,21081,21080,21079,21078,21077,21076,21075,21074,21073,21072,21071,21070,21069,21068,21067,21066,21065,21064,21063,21062,21061,21060,21059,21058,21057,21056,21055,21054,21053,21052,21051,21050,21049,21048,21047,21046,21045,21044,21043,21042,21041,21040,21039,21038,21037,21036,21035,21034,21033,21032,21031,21030,21029,21028,21027,21026,21025,21024,21023,21022,21021,21020,21019,21018,21017,21016,21015,21014,21013,21012,21011,21010,21009,21008,21007,21006,21005,21004,21003,21002,21001,21000,20999,20998,20997,20996,20995,20994,20993,20992,20991,20990,20989,20988,20987,20986,20985,20984,20983,20982,20981,20980,20979,20978,20977,20976,20975,20974,20973,20972,20971,20970,20969,20968,20967,20966,20965,20964,20963,20962,20961,20960,20959,20958,20957,20956,20955,20954,20953,20952,20951,20950,20949,20948,20947,20946,20945,20944,20943,20942,20941,20940,20939,20938,20937,20936,20935,20934,20933,20932,20931,20930,20929,20928,20927,20926,20925,20924,20923,20922,20921,20920,20919,20918,20917,20916,20915,20914,20913,20912,20911,20910,20909,20908,20907,20906,20905,20904,20903,20902,20901,20900,20899,20898,20897,20896,20895,20894,20893,20892,20891,20890,20889,20888,20887,20886,20885,20884,20883,20882,20881,20880,20879,20878,20877,20876,20875,20874,20873,20872,20871,20870,20869,20868,20867,20866,20865,20864,20863,20862,20861,20860,20859,20858,20857,20856,20855,20854,20853,20852,20851,20850,20849,20848,20847,20846,20845,20844,20843,20842,20841,20840,20839,20838,20837,20836,20835,20834,20833,20832,20831,20830,20829,20828,20827,20826,20825,20824,20823,20822,20821,20820,20819,20818,20817,20816,20815,20814,20813,20812,20811,20810,20809,20808,20807,20806,20805,20804,20803,20802,20801,20800,20799,20798,20797,20796,20795,20794,20793,20792,20791,20790,20789,20788,20787,20786,20785,20784,20783,20782,20781,20780,20779,20778,20777,20776,20775,20774,20773,20772,20771,20770,20769,20768,20767,20766,20765,20764,20763,20762,20761,20760,20759,20758,20757,20756,20755,20754,20753,20752,20751,20750,20749,20748,20747,20746,20745,20744,20743,20742,20741,20740,20739,20738,20737,20736,20735,20734,20733,20732,20731,20730,20729,20728,20727,20726,20725,20724,20723,20722,20721,20720,20719,20718,20717,20716,20715,20714,20713,20712,20711,20710,20709,20708,20707,20706,20705,20704,20703,20702,20701,20700,20699,20698,20697,20696,20695,20694,20693,20692,20691,20690,20689,20688,20687,20686,20685,20684,20683,20682,20681,20680,20679,20678,20677,20676,20675,20674,20673,20672,20671,20670,20669,20668,20667,20666,20665,20664,20663,20662,20661,20660,20659,20658,20657,20656,20655,20654,20653,20652,20651,20650,20649,20648,20647,20646,20645,20644,20643,20642,20641,20640,20639,20638,20637,20636,20635,20634,20633,20632,20631,20630,20629,20628,20627,20626,20625,20624,20623,20622,20621,20620,20619,20618,20617,20616,20615,20614,20613,20612,20611,20610,20609,20608,20607,20606,20605,20604,20603,20602,20601,20600,20599,20598,20597,20596,20595,20594,20593,20592,20591,20590,20589,20588,20587,20586,20585,20584,20583,20582,20581,20580,20579,20578,20577,20576,20575,20574,20573,20572,20571,20570,20569,20568,20567,20566,20565,20564,20563,20562,20561,20560,20559,20558,20557,20556,20555,20554,20553,20552,20551,20550,20549,20548,20547,20546,20545,20544,20543,20542,20541,20540,20539,20538,20537,20536,20535,20534,20533,20532,20531,20530,20529,20528,20527,20526,20525,20524,20523,20522,20521,20520,20519,20518,20517,20516,20515,20514,20513,20512,20511,20510,20509,20508,20507,20506,20505,20504,20503,20502,20501,20500,20499,20498,20497,20496,20495,20494,20493,20492,20491,20490,20489,20488,20487,20486,20485,20484,20483,20482,20481,20480,20479,20478,20477,20476,20475,20474,20473,20472,20471,20470,20469,20468,20467,20466,20465,20464,20463,20462,20461,20460,20459,20458,20457,20456,20455,20454,20453,20452,20451,20450,20449,20448,20447,20446,20445,20444,20443,20442,20441,20440,20439,20438,20437,20436,20435,20434,20433,20432,20431,20430,20429,20428,20427,20426,20425,20424,20423,20422,20421,20420,20419,20418,20417,20416,20415,20414,20413,20412,20411,20410,20409,20408,20407,20406,20405,20404,20403,20402,20401,20400,20399,20398,20397,20396,20395,20394,20393,20392,20391,20390,20389,20388,20387,20386,20385,20384,20383,20382,20381,20380,20379,20378,20377,20376,20375,20374,20373,20372,20371,20370,20369,20368,20367,20366,20365,20364,20363,20362,20361,20360,20359,20358,20357,20356,20355,20354,20353,20352,20351,20350,20349,20348,20347,20346,20345,20344,20343,20342,20341,20340,20339,20338,20337,20336,20335,20334,20333,20332,20331,20330,20329,20328,20327,20326,20325,20324,20323,20322,20321,20320,20319,20318,20317,20316,20315,20314,20313,20312,20311,20310,20309,20308,20307,20306,20305,20304,20303,20302,20301,20300,20299,20298,20297,20296,20295,20294,20293,20292,20291,20290,20289,20288,20287,20286,20285,20284,20283,20282,20281,20280,20279,20278,20277,20276,20275,20274,20273,20272,20271,20270,20269,20268,20267,20266,20265,20264,20263,20262,20261,20260,20259,20258,20257,20256,20255,20254,20253,20252,20251,20250,20249,20248,20247,20246,20245,20244,20243,20242,20241,20240,20239,20238,20237,20236,20235,20234,20233,20232,20231,20230,20229,20228,20227,20226,20225,20224,20223,20222,20221,20220,20219,20218,20217,20216,20215,20214,20213,20212,20211,20210,20209,20208,20207,20206,20205,20204,20203,20202,20201,20200,20199,20198,20197,20196,20195,20194,20193,20192,20191,20190,20189,20188,20187,20186,20185,20184,20183,20182,20181,20180,20179,20178,20177,20176,20175,20174,20173,20172,20171,20170,20169,20168,20167,20166,20165,20164,20163,20162,20161,20160,20159,20158,20157,20156,20155,20154,20153,20152,20151,20150,20149,20148,20147,20146,20145,20144,20143,20142,20141,20140,20139,20138,20137,20136,20135,20134,20133,20132,20131,20130,20129,20128,20127,20126,20125,20124,20123,20122,20121,20120,20119,20118,20117,20116,20115,20114,20113,20112,20111,20110,20109,20108,20107,20106,20105,20104,20103,20102,20101,20100,20099,20098,20097,20096,20095,20094,20093,20092,20091,20090,20089,20088,20087,20086,20085,20084,20083,20082,20081,20080,20079,20078,20077,20076,20075,20074,20073,20072,20071,20070,20069,20068,20067,20066,20065,20064,20063,20062,20061,20060,20059,20058,20057,20056,20055,20054,20053,20052,20051,20050,20049,20048,20047,20046,20045,20044,20043,20042,20041,20040,20039,20038,20037,20036,20035,20034,20033,20032,20031,20030,20029,20028,20027,20026,20025,20024,20023,20022,20021,20020,20019,20018,20017,20016,20015,20014,20013,20012,20011,20010,20009,20008,20007,20006,20005,20004,20003,20002,20001,20000,19999,19998,19997,19996,19995,19994,19993,19992,19991,19990,19989,19988,19987,19986,19985,19984,19983,19982,19981,19980,19979,19978,19977,19976,19975,19974,19973,19972,19971,19970,19969,19968,19967,19966,19965,19964,19963,19962,19961,19960,19959,19958,19957,19956,19955,19954,19953,19952,19951,19950,19949,19948,19947,19946,19945,19944,19943,19942,19941,19940,19939,19938,19937,19936,19935,19934,19933,19932,19931,19930,19929,19928,19927,19926,19925,19924,19923,19922,19921,19920,19919,19918,19917,19916,19915,19914,19913,19912,19911,19910,19909,19908,19907,19906,19905,19904,19903,19902,19901,19900,19899,19898,19897,19896,19895,19894,19893,19892,19891,19890,19889,19888,19887,19886,19885,19884,19883,19882,19881,19880,19879,19878,19877,19876,19875,19874,19873,19872,19871,19870,19869,19868,19867,19866,19865,19864,19863,19862,19861,19860,19859,19858,19857,19856,19855,19854,19853,19852,19851,19850,19849,19848,19847,19846,19845,19844,19843,19842,19841,19840,19839,19838,19837,19836,19835,19834,19833,19832,19831,19830,19829,19828,19827,19826,19825,19824,19823,19822,19821,19820,19819,19818,19817,19816,19815,19814,19813,19812,19811,19810,19809,19808,19807,19806,19805,19804,19803,19802,19801,19800,19799,19798,19797,19796,19795,19794,19793,19792,19791,19790,19789,19788,19787,19786,19785,19784,19783,19782,19781,19780,19779,19778,19777,19776,19775,19774,19773,19772,19771,19770,19769,19768,19767,19766,19765,19764,19763,19762,19761,19760,19759,19758,19757,19756,19755,19754,19753,19752,19751,19750,19749,19748,19747,19746,19745,19744,19743,19742,19741,19740,19739,19738,19737,19736,19735,19734,19733,19732,19731,19730,19729,19728,19727,19726,19725,19724,19723,19722,19721,19720,19719,19718,19717,19716,19715,19714,19713,19712,19711,19710,19709,19708,19707,19706,19705,19704,19703,19702,19701,19700,19699,19698,19697,19696,19695,19694,19693,19692,19691,19690,19689,19688,19687,19686,19685,19684,19683,19682,19681,19680,19679,19678,19677,19676,19675,19674,19673,19672,19671,19670,19669,19668,19667,19666,19665,19664,19663,19662,19661,19660,19659,19658,19657,19656,19655,19654,19653,19652,19651,19650,19649,19648,19647,19646,19645,19644,19643,19642,19641,19640,19639,19638,19637,19636,19635,19634,19633,19632,19631,19630,19629,19628,19627,19626,19625,19624,19623,19622,19621,19620,19619,19618,19617,19616,19615,19614,19613,19612,19611,19610,19609,19608,19607,19606,19605,19604,19603,19602,19601,19600,19599,19598,19597,19596,19595,19594,19593,19592,19591,19590,19589,19588,19587,19586,19585,19584,19583,19582,19581,19580,19579,19578,19577,19576,19575,19574,19573,19572,19571,19570,19569,19568,19567,19566,19565,19564,19563,19562,19561,19560,19559,19558,19557,19556,19555,19554,19553,19552,19551,19550,19549,19548,19547,19546,19545,19544,19543,19542,19541,19540,19539,19538,19537,19536,19535,19534,19533,19532,19531,19530,19529,19528,19527,19526,19525,19524,19523,19522,19521,19520,19519,19518,19517,19516,19515,19514,19513,19512,19511,19510,19509,19508,19507,19506,19505,19504,19503,19502,19501,19500,19499,19498,19497,19496,19495,19494,19493,19492,19491,19490,19489,19488,19487,19486,19485,19484,19483,19482,19481,19480,19479,19478,19477,19476,19475,19474,19473,19472,19471,19470,19469,19468,19467,19466,19465,19464,19463,19462,19461,19460,19459,19458,19457,19456,19455,19454,19453,19452,19451,19450,19449,19448,19447,19446,19445,19444,19443,19442,19441,19440,19439,19438,19437,19436,19435,19434,19433,19432,19431,19430,19429,19428,19427,19426,19425,19424,19423,19422,19421,19420,19419,19418,19417,19416,19415,19414,19413,19412,19411,19410,19409,19408,19407,19406,19405,19404,19403,19402,19401,19400,19399,19398,19397,19396,19395,19394,19393,19392,19391,19390,19389,19388,19387,19386,19385,19384,19383,19382,19381,19380,19379,19378,19377,19376,19375,19374,19373,19372,19371,19370,19369,19368,19367,19366,19365,19364,19363,19362,19361,19360,19359,19358,19357,19356,19355,19354,19353,19352,19351,19350,19349,19348,19347,19346,19345,19344,19343,19342,19341,19340,19339,19338,19337,19336,19335,19334,19333,19332,19331,19330,19329,19328,19327,19326,19325,19324,19323,19322,19321,19320,19319,19318,19317,19316,19315,19314,19313,19312,19311,19310,19309,19308,19307,19306,19305,19304,19303,19302,19301,19300,19299,19298,19297,19296,19295,19294,19293,19292,19291,19290,19289,19288,19287,19286,19285,19284,19283,19282,19281,19280,19279,19278,19277,19276,19275,19274,19273,19272,19271,19270,19269,19268,19267,19266,19265,19264,19263,19262,19261,19260,19259,19258,19257,19256,19255,19254,19253,19252,19251,19250,19249,19248,19247,19246,19245,19244,19243,19242,19241,19240,19239,19238,19237,19236,19235,19234,19233,19232,19231,19230,19229,19228,19227,19226,19225,19224,19223,19222,19221,19220,19219,19218,19217,19216,19215,19214,19213,19212,19211,19210,19209,19208,19207,19206,19205,19204,19203,19202,19201,19200,19199,19198,19197,19196,19195,19194,19193,19192,19191,19190,19189,19188,19187,19186,19185,19184,19183,19182,19181,19180,19179,19178,19177,19176,19175,19174,19173,19172,19171,19170,19169,19168,19167,19166,19165,19164,19163,19162,19161,19160,19159,19158,19157,19156,19155,19154,19153,19152,19151,19150,19149,19148,19147,19146,19145,19144,19143,19142,19141,19140,19139,19138,19137,19136,19135,19134,19133,19132,19131,19130,19129,19128,19127,19126,19125,19124,19123,19122,19121,19120,19119,19118,19117,19116,19115,19114,19113,19112,19111,19110,19109,19108,19107,19106,19105,19104,19103,19102,19101,19100,19099,19098,19097,19096,19095,19094,19093,19092,19091,19090,19089,19088,19087,19086,19085,19084,19083,19082,19081,19080,19079,19078,19077,19076,19075,19074,19073,19072,19071,19070,19069,19068,19067,19066,19065,19064,19063,19062,19061,19060,19059,19058,19057,19056,19055,19054,19053,19052,19051,19050,19049,19048,19047,19046,19045,19044,19043,19042,19041,19040,19039,19038,19037,19036,19035,19034,19033,19032,19031,19030,19029,19028,19027,19026,19025,19024,19023,19022,19021,19020,19019,19018,19017,19016,19015,19014,19013,19012,19011,19010,19009,19008,19007,19006,19005,19004,19003,19002,19001,19000,18999,18998,18997,18996,18995,18994,18993,18992,18991,18990,18989,18988,18987,18986,18985,18984,18983,18982,18981,18980,18979,18978,18977,18976,18975,18974,18973,18972,18971,18970,18969,18968,18967,18966,18965,18964,18963,18962,18961,18960,18959,18958,18957,18956,18955,18954,18953,18952,18951,18950,18949,18948,18947,18946,18945,18944,18943,18942,18941,18940,18939,18938,18937,18936,18935,18934,18933,18932,18931,18930,18929,18928,18927,18926,18925,18924,18923,18922,18921,18920,18919,18918,18917,18916,18915,18914,18913,18912,18911,18910,18909,18908,18907,18906,18905,18904,18903,18902,18901,18900,18899,18898,18897,18896,18895,18894,18893,18892,18891,18890,18889,18888,18887,18886,18885,18884,18883,18882,18881,18880,18879,18878,18877,18876,18875,18874,18873,18872,18871,18870,18869,18868,18867,18866,18865,18864,18863,18862,18861,18860,18859,18858,18857,18856,18855,18854,18853,18852,18851,18850,18849,18848,18847,18846,18845,18844,18843,18842,18841,18840,18839,18838,18837,18836,18835,18834,18833,18832,18831,18830,18829,18828,18827,18826,18825,18824,18823,18822,18821,18820,18819,18818,18817,18816,18815,18814,18813,18812,18811,18810,18809,18808,18807,18806,18805,18804,18803,18802,18801,18800,18799,18798,18797,18796,18795,18794,18793,18792,18791,18790,18789,18788,18787,18786,18785,18784,18783,18782,18781,18780,18779,18778,18777,18776,18775,18774,18773,18772,18771,18770,18769,18768,18767,18766,18765,18764,18763,18762,18761,18760,18759,18758,18757,18756,18755,18754,18753,18752,18751,18750,18749,18748,18747,18746,18745,18744,18743,18742,18741,18740,18739,18738,18737,18736,18735,18734,18733,18732,18731,18730,18729,18728,18727,18726,18725,18724,18723,18722,18721,18720,18719,18718,18717,18716,18715,18714,18713,18712,18711,18710,18709,18708,18707,18706,18705,18704,18703,18702,18701,18700,18699,18698,18697,18696,18695,18694,18693,18692,18691,18690,18689,18688,18687,18686,18685,18684,18683,18682,18681,18680,18679,18678,18677,18676,18675,18674,18673,18672,18671,18670,18669,18668,18667,18666,18665,18664,18663,18662,18661,18660,18659,18658,18657,18656,18655,18654,18653,18652,18651,18650,18649,18648,18647,18646,18645,18644,18643,18642,18641,18640,18639,18638,18637,18636,18635,18634,18633,18632,18631,18630,18629,18628,18627,18626,18625,18624,18623,18622,18621,18620,18619,18618,18617,18616,18615,18614,18613,18612,18611,18610,18609,18608,18607,18606,18605,18604,18603,18602,18601,18600,18599,18598,18597,18596,18595,18594,18593,18592,18591,18590,18589,18588,18587,18586,18585,18584,18583,18582,18581,18580,18579,18578,18577,18576,18575,18574,18573,18572,18571,18570,18569,18568,18567,18566,18565,18564,18563,18562,18561,18560,18559,18558,18557,18556,18555,18554,18553,18552,18551,18550,18549,18548,18547,18546,18545,18544,18543,18542,18541,18540,18539,18538,18537,18536,18535,18534,18533,18532,18531,18530,18529,18528,18527,18526,18525,18524,18523,18522,18521,18520,18519,18518,18517,18516,18515,18514,18513,18512,18511,18510,18509,18508,18507,18506,18505,18504,18503,18502,18501,18500,18499,18498,18497,18496,18495,18494,18493,18492,18491,18490,18489,18488,18487,18486,18485,18484,18483,18482,18481,18480,18479,18478,18477,18476,18475,18474,18473,18472,18471,18470,18469,18468,18467,18466,18465,18464,18463,18462,18461,18460,18459,18458,18457,18456,18455,18454,18453,18452,18451,18450,18449,18448,18447,18446,18445,18444,18443,18442,18441,18440,18439,18438,18437,18436,18435,18434,18433,18432,18431,18430,18429,18428,18427,18426,18425,18424,18423,18422,18421,18420,18419,18418,18417,18416,18415,18414,18413,18412,18411,18410,18409,18408,18407,18406,18405,18404,18403,18402,18401,18400,18399,18398,18397,18396,18395,18394,18393,18392,18391,18390,18389,18388,18387,18386,18385,18384,18383,18382,18381,18380,18379,18378,18377,18376,18375,18374,18373,18372,18371,18370,18369,18368,18367,18366,18365,18364,18363,18362,18361,18360,18359,18358,18357,18356,18355,18354,18353,18352,18351,18350,18349,18348,18347,18346,18345,18344,18343,18342,18341,18340,18339,18338,18337,18336,18335,18334,18333,18332,18331,18330,18329,18328,18327,18326,18325,18324,18323,18322,18321,18320,18319,18318,18317,18316,18315,18314,18313,18312,18311,18310,18309,18308,18307,18306,18305,18304,18303,18302,18301,18300,18299,18298,18297,18296,18295,18294,18293,18292,18291,18290,18289,18288,18287,18286,18285,18284,18283,18282,18281,18280,18279,18278,18277,18276,18275,18274,18273,18272,18271,18270,18269,18268,18267,18266,18265,18264,18263,18262,18261,18260,18259,18258,18257,18256,18255,18254,18253,18252,18251,18250,18249,18248,18247,18246,18245,18244,18243,18242,18241,18240,18239,18238,18237,18236,18235,18234,18233,18232,18231,18230,18229,18228,18227,18226,18225,18224,18223,18222,18221,18220,18219,18218,18217,18216,18215,18214,18213,18212,18211,18210,18209,18208,18207,18206,18205,18204,18203,18202,18201,18200,18199,18198,18197,18196,18195,18194,18193,18192,18191,18190,18189,18188,18187,18186,18185,18184,18183,18182,18181,18180,18179,18178,18177,18176,18175,18174,18173,18172,18171,18170,18169,18168,18167,18166,18165,18164,18163,18162,18161,18160,18159,18158,18157,18156,18155,18154,18153,18152,18151,18150,18149,18148,18147,18146,18145,18144,18143,18142,18141,18140,18139,18138,18137,18136,18135,18134,18133,18132,18131,18130,18129,18128,18127,18126,18125,18124,18123,18122,18121,18120,18119,18118,18117,18116,18115,18114,18113,18112,18111,18110,18109,18108,18107,18106,18105,18104,18103,18102,18101,18100,18099,18098,18097,18096,18095,18094,18093,18092,18091,18090,18089,18088,18087,18086,18085,18084,18083,18082,18081,18080,18079,18078,18077,18076,18075,18074,18073,18072,18071,18070,18069,18068,18067,18066,18065,18064,18063,18062,18061,18060,18059,18058,18057,18056,18055,18054,18053,18052,18051,18050,18049,18048,18047,18046,18045,18044,18043,18042,18041,18040,18039,18038,18037,18036,18035,18034,18033,18032,18031,18030,18029,18028,18027,18026,18025,18024,18023,18022,18021,18020,18019,18018,18017,18016,18015,18014,18013,18012,18011,18010,18009,18008,18007,18006,18005,18004,18003,18002,18001,18000,17999,17998,17997,17996,17995,17994,17993,17992,17991,17990,17989,17988,17987,17986,17985,17984,17983,17982,17981,17980,17979,17978,17977,17976,17975,17974,17973,17972,17971,17970,17969,17968,17967,17966,17965,17964,17963,17962,17961,17960,17959,17958,17957,17956,17955,17954,17953,17952,17951,17950,17949,17948,17947,17946,17945,17944,17943,17942,17941,17940,17939,17938,17937,17936,17935,17934,17933,17932,17931,17930,17929,17928,17927,17926,17925,17924,17923,17922,17921,17920,17919,17918,17917,17916,17915,17914,17913,17912,17911,17910,17909,17908,17907,17906,17905,17904,17903,17902,17901,17900,17899,17898,17897,17896,17895,17894,17893,17892,17891,17890,17889,17888,17887,17886,17885,17884,17883,17882,17881,17880,17879,17878,17877,17876,17875,17874,17873,17872,17871,17870,17869,17868,17867,17866,17865,17864,17863,17862,17861,17860,17859,17858,17857,17856,17855,17854,17853,17852,17851,17850,17849,17848,17847,17846,17845,17844,17843,17842,17841,17840,17839,17838,17837,17836,17835,17834,17833,17832,17831,17830,17829,17828,17827,17826,17825,17824,17823,17822,17821,17820,17819,17818,17817,17816,17815,17814,17813,17812,17811,17810,17809,17808,17807,17806,17805,17804,17803,17802,17801,17800,17799,17798,17797,17796,17795,17794,17793,17792,17791,17790,17789,17788,17787,17786,17785,17784,17783,17782,17781,17780,17779,17778,17777,17776,17775,17774,17773,17772,17771,17770,17769,17768,17767,17766,17765,17764,17763,17762,17761,17760,17759,17758,17757,17756,17755,17754,17753,17752,17751,17750,17749,17748,17747,17746,17745,17744,17743,17742,17741,17740,17739,17738,17737,17736,17735,17734,17733,17732,17731,17730,17729,17728,17727,17726,17725,17724,17723,17722,17721,17720,17719,17718,17717,17716,17715,17714,17713,17712,17711,17710,17709,17708,17707,17706,17705,17704,17703,17702,17701,17700,17699,17698,17697,17696,17695,17694,17693,17692,17691,17690,17689,17688,17687,17686,17685,17684,17683,17682,17681,17680,17679,17678,17677,17676,17675,17674,17673,17672,17671,17670,17669,17668,17667,17666,17665,17664,17663,17662,17661,17660,17659,17658,17657,17656,17655,17654,17653,17652,17651,17650,17649,17648,17647,17646,17645,17644,17643,17642,17641,17640,17639,17638,17637,17636,17635,17634,17633,17632,17631,17630,17629,17628,17627,17626,17625,17624,17623,17622,17621,17620,17619,17618,17617,17616,17615,17614,17613,17612,17611,17610,17609,17608,17607,17606,17605,17604,17603,17602,17601,17600,17599,17598,17597,17596,17595,17594,17593,17592,17591,17590,17589,17588,17587,17586,17585,17584,17583,17582,17581,17580,17579,17578,17577,17576,17575,17574,17573,17572,17571,17570,17569,17568,17567,17566,17565,17564,17563,17562,17561,17560,17559,17558,17557,17556,17555,17554,17553,17552,17551,17550,17549,17548,17547,17546,17545,17544,17543,17542,17541,17540,17539,17538,17537,17536,17535,17534,17533,17532,17531,17530,17529,17528,17527,17526,17525,17524,17523,17522,17521,17520,17519,17518,17517,17516,17515,17514,17513,17512,17511,17510,17509,17508,17507,17506,17505,17504,17503,17502,17501,17500,17499,17498,17497,17496,17495,17494,17493,17492,17491,17490,17489,17488,17487,17486,17485,17484,17483,17482,17481,17480,17479,17478,17477,17476,17475,17474,17473,17472,17471,17470,17469,17468,17467,17466,17465,17464,17463,17462,17461,17460,17459,17458,17457,17456,17455,17454,17453,17452,17451,17450,17449,17448,17447,17446,17445,17444,17443,17442,17441,17440,17439,17438,17437,17436,17435,17434,17433,17432,17431,17430,17429,17428,17427,17426,17425,17424,17423,17422,17421,17420,17419,17418,17417,17416,17415,17414,17413,17412,17411,17410,17409,17408,17407,17406,17405,17404,17403,17402,17401,17400,17399,17398,17397,17396,17395,17394,17393,17392,17391,17390,17389,17388,17387,17386,17385,17384,17383,17382,17381,17380,17379,17378,17377,17376,17375,17374,17373,17372,17371,17370,17369,17368,17367,17366,17365,17364,17363,17362,17361,17360,17359,17358,17357,17356,17355,17354,17353,17352,17351,17350,17349,17348,17347,17346,17345,17344,17343,17342,17341,17340,17339,17338,17337,17336,17335,17334,17333,17332,17331,17330,17329,17328,17327,17326,17325,17324,17323,17322,17321,17320,17319,17318,17317,17316,17315,17314,17313,17312,17311,17310,17309,17308,17307,17306,17305,17304,17303,17302,17301,17300,17299,17298,17297,17296,17295,17294,17293,17292,17291,17290,17289,17288,17287,17286,17285,17284,17283,17282,17281,17280,17279,17278,17277,17276,17275,17274,17273,17272,17271,17270,17269,17268,17267,17266,17265,17264,17263,17262,17261,17260,17259,17258,17257,17256,17255,17254,17253,17252,17251,17250,17249,17248,17247,17246,17245,17244,17243,17242,17241,17240,17239,17238,17237,17236,17235,17234,17233,17232,17231,17230,17229,17228,17227,17226,17225,17224,17223,17222,17221,17220,17219,17218,17217,17216,17215,17214,17213,17212,17211,17210,17209,17208,17207,17206,17205,17204,17203,17202,17201,17200,17199,17198,17197,17196,17195,17194,17193,17192,17191,17190,17189,17188,17187,17186,17185,17184,17183,17182,17181,17180,17179,17178,17177,17176,17175,17174,17173,17172,17171,17170,17169,17168,17167,17166,17165,17164,17163,17162,17161,17160,17159,17158,17157,17156,17155,17154,17153,17152,17151,17150,17149,17148,17147,17146,17145,17144,17143,17142,17141,17140,17139,17138,17137,17136,17135,17134,17133,17132,17131,17130,17129,17128,17127,17126,17125,17124,17123,17122,17121,17120,17119,17118,17117,17116,17115,17114,17113,17112,17111,17110,17109,17108,17107,17106,17105,17104,17103,17102,17101,17100,17099,17098,17097,17096,17095,17094,17093,17092,17091,17090,17089,17088,17087,17086,17085,17084,17083,17082,17081,17080,17079,17078,17077,17076,17075,17074,17073,17072,17071,17070,17069,17068,17067,17066,17065,17064,17063,17062,17061,17060,17059,17058,17057,17056,17055,17054,17053,17052,17051,17050,17049,17048,17047,17046,17045,17044,17043,17042,17041,17040,17039,17038,17037,17036,17035,17034,17033,17032,17031,17030,17029,17028,17027,17026,17025,17024,17023,17022,17021,17020,17019,17018,17017,17016,17015,17014,17013,17012,17011,17010,17009,17008,17007,17006,17005,17004,17003,17002,17001,17000,16999,16998,16997,16996,16995,16994,16993,16992,16991,16990,16989,16988,16987,16986,16985,16984,16983,16982,16981,16980,16979,16978,16977,16976,16975,16974,16973,16972,16971,16970,16969,16968,16967,16966,16965,16964,16963,16962,16961,16960,16959,16958,16957,16956,16955,16954,16953,16952,16951,16950,16949,16948,16947,16946,16945,16944,16943,16942,16941,16940,16939,16938,16937,16936,16935,16934,16933,16932,16931,16930,16929,16928,16927,16926,16925,16924,16923,16922,16921,16920,16919,16918,16917,16916,16915,16914,16913,16912,16911,16910,16909,16908,16907,16906,16905,16904,16903,16902,16901,16900,16899,16898,16897,16896,16895,16894,16893,16892,16891,16890,16889,16888,16887,16886,16885,16884,16883,16882,16881,16880,16879,16878,16877,16876,16875,16874,16873,16872,16871,16870,16869,16868,16867,16866,16865,16864,16863,16862,16861,16860,16859,16858,16857,16856,16855,16854,16853,16852,16851,16850,16849,16848,16847,16846,16845,16844,16843,16842,16841,16840,16839,16838,16837,16836,16835,16834,16833,16832,16831,16830,16829,16828,16827,16826,16825,16824,16823,16822,16821,16820,16819,16818,16817,16816,16815,16814,16813,16812,16811,16810,16809,16808,16807,16806,16805,16804,16803,16802,16801,16800,16799,16798,16797,16796,16795,16794,16793,16792,16791,16790,16789,16788,16787,16786,16785,16784,16783,16782,16781,16780,16779,16778,16777,16776,16775,16774,16773,16772,16771,16770,16769,16768,16767,16766,16765,16764,16763,16762,16761,16760,16759,16758,16757,16756,16755,16754,16753,16752,16751,16750,16749,16748,16747,16746,16745,16744,16743,16742,16741,16740,16739,16738,16737,16736,16735,16734,16733,16732,16731,16730,16729,16728,16727,16726,16725,16724,16723,16722,16721,16720,16719,16718,16717,16716,16715,16714,16713,16712,16711,16710,16709,16708,16707,16706,16705,16704,16703,16702,16701,16700,16699,16698,16697,16696,16695,16694,16693,16692,16691,16690,16689,16688,16687,16686,16685,16684,16683,16682,16681,16680,16679,16678,16677,16676,16675,16674,16673,16672,16671,16670,16669,16668,16667,16666,16665,16664,16663,16662,16661,16660,16659,16658,16657,16656,16655,16654,16653,16652,16651,16650,16649,16648,16647,16646,16645,16644,16643,16642,16641,16640,16639,16638,16637,16636,16635,16634,16633,16632,16631,16630,16629,16628,16627,16626,16625,16624,16623,16622,16621,16620,16619,16618,16617,16616,16615,16614,16613,16612,16611,16610,16609,16608,16607,16606,16605,16604,16603,16602,16601,16600,16599,16598,16597,16596,16595,16594,16593,16592,16591,16590,16589,16588,16587,16586,16585,16584,16583,16582,16581,16580,16579,16578,16577,16576,16575,16574,16573,16572,16571,16570,16569,16568,16567,16566,16565,16564,16563,16562,16561,16560,16559,16558,16557,16556,16555,16554,16553,16552,16551,16550,16549,16548,16547,16546,16545,16544,16543,16542,16541,16540,16539,16538,16537,16536,16535,16534,16533,16532,16531,16530,16529,16528,16527,16526,16525,16524,16523,16522,16521,16520,16519,16518,16517,16516,16515,16514,16513,16512,16511,16510,16509,16508,16507,16506,16505,16504,16503,16502,16501,16500,16499,16498,16497,16496,16495,16494,16493,16492,16491,16490,16489,16488,16487,16486,16485,16484,16483,16482,16481,16480,16479,16478,16477,16476,16475,16474,16473,16472,16471,16470,16469,16468,16467,16466,16465,16464,16463,16462,16461,16460,16459,16458,16457,16456,16455,16454,16453,16452,16451,16450,16449,16448,16447,16446,16445,16444,16443,16442,16441,16440,16439,16438,16437,16436,16435,16434,16433,16432,16431,16430,16429,16428,16427,16426,16425,16424,16423,16422,16421,16420,16419,16418,16417,16416,16415,16414,16413,16412,16411,16410,16409,16408,16407,16406,16405,16404,16403,16402,16401,16400,16399,16398,16397,16396,16395,16394,16393,16392,16391,16390,16389,16388,16387,16386,16385,16384,16383,16382,16381,16380,16379,16378,16377,16376,16375,16374,16373,16372,16371,16370,16369,16368,16367,16366,16365,16364,16363,16362,16361,16360,16359,16358,16357,16356,16355,16354,16353,16352,16351,16350,16349,16348,16347,16346,16345,16344,16343,16342,16341,16340,16339,16338,16337,16336,16335,16334,16333,16332,16331,16330,16329,16328,16327,16326,16325,16324,16323,16322,16321,16320,16319,16318,16317,16316,16315,16314,16313,16312,16311,16310,16309,16308,16307,16306,16305,16304,16303,16302,16301,16300,16299,16298,16297,16296,16295,16294,16293,16292,16291,16290,16289,16288,16287,16286,16285,16284,16283,16282,16281,16280,16279,16278,16277,16276,16275,16274,16273,16272,16271,16270,16269,16268,16267,16266,16265,16264,16263,16262,16261,16260,16259,16258,16257,16256,16255,16254,16253,16252,16251,16250,16249,16248,16247,16246,16245,16244,16243,16242,16241,16240,16239,16238,16237,16236,16235,16234,16233,16232,16231,16230,16229,16228,16227,16226,16225,16224,16223,16222,16221,16220,16219,16218,16217,16216,16215,16214,16213,16212,16211,16210,16209,16208,16207,16206,16205,16204,16203,16202,16201,16200,16199,16198,16197,16196,16195,16194,16193,16192,16191,16190,16189,16188,16187,16186,16185,16184,16183,16182,16181,16180,16179,16178,16177,16176,16175,16174,16173,16172,16171,16170,16169,16168,16167,16166,16165,16164,16163,16162,16161,16160,16159,16158,16157,16156,16155,16154,16153,16152,16151,16150,16149,16148,16147,16146,16145,16144,16143,16142,16141,16140,16139,16138,16137,16136,16135,16134,16133,16132,16131,16130,16129,16128,16127,16126,16125,16124,16123,16122,16121,16120,16119,16118,16117,16116,16115,16114,16113,16112,16111,16110,16109,16108,16107,16106,16105,16104,16103,16102,16101,16100,16099,16098,16097,16096,16095,16094,16093,16092,16091,16090,16089,16088,16087,16086,16085,16084,16083,16082,16081,16080,16079,16078,16077,16076,16075,16074,16073,16072,16071,16070,16069,16068,16067,16066,16065,16064,16063,16062,16061,16060,16059,16058,16057,16056,16055,16054,16053,16052,16051,16050,16049,16048,16047,16046,16045,16044,16043,16042,16041,16040,16039,16038,16037,16036,16035,16034,16033,16032,16031,16030,16029,16028,16027,16026,16025,16024,16023,16022,16021,16020,16019,16018,16017,16016,16015,16014,16013,16012,16011,16010,16009,16008,16007,16006,16005,16004,16003,16002,16001,16000,15999,15998,15997,15996,15995,15994,15993,15992,15991,15990,15989,15988,15987,15986,15985,15984,15983,15982,15981,15980,15979,15978,15977,15976,15975,15974,15973,15972,15971,15970,15969,15968,15967,15966,15965,15964,15963,15962,15961,15960,15959,15958,15957,15956,15955,15954,15953,15952,15951,15950,15949,15948,15947,15946,15945,15944,15943,15942,15941,15940,15939,15938,15937,15936,15935,15934,15933,15932,15931,15930,15929,15928,15927,15926,15925,15924,15923,15922,15921,15920,15919,15918,15917,15916,15915,15914,15913,15912,15911,15910,15909,15908,15907,15906,15905,15904,15903,15902,15901,15900,15899,15898,15897,15896,15895,15894,15893,15892,15891,15890,15889,15888,15887,15886,15885,15884,15883,15882,15881,15880,15879,15878,15877,15876,15875,15874,15873,15872,15871,15870,15869,15868,15867,15866,15865,15864,15863,15862,15861,15860,15859,15858,15857,15856,15855,15854,15853,15852,15851,15850,15849,15848,15847,15846,15845,15844,15843,15842,15841,15840,15839,15838,15837,15836,15835,15834,15833,15832,15831,15830,15829,15828,15827,15826,15825,15824,15823,15822,15821,15820,15819,15818,15817,15816,15815,15814,15813,15812,15811,15810,15809,15808,15807,15806,15805,15804,15803,15802,15801,15800,15799,15798,15797,15796,15795,15794,15793,15792,15791,15790,15789,15788,15787,15786,15785,15784,15783,15782,15781,15780,15779,15778,15777,15776,15775,15774,15773,15772,15771,15770,15769,15768,15767,15766,15765,15764,15763,15762,15761,15760,15759,15758,15757,15756,15755,15754,15753,15752,15751,15750,15749,15748,15747,15746,15745,15744,15743,15742,15741,15740,15739,15738,15737,15736,15735,15734,15733,15732,15731,15730,15729,15728,15727,15726,15725,15724,15723,15722,15721,15720,15719,15718,15717,15716,15715,15714,15713,15712,15711,15710,15709,15708,15707,15706,15705,15704,15703,15702,15701,15700,15699,15698,15697,15696,15695,15694,15693,15692,15691,15690,15689,15688,15687,15686,15685,15684,15683,15682,15681,15680,15679,15678,15677,15676,15675,15674,15673,15672,15671,15670,15669,15668,15667,15666,15665,15664,15663,15662,15661,15660,15659,15658,15657,15656,15655,15654,15653,15652,15651,15650,15649,15648,15647,15646,15645,15644,15643,15642,15641,15640,15639,15638,15637,15636,15635,15634,15633,15632,15631,15630,15629,15628,15627,15626,15625,15624,15623,15622,15621,15620,15619,15618,15617,15616,15615,15614,15613,15612,15611,15610,15609,15608,15607,15606,15605,15604,15603,15602,15601,15600,15599,15598,15597,15596,15595,15594,15593,15592,15591,15590,15589,15588,15587,15586,15585,15584,15583,15582,15581,15580,15579,15578,15577,15576,15575,15574,15573,15572,15571,15570,15569,15568,15567,15566,15565,15564,15563,15562,15561,15560,15559,15558,15557,15556,15555,15554,15553,15552,15551,15550,15549,15548,15547,15546,15545,15544,15543,15542,15541,15540,15539,15538,15537,15536,15535,15534,15533,15532,15531,15530,15529,15528,15527,15526,15525,15524,15523,15522,15521,15520,15519,15518,15517,15516,15515,15514,15513,15512,15511,15510,15509,15508,15507,15506,15505,15504,15503,15502,15501,15500,15499,15498,15497,15496,15495,15494,15493,15492,15491,15490,15489,15488,15487,15486,15485,15484,15483,15482,15481,15480,15479,15478,15477,15476,15475,15474,15473,15472,15471,15470,15469,15468,15467,15466,15465,15464,15463,15462,15461,15460,15459,15458,15457,15456,15455,15454,15453,15452,15451,15450,15449,15448,15447,15446,15445,15444,15443,15442,15441,15440,15439,15438,15437,15436,15435,15434,15433,15432,15431,15430,15429,15428,15427,15426,15425,15424,15423,15422,15421,15420,15419,15418,15417,15416,15415,15414,15413,15412,15411,15410,15409,15408,15407,15406,15405,15404,15403,15402,15401,15400,15399,15398,15397,15396,15395,15394,15393,15392,15391,15390,15389,15388,15387,15386,15385,15384,15383,15382,15381,15380,15379,15378,15377,15376,15375,15374,15373,15372,15371,15370,15369,15368,15367,15366,15365,15364,15363,15362,15361,15360,15359,15358,15357,15356,15355,15354,15353,15352,15351,15350,15349,15348,15347,15346,15345,15344,15343,15342,15341,15340,15339,15338,15337,15336,15335,15334,15333,15332,15331,15330,15329,15328,15327,15326,15325,15324,15323,15322,15321,15320,15319,15318,15317,15316,15315,15314,15313,15312,15311,15310,15309,15308,15307,15306,15305,15304,15303,15302,15301,15300,15299,15298,15297,15296,15295,15294,15293,15292,15291,15290,15289,15288,15287,15286,15285,15284,15283,15282,15281,15280,15279,15278,15277,15276,15275,15274,15273,15272,15271,15270,15269,15268,15267,15266,15265,15264,15263,15262,15261,15260,15259,15258,15257,15256,15255,15254,15253,15252,15251,15250,15249,15248,15247,15246,15245,15244,15243,15242,15241,15240,15239,15238,15237,15236,15235,15234,15233,15232,15231,15230,15229,15228,15227,15226,15225,15224,15223,15222,15221,15220,15219,15218,15217,15216,15215,15214,15213,15212,15211,15210,15209,15208,15207,15206,15205,15204,15203,15202,15201,15200,15199,15198,15197,15196,15195,15194,15193,15192,15191,15190,15189,15188,15187,15186,15185,15184,15183,15182,15181,15180,15179,15178,15177,15176,15175,15174,15173,15172,15171,15170,15169,15168,15167,15166,15165,15164,15163,15162,15161,15160,15159,15158,15157,15156,15155,15154,15153,15152,15151,15150,15149,15148,15147,15146,15145,15144,15143,15142,15141,15140,15139,15138,15137,15136,15135,15134,15133,15132,15131,15130,15129,15128,15127,15126,15125,15124,15123,15122,15121,15120,15119,15118,15117,15116,15115,15114,15113,15112,15111,15110,15109,15108,15107,15106,15105,15104,15103,15102,15101,15100,15099,15098,15097,15096,15095,15094,15093,15092,15091,15090,15089,15088,15087,15086,15085,15084,15083,15082,15081,15080,15079,15078,15077,15076,15075,15074,15073,15072,15071,15070,15069,15068,15067,15066,15065,15064,15063,15062,15061,15060,15059,15058,15057,15056,15055,15054,15053,15052,15051,15050,15049,15048,15047,15046,15045,15044,15043,15042,15041,15040,15039,15038,15037,15036,15035,15034,15033,15032,15031,15030,15029,15028,15027,15026,15025,15024,15023,15022,15021,15020,15019,15018,15017,15016,15015,15014,15013,15012,15011,15010,15009,15008,15007,15006,15005,15004,15003,15002,15001,15000,14999,14998,14997,14996,14995,14994,14993,14992,14991,14990,14989,14988,14987,14986,14985,14984,14983,14982,14981,14980,14979,14978,14977,14976,14975,14974,14973,14972,14971,14970,14969,14968,14967,14966,14965,14964,14963,14962,14961,14960,14959,14958,14957,14956,14955,14954,14953,14952,14951,14950,14949,14948,14947,14946,14945,14944,14943,14942,14941,14940,14939,14938,14937,14936,14935,14934,14933,14932,14931,14930,14929,14928,14927,14926,14925,14924,14923,14922,14921,14920,14919,14918,14917,14916,14915,14914,14913,14912,14911,14910,14909,14908,14907,14906,14905,14904,14903,14902,14901,14900,14899,14898,14897,14896,14895,14894,14893,14892,14891,14890,14889,14888,14887,14886,14885,14884,14883,14882,14881,14880,14879,14878,14877,14876,14875,14874,14873,14872,14871,14870,14869,14868,14867,14866,14865,14864,14863,14862,14861,14860,14859,14858,14857,14856,14855,14854,14853,14852,14851,14850,14849,14848,14847,14846,14845,14844,14843,14842,14841,14840,14839,14838,14837,14836,14835,14834,14833,14832,14831,14830,14829,14828,14827,14826,14825,14824,14823,14822,14821,14820,14819,14818,14817,14816,14815,14814,14813,14812,14811,14810,14809,14808,14807,14806,14805,14804,14803,14802,14801,14800,14799,14798,14797,14796,14795,14794,14793,14792,14791,14790,14789,14788,14787,14786,14785,14784,14783,14782,14781,14780,14779,14778,14777,14776,14775,14774,14773,14772,14771,14770,14769,14768,14767,14766,14765,14764,14763,14762,14761,14760,14759,14758,14757,14756,14755,14754,14753,14752,14751,14750,14749,14748,14747,14746,14745,14744,14743,14742,14741,14740,14739,14738,14737,14736,14735,14734,14733,14732,14731,14730,14729,14728,14727,14726,14725,14724,14723,14722,14721,14720,14719,14718,14717,14716,14715,14714,14713,14712,14711,14710,14709,14708,14707,14706,14705,14704,14703,14702,14701,14700,14699,14698,14697,14696,14695,14694,14693,14692,14691,14690,14689,14688,14687,14686,14685,14684,14683,14682,14681,14680,14679,14678,14677,14676,14675,14674,14673,14672,14671,14670,14669,14668,14667,14666,14665,14664,14663,14662,14661,14660,14659,14658,14657,14656,14655,14654,14653,14652,14651,14650,14649,14648,14647,14646,14645,14644,14643,14642,14641,14640,14639,14638,14637,14636,14635,14634,14633,14632,14631,14630,14629,14628,14627,14626,14625,14624,14623,14622,14621,14620,14619,14618,14617,14616,14615,14614,14613,14612,14611,14610,14609,14608,14607,14606,14605,14604,14603,14602,14601,14600,14599,14598,14597,14596,14595,14594,14593,14592,14591,14590,14589,14588,14587,14586,14585,14584,14583,14582,14581,14580,14579,14578,14577,14576,14575,14574,14573,14572,14571,14570,14569,14568,14567,14566,14565,14564,14563,14562,14561,14560,14559,14558,14557,14556,14555,14554,14553,14552,14551,14550,14549,14548,14547,14546,14545,14544,14543,14542,14541,14540,14539,14538,14537,14536,14535,14534,14533,14532,14531,14530,14529,14528,14527,14526,14525,14524,14523,14522,14521,14520,14519,14518,14517,14516,14515,14514,14513,14512,14511,14510,14509,14508,14507,14506,14505,14504,14503,14502,14501,14500,14499,14498,14497,14496,14495,14494,14493,14492,14491,14490,14489,14488,14487,14486,14485,14484,14483,14482,14481,14480,14479,14478,14477,14476,14475,14474,14473,14472,14471,14470,14469,14468,14467,14466,14465,14464,14463,14462,14461,14460,14459,14458,14457,14456,14455,14454,14453,14452,14451,14450,14449,14448,14447,14446,14445,14444,14443,14442,14441,14440,14439,14438,14437,14436,14435,14434,14433,14432,14431,14430,14429,14428,14427,14426,14425,14424,14423,14422,14421,14420,14419,14418,14417,14416,14415,14414,14413,14412,14411,14410,14409,14408,14407,14406,14405,14404,14403,14402,14401,14400,14399,14398,14397,14396,14395,14394,14393,14392,14391,14390,14389,14388,14387,14386,14385,14384,14383,14382,14381,14380,14379,14378,14377,14376,14375,14374,14373,14372,14371,14370,14369,14368,14367,14366,14365,14364,14363,14362,14361,14360,14359,14358,14357,14356,14355,14354,14353,14352,14351,14350,14349,14348,14347,14346,14345,14344,14343,14342,14341,14340,14339,14338,14337,14336,14335,14334,14333,14332,14331,14330,14329,14328,14327,14326,14325,14324,14323,14322,14321,14320,14319,14318,14317,14316,14315,14314,14313,14312,14311,14310,14309,14308,14307,14306,14305,14304,14303,14302,14301,14300,14299,14298,14297,14296,14295,14294,14293,14292,14291,14290,14289,14288,14287,14286,14285,14284,14283,14282,14281,14280,14279,14278,14277,14276,14275,14274,14273,14272,14271,14270,14269,14268,14267,14266,14265,14264,14263,14262,14261,14260,14259,14258,14257,14256,14255,14254,14253,14252,14251,14250,14249,14248,14247,14246,14245,14244,14243,14242,14241,14240,14239,14238,14237,14236,14235,14234,14233,14232,14231,14230,14229,14228,14227,14226,14225,14224,14223,14222,14221,14220,14219,14218,14217,14216,14215,14214,14213,14212,14211,14210,14209,14208,14207,14206,14205,14204,14203,14202,14201,14200,14199,14198,14197,14196,14195,14194,14193,14192,14191,14190,14189,14188,14187,14186,14185,14184,14183,14182,14181,14180,14179,14178,14177,14176,14175,14174,14173,14172,14171,14170,14169,14168,14167,14166,14165,14164,14163,14162,14161,14160,14159,14158,14157,14156,14155,14154,14153,14152,14151,14150,14149,14148,14147,14146,14145,14144,14143,14142,14141,14140,14139,14138,14137,14136,14135,14134,14133,14132,14131,14130,14129,14128,14127,14126,14125,14124,14123,14122,14121,14120,14119,14118,14117,14116,14115,14114,14113,14112,14111,14110,14109,14108,14107,14106,14105,14104,14103,14102,14101,14100,14099,14098,14097,14096,14095,14094,14093,14092,14091,14090,14089,14088,14087,14086,14085,14084,14083,14082,14081,14080,14079,14078,14077,14076,14075,14074,14073,14072,14071,14070,14069,14068,14067,14066,14065,14064,14063,14062,14061,14060,14059,14058,14057,14056,14055,14054,14053,14052,14051,14050,14049,14048,14047,14046,14045,14044,14043,14042,14041,14040,14039,14038,14037,14036,14035,14034,14033,14032,14031,14030,14029,14028,14027,14026,14025,14024,14023,14022,14021,14020,14019,14018,14017,14016,14015,14014,14013,14012,14011,14010,14009,14008,14007,14006,14005,14004,14003,14002,14001,14000,13999,13998,13997,13996,13995,13994,13993,13992,13991,13990,13989,13988,13987,13986,13985,13984,13983,13982,13981,13980,13979,13978,13977,13976,13975,13974,13973,13972,13971,13970,13969,13968,13967,13966,13965,13964,13963,13962,13961,13960,13959,13958,13957,13956,13955,13954,13953,13952,13951,13950,13949,13948,13947,13946,13945,13944,13943,13942,13941,13940,13939,13938,13937,13936,13935,13934,13933,13932,13931,13930,13929,13928,13927,13926,13925,13924,13923,13922,13921,13920,13919,13918,13917,13916,13915,13914,13913,13912,13911,13910,13909,13908,13907,13906,13905,13904,13903,13902,13901,13900,13899,13898,13897,13896,13895,13894,13893,13892,13891,13890,13889,13888,13887,13886,13885,13884,13883,13882,13881,13880,13879,13878,13877,13876,13875,13874,13873,13872,13871,13870,13869,13868,13867,13866,13865,13864,13863,13862,13861,13860,13859,13858,13857,13856,13855,13854,13853,13852,13851,13850,13849,13848,13847,13846,13845,13844,13843,13842,13841,13840,13839,13838,13837,13836,13835,13834,13833,13832,13831,13830,13829,13828,13827,13826,13825,13824,13823,13822,13821,13820,13819,13818,13817,13816,13815,13814,13813,13812,13811,13810,13809,13808,13807,13806,13805,13804,13803,13802,13801,13800,13799,13798,13797,13796,13795,13794,13793,13792,13791,13790,13789,13788,13787,13786,13785,13784,13783,13782,13781,13780,13779,13778,13777,13776,13775,13774,13773,13772,13771,13770,13769,13768,13767,13766,13765,13764,13763,13762,13761,13760,13759,13758,13757,13756,13755,13754,13753,13752,13751,13750,13749,13748,13747,13746,13745,13744,13743,13742,13741,13740,13739,13738,13737,13736,13735,13734,13733,13732,13731,13730,13729,13728,13727,13726,13725,13724,13723,13722,13721,13720,13719,13718,13717,13716,13715,13714,13713,13712,13711,13710,13709,13708,13707,13706,13705,13704,13703,13702,13701,13700,13699,13698,13697,13696,13695,13694,13693,13692,13691,13690,13689,13688,13687,13686,13685,13684,13683,13682,13681,13680,13679,13678,13677,13676,13675,13674,13673,13672,13671,13670,13669,13668,13667,13666,13665,13664,13663,13662,13661,13660,13659,13658,13657,13656,13655,13654,13653,13652,13651,13650,13649,13648,13647,13646,13645,13644,13643,13642,13641,13640,13639,13638,13637,13636,13635,13634,13633,13632,13631,13630,13629,13628,13627,13626,13625,13624,13623,13622,13621,13620,13619,13618,13617,13616,13615,13614,13613,13612,13611,13610,13609,13608,13607,13606,13605,13604,13603,13602,13601,13600,13599,13598,13597,13596,13595,13594,13593,13592,13591,13590,13589,13588,13587,13586,13585,13584,13583,13582,13581,13580,13579,13578,13577,13576,13575,13574,13573,13572,13571,13570,13569,13568,13567,13566,13565,13564,13563,13562,13561,13560,13559,13558,13557,13556,13555,13554,13553,13552,13551,13550,13549,13548,13547,13546,13545,13544,13543,13542,13541,13540,13539,13538,13537,13536,13535,13534,13533,13532,13531,13530,13529,13528,13527,13526,13525,13524,13523,13522,13521,13520,13519,13518,13517,13516,13515,13514,13513,13512,13511,13510,13509,13508,13507,13506,13505,13504,13503,13502,13501,13500,13499,13498,13497,13496,13495,13494,13493,13492,13491,13490,13489,13488,13487,13486,13485,13484,13483,13482,13481,13480,13479,13478,13477,13476,13475,13474,13473,13472,13471,13470,13469,13468,13467,13466,13465,13464,13463,13462,13461,13460,13459,13458,13457,13456,13455,13454,13453,13452,13451,13450,13449,13448,13447,13446,13445,13444,13443,13442,13441,13440,13439,13438,13437,13436,13435,13434,13433,13432,13431,13430,13429,13428,13427,13426,13425,13424,13423,13422,13421,13420,13419,13418,13417,13416,13415,13414,13413,13412,13411,13410,13409,13408,13407,13406,13405,13404,13403,13402,13401,13400,13399,13398,13397,13396,13395,13394,13393,13392,13391,13390,13389,13388,13387,13386,13385,13384,13383,13382,13381,13380,13379,13378,13377,13376,13375,13374,13373,13372,13371,13370,13369,13368,13367,13366,13365,13364,13363,13362,13361,13360,13359,13358,13357,13356,13355,13354,13353,13352,13351,13350,13349,13348,13347,13346,13345,13344,13343,13342,13341,13340,13339,13338,13337,13336,13335,13334,13333,13332,13331,13330,13329,13328,13327,13326,13325,13324,13323,13322,13321,13320,13319,13318,13317,13316,13315,13314,13313,13312,13311,13310,13309,13308,13307,13306,13305,13304,13303,13302,13301,13300,13299,13298,13297,13296,13295,13294,13293,13292,13291,13290,13289,13288,13287,13286,13285,13284,13283,13282,13281,13280,13279,13278,13277,13276,13275,13274,13273,13272,13271,13270,13269,13268,13267,13266,13265,13264,13263,13262,13261,13260,13259,13258,13257,13256,13255,13254,13253,13252,13251,13250,13249,13248,13247,13246,13245,13244,13243,13242,13241,13240,13239,13238,13237,13236,13235,13234,13233,13232,13231,13230,13229,13228,13227,13226,13225,13224,13223,13222,13221,13220,13219,13218,13217,13216,13215,13214,13213,13212,13211,13210,13209,13208,13207,13206,13205,13204,13203,13202,13201,13200,13199,13198,13197,13196,13195,13194,13193,13192,13191,13190,13189,13188,13187,13186,13185,13184,13183,13182,13181,13180,13179,13178,13177,13176,13175,13174,13173,13172,13171,13170,13169,13168,13167,13166,13165,13164,13163,13162,13161,13160,13159,13158,13157,13156,13155,13154,13153,13152,13151,13150,13149,13148,13147,13146,13145,13144,13143,13142,13141,13140,13139,13138,13137,13136,13135,13134,13133,13132,13131,13130,13129,13128,13127,13126,13125,13124,13123,13122,13121,13120,13119,13118,13117,13116,13115,13114,13113,13112,13111,13110,13109,13108,13107,13106,13105,13104,13103,13102,13101,13100,13099,13098,13097,13096,13095,13094,13093,13092,13091,13090,13089,13088,13087,13086,13085,13084,13083,13082,13081,13080,13079,13078,13077,13076,13075,13074,13073,13072,13071,13070,13069,13068,13067,13066,13065,13064,13063,13062,13061,13060,13059,13058,13057,13056,13055,13054,13053,13052,13051,13050,13049,13048,13047,13046,13045,13044,13043,13042,13041,13040,13039,13038,13037,13036,13035,13034,13033,13032,13031,13030,13029,13028,13027,13026,13025,13024,13023,13022,13021,13020,13019,13018,13017,13016,13015,13014,13013,13012,13011,13010,13009,13008,13007,13006,13005,13004,13003,13002,13001,13000,12999,12998,12997,12996,12995,12994,12993,12992,12991,12990,12989,12988,12987,12986,12985,12984,12983,12982,12981,12980,12979,12978,12977,12976,12975,12974,12973,12972,12971,12970,12969,12968,12967,12966,12965,12964,12963,12962,12961,12960,12959,12958,12957,12956,12955,12954,12953,12952,12951,12950,12949,12948,12947,12946,12945,12944,12943,12942,12941,12940,12939,12938,12937,12936,12935,12934,12933,12932,12931,12930,12929,12928,12927,12926,12925,12924,12923,12922,12921,12920,12919,12918,12917,12916,12915,12914,12913,12912,12911,12910,12909,12908,12907,12906,12905,12904,12903,12902,12901,12900,12899,12898,12897,12896,12895,12894,12893,12892,12891,12890,12889,12888,12887,12886,12885,12884,12883,12882,12881,12880,12879,12878,12877,12876,12875,12874,12873,12872,12871,12870,12869,12868,12867,12866,12865,12864,12863,12862,12861,12860,12859,12858,12857,12856,12855,12854,12853,12852,12851,12850,12849,12848,12847,12846,12845,12844,12843,12842,12841,12840,12839,12838,12837,12836,12835,12834,12833,12832,12831,12830,12829,12828,12827,12826,12825,12824,12823,12822,12821,12820,12819,12818,12817,12816,12815,12814,12813,12812,12811,12810,12809,12808,12807,12806,12805,12804,12803,12802,12801,12800,12799,12798,12797,12796,12795,12794,12793,12792,12791,12790,12789,12788,12787,12786,12785,12784,12783,12782,12781,12780,12779,12778,12777,12776,12775,12774,12773,12772,12771,12770,12769,12768,12767,12766,12765,12764,12763,12762,12761,12760,12759,12758,12757,12756,12755,12754,12753,12752,12751,12750,12749,12748,12747,12746,12745,12744,12743,12742,12741,12740,12739,12738,12737,12736,12735,12734,12733,12732,12731,12730,12729,12728,12727,12726,12725,12724,12723,12722,12721,12720,12719,12718,12717,12716,12715,12714,12713,12712,12711,12710,12709,12708,12707,12706,12705,12704,12703,12702,12701,12700,12699,12698,12697,12696,12695,12694,12693,12692,12691,12690,12689,12688,12687,12686,12685,12684,12683,12682,12681,12680,12679,12678,12677,12676,12675,12674,12673,12672,12671,12670,12669,12668,12667,12666,12665,12664,12663,12662,12661,12660,12659,12658,12657,12656,12655,12654,12653,12652,12651,12650,12649,12648,12647,12646,12645,12644,12643,12642,12641,12640,12639,12638,12637,12636,12635,12634,12633,12632,12631,12630,12629,12628,12627,12626,12625,12624,12623,12622,12621,12620,12619,12618,12617,12616,12615,12614,12613,12612,12611,12610,12609,12608,12607,12606,12605,12604,12603,12602,12601,12600,12599,12598,12597,12596,12595,12594,12593,12592,12591,12590,12589,12588,12587,12586,12585,12584,12583,12582,12581,12580,12579,12578,12577,12576,12575,12574,12573,12572,12571,12570,12569,12568,12567,12566,12565,12564,12563,12562,12561,12560,12559,12558,12557,12556,12555,12554,12553,12552,12551,12550,12549,12548,12547,12546,12545,12544,12543,12542,12541,12540,12539,12538,12537,12536,12535,12534,12533,12532,12531,12530,12529,12528,12527,12526,12525,12524,12523,12522,12521,12520,12519,12518,12517,12516,12515,12514,12513,12512,12511,12510,12509,12508,12507,12506,12505,12504,12503,12502,12501,12500,12499,12498,12497,12496,12495,12494,12493,12492,12491,12490,12489,12488,12487,12486,12485,12484,12483,12482,12481,12480,12479,12478,12477,12476,12475,12474,12473,12472,12471,12470,12469,12468,12467,12466,12465,12464,12463,12462,12461,12460,12459,12458,12457,12456,12455,12454,12453,12452,12451,12450,12449,12448,12447,12446,12445,12444,12443,12442,12441,12440,12439,12438,12437,12436,12435,12434,12433,12432,12431,12430,12429,12428,12427,12426,12425,12424,12423,12422,12421,12420,12419,12418,12417,12416,12415,12414,12413,12412,12411,12410,12409,12408,12407,12406,12405,12404,12403,12402,12401,12400,12399,12398,12397,12396,12395,12394,12393,12392,12391,12390,12389,12388,12387,12386,12385,12384,12383,12382,12381,12380,12379,12378,12377,12376,12375,12374,12373,12372,12371,12370,12369,12368,12367,12366,12365,12364,12363,12362,12361,12360,12359,12358,12357,12356,12355,12354,12353,12352,12351,12350,12349,12348,12347,12346,12345,12344,12343,12342,12341,12340,12339,12338,12337,12336,12335,12334,12333,12332,12331,12330,12329,12328,12327,12326,12325,12324,12323,12322,12321,12320,12319,12318,12317,12316,12315,12314,12313,12312,12311,12310,12309,12308,12307,12306,12305,12304,12303,12302,12301,12300,12299,12298,12297,12296,12295,12294,12293,12292,12291,12290,12289,12288,12287,12286,12285,12284,12283,12282,12281,12280,12279,12278,12277,12276,12275,12274,12273,12272,12271,12270,12269,12268,12267,12266,12265,12264,12263,12262,12261,12260,12259,12258,12257,12256,12255,12254,12253,12252,12251,12250,12249,12248,12247,12246,12245,12244,12243,12242,12241,12240,12239,12238,12237,12236,12235,12234,12233,12232,12231,12230,12229,12228,12227,12226,12225,12224,12223,12222,12221,12220,12219,12218,12217,12216,12215,12214,12213,12212,12211,12210,12209,12208,12207,12206,12205,12204,12203,12202,12201,12200,12199,12198,12197,12196,12195,12194,12193,12192,12191,12190,12189,12188,12187,12186,12185,12184,12183,12182,12181,12180,12179,12178,12177,12176,12175,12174,12173,12172,12171,12170,12169,12168,12167,12166,12165,12164,12163,12162,12161,12160,12159,12158,12157,12156,12155,12154,12153,12152,12151,12150,12149,12148,12147,12146,12145,12144,12143,12142,12141,12140,12139,12138,12137,12136,12135,12134,12133,12132,12131,12130,12129,12128,12127,12126,12125,12124,12123,12122,12121,12120,12119,12118,12117,12116,12115,12114,12113,12112,12111,12110,12109,12108,12107,12106,12105,12104,12103,12102,12101,12100,12099,12098,12097,12096,12095,12094,12093,12092,12091,12090,12089,12088,12087,12086,12085,12084,12083,12082,12081,12080,12079,12078,12077,12076,12075,12074,12073,12072,12071,12070,12069,12068,12067,12066,12065,12064,12063,12062,12061,12060,12059,12058,12057,12056,12055,12054,12053,12052,12051,12050,12049,12048,12047,12046,12045,12044,12043,12042,12041,12040,12039,12038,12037,12036,12035,12034,12033,12032,12031,12030,12029,12028,12027,12026,12025,12024,12023,12022,12021,12020,12019,12018,12017,12016,12015,12014,12013,12012,12011,12010,12009,12008,12007,12006,12005,12004,12003,12002,12001,12000,11999,11998,11997,11996,11995,11994,11993,11992,11991,11990,11989,11988,11987,11986,11985,11984,11983,11982,11981,11980,11979,11978,11977,11976,11975,11974,11973,11972,11971,11970,11969,11968,11967,11966,11965,11964,11963,11962,11961,11960,11959,11958,11957,11956,11955,11954,11953,11952,11951,11950,11949,11948,11947,11946,11945,11944,11943,11942,11941,11940,11939,11938,11937,11936,11935,11934,11933,11932,11931,11930,11929,11928,11927,11926,11925,11924,11923,11922,11921,11920,11919,11918,11917,11916,11915,11914,11913,11912,11911,11910,11909,11908,11907,11906,11905,11904,11903,11902,11901,11900,11899,11898,11897,11896,11895,11894,11893,11892,11891,11890,11889,11888,11887,11886,11885,11884,11883,11882,11881,11880,11879,11878,11877,11876,11875,11874,11873,11872,11871,11870,11869,11868,11867,11866,11865,11864,11863,11862,11861,11860,11859,11858,11857,11856,11855,11854,11853,11852,11851,11850,11849,11848,11847,11846,11845,11844,11843,11842,11841,11840,11839,11838,11837,11836,11835,11834,11833,11832,11831,11830,11829,11828,11827,11826,11825,11824,11823,11822,11821,11820,11819,11818,11817,11816,11815,11814,11813,11812,11811,11810,11809,11808,11807,11806,11805,11804,11803,11802,11801,11800,11799,11798,11797,11796,11795,11794,11793,11792,11791,11790,11789,11788,11787,11786,11785,11784,11783,11782,11781,11780,11779,11778,11777,11776,11775,11774,11773,11772,11771,11770,11769,11768,11767,11766,11765,11764,11763,11762,11761,11760,11759,11758,11757,11756,11755,11754,11753,11752,11751,11750,11749,11748,11747,11746,11745,11744,11743,11742,11741,11740,11739,11738,11737,11736,11735,11734,11733,11732,11731,11730,11729,11728,11727,11726,11725,11724,11723,11722,11721,11720,11719,11718,11717,11716,11715,11714,11713,11712,11711,11710,11709,11708,11707,11706,11705,11704,11703,11702,11701,11700,11699,11698,11697,11696,11695,11694,11693,11692,11691,11690,11689,11688,11687,11686,11685,11684,11683,11682,11681,11680,11679,11678,11677,11676,11675,11674,11673,11672,11671,11670,11669,11668,11667,11666,11665,11664,11663,11662,11661,11660,11659,11658,11657,11656,11655,11654,11653,11652,11651,11650,11649,11648,11647,11646,11645,11644,11643,11642,11641,11640,11639,11638,11637,11636,11635,11634,11633,11632,11631,11630,11629,11628,11627,11626,11625,11624,11623,11622,11621,11620,11619,11618,11617,11616,11615,11614,11613,11612,11611,11610,11609,11608,11607,11606,11605,11604,11603,11602,11601,11600,11599,11598,11597,11596,11595,11594,11593,11592,11591,11590,11589,11588,11587,11586,11585,11584,11583,11582,11581,11580,11579,11578,11577,11576,11575,11574,11573,11572,11571,11570,11569,11568,11567,11566,11565,11564,11563,11562,11561,11560,11559,11558,11557,11556,11555,11554,11553,11552,11551,11550,11549,11548,11547,11546,11545,11544,11543,11542,11541,11540,11539,11538,11537,11536,11535,11534,11533,11532,11531,11530,11529,11528,11527,11526,11525,11524,11523,11522,11521,11520,11519,11518,11517,11516,11515,11514,11513,11512,11511,11510,11509,11508,11507,11506,11505,11504,11503,11502,11501,11500,11499,11498,11497,11496,11495,11494,11493,11492,11491,11490,11489,11488,11487,11486,11485,11484,11483,11482,11481,11480,11479,11478,11477,11476,11475,11474,11473,11472,11471,11470,11469,11468,11467,11466,11465,11464,11463,11462,11461,11460,11459,11458,11457,11456,11455,11454,11453,11452,11451,11450,11449,11448,11447,11446,11445,11444,11443,11442,11441,11440,11439,11438,11437,11436,11435,11434,11433,11432,11431,11430,11429,11428,11427,11426,11425,11424,11423,11422,11421,11420,11419,11418,11417,11416,11415,11414,11413,11412,11411,11410,11409,11408,11407,11406,11405,11404,11403,11402,11401,11400,11399,11398,11397,11396,11395,11394,11393,11392,11391,11390,11389,11388,11387,11386,11385,11384,11383,11382,11381,11380,11379,11378,11377,11376,11375,11374,11373,11372,11371,11370,11369,11368,11367,11366,11365,11364,11363,11362,11361,11360,11359,11358,11357,11356,11355,11354,11353,11352,11351,11350,11349,11348,11347,11346,11345,11344,11343,11342,11341,11340,11339,11338,11337,11336,11335,11334,11333,11332,11331,11330,11329,11328,11327,11326,11325,11324,11323,11322,11321,11320,11319,11318,11317,11316,11315,11314,11313,11312,11311,11310,11309,11308,11307,11306,11305,11304,11303,11302,11301,11300,11299,11298,11297,11296,11295,11294,11293,11292,11291,11290,11289,11288,11287,11286,11285,11284,11283,11282,11281,11280,11279,11278,11277,11276,11275,11274,11273,11272,11271,11270,11269,11268,11267,11266,11265,11264,11263,11262,11261,11260,11259,11258,11257,11256,11255,11254,11253,11252,11251,11250,11249,11248,11247,11246,11245,11244,11243,11242,11241,11240,11239,11238,11237,11236,11235,11234,11233,11232,11231,11230,11229,11228,11227,11226,11225,11224,11223,11222,11221,11220,11219,11218,11217,11216,11215,11214,11213,11212,11211,11210,11209,11208,11207,11206,11205,11204,11203,11202,11201,11200,11199,11198,11197,11196,11195,11194,11193,11192,11191,11190,11189,11188,11187,11186,11185,11184,11183,11182,11181,11180,11179,11178,11177,11176,11175,11174,11173,11172,11171,11170,11169,11168,11167,11166,11165,11164,11163,11162,11161,11160,11159,11158,11157,11156,11155,11154,11153,11152,11151,11150,11149,11148,11147,11146,11145,11144,11143,11142,11141,11140,11139,11138,11137,11136,11135,11134,11133,11132,11131,11130,11129,11128,11127,11126,11125,11124,11123,11122,11121,11120,11119,11118,11117,11116,11115,11114,11113,11112,11111,11110,11109,11108,11107,11106,11105,11104,11103,11102,11101,11100,11099,11098,11097,11096,11095,11094,11093,11092,11091,11090,11089,11088,11087,11086,11085,11084,11083,11082,11081,11080,11079,11078,11077,11076,11075,11074,11073,11072,11071,11070,11069,11068,11067,11066,11065,11064,11063,11062,11061,11060,11059,11058,11057,11056,11055,11054,11053,11052,11051,11050,11049,11048,11047,11046,11045,11044,11043,11042,11041,11040,11039,11038,11037,11036,11035,11034,11033,11032,11031,11030,11029,11028,11027,11026,11025,11024,11023,11022,11021,11020,11019,11018,11017,11016,11015,11014,11013,11012,11011,11010,11009,11008,11007,11006,11005,11004,11003,11002,11001,11000,10999,10998,10997,10996,10995,10994,10993,10992,10991,10990,10989,10988,10987,10986,10985,10984,10983,10982,10981,10980,10979,10978,10977,10976,10975,10974,10973,10972,10971,10970,10969,10968,10967,10966,10965,10964,10963,10962,10961,10960,10959,10958,10957,10956,10955,10954,10953,10952,10951,10950,10949,10948,10947,10946,10945,10944,10943,10942,10941,10940,10939,10938,10937,10936,10935,10934,10933,10932,10931,10930,10929,10928,10927,10926,10925,10924,10923,10922,10921,10920,10919,10918,10917,10916,10915,10914,10913,10912,10911,10910,10909,10908,10907,10906,10905,10904,10903,10902,10901,10900,10899,10898,10897,10896,10895,10894,10893,10892,10891,10890,10889,10888,10887,10886,10885,10884,10883,10882,10881,10880,10879,10878,10877,10876,10875,10874,10873,10872,10871,10870,10869,10868,10867,10866,10865,10864,10863,10862,10861,10860,10859,10858,10857,10856,10855,10854,10853,10852,10851,10850,10849,10848,10847,10846,10845,10844,10843,10842,10841,10840,10839,10838,10837,10836,10835,10834,10833,10832,10831,10830,10829,10828,10827,10826,10825,10824,10823,10822,10821,10820,10819,10818,10817,10816,10815,10814,10813,10812,10811,10810,10809,10808,10807,10806,10805,10804,10803,10802,10801,10800,10799,10798,10797,10796,10795,10794,10793,10792,10791,10790,10789,10788,10787,10786,10785,10784,10783,10782,10781,10780,10779,10778,10777,10776,10775,10774,10773,10772,10771,10770,10769,10768,10767,10766,10765,10764,10763,10762,10761,10760,10759,10758,10757,10756,10755,10754,10753,10752,10751,10750,10749,10748,10747,10746,10745,10744,10743,10742,10741,10740,10739,10738,10737,10736,10735,10734,10733,10732,10731,10730,10729,10728,10727,10726,10725,10724,10723,10722,10721,10720,10719,10718,10717,10716,10715,10714,10713,10712,10711,10710,10709,10708,10707,10706,10705,10704,10703,10702,10701,10700,10699,10698,10697,10696,10695,10694,10693,10692,10691,10690,10689,10688,10687,10686,10685,10684,10683,10682,10681,10680,10679,10678,10677,10676,10675,10674,10673,10672,10671,10670,10669,10668,10667,10666,10665,10664,10663,10662,10661,10660,10659,10658,10657,10656,10655,10654,10653,10652,10651,10650,10649,10648,10647,10646,10645,10644,10643,10642,10641,10640,10639,10638,10637,10636,10635,10634,10633,10632,10631,10630,10629,10628,10627,10626,10625,10624,10623,10622,10621,10620,10619,10618,10617,10616,10615,10614,10613,10612,10611,10610,10609,10608,10607,10606,10605,10604,10603,10602,10601,10600,10599,10598,10597,10596,10595,10594,10593,10592,10591,10590,10589,10588,10587,10586,10585,10584,10583,10582,10581,10580,10579,10578,10577,10576,10575,10574,10573,10572,10571,10570,10569,10568,10567,10566,10565,10564,10563,10562,10561,10560,10559,10558,10557,10556,10555,10554,10553,10552,10551,10550,10549,10548,10547,10546,10545,10544,10543,10542,10541,10540,10539,10538,10537,10536,10535,10534,10533,10532,10531,10530,10529,10528,10527,10526,10525,10524,10523,10522,10521,10520,10519,10518,10517,10516,10515,10514,10513,10512,10511,10510,10509,10508,10507,10506,10505,10504,10503,10502,10501,10500,10499,10498,10497,10496,10495,10494,10493,10492,10491,10490,10489,10488,10487,10486,10485,10484,10483,10482,10481,10480,10479,10478,10477,10476,10475,10474,10473,10472,10471,10470,10469,10468,10467,10466,10465,10464,10463,10462,10461,10460,10459,10458,10457,10456,10455,10454,10453,10452,10451,10450,10449,10448,10447,10446,10445,10444,10443,10442,10441,10440,10439,10438,10437,10436,10435,10434,10433,10432,10431,10430,10429,10428,10427,10426,10425,10424,10423,10422,10421,10420,10419,10418,10417,10416,10415,10414,10413,10412,10411,10410,10409,10408,10407,10406,10405,10404,10403,10402,10401,10400,10399,10398,10397,10396,10395,10394,10393,10392,10391,10390,10389,10388,10387,10386,10385,10384,10383,10382,10381,10380,10379,10378,10377,10376,10375,10374,10373,10372,10371,10370,10369,10368,10367,10366,10365,10364,10363,10362,10361,10360,10359,10358,10357,10356,10355,10354,10353,10352,10351,10350,10349,10348,10347,10346,10345,10344,10343,10342,10341,10340,10339,10338,10337,10336,10335,10334,10333,10332,10331,10330,10329,10328,10327,10326,10325,10324,10323,10322,10321,10320,10319,10318,10317,10316,10315,10314,10313,10312,10311,10310,10309,10308,10307,10306,10305,10304,10303,10302,10301,10300,10299,10298,10297,10296,10295,10294,10293,10292,10291,10290,10289,10288,10287,10286,10285,10284,10283,10282,10281,10280,10279,10278,10277,10276,10275,10274,10273,10272,10271,10270,10269,10268,10267,10266,10265,10264,10263,10262,10261,10260,10259,10258,10257,10256,10255,10254,10253,10252,10251,10250,10249,10248,10247,10246,10245,10244,10243,10242,10241,10240,10239,10238,10237,10236,10235,10234,10233,10232,10231,10230,10229,10228,10227,10226,10225,10224,10223,10222,10221,10220,10219,10218,10217,10216,10215,10214,10213,10212,10211,10210,10209,10208,10207,10206,10205,10204,10203,10202,10201,10200,10199,10198,10197,10196,10195,10194,10193,10192,10191,10190,10189,10188,10187,10186,10185,10184,10183,10182,10181,10180,10179,10178,10177,10176,10175,10174,10173,10172,10171,10170,10169,10168,10167,10166,10165,10164,10163,10162,10161,10160,10159,10158,10157,10156,10155,10154,10153,10152,10151,10150,10149,10148,10147,10146,10145,10144,10143,10142,10141,10140,10139,10138,10137,10136,10135,10134,10133,10132,10131,10130,10129,10128,10127,10126,10125,10124,10123,10122,10121,10120,10119,10118,10117,10116,10115,10114,10113,10112,10111,10110,10109,10108,10107,10106,10105,10104,10103,10102,10101,10100,10099,10098,10097,10096,10095,10094,10093,10092,10091,10090,10089,10088,10087,10086,10085,10084,10083,10082,10081,10080,10079,10078,10077,10076,10075,10074,10073,10072,10071,10070,10069,10068,10067,10066,10065,10064,10063,10062,10061,10060,10059,10058,10057,10056,10055,10054,10053,10052,10051,10050,10049,10048,10047,10046,10045,10044,10043,10042,10041,10040,10039,10038,10037,10036,10035,10034,10033,10032,10031,10030,10029,10028,10027,10026,10025,10024,10023,10022,10021,10020,10019,10018,10017,10016,10015,10014,10013,10012,10011,10010,10009,10008,10007,10006,10005,10004,10003,10002,10001,10000,9999,9998,9997,9996,9995,9994,9993,9992,9991,9990,9989,9988,9987,9986,9985,9984,9983,9982,9981,9980,9979,9978,9977,9976,9975,9974,9973,9972,9971,9970,9969,9968,9967,9966,9965,9964,9963,9962,9961,9960,9959,9958,9957,9956,9955,9954,9953,9952,9951,9950,9949,9948,9947,9946,9945,9944,9943,9942,9941,9940,9939,9938,9937,9936,9935,9934,9933,9932,9931,9930,9929,9928,9927,9926,9925,9924,9923,9922,9921,9920,9919,9918,9917,9916,9915,9914,9913,9912,9911,9910,9909,9908,9907,9906,9905,9904,9903,9902,9901,9900,9899,9898,9897,9896,9895,9894,9893,9892,9891,9890,9889,9888,9887,9886,9885,9884,9883,9882,9881,9880,9879,9878,9877,9876,9875,9874,9873,9872,9871,9870,9869,9868,9867,9866,9865,9864,9863,9862,9861,9860,9859,9858,9857,9856,9855,9854,9853,9852,9851,9850,9849,9848,9847,9846,9845,9844,9843,9842,9841,9840,9839,9838,9837,9836,9835,9834,9833,9832,9831,9830,9829,9828,9827,9826,9825,9824,9823,9822,9821,9820,9819,9818,9817,9816,9815,9814,9813,9812,9811,9810,9809,9808,9807,9806,9805,9804,9803,9802,9801,9800,9799,9798,9797,9796,9795,9794,9793,9792,9791,9790,9789,9788,9787,9786,9785,9784,9783,9782,9781,9780,9779,9778,9777,9776,9775,9774,9773,9772,9771,9770,9769,9768,9767,9766,9765,9764,9763,9762,9761,9760,9759,9758,9757,9756,9755,9754,9753,9752,9751,9750,9749,9748,9747,9746,9745,9744,9743,9742,9741,9740,9739,9738,9737,9736,9735,9734,9733,9732,9731,9730,9729,9728,9727,9726,9725,9724,9723,9722,9721,9720,9719,9718,9717,9716,9715,9714,9713,9712,9711,9710,9709,9708,9707,9706,9705,9704,9703,9702,9701,9700,9699,9698,9697,9696,9695,9694,9693,9692,9691,9690,9689,9688,9687,9686,9685,9684,9683,9682,9681,9680,9679,9678,9677,9676,9675,9674,9673,9672,9671,9670,9669,9668,9667,9666,9665,9664,9663,9662,9661,9660,9659,9658,9657,9656,9655,9654,9653,9652,9651,9650,9649,9648,9647,9646,9645,9644,9643,9642,9641,9640,9639,9638,9637,9636,9635,9634,9633,9632,9631,9630,9629,9628,9627,9626,9625,9624,9623,9622,9621,9620,9619,9618,9617,9616,9615,9614,9613,9612,9611,9610,9609,9608,9607,9606,9605,9604,9603,9602,9601,9600,9599,9598,9597,9596,9595,9594,9593,9592,9591,9590,9589,9588,9587,9586,9585,9584,9583,9582,9581,9580,9579,9578,9577,9576,9575,9574,9573,9572,9571,9570,9569,9568,9567,9566,9565,9564,9563,9562,9561,9560,9559,9558,9557,9556,9555,9554,9553,9552,9551,9550,9549,9548,9547,9546,9545,9544,9543,9542,9541,9540,9539,9538,9537,9536,9535,9534,9533,9532,9531,9530,9529,9528,9527,9526,9525,9524,9523,9522,9521,9520,9519,9518,9517,9516,9515,9514,9513,9512,9511,9510,9509,9508,9507,9506,9505,9504,9503,9502,9501,9500,9499,9498,9497,9496,9495,9494,9493,9492,9491,9490,9489,9488,9487,9486,9485,9484,9483,9482,9481,9480,9479,9478,9477,9476,9475,9474,9473,9472,9471,9470,9469,9468,9467,9466,9465,9464,9463,9462,9461,9460,9459,9458,9457,9456,9455,9454,9453,9452,9451,9450,9449,9448,9447,9446,9445,9444,9443,9442,9441,9440,9439,9438,9437,9436,9435,9434,9433,9432,9431,9430,9429,9428,9427,9426,9425,9424,9423,9422,9421,9420,9419,9418,9417,9416,9415,9414,9413,9412,9411,9410,9409,9408,9407,9406,9405,9404,9403,9402,9401,9400,9399,9398,9397,9396,9395,9394,9393,9392,9391,9390,9389,9388,9387,9386,9385,9384,9383,9382,9381,9380,9379,9378,9377,9376,9375,9374,9373,9372,9371,9370,9369,9368,9367,9366,9365,9364,9363,9362,9361,9360,9359,9358,9357,9356,9355,9354,9353,9352,9351,9350,9349,9348,9347,9346,9345,9344,9343,9342,9341,9340,9339,9338,9337,9336,9335,9334,9333,9332,9331,9330,9329,9328,9327,9326,9325,9324,9323,9322,9321,9320,9319,9318,9317,9316,9315,9314,9313,9312,9311,9310,9309,9308,9307,9306,9305,9304,9303,9302,9301,9300,9299,9298,9297,9296,9295,9294,9293,9292,9291,9290,9289,9288,9287,9286,9285,9284,9283,9282,9281,9280,9279,9278,9277,9276,9275,9274,9273,9272,9271,9270,9269,9268,9267,9266,9265,9264,9263,9262,9261,9260,9259,9258,9257,9256,9255,9254,9253,9252,9251,9250,9249,9248,9247,9246,9245,9244,9243,9242,9241,9240,9239,9238,9237,9236,9235,9234,9233,9232,9231,9230,9229,9228,9227,9226,9225,9224,9223,9222,9221,9220,9219,9218,9217,9216,9215,9214,9213,9212,9211,9210,9209,9208,9207,9206,9205,9204,9203,9202,9201,9200,9199,9198,9197,9196,9195,9194,9193,9192,9191,9190,9189,9188,9187,9186,9185,9184,9183,9182,9181,9180,9179,9178,9177,9176,9175,9174,9173,9172,9171,9170,9169,9168,9167,9166,9165,9164,9163,9162,9161,9160,9159,9158,9157,9156,9155,9154,9153,9152,9151,9150,9149,9148,9147,9146,9145,9144,9143,9142,9141,9140,9139,9138,9137,9136,9135,9134,9133,9132,9131,9130,9129,9128,9127,9126,9125,9124,9123,9122,9121,9120,9119,9118,9117,9116,9115,9114,9113,9112,9111,9110,9109,9108,9107,9106,9105,9104,9103,9102,9101,9100,9099,9098,9097,9096,9095,9094,9093,9092,9091,9090,9089,9088,9087,9086,9085,9084,9083,9082,9081,9080,9079,9078,9077,9076,9075,9074,9073,9072,9071,9070,9069,9068,9067,9066,9065,9064,9063,9062,9061,9060,9059,9058,9057,9056,9055,9054,9053,9052,9051,9050,9049,9048,9047,9046,9045,9044,9043,9042,9041,9040,9039,9038,9037,9036,9035,9034,9033,9032,9031,9030,9029,9028,9027,9026,9025,9024,9023,9022,9021,9020,9019,9018,9017,9016,9015,9014,9013,9012,9011,9010,9009,9008,9007,9006,9005,9004,9003,9002,9001,9000,8999,8998,8997,8996,8995,8994,8993,8992,8991,8990,8989,8988,8987,8986,8985,8984,8983,8982,8981,8980,8979,8978,8977,8976,8975,8974,8973,8972,8971,8970,8969,8968,8967,8966,8965,8964,8963,8962,8961,8960,8959,8958,8957,8956,8955,8954,8953,8952,8951,8950,8949,8948,8947,8946,8945,8944,8943,8942,8941,8940,8939,8938,8937,8936,8935,8934,8933,8932,8931,8930,8929,8928,8927,8926,8925,8924,8923,8922,8921,8920,8919,8918,8917,8916,8915,8914,8913,8912,8911,8910,8909,8908,8907,8906,8905,8904,8903,8902,8901,8900,8899,8898,8897,8896,8895,8894,8893,8892,8891,8890,8889,8888,8887,8886,8885,8884,8883,8882,8881,8880,8879,8878,8877,8876,8875,8874,8873,8872,8871,8870,8869,8868,8867,8866,8865,8864,8863,8862,8861,8860,8859,8858,8857,8856,8855,8854,8853,8852,8851,8850,8849,8848,8847,8846,8845,8844,8843,8842,8841,8840,8839,8838,8837,8836,8835,8834,8833,8832,8831,8830,8829,8828,8827,8826,8825,8824,8823,8822,8821,8820,8819,8818,8817,8816,8815,8814,8813,8812,8811,8810,8809,8808,8807,8806,8805,8804,8803,8802,8801,8800,8799,8798,8797,8796,8795,8794,8793,8792,8791,8790,8789,8788,8787,8786,8785,8784,8783,8782,8781,8780,8779,8778,8777,8776,8775,8774,8773,8772,8771,8770,8769,8768,8767,8766,8765,8764,8763,8762,8761,8760,8759,8758,8757,8756,8755,8754,8753,8752,8751,8750,8749,8748,8747,8746,8745,8744,8743,8742,8741,8740,8739,8738,8737,8736,8735,8734,8733,8732,8731,8730,8729,8728,8727,8726,8725,8724,8723,8722,8721,8720,8719,8718,8717,8716,8715,8714,8713,8712,8711,8710,8709,8708,8707,8706,8705,8704,8703,8702,8701,8700,8699,8698,8697,8696,8695,8694,8693,8692,8691,8690,8689,8688,8687,8686,8685,8684,8683,8682,8681,8680,8679,8678,8677,8676,8675,8674,8673,8672,8671,8670,8669,8668,8667,8666,8665,8664,8663,8662,8661,8660,8659,8658,8657,8656,8655,8654,8653,8652,8651,8650,8649,8648,8647,8646,8645,8644,8643,8642,8641,8640,8639,8638,8637,8636,8635,8634,8633,8632,8631,8630,8629,8628,8627,8626,8625,8624,8623,8622,8621,8620,8619,8618,8617,8616,8615,8614,8613,8612,8611,8610,8609,8608,8607,8606,8605,8604,8603,8602,8601,8600,8599,8598,8597,8596,8595,8594,8593,8592,8591,8590,8589,8588,8587,8586,8585,8584,8583,8582,8581,8580,8579,8578,8577,8576,8575,8574,8573,8572,8571,8570,8569,8568,8567,8566,8565,8564,8563,8562,8561,8560,8559,8558,8557,8556,8555,8554,8553,8552,8551,8550,8549,8548,8547,8546,8545,8544,8543,8542,8541,8540,8539,8538,8537,8536,8535,8534,8533,8532,8531,8530,8529,8528,8527,8526,8525,8524,8523,8522,8521,8520,8519,8518,8517,8516,8515,8514,8513,8512,8511,8510,8509,8508,8507,8506,8505,8504,8503,8502,8501,8500,8499,8498,8497,8496,8495,8494,8493,8492,8491,8490,8489,8488,8487,8486,8485,8484,8483,8482,8481,8480,8479,8478,8477,8476,8475,8474,8473,8472,8471,8470,8469,8468,8467,8466,8465,8464,8463,8462,8461,8460,8459,8458,8457,8456,8455,8454,8453,8452,8451,8450,8449,8448,8447,8446,8445,8444,8443,8442,8441,8440,8439,8438,8437,8436,8435,8434,8433,8432,8431,8430,8429,8428,8427,8426,8425,8424,8423,8422,8421,8420,8419,8418,8417,8416,8415,8414,8413,8412,8411,8410,8409,8408,8407,8406,8405,8404,8403,8402,8401,8400,8399,8398,8397,8396,8395,8394,8393,8392,8391,8390,8389,8388,8387,8386,8385,8384,8383,8382,8381,8380,8379,8378,8377,8376,8375,8374,8373,8372,8371,8370,8369,8368,8367,8366,8365,8364,8363,8362,8361,8360,8359,8358,8357,8356,8355,8354,8353,8352,8351,8350,8349,8348,8347,8346,8345,8344,8343,8342,8341,8340,8339,8338,8337,8336,8335,8334,8333,8332,8331,8330,8329,8328,8327,8326,8325,8324,8323,8322,8321,8320,8319,8318,8317,8316,8315,8314,8313,8312,8311,8310,8309,8308,8307,8306,8305,8304,8303,8302,8301,8300,8299,8298,8297,8296,8295,8294,8293,8292,8291,8290,8289,8288,8287,8286,8285,8284,8283,8282,8281,8280,8279,8278,8277,8276,8275,8274,8273,8272,8271,8270,8269,8268,8267,8266,8265,8264,8263,8262,8261,8260,8259,8258,8257,8256,8255,8254,8253,8252,8251,8250,8249,8248,8247,8246,8245,8244,8243,8242,8241,8240,8239,8238,8237,8236,8235,8234,8233,8232,8231,8230,8229,8228,8227,8226,8225,8224,8223,8222,8221,8220,8219,8218,8217,8216,8215,8214,8213,8212,8211,8210,8209,8208,8207,8206,8205,8204,8203,8202,8201,8200,8199,8198,8197,8196,8195,8194,8193,8192,8191,8190,8189,8188,8187,8186,8185,8184,8183,8182,8181,8180,8179,8178,8177,8176,8175,8174,8173,8172,8171,8170,8169,8168,8167,8166,8165,8164,8163,8162,8161,8160,8159,8158,8157,8156,8155,8154,8153,8152,8151,8150,8149,8148,8147,8146,8145,8144,8143,8142,8141,8140,8139,8138,8137,8136,8135,8134,8133,8132,8131,8130,8129,8128,8127,8126,8125,8124,8123,8122,8121,8120,8119,8118,8117,8116,8115,8114,8113,8112,8111,8110,8109,8108,8107,8106,8105,8104,8103,8102,8101,8100,8099,8098,8097,8096,8095,8094,8093,8092,8091,8090,8089,8088,8087,8086,8085,8084,8083,8082,8081,8080,8079,8078,8077,8076,8075,8074,8073,8072,8071,8070,8069,8068,8067,8066,8065,8064,8063,8062,8061,8060,8059,8058,8057,8056,8055,8054,8053,8052,8051,8050,8049,8048,8047,8046,8045,8044,8043,8042,8041,8040,8039,8038,8037,8036,8035,8034,8033,8032,8031,8030,8029,8028,8027,8026,8025,8024,8023,8022,8021,8020,8019,8018,8017,8016,8015,8014,8013,8012,8011,8010,8009,8008,8007,8006,8005,8004,8003,8002,8001,8000,7999,7998,7997,7996,7995,7994,7993,7992,7991,7990,7989,7988,7987,7986,7985,7984,7983,7982,7981,7980,7979,7978,7977,7976,7975,7974,7973,7972,7971,7970,7969,7968,7967,7966,7965,7964,7963,7962,7961,7960,7959,7958,7957,7956,7955,7954,7953,7952,7951,7950,7949,7948,7947,7946,7945,7944,7943,7942,7941,7940,7939,7938,7937,7936,7935,7934,7933,7932,7931,7930,7929,7928,7927,7926,7925,7924,7923,7922,7921,7920,7919,7918,7917,7916,7915,7914,7913,7912,7911,7910,7909,7908,7907,7906,7905,7904,7903,7902,7901,7900,7899,7898,7897,7896,7895,7894,7893,7892,7891,7890,7889,7888,7887,7886,7885,7884,7883,7882,7881,7880,7879,7878,7877,7876,7875,7874,7873,7872,7871,7870,7869,7868,7867,7866,7865,7864,7863,7862,7861,7860,7859,7858,7857,7856,7855,7854,7853,7852,7851,7850,7849,7848,7847,7846,7845,7844,7843,7842,7841,7840,7839,7838,7837,7836,7835,7834,7833,7832,7831,7830,7829,7828,7827,7826,7825,7824,7823,7822,7821,7820,7819,7818,7817,7816,7815,7814,7813,7812,7811,7810,7809,7808,7807,7806,7805,7804,7803,7802,7801,7800,7799,7798,7797,7796,7795,7794,7793,7792,7791,7790,7789,7788,7787,7786,7785,7784,7783,7782,7781,7780,7779,7778,7777,7776,7775,7774,7773,7772,7771,7770,7769,7768,7767,7766,7765,7764,7763,7762,7761,7760,7759,7758,7757,7756,7755,7754,7753,7752,7751,7750,7749,7748,7747,7746,7745,7744,7743,7742,7741,7740,7739,7738,7737,7736,7735,7734,7733,7732,7731,7730,7729,7728,7727,7726,7725,7724,7723,7722,7721,7720,7719,7718,7717,7716,7715,7714,7713,7712,7711,7710,7709,7708,7707,7706,7705,7704,7703,7702,7701,7700,7699,7698,7697,7696,7695,7694,7693,7692,7691,7690,7689,7688,7687,7686,7685,7684,7683,7682,7681,7680,7679,7678,7677,7676,7675,7674,7673,7672,7671,7670,7669,7668,7667,7666,7665,7664,7663,7662,7661,7660,7659,7658,7657,7656,7655,7654,7653,7652,7651,7650,7649,7648,7647,7646,7645,7644,7643,7642,7641,7640,7639,7638,7637,7636,7635,7634,7633,7632,7631,7630,7629,7628,7627,7626,7625,7624,7623,7622,7621,7620,7619,7618,7617,7616,7615,7614,7613,7612,7611,7610,7609,7608,7607,7606,7605,7604,7603,7602,7601,7600,7599,7598,7597,7596,7595,7594,7593,7592,7591,7590,7589,7588,7587,7586,7585,7584,7583,7582,7581,7580,7579,7578,7577,7576,7575,7574,7573,7572,7571,7570,7569,7568,7567,7566,7565,7564,7563,7562,7561,7560,7559,7558,7557,7556,7555,7554,7553,7552,7551,7550,7549,7548,7547,7546,7545,7544,7543,7542,7541,7540,7539,7538,7537,7536,7535,7534,7533,7532,7531,7530,7529,7528,7527,7526,7525,7524,7523,7522,7521,7520,7519,7518,7517,7516,7515,7514,7513,7512,7511,7510,7509,7508,7507,7506,7505,7504,7503,7502,7501,7500,7499,7498,7497,7496,7495,7494,7493,7492,7491,7490,7489,7488,7487,7486,7485,7484,7483,7482,7481,7480,7479,7478,7477,7476,7475,7474,7473,7472,7471,7470,7469,7468,7467,7466,7465,7464,7463,7462,7461,7460,7459,7458,7457,7456,7455,7454,7453,7452,7451,7450,7449,7448,7447,7446,7445,7444,7443,7442,7441,7440,7439,7438,7437,7436,7435,7434,7433,7432,7431,7430,7429,7428,7427,7426,7425,7424,7423,7422,7421,7420,7419,7418,7417,7416,7415,7414,7413,7412,7411,7410,7409,7408,7407,7406,7405,7404,7403,7402,7401,7400,7399,7398,7397,7396,7395,7394,7393,7392,7391,7390,7389,7388,7387,7386,7385,7384,7383,7382,7381,7380,7379,7378,7377,7376,7375,7374,7373,7372,7371,7370,7369,7368,7367,7366,7365,7364,7363,7362,7361,7360,7359,7358,7357,7356,7355,7354,7353,7352,7351,7350,7349,7348,7347,7346,7345,7344,7343,7342,7341,7340,7339,7338,7337,7336,7335,7334,7333,7332,7331,7330,7329,7328,7327,7326,7325,7324,7323,7322,7321,7320,7319,7318,7317,7316,7315,7314,7313,7312,7311,7310,7309,7308,7307,7306,7305,7304,7303,7302,7301,7300,7299,7298,7297,7296,7295,7294,7293,7292,7291,7290,7289,7288,7287,7286,7285,7284,7283,7282,7281,7280,7279,7278,7277,7276,7275,7274,7273,7272,7271,7270,7269,7268,7267,7266,7265,7264,7263,7262,7261,7260,7259,7258,7257,7256,7255,7254,7253,7252,7251,7250,7249,7248,7247,7246,7245,7244,7243,7242,7241,7240,7239,7238,7237,7236,7235,7234,7233,7232,7231,7230,7229,7228,7227,7226,7225,7224,7223,7222,7221,7220,7219,7218,7217,7216,7215,7214,7213,7212,7211,7210,7209,7208,7207,7206,7205,7204,7203,7202,7201,7200,7199,7198,7197,7196,7195,7194,7193,7192,7191,7190,7189,7188,7187,7186,7185,7184,7183,7182,7181,7180,7179,7178,7177,7176,7175,7174,7173,7172,7171,7170,7169,7168,7167,7166,7165,7164,7163,7162,7161,7160,7159,7158,7157,7156,7155,7154,7153,7152,7151,7150,7149,7148,7147,7146,7145,7144,7143,7142,7141,7140,7139,7138,7137,7136,7135,7134,7133,7132,7131,7130,7129,7128,7127,7126,7125,7124,7123,7122,7121,7120,7119,7118,7117,7116,7115,7114,7113,7112,7111,7110,7109,7108,7107,7106,7105,7104,7103,7102,7101,7100,7099,7098,7097,7096,7095,7094,7093,7092,7091,7090,7089,7088,7087,7086,7085,7084,7083,7082,7081,7080,7079,7078,7077,7076,7075,7074,7073,7072,7071,7070,7069,7068,7067,7066,7065,7064,7063,7062,7061,7060,7059,7058,7057,7056,7055,7054,7053,7052,7051,7050,7049,7048,7047,7046,7045,7044,7043,7042,7041,7040,7039,7038,7037,7036,7035,7034,7033,7032,7031,7030,7029,7028,7027,7026,7025,7024,7023,7022,7021,7020,7019,7018,7017,7016,7015,7014,7013,7012,7011,7010,7009,7008,7007,7006,7005,7004,7003,7002,7001,7000,6999,6998,6997,6996,6995,6994,6993,6992,6991,6990,6989,6988,6987,6986,6985,6984,6983,6982,6981,6980,6979,6978,6977,6976,6975,6974,6973,6972,6971,6970,6969,6968,6967,6966,6965,6964,6963,6962,6961,6960,6959,6958,6957,6956,6955,6954,6953,6952,6951,6950,6949,6948,6947,6946,6945,6944,6943,6942,6941,6940,6939,6938,6937,6936,6935,6934,6933,6932,6931,6930,6929,6928,6927,6926,6925,6924,6923,6922,6921,6920,6919,6918,6917,6916,6915,6914,6913,6912,6911,6910,6909,6908,6907,6906,6905,6904,6903,6902,6901,6900,6899,6898,6897,6896,6895,6894,6893,6892,6891,6890,6889,6888,6887,6886,6885,6884,6883,6882,6881,6880,6879,6878,6877,6876,6875,6874,6873,6872,6871,6870,6869,6868,6867,6866,6865,6864,6863,6862,6861,6860,6859,6858,6857,6856,6855,6854,6853,6852,6851,6850,6849,6848,6847,6846,6845,6844,6843,6842,6841,6840,6839,6838,6837,6836,6835,6834,6833,6832,6831,6830,6829,6828,6827,6826,6825,6824,6823,6822,6821,6820,6819,6818,6817,6816,6815,6814,6813,6812,6811,6810,6809,6808,6807,6806,6805,6804,6803,6802,6801,6800,6799,6798,6797,6796,6795,6794,6793,6792,6791,6790,6789,6788,6787,6786,6785,6784,6783,6782,6781,6780,6779,6778,6777,6776,6775,6774,6773,6772,6771,6770,6769,6768,6767,6766,6765,6764,6763,6762,6761,6760,6759,6758,6757,6756,6755,6754,6753,6752,6751,6750,6749,6748,6747,6746,6745,6744,6743,6742,6741,6740,6739,6738,6737,6736,6735,6734,6733,6732,6731,6730,6729,6728,6727,6726,6725,6724,6723,6722,6721,6720,6719,6718,6717,6716,6715,6714,6713,6712,6711,6710,6709,6708,6707,6706,6705,6704,6703,6702,6701,6700,6699,6698,6697,6696,6695,6694,6693,6692,6691,6690,6689,6688,6687,6686,6685,6684,6683,6682,6681,6680,6679,6678,6677,6676,6675,6674,6673,6672,6671,6670,6669,6668,6667,6666,6665,6664,6663,6662,6661,6660,6659,6658,6657,6656,6655,6654,6653,6652,6651,6650,6649,6648,6647,6646,6645,6644,6643,6642,6641,6640,6639,6638,6637,6636,6635,6634,6633,6632,6631,6630,6629,6628,6627,6626,6625,6624,6623,6622,6621,6620,6619,6618,6617,6616,6615,6614,6613,6612,6611,6610,6609,6608,6607,6606,6605,6604,6603,6602,6601,6600,6599,6598,6597,6596,6595,6594,6593,6592,6591,6590,6589,6588,6587,6586,6585,6584,6583,6582,6581,6580,6579,6578,6577,6576,6575,6574,6573,6572,6571,6570,6569,6568,6567,6566,6565,6564,6563,6562,6561,6560,6559,6558,6557,6556,6555,6554,6553,6552,6551,6550,6549,6548,6547,6546,6545,6544,6543,6542,6541,6540,6539,6538,6537,6536,6535,6534,6533,6532,6531,6530,6529,6528,6527,6526,6525,6524,6523,6522,6521,6520,6519,6518,6517,6516,6515,6514,6513,6512,6511,6510,6509,6508,6507,6506,6505,6504,6503,6502,6501,6500,6499,6498,6497,6496,6495,6494,6493,6492,6491,6490,6489,6488,6487,6486,6485,6484,6483,6482,6481,6480,6479,6478,6477,6476,6475,6474,6473,6472,6471,6470,6469,6468,6467,6466,6465,6464,6463,6462,6461,6460,6459,6458,6457,6456,6455,6454,6453,6452,6451,6450,6449,6448,6447,6446,6445,6444,6443,6442,6441,6440,6439,6438,6437,6436,6435,6434,6433,6432,6431,6430,6429,6428,6427,6426,6425,6424,6423,6422,6421,6420,6419,6418,6417,6416,6415,6414,6413,6412,6411,6410,6409,6408,6407,6406,6405,6404,6403,6402,6401,6400,6399,6398,6397,6396,6395,6394,6393,6392,6391,6390,6389,6388,6387,6386,6385,6384,6383,6382,6381,6380,6379,6378,6377,6376,6375,6374,6373,6372,6371,6370,6369,6368,6367,6366,6365,6364,6363,6362,6361,6360,6359,6358,6357,6356,6355,6354,6353,6352,6351,6350,6349,6348,6347,6346,6345,6344,6343,6342,6341,6340,6339,6338,6337,6336,6335,6334,6333,6332,6331,6330,6329,6328,6327,6326,6325,6324,6323,6322,6321,6320,6319,6318,6317,6316,6315,6314,6313,6312,6311,6310,6309,6308,6307,6306,6305,6304,6303,6302,6301,6300,6299,6298,6297,6296,6295,6294,6293,6292,6291,6290,6289,6288,6287,6286,6285,6284,6283,6282,6281,6280,6279,6278,6277,6276,6275,6274,6273,6272,6271,6270,6269,6268,6267,6266,6265,6264,6263,6262,6261,6260,6259,6258,6257,6256,6255,6254,6253,6252,6251,6250,6249,6248,6247,6246,6245,6244,6243,6242,6241,6240,6239,6238,6237,6236,6235,6234,6233,6232,6231,6230,6229,6228,6227,6226,6225,6224,6223,6222,6221,6220,6219,6218,6217,6216,6215,6214,6213,6212,6211,6210,6209,6208,6207,6206,6205,6204,6203,6202,6201,6200,6199,6198,6197,6196,6195,6194,6193,6192,6191,6190,6189,6188,6187,6186,6185,6184,6183,6182,6181,6180,6179,6178,6177,6176,6175,6174,6173,6172,6171,6170,6169,6168,6167,6166,6165,6164,6163,6162,6161,6160,6159,6158,6157,6156,6155,6154,6153,6152,6151,6150,6149,6148,6147,6146,6145,6144,6143,6142,6141,6140,6139,6138,6137,6136,6135,6134,6133,6132,6131,6130,6129,6128,6127,6126,6125,6124,6123,6122,6121,6120,6119,6118,6117,6116,6115,6114,6113,6112,6111,6110,6109,6108,6107,6106,6105,6104,6103,6102,6101,6100,6099,6098,6097,6096,6095,6094,6093,6092,6091,6090,6089,6088,6087,6086,6085,6084,6083,6082,6081,6080,6079,6078,6077,6076,6075,6074,6073,6072,6071,6070,6069,6068,6067,6066,6065,6064,6063,6062,6061,6060,6059,6058,6057,6056,6055,6054,6053,6052,6051,6050,6049,6048,6047,6046,6045,6044,6043,6042,6041,6040,6039,6038,6037,6036,6035,6034,6033,6032,6031,6030,6029,6028,6027,6026,6025,6024,6023,6022,6021,6020,6019,6018,6017,6016,6015,6014,6013,6012,6011,6010,6009,6008,6007,6006,6005,6004,6003,6002,6001,6000,5999,5998,5997,5996,5995,5994,5993,5992,5991,5990,5989,5988,5987,5986,5985,5984,5983,5982,5981,5980,5979,5978,5977,5976,5975,5974,5973,5972,5971,5970,5969,5968,5967,5966,5965,5964,5963,5962,5961,5960,5959,5958,5957,5956,5955,5954,5953,5952,5951,5950,5949,5948,5947,5946,5945,5944,5943,5942,5941,5940,5939,5938,5937,5936,5935,5934,5933,5932,5931,5930,5929,5928,5927,5926,5925,5924,5923,5922,5921,5920,5919,5918,5917,5916,5915,5914,5913,5912,5911,5910,5909,5908,5907,5906,5905,5904,5903,5902,5901,5900,5899,5898,5897,5896,5895,5894,5893,5892,5891,5890,5889,5888,5887,5886,5885,5884,5883,5882,5881,5880,5879,5878,5877,5876,5875,5874,5873,5872,5871,5870,5869,5868,5867,5866,5865,5864,5863,5862,5861,5860,5859,5858,5857,5856,5855,5854,5853,5852,5851,5850,5849,5848,5847,5846,5845,5844,5843,5842,5841,5840,5839,5838,5837,5836,5835,5834,5833,5832,5831,5830,5829,5828,5827,5826,5825,5824,5823,5822,5821,5820,5819,5818,5817,5816,5815,5814,5813,5812,5811,5810,5809,5808,5807,5806,5805,5804,5803,5802,5801,5800,5799,5798,5797,5796,5795,5794,5793,5792,5791,5790,5789,5788,5787,5786,5785,5784,5783,5782,5781,5780,5779,5778,5777,5776,5775,5774,5773,5772,5771,5770,5769,5768,5767,5766,5765,5764,5763,5762,5761,5760,5759,5758,5757,5756,5755,5754,5753,5752,5751,5750,5749,5748,5747,5746,5745,5744,5743,5742,5741,5740,5739,5738,5737,5736,5735,5734,5733,5732,5731,5730,5729,5728,5727,5726,5725,5724,5723,5722,5721,5720,5719,5718,5717,5716,5715,5714,5713,5712,5711,5710,5709,5708,5707,5706,5705,5704,5703,5702,5701,5700,5699,5698,5697,5696,5695,5694,5693,5692,5691,5690,5689,5688,5687,5686,5685,5684,5683,5682,5681,5680,5679,5678,5677,5676,5675,5674,5673,5672,5671,5670,5669,5668,5667,5666,5665,5664,5663,5662,5661,5660,5659,5658,5657,5656,5655,5654,5653,5652,5651,5650,5649,5648,5647,5646,5645,5644,5643,5642,5641,5640,5639,5638,5637,5636,5635,5634,5633,5632,5631,5630,5629,5628,5627,5626,5625,5624,5623,5622,5621,5620,5619,5618,5617,5616,5615,5614,5613,5612,5611,5610,5609,5608,5607,5606,5605,5604,5603,5602,5601,5600,5599,5598,5597,5596,5595,5594,5593,5592,5591,5590,5589,5588,5587,5586,5585,5584,5583,5582,5581,5580,5579,5578,5577,5576,5575,5574,5573,5572,5571,5570,5569,5568,5567,5566,5565,5564,5563,5562,5561,5560,5559,5558,5557,5556,5555,5554,5553,5552,5551,5550,5549,5548,5547,5546,5545,5544,5543,5542,5541,5540,5539,5538,5537,5536,5535,5534,5533,5532,5531,5530,5529,5528,5527,5526,5525,5524,5523,5522,5521,5520,5519,5518,5517,5516,5515,5514,5513,5512,5511,5510,5509,5508,5507,5506,5505,5504,5503,5502,5501,5500,5499,5498,5497,5496,5495,5494,5493,5492,5491,5490,5489,5488,5487,5486,5485,5484,5483,5482,5481,5480,5479,5478,5477,5476,5475,5474,5473,5472,5471,5470,5469,5468,5467,5466,5465,5464,5463,5462,5461,5460,5459,5458,5457,5456,5455,5454,5453,5452,5451,5450,5449,5448,5447,5446,5445,5444,5443,5442,5441,5440,5439,5438,5437,5436,5435,5434,5433,5432,5431,5430,5429,5428,5427,5426,5425,5424,5423,5422,5421,5420,5419,5418,5417,5416,5415,5414,5413,5412,5411,5410,5409,5408,5407,5406,5405,5404,5403,5402,5401,5400,5399,5398,5397,5396,5395,5394,5393,5392,5391,5390,5389,5388,5387,5386,5385,5384,5383,5382,5381,5380,5379,5378,5377,5376,5375,5374,5373,5372,5371,5370,5369,5368,5367,5366,5365,5364,5363,5362,5361,5360,5359,5358,5357,5356,5355,5354,5353,5352,5351,5350,5349,5348,5347,5346,5345,5344,5343,5342,5341,5340,5339,5338,5337,5336,5335,5334,5333,5332,5331,5330,5329,5328,5327,5326,5325,5324,5323,5322,5321,5320,5319,5318,5317,5316,5315,5314,5313,5312,5311,5310,5309,5308,5307,5306,5305,5304,5303,5302,5301,5300,5299,5298,5297,5296,5295,5294,5293,5292,5291,5290,5289,5288,5287,5286,5285,5284,5283,5282,5281,5280,5279,5278,5277,5276,5275,5274,5273,5272,5271,5270,5269,5268,5267,5266,5265,5264,5263,5262,5261,5260,5259,5258,5257,5256,5255,5254,5253,5252,5251,5250,5249,5248,5247,5246,5245,5244,5243,5242,5241,5240,5239,5238,5237,5236,5235,5234,5233,5232,5231,5230,5229,5228,5227,5226,5225,5224,5223,5222,5221,5220,5219,5218,5217,5216,5215,5214,5213,5212,5211,5210,5209,5208,5207,5206,5205,5204,5203,5202,5201,5200,5199,5198,5197,5196,5195,5194,5193,5192,5191,5190,5189,5188,5187,5186,5185,5184,5183,5182,5181,5180,5179,5178,5177,5176,5175,5174,5173,5172,5171,5170,5169,5168,5167,5166,5165,5164,5163,5162,5161,5160,5159,5158,5157,5156,5155,5154,5153,5152,5151,5150,5149,5148,5147,5146,5145,5144,5143,5142,5141,5140,5139,5138,5137,5136,5135,5134,5133,5132,5131,5130,5129,5128,5127,5126,5125,5124,5123,5122,5121,5120,5119,5118,5117,5116,5115,5114,5113,5112,5111,5110,5109,5108,5107,5106,5105,5104,5103,5102,5101,5100,5099,5098,5097,5096,5095,5094,5093,5092,5091,5090,5089,5088,5087,5086,5085,5084,5083,5082,5081,5080,5079,5078,5077,5076,5075,5074,5073,5072,5071,5070,5069,5068,5067,5066,5065,5064,5063,5062,5061,5060,5059,5058,5057,5056,5055,5054,5053,5052,5051,5050,5049,5048,5047,5046,5045,5044,5043,5042,5041,5040,5039,5038,5037,5036,5035,5034,5033,5032,5031,5030,5029,5028,5027,5026,5025,5024,5023,5022,5021,5020,5019,5018,5017,5016,5015,5014,5013,5012,5011,5010,5009,5008,5007,5006,5005,5004,5003,5002,5001,5000,4999,4998,4997,4996,4995,4994,4993,4992,4991,4990,4989,4988,4987,4986,4985,4984,4983,4982,4981,4980,4979,4978,4977,4976,4975,4974,4973,4972,4971,4970,4969,4968,4967,4966,4965,4964,4963,4962,4961,4960,4959,4958,4957,4956,4955,4954,4953,4952,4951,4950,4949,4948,4947,4946,4945,4944,4943,4942,4941,4940,4939,4938,4937,4936,4935,4934,4933,4932,4931,4930,4929,4928,4927,4926,4925,4924,4923,4922,4921,4920,4919,4918,4917,4916,4915,4914,4913,4912,4911,4910,4909,4908,4907,4906,4905,4904,4903,4902,4901,4900,4899,4898,4897,4896,4895,4894,4893,4892,4891,4890,4889,4888,4887,4886,4885,4884,4883,4882,4881,4880,4879,4878,4877,4876,4875,4874,4873,4872,4871,4870,4869,4868,4867,4866,4865,4864,4863,4862,4861,4860,4859,4858,4857,4856,4855,4854,4853,4852,4851,4850,4849,4848,4847,4846,4845,4844,4843,4842,4841,4840,4839,4838,4837,4836,4835,4834,4833,4832,4831,4830,4829,4828,4827,4826,4825,4824,4823,4822,4821,4820,4819,4818,4817,4816,4815,4814,4813,4812,4811,4810,4809,4808,4807,4806,4805,4804,4803,4802,4801,4800,4799,4798,4797,4796,4795,4794,4793,4792,4791,4790,4789,4788,4787,4786,4785,4784,4783,4782,4781,4780,4779,4778,4777,4776,4775,4774,4773,4772,4771,4770,4769,4768,4767,4766,4765,4764,4763,4762,4761,4760,4759,4758,4757,4756,4755,4754,4753,4752,4751,4750,4749,4748,4747,4746,4745,4744,4743,4742,4741,4740,4739,4738,4737,4736,4735,4734,4733,4732,4731,4730,4729,4728,4727,4726,4725,4724,4723,4722,4721,4720,4719,4718,4717,4716,4715,4714,4713,4712,4711,4710,4709,4708,4707,4706,4705,4704,4703,4702,4701,4700,4699,4698,4697,4696,4695,4694,4693,4692,4691,4690,4689,4688,4687,4686,4685,4684,4683,4682,4681,4680,4679,4678,4677,4676,4675,4674,4673,4672,4671,4670,4669,4668,4667,4666,4665,4664,4663,4662,4661,4660,4659,4658,4657,4656,4655,4654,4653,4652,4651,4650,4649,4648,4647,4646,4645,4644,4643,4642,4641,4640,4639,4638,4637,4636,4635,4634,4633,4632,4631,4630,4629,4628,4627,4626,4625,4624,4623,4622,4621,4620,4619,4618,4617,4616,4615,4614,4613,4612,4611,4610,4609,4608,4607,4606,4605,4604,4603,4602,4601,4600,4599,4598,4597,4596,4595,4594,4593,4592,4591,4590,4589,4588,4587,4586,4585,4584,4583,4582,4581,4580,4579,4578,4577,4576,4575,4574,4573,4572,4571,4570,4569,4568,4567,4566,4565,4564,4563,4562,4561,4560,4559,4558,4557,4556,4555,4554,4553,4552,4551,4550,4549,4548,4547,4546,4545,4544,4543,4542,4541,4540,4539,4538,4537,4536,4535,4534,4533,4532,4531,4530,4529,4528,4527,4526,4525,4524,4523,4522,4521,4520,4519,4518,4517,4516,4515,4514,4513,4512,4511,4510,4509,4508,4507,4506,4505,4504,4503,4502,4501,4500,4499,4498,4497,4496,4495,4494,4493,4492,4491,4490,4489,4488,4487,4486,4485,4484,4483,4482,4481,4480,4479,4478,4477,4476,4475,4474,4473,4472,4471,4470,4469,4468,4467,4466,4465,4464,4463,4462,4461,4460,4459,4458,4457,4456,4455,4454,4453,4452,4451,4450,4449,4448,4447,4446,4445,4444,4443,4442,4441,4440,4439,4438,4437,4436,4435,4434,4433,4432,4431,4430,4429,4428,4427,4426,4425,4424,4423,4422,4421,4420,4419,4418,4417,4416,4415,4414,4413,4412,4411,4410,4409,4408,4407,4406,4405,4404,4403,4402,4401,4400,4399,4398,4397,4396,4395,4394,4393,4392,4391,4390,4389,4388,4387,4386,4385,4384,4383,4382,4381,4380,4379,4378,4377,4376,4375,4374,4373,4372,4371,4370,4369,4368,4367,4366,4365,4364,4363,4362,4361,4360,4359,4358,4357,4356,4355,4354,4353,4352,4351,4350,4349,4348,4347,4346,4345,4344,4343,4342,4341,4340,4339,4338,4337,4336,4335,4334,4333,4332,4331,4330,4329,4328,4327,4326,4325,4324,4323,4322,4321,4320,4319,4318,4317,4316,4315,4314,4313,4312,4311,4310,4309,4308,4307,4306,4305,4304,4303,4302,4301,4300,4299,4298,4297,4296,4295,4294,4293,4292,4291,4290,4289,4288,4287,4286,4285,4284,4283,4282,4281,4280,4279,4278,4277,4276,4275,4274,4273,4272,4271,4270,4269,4268,4267,4266,4265,4264,4263,4262,4261,4260,4259,4258,4257,4256,4255,4254,4253,4252,4251,4250,4249,4248,4247,4246,4245,4244,4243,4242,4241,4240,4239,4238,4237,4236,4235,4234,4233,4232,4231,4230,4229,4228,4227,4226,4225,4224,4223,4222,4221,4220,4219,4218,4217,4216,4215,4214,4213,4212,4211,4210,4209,4208,4207,4206,4205,4204,4203,4202,4201,4200,4199,4198,4197,4196,4195,4194,4193,4192,4191,4190,4189,4188,4187,4186,4185,4184,4183,4182,4181,4180,4179,4178,4177,4176,4175,4174,4173,4172,4171,4170,4169,4168,4167,4166,4165,4164,4163,4162,4161,4160,4159,4158,4157,4156,4155,4154,4153,4152,4151,4150,4149,4148,4147,4146,4145,4144,4143,4142,4141,4140,4139,4138,4137,4136,4135,4134,4133,4132,4131,4130,4129,4128,4127,4126,4125,4124,4123,4122,4121,4120,4119,4118,4117,4116,4115,4114,4113,4112,4111,4110,4109,4108,4107,4106,4105,4104,4103,4102,4101,4100,4099,4098,4097,4096,4095,4094,4093,4092,4091,4090,4089,4088,4087,4086,4085,4084,4083,4082,4081,4080,4079,4078,4077,4076,4075,4074,4073,4072,4071,4070,4069,4068,4067,4066,4065,4064,4063,4062,4061,4060,4059,4058,4057,4056,4055,4054,4053,4052,4051,4050,4049,4048,4047,4046,4045,4044,4043,4042,4041,4040,4039,4038,4037,4036,4035,4034,4033,4032,4031,4030,4029,4028,4027,4026,4025,4024,4023,4022,4021,4020,4019,4018,4017,4016,4015,4014,4013,4012,4011,4010,4009,4008,4007,4006,4005,4004,4003,4002,4001,4000,3999,3998,3997,3996,3995,3994,3993,3992,3991,3990,3989,3988,3987,3986,3985,3984,3983,3982,3981,3980,3979,3978,3977,3976,3975,3974,3973,3972,3971,3970,3969,3968,3967,3966,3965,3964,3963,3962,3961,3960,3959,3958,3957,3956,3955,3954,3953,3952,3951,3950,3949,3948,3947,3946,3945,3944,3943,3942,3941,3940,3939,3938,3937,3936,3935,3934,3933,3932,3931,3930,3929,3928,3927,3926,3925,3924,3923,3922,3921,3920,3919,3918,3917,3916,3915,3914,3913,3912,3911,3910,3909,3908,3907,3906,3905,3904,3903,3902,3901,3900,3899,3898,3897,3896,3895,3894,3893,3892,3891,3890,3889,3888,3887,3886,3885,3884,3883,3882,3881,3880,3879,3878,3877,3876,3875,3874,3873,3872,3871,3870,3869,3868,3867,3866,3865,3864,3863,3862,3861,3860,3859,3858,3857,3856,3855,3854,3853,3852,3851,3850,3849,3848,3847,3846,3845,3844,3843,3842,3841,3840,3839,3838,3837,3836,3835,3834,3833,3832,3831,3830,3829,3828,3827,3826,3825,3824,3823,3822,3821,3820,3819,3818,3817,3816,3815,3814,3813,3812,3811,3810,3809,3808,3807,3806,3805,3804,3803,3802,3801,3800,3799,3798,3797,3796,3795,3794,3793,3792,3791,3790,3789,3788,3787,3786,3785,3784,3783,3782,3781,3780,3779,3778,3777,3776,3775,3774,3773,3772,3771,3770,3769,3768,3767,3766,3765,3764,3763,3762,3761,3760,3759,3758,3757,3756,3755,3754,3753,3752,3751,3750,3749,3748,3747,3746,3745,3744,3743,3742,3741,3740,3739,3738,3737,3736,3735,3734,3733,3732,3731,3730,3729,3728,3727,3726,3725,3724,3723,3722,3721,3720,3719,3718,3717,3716,3715,3714,3713,3712,3711,3710,3709,3708,3707,3706,3705,3704,3703,3702,3701,3700,3699,3698,3697,3696,3695,3694,3693,3692,3691,3690,3689,3688,3687,3686,3685,3684,3683,3682,3681,3680,3679,3678,3677,3676,3675,3674,3673,3672,3671,3670,3669,3668,3667,3666,3665,3664,3663,3662,3661,3660,3659,3658,3657,3656,3655,3654,3653,3652,3651,3650,3649,3648,3647,3646,3645,3644,3643,3642,3641,3640,3639,3638,3637,3636,3635,3634,3633,3632,3631,3630,3629,3628,3627,3626,3625,3624,3623,3622,3621,3620,3619,3618,3617,3616,3615,3614,3613,3612,3611,3610,3609,3608,3607,3606,3605,3604,3603,3602,3601,3600,3599,3598,3597,3596,3595,3594,3593,3592,3591,3590,3589,3588,3587,3586,3585,3584,3583,3582,3581,3580,3579,3578,3577,3576,3575,3574,3573,3572,3571,3570,3569,3568,3567,3566,3565,3564,3563,3562,3561,3560,3559,3558,3557,3556,3555,3554,3553,3552,3551,3550,3549,3548,3547,3546,3545,3544,3543,3542,3541,3540,3539,3538,3537,3536,3535,3534,3533,3532,3531,3530,3529,3528,3527,3526,3525,3524,3523,3522,3521,3520,3519,3518,3517,3516,3515,3514,3513,3512,3511,3510,3509,3508,3507,3506,3505,3504,3503,3502,3501,3500,3499,3498,3497,3496,3495,3494,3493,3492,3491,3490,3489,3488,3487,3486,3485,3484,3483,3482,3481,3480,3479,3478,3477,3476,3475,3474,3473,3472,3471,3470,3469,3468,3467,3466,3465,3464,3463,3462,3461,3460,3459,3458,3457,3456,3455,3454,3453,3452,3451,3450,3449,3448,3447,3446,3445,3444,3443,3442,3441,3440,3439,3438,3437,3436,3435,3434,3433,3432,3431,3430,3429,3428,3427,3426,3425,3424,3423,3422,3421,3420,3419,3418,3417,3416,3415,3414,3413,3412,3411,3410,3409,3408,3407,3406,3405,3404,3403,3402,3401,3400,3399,3398,3397,3396,3395,3394,3393,3392,3391,3390,3389,3388,3387,3386,3385,3384,3383,3382,3381,3380,3379,3378,3377,3376,3375,3374,3373,3372,3371,3370,3369,3368,3367,3366,3365,3364,3363,3362,3361,3360,3359,3358,3357,3356,3355,3354,3353,3352,3351,3350,3349,3348,3347,3346,3345,3344,3343,3342,3341,3340,3339,3338,3337,3336,3335,3334,3333,3332,3331,3330,3329,3328,3327,3326,3325,3324,3323,3322,3321,3320,3319,3318,3317,3316,3315,3314,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301,3300,3299,3298,3297,3296,3295,3294,3293,3292,3291,3290,3289,3288,3287,3286,3285,3284,3283,3282,3281,3280,3279,3278,3277,3276,3275,3274,3273,3272,3271,3270,3269,3268,3267,3266,3265,3264,3263,3262,3261,3260,3259,3258,3257,3256,3255,3254,3253,3252,3251,3250,3249,3248,3247,3246,3245,3244,3243,3242,3241,3240,3239,3238,3237,3236,3235,3234,3233,3232,3231,3230,3229,3228,3227,3226,3225,3224,3223,3222,3221,3220,3219,3218,3217,3216,3215,3214,3213,3212,3211,3210,3209,3208,3207,3206,3205,3204,3203,3202,3201,3200,3199,3198,3197,3196,3195,3194,3193,3192,3191,3190,3189,3188,3187,3186,3185,3184,3183,3182,3181,3180,3179,3178,3177,3176,3175,3174,3173,3172,3171,3170,3169,3168,3167,3166,3165,3164,3163,3162,3161,3160,3159,3158,3157,3156,3155,3154,3153,3152,3151,3150,3149,3148,3147,3146,3145,3144,3143,3142,3141,3140,3139,3138,3137,3136,3135,3134,3133,3132,3131,3130,3129,3128,3127,3126,3125,3124,3123,3122,3121,3120,3119,3118,3117,3116,3115,3114,3113,3112,3111,3110,3109,3108,3107,3106,3105,3104,3103,3102,3101,3100,3099,3098,3097,3096,3095,3094,3093,3092,3091,3090,3089,3088,3087,3086,3085,3084,3083,3082,3081,3080,3079,3078,3077,3076,3075,3074,3073,3072,3071,3070,3069,3068,3067,3066,3065,3064,3063,3062,3061,3060,3059,3058,3057,3056,3055,3054,3053,3052,3051,3050,3049,3048,3047,3046,3045,3044,3043,3042,3041,3040,3039,3038,3037,3036,3035,3034,3033,3032,3031,3030,3029,3028,3027,3026,3025,3024,3023,3022,3021,3020,3019,3018,3017,3016,3015,3014,3013,3012,3011,3010,3009,3008,3007,3006,3005,3004,3003,3002,3001,3000,2999,2998,2997,2996,2995,2994,2993,2992,2991,2990,2989,2988,2987,2986,2985,2984,2983,2982,2981,2980,2979,2978,2977,2976,2975,2974,2973,2972,2971,2970,2969,2968,2967,2966,2965,2964,2963,2962,2961,2960,2959,2958,2957,2956,2955,2954,2953,2952,2951,2950,2949,2948,2947,2946,2945,2944,2943,2942,2941,2940,2939,2938,2937,2936,2935,2934,2933,2932,2931,2930,2929,2928,2927,2926,2925,2924,2923,2922,2921,2920,2919,2918,2917,2916,2915,2914,2913,2912,2911,2910,2909,2908,2907,2906,2905,2904,2903,2902,2901,2900,2899,2898,2897,2896,2895,2894,2893,2892,2891,2890,2889,2888,2887,2886,2885,2884,2883,2882,2881,2880,2879,2878,2877,2876,2875,2874,2873,2872,2871,2870,2869,2868,2867,2866,2865,2864,2863,2862,2861,2860,2859,2858,2857,2856,2855,2854,2853,2852,2851,2850,2849,2848,2847,2846,2845,2844,2843,2842,2841,2840,2839,2838,2837,2836,2835,2834,2833,2832,2831,2830,2829,2828,2827,2826,2825,2824,2823,2822,2821,2820,2819,2818,2817,2816,2815,2814,2813,2812,2811,2810,2809,2808,2807,2806,2805,2804,2803,2802,2801,2800,2799,2798,2797,2796,2795,2794,2793,2792,2791,2790,2789,2788,2787,2786,2785,2784,2783,2782,2781,2780,2779,2778,2777,2776,2775,2774,2773,2772,2771,2770,2769,2768,2767,2766,2765,2764,2763,2762,2761,2760,2759,2758,2757,2756,2755,2754,2753,2752,2751,2750,2749,2748,2747,2746,2745,2744,2743,2742,2741,2740,2739,2738,2737,2736,2735,2734,2733,2732,2731,2730,2729,2728,2727,2726,2725,2724,2723,2722,2721,2720,2719,2718,2717,2716,2715,2714,2713,2712,2711,2710,2709,2708,2707,2706,2705,2704,2703,2702,2701,2700,2699,2698,2697,2696,2695,2694,2693,2692,2691,2690,2689,2688,2687,2686,2685,2684,2683,2682,2681,2680,2679,2678,2677,2676,2675,2674,2673,2672,2671,2670,2669,2668,2667,2666,2665,2664,2663,2662,2661,2660,2659,2658,2657,2656,2655,2654,2653,2652,2651,2650,2649,2648,2647,2646,2645,2644,2643,2642,2641,2640,2639,2638,2637,2636,2635,2634,2633,2632,2631,2630,2629,2628,2627,2626,2625,2624,2623,2622,2621,2620,2619,2618,2617,2616,2615,2614,2613,2612,2611,2610,2609,2608,2607,2606,2605,2604,2603,2602,2601,2600,2599,2598,2597,2596,2595,2594,2593,2592,2591,2590,2589,2588,2587,2586,2585,2584,2583,2582,2581,2580,2579,2578,2577,2576,2575,2574,2573,2572,2571,2570,2569,2568,2567,2566,2565,2564,2563,2562,2561,2560,2559,2558,2557,2556,2555,2554,2553,2552,2551,2550,2549,2548,2547,2546,2545,2544,2543,2542,2541,2540,2539,2538,2537,2536,2535,2534,2533,2532,2531,2530,2529,2528,2527,2526,2525,2524,2523,2522,2521,2520,2519,2518,2517,2516,2515,2514,2513,2512,2511,2510,2509,2508,2507,2506,2505,2504,2503,2502,2501,2500,2499,2498,2497,2496,2495,2494,2493,2492,2491,2490,2489,2488,2487,2486,2485,2484,2483,2482,2481,2480,2479,2478,2477,2476,2475,2474,2473,2472,2471,2470,2469,2468,2467,2466,2465,2464,2463,2462,2461,2460,2459,2458,2457,2456,2455,2454,2453,2452,2451,2450,2449,2448,2447,2446,2445,2444,2443,2442,2441,2440,2439,2438,2437,2436,2435,2434,2433,2432,2431,2430,2429,2428,2427,2426,2425,2424,2423,2422,2421,2420,2419,2418,2417,2416,2415,2414,2413,2412,2411,2410,2409,2408,2407,2406,2405,2404,2403,2402,2401,2400,2399,2398,2397,2396,2395,2394,2393,2392,2391,2390,2389,2388,2387,2386,2385,2384,2383,2382,2381,2380,2379,2378,2377,2376,2375,2374,2373,2372,2371,2370,2369,2368,2367,2366,2365,2364,2363,2362,2361,2360,2359,2358,2357,2356,2355,2354,2353,2352,2351,2350,2349,2348,2347,2346,2345,2344,2343,2342,2341,2340,2339,2338,2337,2336,2335,2334,2333,2332,2331,2330,2329,2328,2327,2326,2325,2324,2323,2322,2321,2320,2319,2318,2317,2316,2315,2314,2313,2312,2311,2310,2309,2308,2307,2306,2305,2304,2303,2302,2301,2300,2299,2298,2297,2296,2295,2294,2293,2292,2291,2290,2289,2288,2287,2286,2285,2284,2283,2282,2281,2280,2279,2278,2277,2276,2275,2274,2273,2272,2271,2270,2269,2268,2267,2266,2265,2264,2263,2262,2261,2260,2259,2258,2257,2256,2255,2254,2253,2252,2251,2250,2249,2248,2247,2246,2245,2244,2243,2242,2241,2240,2239,2238,2237,2236,2235,2234,2233,2232,2231,2230,2229,2228,2227,2226,2225,2224,2223,2222,2221,2220,2219,2218,2217,2216,2215,2214,2213,2212,2211,2210,2209,2208,2207,2206,2205,2204,2203,2202,2201,2200,2199,2198,2197,2196,2195,2194,2193,2192,2191,2190,2189,2188,2187,2186,2185,2184,2183,2182,2181,2180,2179,2178,2177,2176,2175,2174,2173,2172,2171,2170,2169,2168,2167,2166,2165,2164,2163,2162,2161,2160,2159,2158,2157,2156,2155,2154,2153,2152,2151,2150,2149,2148,2147,2146,2145,2144,2143,2142,2141,2140,2139,2138,2137,2136,2135,2134,2133,2132,2131,2130,2129,2128,2127,2126,2125,2124,2123,2122,2121,2120,2119,2118,2117,2116,2115,2114,2113,2112,2111,2110,2109,2108,2107,2106,2105,2104,2103,2102,2101,2100,2099,2098,2097,2096,2095,2094,2093,2092,2091,2090,2089,2088,2087,2086,2085,2084,2083,2082,2081,2080,2079,2078,2077,2076,2075,2074,2073,2072,2071,2070,2069,2068,2067,2066,2065,2064,2063,2062,2061,2060,2059,2058,2057,2056,2055,2054,2053,2052,2051,2050,2049,2048,2047,2046,2045,2044,2043,2042,2041,2040,2039,2038,2037,2036,2035,2034,2033,2032,2031,2030,2029,2028,2027,2026,2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999,1998,1997,1996,1995,1994,1993,1992,1991,1990,1989,1988,1987,1986,1985,1984,1983,1982,1981,1980,1979,1978,1977,1976,1975,1974,1973,1972,1971,1970,1969,1968,1967,1966,1965,1964,1963,1962,1961,1960,1959,1958,1957,1956,1955,1954,1953,1952,1951,1950,1949,1948,1947,1946,1945,1944,1943,1942,1941,1940,1939,1938,1937,1936,1935,1934,1933,1932,1931,1930,1929,1928,1927,1926,1925,1924,1923,1922,1921,1920,1919,1918,1917,1916,1915,1914,1913,1912,1911,1910,1909,1908,1907,1906,1905,1904,1903,1902,1901,1900,1899,1898,1897,1896,1895,1894,1893,1892,1891,1890,1889,1888,1887,1886,1885,1884,1883,1882,1881,1880,1879,1878,1877,1876,1875,1874,1873,1872,1871,1870,1869,1868,1867,1866,1865,1864,1863,1862,1861,1860,1859,1858,1857,1856,1855,1854,1853,1852,1851,1850,1849,1848,1847,1846,1845,1844,1843,1842,1841,1840,1839,1838,1837,1836,1835,1834,1833,1832,1831,1830,1829,1828,1827,1826,1825,1824,1823,1822,1821,1820,1819,1818,1817,1816,1815,1814,1813,1812,1811,1810,1809,1808,1807,1806,1805,1804,1803,1802,1801,1800,1799,1798,1797,1796,1795,1794,1793,1792,1791,1790,1789,1788,1787,1786,1785,1784,1783,1782,1781,1780,1779,1778,1777,1776,1775,1774,1773,1772,1771,1770,1769,1768,1767,1766,1765,1764,1763,1762,1761,1760,1759,1758,1757,1756,1755,1754,1753,1752,1751,1750,1749,1748,1747,1746,1745,1744,1743,1742,1741,1740,1739,1738,1737,1736,1735,1734,1733,1732,1731,1730,1729,1728,1727,1726,1725,1724,1723,1722,1721,1720,1719,1718,1717,1716,1715,1714,1713,1712,1711,1710,1709,1708,1707,1706,1705,1704,1703,1702,1701,1700,1699,1698,1697,1696,1695,1694,1693,1692,1691,1690,1689,1688,1687,1686,1685,1684,1683,1682,1681,1680,1679,1678,1677,1676,1675,1674,1673,1672,1671,1670,1669,1668,1667,1666,1665,1664,1663,1662,1661,1660,1659,1658,1657,1656,1655,1654,1653,1652,1651,1650,1649,1648,1647,1646,1645,1644,1643,1642,1641,1640,1639,1638,1637,1636,1635,1634,1633,1632,1631,1630,1629,1628,1627,1626,1625,1624,1623,1622,1621,1620,1619,1618,1617,1616,1615,1614,1613,1612,1611,1610,1609,1608,1607,1606,1605,1604,1603,1602,1601,1600,1599,1598,1597,1596,1595,1594,1593,1592,1591,1590,1589,1588,1587,1586,1585,1584,1583,1582,1581,1580,1579,1578,1577,1576,1575,1574,1573,1572,1571,1570,1569,1568,1567,1566,1565,1564,1563,1562,1561,1560,1559,1558,1557,1556,1555,1554,1553,1552,1551,1550,1549,1548,1547,1546,1545,1544,1543,1542,1541,1540,1539,1538,1537,1536,1535,1534,1533,1532,1531,1530,1529,1528,1527,1526,1525,1524,1523,1522,1521,1520,1519,1518,1517,1516,1515,1514,1513,1512,1511,1510,1509,1508,1507,1506,1505,1504,1503,1502,1501,1500,1499,1498,1497,1496,1495,1494,1493,1492,1491,1490,1489,1488,1487,1486,1485,1484,1483,1482,1481,1480,1479,1478,1477,1476,1475,1474,1473,1472,1471,1470,1469,1468,1467,1466,1465,1464,1463,1462,1461,1460,1459,1458,1457,1456,1455,1454,1453,1452,1451,1450,1449,1448,1447,1446,1445,1444,1443,1442,1441,1440,1439,1438,1437,1436,1435,1434,1433,1432,1431,1430,1429,1428,1427,1426,1425,1424,1423,1422,1421,1420,1419,1418,1417,1416,1415,1414,1413,1412,1411,1410,1409,1408,1407,1406,1405,1404,1403,1402,1401,1400,1399,1398,1397,1396,1395,1394,1393,1392,1391,1390,1389,1388,1387,1386,1385,1384,1383,1382,1381,1380,1379,1378,1377,1376,1375,1374,1373,1372,1371,1370,1369,1368,1367,1366,1365,1364,1363,1362,1361,1360,1359,1358,1357,1356,1355,1354,1353,1352,1351,1350,1349,1348,1347,1346,1345,1344,1343,1342,1341,1340,1339,1338,1337,1336,1335,1334,1333,1332,1331,1330,1329,1328,1327,1326,1325,1324,1323,1322,1321,1320,1319,1318,1317,1316,1315,1314,1313,1312,1311,1310,1309,1308,1307,1306,1305,1304,1303,1302,1301,1300,1299,1298,1297,1296,1295,1294,1293,1292,1291,1290,1289,1288,1287,1286,1285,1284,1283,1282,1281,1280,1279,1278,1277,1276,1275,1274,1273,1272,1271,1270,1269,1268,1267,1266,1265,1264,1263,1262,1261,1260,1259,1258,1257,1256,1255,1254,1253,1252,1251,1250,1249,1248,1247,1246,1245,1244,1243,1242,1241,1240,1239,1238,1237,1236,1235,1234,1233,1232,1231,1230,1229,1228,1227,1226,1225,1224,1223,1222,1221,1220,1219,1218,1217,1216,1215,1214,1213,1212,1211,1210,1209,1208,1207,1206,1205,1204,1203,1202,1201,1200,1199,1198,1197,1196,1195,1194,1193,1192,1191,1190,1189,1188,1187,1186,1185,1184,1183,1182,1181,1180,1179,1178,1177,1176,1175,1174,1173,1172,1171,1170,1169,1168,1167,1166,1165,1164,1163,1162,1161,1160,1159,1158,1157,1156,1155,1154,1153,1152,1151,1150,1149,1148,1147,1146,1145,1144,1143,1142,1141,1140,1139,1138,1137,1136,1135,1134,1133,1132,1131,1130,1129,1128,1127,1126,1125,1124,1123,1122,1121,1120,1119,1118,1117,1116,1115,1114,1113,1112,1111,1110,1109,1108,1107,1106,1105,1104,1103,1102,1101,1100,1099,1098,1097,1096,1095,1094,1093,1092,1091,1090,1089,1088,1087,1086,1085,1084,1083,1082,1081,1080,1079,1078,1077,1076,1075,1074,1073,1072,1071,1070,1069,1068,1067,1066,1065,1064,1063,1062,1061,1060,1059,1058,1057,1056,1055,1054,1053,1052,1051,1050,1049,1048,1047,1046,1045,1044,1043,1042,1041,1040,1039,1038,1037,1036,1035,1034,1033,1032,1031,1030,1029,1028,1027,1026,1025,1024,1023,1022,1021,1020,1019,1018,1017,1016,1015,1014,1013,1012,1011,1010,1009,1008,1007,1006,1005,1004,1003,1002,1001,1000,999,998,997,996,995,994,993,992,991,990,989,988,987,986,985,984,983,982,981,980,979,978,977,976,975,974,973,972,971,970,969,968,967,966,965,964,963,962,961,960,959,958,957,956,955,954,953,952,951,950,949,948,947,946,945,944,943,942,941,940,939,938,937,936,935,934,933,932,931,930,929,928,927,926,925,924,923,922,921,920,919,918,917,916,915,914,913,912,911,910,909,908,907,906,905,904,903,902,901,900,899,898,897,896,895,894,893,892,891,890,889,888,887,886,885,884,883,882,881,880,879,878,877,876,875,874,873,872,871,870,869,868,867,866,865,864,863,862,861,860,859,858,857,856,855,854,853,852,851,850,849,848,847,846,845,844,843,842,841,840,839,838,837,836,835,834,833,832,831,830,829,828,827,826,825,824,823,822,821,820,819,818,817,816,815,814,813,812,811,810,809,808,807,806,805,804,803,802,801,800,799,798,797,796,795,794,793,792,791,790,789,788,787,786,785,784,783,782,781,780,779,778,777,776,775,774,773,772,771,770,769,768,767,766,765,764,763,762,761,760,759,758,757,756,755,754,753,752,751,750,749,748,747,746,745,744,743,742,741,740,739,738,737,736,735,734,733,732,731,730,729,728,727,726,725,724,723,722,721,720,719,718,717,716,715,714,713,712,711,710,709,708,707,706,705,704,703,702,701,700,699,698,697,696,695,694,693,692,691,690,689,688,687,686,685,684,683,682,681,680,679,678,677,676,675,674,673,672,671,670,669,668,667,666,665,664,663,662,661,660,659,658,657,656,655,654,653,652,651,650,649,648,647,646,645,644,643,642,641,640,639,638,637,636,635,634,633,632,631,630,629,628,627,626,625,624,623,622,621,620,619,618,617,616,615,614,613,612,611,610,609,608,607,606,605,604,603,602,601,600,599,598,597,596,595,594,593,592,591,590,589,588,587,586,585,584,583,582,581,580,579,578,577,576,575,574,573,572,571,570,569,568,567,566,565,564,563,562,561,560,559,558,557,556,555,554,553,552,551,550,549,548,547,546,545,544,543,542,541,540,539,538,537,536,535,534,533,532,531,530,529,528,527,526,525,524,523,522,521,520,519,518,517,516,515,514,513,512,511,510,509,508,507,506,505,504,503,502,501,500,499,498,497,496,495,494,493,492,491,490,489,488,487,486,485,484,483,482,481,480,479,478,477,476,475,474,473,472,471,470,469,468,467,466,465,464,463,462,461,460,459,458,457,456,455,454,453,452,451,450,449,448,447,446,445,444,443,442,441,440,439,438,437,436,435,434,433,432,431,430,429,428,427,426,425,424,423,422,421,420,419,418,417,416,415,414,413,412,411,410,409,408,407,406,405,404,403,402,401,400,399,398,397,396,395,394,393,392,391,390,389,388,387,386,385,384,383,382,381,380,379,378,377,376,375,374,373,372,371,370,369,368,367,366,365,364,363,362,361,360,359,358,357,356,355,354,353,352,351,350,349,348,347,346,345,344,343,342,341,340,339,338,337,336,335,334,333,332,331,330,329,328,327,326,325,324,323,322,321,320,319,318,317,316,315,314,313,312,311,310,309,308,307,306,305,304,303,302,301,300,299,298,297,296,295,294,293,292,291,290,289,288,287,286,285,284,283,282,281,280,279,278,277,276,275,274,273,272,271,270,269,268,267,266,265,264,263,262,261,260,259,258,257,256,255,254,253,252,251,250,249,248,247,246,245,244,243,242,241,240,239,238,237,236,235,234,233,232,231,230,229,228,227,226,225,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,207,206,205,204,203,202,201,200,199,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,111,110,109,108,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,1,0]
        