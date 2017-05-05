###############################################################################
# 208. Implement Trie (Prefix Tree)
###############################################################################

class TrieNode(object):
    def __init__(self):
        #self.val = char
        self.children = {}
        self.isLeaf = False

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        
        if (not word):
            return
        
        ln = len(word)
        
        tmp = self.root
        i = 0
        while(i < ln):
            if (word[i] in tmp.children):
                tmp = tmp.children[word[i]]
            else:
                tmp.children[word[i]] = TrieNode()
                tmp = tmp.children[word[i]]
            i += 1
        
        tmp.isLeaf = True    
        
        return
                

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        
        if (not word):
            return True
            
        ln = len(word)
        
        tmp = self.root
        
        i = 0
        while (i < ln):
            if (word[i] not in tmp.children):
                return False
            tmp = tmp.children[word[i]]
            i += 1
            
        return tmp.isLeaf
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        if (not prefix):
            return True
            
        ln = len(prefix)
        
        tmp = self.root
        
        i = 0
        while (i < ln):
            if (prefix[i] not in tmp.children):
                return False
            tmp = tmp.children[prefix[i]]
            i += 1
            
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)