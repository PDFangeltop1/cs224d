from numpy import *
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def sigmoidGrad(x):
    return sigmoid(x)*(1-sigmoid(x))


def softmax(x):
    #x = x-x.max(axis=1).reshape(x.shape[0],1)
    #b = np.sum(np.exp(x),axis=1)
    #x = np.exp(x)/b.reshape(x.shape[0],1)
    x = x-x.max()
    b = np.sum(np.exp(x))
    x = np.exp(x)/b
    return x

def make_onehot(i, n):
    y = np.zeros(n)
    y[i] = 1
    return y

def random_weight_matrix(m,n):
    A0 = np.random.random((m,n))
    A0 = A0*np.sqrt(6.0/(m+n))
    assert (A0.shape == (m,n))
    return A0

class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = cumsum(p)

    def sample(self, k=1):
        rs = np.random.random(k)
        # binary search to get indices
        return np.searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = np.zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p


def multinomial_sample(p):
    """
    Wrapper to generate a single sample,
    using the above class.
    """
    return MultinomialSampler(p).sample(1)[0]


class Node:
    def __init__(self, label, word=None):
        self.label = label
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.isLeft = False
        self.hActs = None
        self.grad = None
    
class HierarchicalSoftmaxTree:
    
    def __init__(self, vocabsize,hDim,word2node):
        self.cnt = 0
        self.hDim = hDim
        self.vocabsize = vocabsize
        self.root = self.parse(0,vocabsize,word2node)

    def parse(self,l,r,word2node,parent=None,isLeft=False):
        node = Node(self.cnt)
        self.cnt += 1
        node.isLeft = isLeft
        node.parent = parent
        if l == r:
            node.isLeaf = True
            word2node[l] = node
            return node
            
        node.hActs = np.random.random(self.hDim)
        mid = (l+r)/2
        node.left = self.parse(l,mid,word2node,parent=node,isLeft=True)
        node.right = self.parse(mid+1,r,word2node,parent=node)
        return node
        
    def getSumSquareU(self, node):
        if node.isLeaf == True:
            return 0
        if node.grad == None:
            return 0

        sum1 = np.sum(node.hActs**2)
        sum1 = sum1 + self.getSumSquareU(node.right)
        sum1 = sum1 + self.getSumSquareU(node.left)
        return sum1

    def regularizedGrad(self,node, lreg):    
        if node.isLeaf == True:
            return
        if node.grad == None:
            return 

        node.grad = node.grad + lreg*node.hActs
        self.regularizedGrad(node.left,lreg)
        self.regularizedGrad(node.right,lreg)

    def reset(self,node):
        if node.isLeaf == True:
            return
        if node.grad == None:
            return
 
        node.grad = None
        self.reset(node.left)
        self.reset(node.right)

    def apply_grad_acc(self,node,alpha):
        if node.isLeaf == True:
            return 
        if node.grad == None:
            return         

        node.hActs = node.hActs + alpha*node.grad
        self.apply_grad_acc(node.left,alpha)
        self.apply_grad_acc(node.right,alpha)
        
    #####  For generating sequence ######
    def getDistributionRecursive(self,l,r,pArray,h,tmpProbability,node):
        if l == r:
            pArray.append(tmpProbability)
            return
        mid = (l+r)/2
        self.getDistributionRecursive(l,mid,pArray,h, tmpProbability*sigmoid(node.hActs*h),node.left)
        self.getDistributionRecursive(mid+1,r,pArray,h,tmpProbability*sigmoid(-node.hActs*h),node.right)
            
    def getDistribution(self,h):
        pArray = []
        self.getDistributionRecursive(0,self.vocabsize,pArray,h,1.0,self.root)
        return pArray

        


