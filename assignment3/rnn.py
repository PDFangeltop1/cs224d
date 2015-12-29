#ESC + shift + 5:  substitude query , press y to commit !

import sys
sys.path.append('./nn/')

import numpy as np
import collections
from nn.math import softmax, make_onehot



# This is a simple Recursive Neural Netowrk with one ReLU Layer and a softmax layer
# TODO: You must update the forward and backward propogation functions of this file.

# You can run this file via 'python rnn.py' to perform a gradient check! 

# tip: insert pdb.set_trace() in places where you are unsure whats going on

class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden layer parameters
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        #self.Ws = 0.01*np.random.randn(self.wvecDim, self.outputDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
           
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def ReLU(self,x,label=0):
        if label == 0:
            return x*(x > 0)
        else:
            return 1.0/(1+np.exp(-x))

    def df(self,x,label=0):
        if(label == 0):
            return 1.0*(x > 0)
        else:
            f = self.ReLU(x)
            return f*(1-f)
        
    def forwardProp(self,node,correct=[], guess=[]):
        cost  =  total = 0.0 # cost should be a running number and total is the total examples we have seen used in accuracy reporting later
        ################
        # TODO: Implement the recursive forwardProp function
        #  - you should update node.probs, node.hActs1, node.fprop, and cost
        #  - node: your current node in the parse tree
        #  - correct: this is a running list of truth labels
        #  - guess: this is a running list of guess that our model makes
        #     (we will use both correct and guess to make our confusion matrix)
        ################

        if node.isLeaf == True:
            node.fprop = True
            node.hActs1 = self.L[:, node.word]
            node.probs = softmax(self.Ws.dot(node.hActs1) + self.bs)
            p = node.probs*make_onehot(node.label, len(self.bs))
            cost = -np.log(np.sum(p))
            correct.append(node.label)
            guess.append(np.argmax(node.probs))
            return cost, 1
        
        c1,t1 = self.forwardProp(node.left,correct,guess)
        c2,t2 = self.forwardProp(node.right,correct,guess)
        if node.left.fprop and node.right.fprop:
            node.fprop = True
            h = np.hstack([node.left.hActs1, node.right.hActs1])
            node.hActs1 = self.ReLU(self.W.dot(h) + self.b)
            node.probs = softmax(self.Ws.dot(node.hActs1)+self.bs)
            p = node.probs*make_onehot(node.label,len(self.bs))
            cost = -np.log(np.sum(p))
            correct.append(node.label)
            guess.append(np.argmax(node.probs))
            
        cost += c1
        cost += c2
        total += t1
        total += t2
        return cost, total + 1

    
    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False
        ################
        # TODO: Implement the recursive backProp function
        #  - you should update self.dWs, self.dbs, self.dW, self.db, and self.dL[node.word] accordingly
        #  - node: your current node in the parse tree
        #  - error: error that has been passed down from a previous iteration
        ################

        errorCur = node.probs - make_onehot(node.label,len(self.bs))
        self.dWs += np.outer(errorCur, node.hActs1)
        self.dbs += errorCur

        errorCur = errorCur.dot(self.Ws)
        if error is not None:
            errorCur += error

        if node.isLeaf == True:
            self.dL[node.word] += errorCur
            return

        errorCur = errorCur*self.df(node.hActs1)
        self.dW += np.outer(errorCur,np.hstack([node.left.hActs1, node.right.hActs1]))
        self.db += errorCur
        errorDown = errorCur.dot(self.W)        
        self.backProp(node.left,errorDown[:self.wvecDim])
        self.backProp(node.right,errorDown[self.wvecDim:])
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)        
        err1 = 0.0
        count = 0.0
        print "Checking dW..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
                    if err > 0.00001:
                        print " -> ",W.shape,"W[%d, %d] = %.12f" %(i,j, err)

        if 0.0001 > err1/count:
            print "Grad Check Passed for dW Sum of Error = %.12f" % (err1/count)
        else:
            print "Grad Check Failed for dW: Sum of Error = %.12f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1

        if 0.0001 > err2/count:
            print "Grad Check Passed for dL Sum of Error = %.12f" % (err2/count)
        else:
            print "Grad Check Failed for dL: Sum of Error = %.12f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())
        
    wvecDim = 10
    outputDim = 5

    np.random.seed(10)
    rnn = RNN(wvecDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






