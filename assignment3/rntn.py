import numpy as np
import collections
np.seterr(over='raise',under='raise')

from nn.math import softmax, make_onehot
class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
        # Word vectors
        self.L = 0.001*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.001*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.001*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.001*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        cost = 0.0
        correct = []
        guess = []
        total = 0.0
        self.L, self.V, self.W, self.b, self.Ws, self.bs = self.stack

        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dV[:] = 0
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
        cost += (self.rho/2)*np.sum(self.V**2)
        #for i in range(len(self.V)):
        #    cost += (self.rho/2)*np.sum(self.V[i])
            
        return cost*scale, [self.dL, scale*(self.dV + self.rho*self.V),scale*(self.dW + self.rho*self.dW), scale*self.db, scale*(self.dWs+self.rho*self.Ws), scale*(self.dbs)]

    def ReLU(self,x):
        #return x*(x > 0)
        #return 1.0/(1+np.exp(-x))
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def df(self,x):
        #return 1.0*(x > 0)
        f = self.ReLU(x)
        #return f*(1-f)
        return 1 - f*f

    def forwardProp(self,node,correct, guess):
        cost = total = 0.0
        if node.isLeaf == True:
            node.fprop = True
            node.hActs1 = self.L[:, node.word]
            node.probs = softmax(self.Ws.dot(node.hActs1)+self.bs)
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
            tmp = np.zeros(len(node.left.hActs1))
            for i in range(len(tmp)):
                tmp[i] = h.dot(self.V[i]).dot(h)
            node.hActs1 = self.ReLU(self.W.dot(h) + self.b + tmp)
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
        LR = np.hstack([node.left.hActs1, node.right.hActs1])
        self.dW += np.outer(errorCur,LR)
        self.db += errorCur

        S = np.zeros(len(LR))
        for i in range(len(self.V)):
            self.dV[i] += errorCur[i]*np.outer(LR,LR)
            S += (self.V[i]+self.V[i].T).dot(LR)*errorCur[i]
        
        errorDown = errorCur.dot(self.W) + S        
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

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
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
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
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

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






