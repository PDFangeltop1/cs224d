from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot ,random_weight_matrix
import numpy as np
import warnings

##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda) 
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate
        
        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####
        
        # any other initialization you need
        self.nclass = dims[2]
        self.sparams.L = wv.copy()
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)

        #### END YOUR CODE ####

    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))*1.0/(np.exp(x)+np.exp(-x))

    def tanhGrad(self,x):
        return 1-self.tanh(x)**2
    
    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####
        
        ##
        # Forward propagation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.sparams.L[window[0]]
            for i in range(len(window)-1):
                x = np.concatenate((x,self.sparams.L[window[i+1]]),axis=1)
        
        z = self.params.W.dot(x)+self.params.b1
        h = self.tanh(z)
        y = make_onehot(label,self.params.b2.shape[0])
        yP = softmax(self.params.U.dot(h)+self.params.b2)

        ##
        # Backpropagation
        self.grads.U += np.outer(yP-y,h) + self.lreg*self.params.U
        self.grads.b2 += yP-y
        self.grads.W += np.outer((self.params.U.T.dot(yP-y)*self.tanhGrad(z)),x) + self.lreg*self.params.W
        self.grads.b1 += self.params.U.T.dot(yP-y)*self.tanhGrad(z)

        gL = self.params.W.T.dot(self.params.U.T.dot(yP-y)*self.tanhGrad(z))
        #print len(gL)
        for i in range(len(window)):
            self.sgrads.L[window[i]] = gL[len(gL)*i/len(window):len(gL)*(i+1)/len(window)]
        #### END YOUR CODE ####

    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        x = self.sparams.L[windows[:,0]]
        for i in range(len(windows[0])-1):
            x = np.concatenate((x,self.sparams.L[windows[:,i+1]]),axis=1)
            
        z = self.params.W.dot(x.T)+self.params.b1.reshape((self.params.b1.shape[0],1))
        h = self.tanh(z)
        p = softmax(self.params.U.dot(h)+self.params.b2.reshape((self.params.b2.shape[0],1)))
        #### END YOUR CODE ####
        return p # rows are output for each input

    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        p = self.predict_proba(windows)
        c = np.argmax(p,axis=0)  #axis=0, column-wise
        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####

        print "windows shape ", windows.shape 
        x = self.sparams.L[windows[:,0]]
        for i in range(len(windows[0])-1):
            x = np.concatenate((x,self.sparams.L[windows[:,i+1]]),axis=1)

        z = self.params.W.dot(x.T)+self.params.b1.reshape((self.params.b1.shape[0],1))
        h = tanh(z)
        p = softmax(self.params.U.dot(h)+self.params.b2.reshape((self.params.b2.shape[0],1)))
        labelArray = np.zeros((len(labels),self.params.b2.shape[0]))
        for i in range(len(labels)):
            labelArray[i] = make_onehot(labels[i],self.params.b2.shape[0])
        batch = len(labels)
        p = p*labelArray.T
        p = np.sum(p,axis=0)
        J = np.sum(-np.log(p))
        Jreg = batch*(self.lreg/2.0)*(np.sum(self.params.W**2)+np.sum(self.params.U**2))
        J += Jreg                    
        #### END YOUR CODE ####
        return J


def get_precision(clf,X_dev,y_dev):
    count = 0
    for i,w in enumerate(X_dev):
        labelP = clf.predict(w)
        if labelP == y_dev[i]:
            count += 1

    print "Precision: --> ",count*100.0/len(X_dev)
##############################################################    
def Generator(y_train):
    nepoch = 5
    N = nepoch*len(y_train)
    k = 5
    random.seed(10)
    for i in range(N):
        yield np.random.randint(0,len(y_train),k)   
##############################################################
def makePlot(trainCosts):
    import matplotlib.pyplot as plt
    counts, costs = zip(*trainCosts)
    plt.figure(figsize=(6,4))
    plt.plot(5*np.array(counts),costs,color='b',marker='o',linestyle=
'-')
    plt.title(r"Learning Curve ($\alpha$=%g,$\lambda$=%g)"%(clf.alpha,clf.lreg))
    plt.xlabel("SGD Iterations");plt.ylabel(r"Average $J(\theta)$");
    plt.ylim(ymin=0,ymax=max(1.1*max(costs),3*min(costs)))
    plt.savefig("ner.learningcurve.png")

###############################################################
import collections    
def print_scores(scores,words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i]*10000, words[i])

def numericalCmp(x,y):
    if(y[0] > x[0]):
        return 1;
    else:
        return -1;

def HiddenLayerCenterWord(clf,word_to_num,num_to_word,windowsize):
    #clf.sparams.L
    #clf.params.W[:,]
    #clf.params.b1
    #hiddenLayer [C,N]
    hiddenLayer = collections.defaultdict(list)
    for w,n in word_to_num.iteritems():
        d = len(clf.sparams.L[n])
        h = clf.params.W[:,(windowsize/2)*d:(windowsize/2+1)*d].dot(clf.sparams.L[n])+clf.params.b1
        for i,v in enumerate(h):            
            hiddenLayer[i].append((v,n))

    topN = 10
    topscores = np.zeros((len(clf.params.b1),topN))
    topwords = np.zeros((len(clf.params.b1),topN))
    for i in range(len(hiddenLayer)):
        a = sorted(hiddenLayer[i],numericalCmp)
        for j in range(topN):
            topscores[i][j] = a[j][0]
            topwords[i][j] = a[j][1]
        
    neurons = [1,3,4,5,8]
    for i in neurons:
        print "Neuron %d" %i
        words = []
        for j in topwords[i]:
            words.append(num_to_word[j])
        print_scores(topscores[i],words)

def ModelOutputCenterWord(clf,word_to_num,num_to_word,num_to_tag,windowsize):
    d = len(clf.sparams.L[0])
    partW = clf.params.W[:,(windowsize/2)*d:(windowsize/2+1)*d]
    z = clf.sparams.L.dot(partW.T) + clf.params.b1 # z -> (N,h)
    h = clf.tanh(z)
    p = softmax(h.dot(clf.params.U.T)+clf.params.b2)  #p ->(N,C)
    
    outputLayer = collections.defaultdict(list)
    for i in range(len(p)):
        for j in range(len(p[i])):
            outputLayer[j].append((p[i][j],i))

    topN = 10
    topscores = np.zeros((len(clf.params.b2),topN))
    topwords = np.zeros((len(clf.params.b2),topN))
    for i in range(len(outputLayer)):
        a = sorted(outputLayer[i],numericalCmp)
        for j in range(topN):
            topscores[i][j] = a[j][0]
            topwords[i][j] = a[j][1]

    print "topscores -->"
    print topscores
    for i in range(1,5):
        print "Output Neuron %d: %s" % (i,num_to_tag[i])
        words = []
        for j in topwords[i]:
            words.append(num_to_word[j])
        print_scores(topscores[i],words)


####################################################################

if __name__ == "__main__":
    import data_utils.ner as ner
    import data_utils.utils as du
    wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt','data/ner/wordVectors.txt')


    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = du.invert_dict(num_to_tag)
    windowsize = 3
    docs = du.load_dataset('data/ner/train')

    X_train, y_train = du.docs_to_windows(docs,word_to_num,tag_to_num,
                                          wsize=windowsize)
    docs = du.load_dataset('data/ner/dev')
    X_dev, y_dev = du.docs_to_windows(docs,word_to_num,tag_to_num,
                                     wsize=windowsize)
    docs = du.load_dataset('data/ner/test.masked')
    X_test,y_test = du.docs_to_windows(docs,word_to_num,tag_to_num,
                                       wsize=windowsize)


####################################################################
    # 3,100
    # 3,120
    # 5,120
    
    clf = WindowMLP(wv,windowsize=windowsize, dims=[None,100,5],
                    reg=0.001,alpha=0.01)
    #clf.grad_check(X_train[0],y_train[0])
    X_train = X_train[:100000]
    y_train = y_train[:100000]


    print X_train[:5]
    idxiter = Generator(y_train)
    trainCosts = clf.train_sgd(X_train,y_train,idxiter)
    #get_precision(clf,X_dev,y_dev)

    yp = clf.predict(X_dev)
    #full_report(y_dev,yp, tagnames)
    #eval_performance(y_dev,yp,tagnames)
    #makePlot(trainCosts)
    
    #HiddenLayerCenterWord(clf,word_to_num,num_to_word,windowsize)
    #ModelOutputCenterWord(clf,word_to_num,num_to_word,num_to_tag,windowsize)
    





