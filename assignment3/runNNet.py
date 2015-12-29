import optparse
import cPickle as pickle
import sgd as optimizer
from rntn import RNTN
from rnn2deep import RNN2
from rnn2deep_dropout import RNN2Drop
from rnn2deep_dropout_maxout import RNN2DropMaxout
from rnn import RNN
#from dcnn import DCNN
from rnn_changed import RNN3
import tree as tr
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb


# This is the main training function of the codebase. You are intended to run this function via command line 
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here
def plot_accuracies(train_accuracies, dev_accuracies,opts):
    plt.figure(figsize=(6,4))
    plt.title(r"Learning Curve Accuracies of %s on train/dev set" % opts.model)
    plt.xlabel("SGD Iterations");plt.ylabel(r"Accuracy");
    plt.ylim(ymin=min(min(train_accuracies)*0.8, min(dev_accuracies)*0.8),ymax=max(1.2*max(train_accuracies), 1.2*max(dev_accuracies)))
    plt.plot(np.arange(opts.epochs),train_accuracies, color='b', marker='o', linestyle='-')
    plt.annotate("train curve",xy=(1,train_accuracies[1]),
                 xytext=(1,train_accuracies[1]+0.3),
                 arrowprops=dict(facecolor='green'),
                 horizontalalignment='left',verticalalignment='top')
    
    plt.plot(np.arange(opts.epochs),dev_accuracies, color='r', marker='o', linestyle='-')
    plt.annotate("dev curve",xy=(45,dev_accuracies[45]),
                 xytext=(45,dev_accuracies[45]+0.3),
                 arrowprops=dict(facecolor='red'),
                 horizontalalignment='left',verticalalignment='top')
    plt.savefig("./figures/%s/%s_learningCurve_on_train_dev_set_%d_epochs.png" % (opts.model,opts.model,opts.epochs))
    plt.show()
    plt.close()

def plot_cost(train_cost, dev_cost,opts):
    plt.figure(figsize=(6,4))
    plt.title(r"Learning Curve Cost of %s on train/dev set" % opts.model)
    plt.xlabel("SGD Iterations");plt.ylabel(r"Cost");
    plt.ylim(ymin=min(min(train_cost)*0.8, min(dev_cost)*0.8),ymax=max(1.2*max(train_cost), 1.2*max(dev_cost)))
    plt.plot(np.arange(opts.epochs),train_cost, color='b', marker='o', linestyle='-')
    plt.annotate("train curve",xy=(1,train_cost[1]),
                 xytext=(1,train_cost[1]+3),
                 arrowprops=dict(facecolor='green'),
                 horizontalalignment='left',verticalalignment='top')
    
    plt.plot(np.arange(opts.epochs),dev_cost, color='r', marker='o', linestyle='-')
    plt.annotate("dev curve",xy=(45,dev_cost[45]),
                 xytext=(45,dev_cost[45]+3),
                 arrowprops=dict(facecolor='red'),
                 horizontalalignment='left',verticalalignment='top')
    plt.savefig("./figures/%s/%s_learningCurveCost_on_train_dev_set_%d_epochs.png" % (opts.model,opts.model,opts.epochs))
    plt.show()
    plt.close()

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)


    parser.add_option("--middleDim",dest="middleDim",type="int",default=10)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    # for DCNN only
    parser.add_option("--ktop",dest="ktop",type="int",default=5)
    parser.add_option("--m1",dest="m1",type="int",default=10)
    parser.add_option("--m2",dest="m2",type="int",default=7)
    parser.add_option("--n1",dest="n1",type="int",default=6)
    parser.add_option("--n2",dest="n2",type="int",default=12)
    
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    (opts,args)=parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        test(opts.inFile,opts.data,opts.model)
        return
    
    print "Loading data..."
    train_accuracies = []
    train_cost = []
    dev_accuracies = []
    dev_cost = []
    # load training data
    trees = tr.loadTrees('train')
    opts.numWords = len(tr.loadWordMap())

    if (opts.model=='RNTN'):
        nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2Drop'):
        nn = RNN2Drop(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2DropMaxout'):
        nn = RNN2DropMaxout(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN3'):
        nn = RNN3(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='DCNN'):
        nn = DCNN(opts.wvecDim,opts.ktop,opts.m1,opts.m2, opts.n1, opts.n2,0, opts.outputDim,opts.numWords, 2, opts.minibatch,rho=1e-4)
        trees = cnn.tree2matrix(trees)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
    
    nn.initParams()

    sgd = optimizer.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)


    dev_trees = tr.loadTrees("dev")
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d"%e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)
            nn.toFile(fid)
        if evaluate_accuracy_while_training:
            print "testing on training set real quick"
            costPerTrainEpoch, accPerTrainEpoch = test(opts.outFile,"train",opts.model,trees)
            train_accuracies.append(accPerTrainEpoch)
            train_cost.append(costPerTrainEpoch)
            
            print "testing on dev set real quick"
            costPerDevEpoch, accPerDevEpoch = test(opts.outFile,"dev",opts.model,dev_trees)
            dev_accuracies.append(accPerDevEpoch)
            dev_cost.append(costPerDevEpoch)

            # clear the fprop flags in trees and dev_trees
            for tree in trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            for tree in dev_trees:
                tr.leftTraverse(tree.root,nodeFn=tr.clearFprop)
            print "fprop in trees cleared"


    if evaluate_accuracy_while_training:
        pdb.set_trace()
        print train_accuracies
        print dev_accuracies
        
        with open("./param/%s/%s_train_dev_accuracies_cost.bin"%(opts.model,opts.model),'w') as fid:
            pickle.dump(train_accuracies, fid)
            pickle.dump(dev_accuracies, fid)
            pickle.dump(train_cost, fid)
            pickle.dump(dev_cost, fid)
        
        # TODO:
        # Plot train/dev_accuracies here?            
        plot_accuracies(train_accuracies, dev_accuracies,opts)
        plot_cost(train_cost, dev_cost,opts)


def test(netFile,dataSet, model='RNN', trees=None):
    if trees==None:
        trees = tr.loadTrees(dataSet)
    assert netFile is not None, "Must give model to test"
    print "Testing netFile %s"%netFile
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        
        if (model=='RNTN'):
            nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN'):
            nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN2'):
            nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN2Drop'):
            nn = RNN2Drop(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='RNN2DropMaxout'):
            nn = RNN2DropMaxout(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(opts.model=='RNN3'):
            nn = RNN3(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
        elif(model=='DCNN'):
            nn = DCNN(opts.wvecDim,opts.ktop,opts.m1,opts.m2, opts.n1, opts.n2,0, opts.outputDim,opts.numWords, 2, opts.minibatch,rho=1e-4)
            trees = cnn.tree2matrix(trees)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
        
        nn.initParams()
        nn.fromFile(fid)

    print "Testing %s..."%model

    cost,correct, guess, total = nn.costAndGrad(trees,test=True)
    correct_sum = 0
    for i in xrange(0,len(correct)):        
        correct_sum+=(guess[i]==correct[i])
    
    # TODO
    # Plot the confusion matrix?

    confuse_matrix = np.zeros((5,5))
                              
    for i in range(len(correct)):
        confuse_matrix[correct[i]][guess[i]] += 1
    print "Cost %f, Acc %f"%(cost,correct_sum/float(total))
    makeconf(confuse_matrix,model)
    return cost, correct_sum/float(total)


def makeconf(conf_arr,model):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    plt.savefig("./figures/%s/%s_imageConfuseMatrix_200.png"%(model,model))
    plt.show()
    plt.close()

if __name__=='__main__':
    run()


