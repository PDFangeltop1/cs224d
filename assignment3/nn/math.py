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
