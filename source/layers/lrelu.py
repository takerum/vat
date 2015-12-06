import theano
import theano.tensor as T
import numpy
from layer import Layer


class LReLU(Layer):

    def __init__(self,slope):
        self.slope = theano.shared(numpy.asarray(slope,theano.config.floatX))

    def forward(self,x):
        print "Layer/LeakyReLU"
        return T.maximum(self.slope*x, x)

def lrelu(x,slope=0.1):
    return LReLU(slope)(x)