import theano
import theano.tensor as T
from layer import Layer
import numpy

from theano.tensor.shared_randomstreams import RandomStreams

class Dropout(Layer):

    def __init__(self,rate):

        self.p = numpy.array(1-rate).astype(theano.config.floatX)
        self.rng = RandomStreams(numpy.random.randint(1234))

    def forward(self,x):
        print "Layer/Dropout"
        mask = T.cast( ( 1 / self.p ) * self.rng.binomial(n=1,p=self.p,size=x.shape), dtype=theano.config.floatX)
        return mask*x

def dropout(x,rate=0.5,train=True):
    if(train):
        return Dropout(rate)(x)
    else:
        return x
