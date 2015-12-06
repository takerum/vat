import theano
import theano.tensor as T
from layer import Layer
import numpy

from theano.tensor.shared_randomstreams import RandomStreams

class GaussianNoise(Layer):

    def __init__(self,std):
        self.std = numpy.array(std).astype(theano.config.floatX)
        self.rng = RandomStreams(numpy.random.randint(1234))

    def forward(self,x):
        print "Layer/GaussianNoise"
        noise = self.rng.normal(std=self.std,size=x.shape)
        return x + noise

def gaussian_noise(x,std=0.3,train=True):
    if(train):
        return GaussianNoise(std)(x)
    else:
        return x
