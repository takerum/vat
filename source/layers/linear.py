import theano
import theano.tensor as T
import numpy

from layer import LearnableLayer

class Linear(LearnableLayer):

    def __init__(self,size,use_bias=True,initial_W=None,initial_b=None):
        self.use_bias = use_bias
        self.params = []

        if(initial_W is not None):
            assert initial_W.shape == size
            W_values = initial_W
        else:
            W_values = numpy.random.normal(0, numpy.sqrt(1. / size[0]), size=size).astype(theano.config.floatX)
        self.W = theano.shared(W_values)
        self.params.append(self.W)

        if(self.use_bias == True):
            if(initial_b is not None):
                assert initial_b.shape == size[1]
                b_values = initial_b
            else:
                b_values = numpy.zeros((size[1],)).astype(theano.config.floatX)
            self.b = theano.shared(b_values)
            self.params.append(self.b)

    def forward(self,input):
        print "Layer/Linear"
        input = self._as_mat(input)
        output = T.dot(input, self.W)
        if(self.use_bias == True):
            output += self.b
        return output

    def _as_mat(self,x):
        return x.reshape((x.shape[0],x.size//x.shape[0]))