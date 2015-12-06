import theano.tensor as T
from source.theano import Layer

class ReLU(Layer):

    def forward(self,x):
        print "Layer/ReLU"
        return T.maximum(0.0, x)

def relu(x):
    return ReLU()(x)