import theano.tensor as T
from layer import Layer

class ReLU(Layer):

    def forward(self,x):
        print "Layer/ReLU"
        return T.maximum(0.0, x)

def relu(x):
    return ReLU()(x)