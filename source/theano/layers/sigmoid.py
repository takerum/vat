import theano.tensor as T
from layer import Layer

class Sigmoid(Layer):

    def forward(self,x):
        print "Layer/Sigmoid"
        return T.nnet.sigmoid(x)

def sigmoid(x):
    return Sigmoid()(x)