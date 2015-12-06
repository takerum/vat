import theano.tensor as T
from layer import Layer

class Softmax(Layer):

    def __init__(self,stable):
        self.stable = stable

    def forward(self,x):
        print "Layer/Softmax"
        if(self.stable):
            x -= x.max(axis=1,keepdims=True)
        e_x = T.exp(x)
        out = e_x / e_x.sum(axis=1, keepdims=True)   
        return out

def softmax(x,stable=False):
    return Softmax(stable=stable)(x)

