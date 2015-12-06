import theano.tensor as T
from layer import Layer

class TensorToMat(Layer):

    def forward(self,x):
        print "Layer/TensorToMat"
        return x.reshape((x.shape[0],x.size//x.shape[0]))

def tensor_to_mat(x):
    return TensorToMat()(x)