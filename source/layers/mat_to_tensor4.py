import theano.tensor as T
from layer import Layer

class MatToTensor4(Layer):

    def __init__(self,out_maps,h,w):
        self.out_maps = out_maps
        self.h = h
        self.w = w

    def forward(self,x):
        print "Layer/MatToTensor4"
        return x.reshape((x.shape[0],self.out_maps,self.h,self.w))

def mat_to_tensor4(x,out_maps,h,w):
    return MatToTensor4(out_maps,h,w)(x)