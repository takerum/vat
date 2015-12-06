import theano
from theano.tensor.signal import downsample

from source.theano import Layer
import collections
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from theano.sandbox.cuda.basic_ops import gpu_contiguous

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)

class MaxPooling2D(Layer):

    def __init__(self,filter_size,stride,ignore_border):
        self.filter_size = _pair(filter_size)
        self.stride = _pair(stride)
        self.ignore_border = ignore_border
        self.gpu = theano.config.device=='gpu'

    def forward(self,input):
        print "Layer/MaxPooling2D"
        if(self.gpu):
            pool_op = MaxPool(ds=self.filter_size[0], stride=self.stride[0])
            input_shuffled = input.dimshuffle(1, 2, 3, 0)
            contiguous_input = gpu_contiguous(input_shuffled)
            pool_out = pool_op(contiguous_input).dimshuffle(3, 0, 1, 2)
        else:
            pool_out = downsample.max_pool_2d(input,self.filter_size,st=self.stride,ignore_border=self.ignore_border)
        return pool_out

def max_pool_2d(x,filter_size,stride=1,ignore_border=False):
    return MaxPooling2D(filter_size=filter_size,
                          stride=stride,
                          ignore_border=ignore_border)(x)
