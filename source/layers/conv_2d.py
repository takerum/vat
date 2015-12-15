import theano
import theano.tensor as T
import numpy
import collections
from theano.tensor.nnet import conv
from layer import Layer

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)

class Conv_2D(Layer):

    def __init__(self,in_maps,out_maps,filter_size,border_mode='valid',stride=1,use_bias=True,initial_W=None,initial_b=None,use_pylearn=True):
        """
        :param in_maps:
        :param out_maps:
        :param filter_size:
        :param stride:
        :param use_bias:
        :param initial_W:
        :param initial_b:
        :return:
        """

        self.use_bias = use_bias
        self.params = []
        self.filter_size = _pair(filter_size)
        self.border_mode = border_mode
        self.stride = _pair(stride)
        self.size = numpy.asarray([out_maps,in_maps,self.filter_size[0],self.filter_size[1]])
        self.gpu = 'gpu' in theano.config.device
        self.use_pylearn = use_pylearn

        if(initial_W != None):
            assert initial_W.shape == self.size
            W_values = initial_W
        else:
            W_values = numpy.random.normal(0, numpy.sqrt(1. / (self.filter_size[0]*self.filter_size[1]*in_maps) ), size=self.size).astype(theano.config.floatX)

        self.W = theano.shared(W_values)
        self.params.append(self.W)

        if(self.use_bias == True):
            if(initial_b != None):
                assert initial_b.shape == (out_maps,)
                b_values = initial_b
            else:
                b_values = numpy.zeros((out_maps,)).astype(theano.config.floatX)
            self.b = theano.shared(b_values)
            self.params.append(self.b)

    def forward(self,input):
        print "Layer/Convolution_2d"
        if(self.gpu and self.use_pylearn):
            pad = 0
            if(self.border_mode == 'full'):
                pad = self.filter_size[0]-1
            elif(self.border_mode == 'same'):
                assert self.filter_size[0]%2==1
                pad = (self.filter_size[0]-1)/2
            conv_op = FilterActs(partial_sum=1,pad=pad,stride=self.stride[0])
            input_shuffled = input.dimshuffle(1, 2, 3, 0)   #  bc01 -> c01b
            filters_shuffled = self.W.dimshuffle(1, 2, 3, 0)   #  bc01 -> c01b
            contiguous_input = gpu_contiguous(input_shuffled)
            contiguous_filters = gpu_contiguous(filters_shuffled)
            out_shuffled = conv_op(contiguous_input, contiguous_filters)
            conv_out = out_shuffled.dimshuffle(3, 0, 1, 2)   #  c01b -> bc01
        else:
            conv_out = conv.conv2d(input, self.W,subsample=self.stride,border_mode=self.border_mode)
        if(self.use_bias):
            conv_out += self.b.dimshuffle('x', 0, 'x', 'x')
        return conv_out
