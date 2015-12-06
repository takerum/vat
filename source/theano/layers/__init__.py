from source.theano.layers import relu,lrelu
from source.theano.layers import linear
from source.theano.layers import sigmoid
from source.theano.layers import softmax
from source.theano.layers import dropout
from source.theano.layers import batch_normalization
from source.theano.layers import conv_2d,max_pool_2d
from source.theano.layers import mat_to_tensor4
from source.theano.layers import tensor_to_mat
from source.theano.layers import global_average
from source.theano.layers import gaussian_noise

Linear = linear.Linear
Conv_2D = conv_2d.Conv_2D
BatchNormalization = batch_normalization.BatchNormalization

relu = relu.relu
lrelu = lrelu.lrelu
sigmoid = sigmoid.sigmoid
softmax = softmax.softmax
dropout = dropout.dropout
gaussian_noise = gaussian_noise.gaussian_noise
max_pool_2d = max_pool_2d.max_pool_2d
global_average = global_average.global_average

mat_to_tensor4 = mat_to_tensor4.mat_to_tensor4
tensor_to_mat = tensor_to_mat.tensor_to_mat
