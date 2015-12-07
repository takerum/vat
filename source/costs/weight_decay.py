import numpy
import theano
import theano.tensor as T


def weight_decay(params,coeff):
    print "costs/weight_decay"
    cost = 0
    for param in params:
        cost += T.sum(param**2)
    return theano.shared(numpy.array(coeff).astype(theano.config.floatX))*cost
