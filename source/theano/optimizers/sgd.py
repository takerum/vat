from optimizer import Optimizer
from collections import OrderedDict
import theano
import theano.tensor as T
import numpy

class SGD(Optimizer):
    def __init__(self,cost,params,lr=0.1):
        self.lr = theano.shared(numpy.array(lr).astype(theano.config.floatX))
        super(SGD,self).__init__(cost,params)

    def _updates(self):
        updates = OrderedDict()
        for param in self.params:
            gparam = T.grad(self.cost, wrt=param)
            updates[param] = param - self.lr*gparam
        return updates
