from optimizer import Optimizer
from collections import OrderedDict
import theano
import theano.tensor as T
import numpy

class MomentumSGD(Optimizer):
    def __init__(self,cost,params,lr=0.1,momentum_ratio=0.9):
        self.lr = theano.shared(numpy.array(lr).astype(theano.config.floatX))
        self.ratio = theano.shared(numpy.array(momentum_ratio).astype(theano.config.floatX))
        super(MomentumSGD,self).__init__(cost,params)

    def _updates(self):
        updates = OrderedDict()
        g_model_params = []
        g_model_params_mom = []
        for param in self.params:
            gparam = T.grad(self.cost,wrt=param)
            g_model_params.append(gparam)
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            g_model_params_mom.append(gparam_mom)

        for param, gparam_mom, gparam in zip(self.params, g_model_params_mom, g_model_params):
            updates[gparam_mom] = self.ratio * gparam_mom + (1. - self.ratio) * self.lr * gparam
            updates[param] = param-updates[gparam_mom]

        return updates
