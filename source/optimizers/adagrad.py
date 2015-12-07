from optimizer import Optimizer
from collections import OrderedDict
import theano
import theano.tensor as T
import numpy

class AdaGrad(Optimizer):
    def __init__(self,cost,params,lr=0.1):
        self.lr = theano.shared(numpy.array(lr).astype(theano.config.floatX))
        super(AdaGrad,self).__init__(cost,params)

    def _updates(self):
        updates = OrderedDict()
        g_model_params = []
        model_adg_rates = []
        for param in self.params:
            gparam = T.grad(self.cost,wrt=param)
            g_model_params.append(gparam)
            adg_rate= theano.shared(numpy.ones(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            model_adg_rates.append(adg_rate)

        for param, gparam,adg_rate in zip(self.params, g_model_params,model_adg_rates):
            updates[adg_rate] = adg_rate + gparam*gparam
            stepped_param = param - (self.lr/T.sqrt(updates[adg_rate]))*gparam
            updates[param] = stepped_param

        return updates
