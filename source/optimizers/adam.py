from optimizer import Optimizer
from collections import OrderedDict
import theano
import theano.tensor as T
import numpy


class ADAM(Optimizer):
    def __init__(self, cost, params, alpha=0.001):
        self.alpha = theano.shared(numpy.array(alpha).astype(theano.config.floatX))
        super(ADAM, self).__init__(cost, params)

    def _updates(self):
        updates = OrderedDict()
        t = theano.shared(numpy.array(1).astype(theano.config.floatX))
        alpha = self.alpha
        beta_1 = numpy.array(0.9).astype(theano.config.floatX)
        beta_2 = numpy.array(0.999).astype(theano.config.floatX)
        epsilon = numpy.array(1.0 * 10 ** -8.0).astype(theano.config.floatX)
        lam = numpy.array(1.0 - 1.0 * 10 ** -8.0).astype(theano.config.floatX)
        g_model_params = []
        models_m = []
        models_v = []
        for param in self.params:
            gparam = T.grad(self.cost, wrt=param)
            g_model_params.append(gparam)
            m = theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
            v = theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
            models_m.append(m)
            models_v.append(v)
        for param, gparam, m, v in zip(self.params, g_model_params, models_m, models_v):
            beta_1_t = T.cast(beta_1 * lam ** (t - 1), theano.config.floatX)
            updates[m] = T.cast(beta_1_t * m + (1 - beta_1_t) * gparam, theano.config.floatX)
            updates[v] = T.cast(beta_2 * v + (1 - beta_2) * (gparam * gparam), theano.config.floatX)
            m_hat = T.cast(updates[m] / (1 - beta_1 ** t), theano.config.floatX)
            v_hat = T.cast(updates[v] / (1 - beta_2 ** t), theano.config.floatX)
            updates[param] = param - alpha * m_hat / (T.sqrt(v_hat) + epsilon)
        updates[t] = t + 1
        return updates
