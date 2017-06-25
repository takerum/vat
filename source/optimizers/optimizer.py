import theano
import theano.tensor


class Optimizer(object):
    def __init__(self, cost, params):
        self.cost = cost
        self.params = params
        self.updates = self._updates()

    def _updates(self):
        raise NotImplementedError()
