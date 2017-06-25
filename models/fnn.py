import theano
import theano.tensor as T
import numpy


class FNN(object):
    def __init__(self):
        raise NotImplementedError()

    def forward_train(self, input):
        return self.forward(input, train=True)

    def forward_test(self, input):
        return self.forward(input, train=False)

    def forward(self, input, train=True):
        return input
