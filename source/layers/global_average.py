import theano.tensor as T
from layer import Layer


class GlobalAverage(Layer):

    def forward(self,x):
        print "Layer/GlobalAverage"
        return T.mean(T.flatten(x,outdim=3),axis=2)

def global_average(x):
    return GlobalAverage()(x)