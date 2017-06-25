from fnn import FNN
from source import layers as L


class FNN_MNIST(FNN):
    def __init__(self, layer_sizes):
        self.linear_layers = []
        self.bn_layers = []
        self.act_layers = []
        self.params = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            l = L.Linear(size=(m, n))
            bn = L.BatchNormalization(size=(n))
            self.linear_layers.append(l)
            self.bn_layers.append(bn)
            self.params += l.params + bn.params
        for i in xrange(len(self.linear_layers) - 1):
            self.act_layers.append(L.relu)
        self.act_layers.append(L.softmax)

    def forward_for_finetuning_batch_stat(self, input):
        return self.forward(input, finetune=True)

    def forward_no_update_batch_stat(self, input, train=True):
        return self.forward(input, train, False)

    def forward(self, input, train=True, update_batch_stat=True, finetune=False):
        h = input
        for l, bn, act in zip(self.linear_layers, self.bn_layers, self.act_layers):
            h = l(h)
            h = bn(h, train=train, update_batch_stat=update_batch_stat, finetune=finetune)
            h = act(h)
        return h
