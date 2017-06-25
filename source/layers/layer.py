class Layer(object):
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError()


class LearnableLayer(Layer):
    def __init__(self):
        self.params = None
