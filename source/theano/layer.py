
class Layer(object):

    def __call__(self, input):
        return self.forward(input)

    def forward(self,input):
        raise NotImplementedError()
