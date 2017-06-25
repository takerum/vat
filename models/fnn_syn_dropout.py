from fnn import FNN
from source import layers as L


class FNN_syn_dropout(FNN):
    def __init__(self, drate=0.5):
        self.drate = drate
        self.l1 = L.Linear(size=(100, 100))
        self.l2 = L.Linear(size=(100, 2))
        self.params = self.l1.params + self.l2.params

    def forward(self, input, train=True):
        h = input
        h = L.dropout(h, rate=self.drate, train=train)
        h = self.l1(h)
        h = L.relu(h)
        # h = L.dropout(h,rate=self.drate,train=train)
        h = self.l2(h)
        return L.softmax(h)
