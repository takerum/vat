import theano.tensor as T

def cross_entropy_loss(x,t,forward_func):
    print "costs/cross_entropy_loss"
    y = forward_func(x)
    return _cross_entropy_loss(y,t)

def _cross_entropy_loss(y,t):
    return -T.mean(T.log(y)[T.arange(t.shape[0]), t])
