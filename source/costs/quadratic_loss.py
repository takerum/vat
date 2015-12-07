import theano.tensor as T

def quadratic_loss(x,t,forward_func):
    print "costs/quadratic_loss"
    y = forward_func(x)
    return _quadratic_loss(y,t)

def _quadratic_loss(y,t):
    return T.mean(T.sum((y-t)**2,axis=1))
