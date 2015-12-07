import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

from adversarial_training import get_main_obj,get_normalized_vector,get_perturbation
from virtual_adversarial_training import get_kl

def virtual_adversarial_training_finite_diff(x,t,forward_func,
                         main_obj_type,
                         epsilon,
                         lamb = numpy.asarray(1.0,theano.config.floatX),
                         norm_constraint='L2',
                         num_power_iter =1,
                         unchain_y=True,
                         xi = 1e-6,
                         x_for_generating_adversarial_examples=None,
                         forward_func_for_generating_adversarial_examples=None,):
    print "costs/virtual_adversarial_training_finite_diff"
    print "### HyperParameters ###"
    print "epsilon:", str(epsilon)
    print "lambda:", str(lamb)
    print "norm_constraint:", str(norm_constraint)
    print "num_power_iter:", str(num_power_iter)
    print "unchain_y:", str(unchain_y)
    print "xi:", str(xi)
    print "#######################"
    ret = 0
    y = forward_func(x)
    ret += get_main_obj(y,t,main_obj_type)

    if(x_for_generating_adversarial_examples!=None):
        x = x_for_generating_adversarial_examples
        y = forward_func(x)
    if(forward_func_for_generating_adversarial_examples!=None):
        forward_func = forward_func_for_generating_adversarial_examples
        y = forward_func(x)
    rng = RandomStreams(seed=numpy.random.randint(1234))
    d = rng.normal(size=x.shape, dtype=theano.config.floatX)
    #power_iteration
    for power_iter in xrange(num_power_iter):
        d = xi*get_normalized_vector(d)
        y_d = forward_func(x+d)
        Hd = T.grad(get_kl(y_d,y,main_obj_type).mean(),wrt=d)/xi
        Hd = theano.gradient.disconnected_grad(Hd)
        d = Hd
    r_vadv = get_perturbation(d,epsilon,norm_constraint)
    if(unchain_y==True):
        y_hat = theano.gradient.disconnected_grad(y)
        vadv_cost = get_kl(forward_func(x+r_vadv),y_hat,main_obj_type).mean()
    else:
        vadv_cost = get_kl(forward_func(x+r_vadv),y,main_obj_type,include_ent_term=True).mean()
    ret += lamb*vadv_cost

    return ret

def LDS_finite_diff(x,
        forward_func,
        main_obj_type,
        epsilon,
        norm_constraint='L2',
        num_power_iter=1,
        xi = 1e-6):
    rng = RandomStreams(seed=numpy.random.randint(1234))
    y = forward_func(x)
    d = rng.normal(size=x.shape, dtype=theano.config.floatX)
    #power_iteration
    for power_iter in xrange(num_power_iter):
        d = xi*get_normalized_vector(d)
        y_d = forward_func(x+d)
        Hd = T.grad(get_kl(y_d,y,main_obj_type).mean(),wrt=d)/xi
        Hd = theano.gradient.disconnected_grad(Hd)
        d = Hd
    r_vadv = get_perturbation(d,epsilon,norm_constraint)
    return -get_kl(forward_func(x+r_vadv),y,main_obj_type,include_ent_term=True)

def average_LDS_finite_diff(x,
        forward_func,
        main_obj_type,
        epsilon,
        norm_constraint='L2',
        num_power_iter=1):
    return LDS_finite_diff(x=x,
               forward_func=forward_func,
               main_obj_type=main_obj_type,
               epsilon=epsilon,
               norm_constraint=norm_constraint,
               num_power_iter=num_power_iter
               ).mean()
