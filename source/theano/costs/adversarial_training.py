import theano
import theano.tensor as T
import numpy

from cross_entropy_loss import _cross_entropy_loss
from quadratic_loss import _quadratic_loss

def get_main_obj(y,t,nll_type):
    if(nll_type=='CE'):
        return _cross_entropy_loss(y,t)
    elif(nll_type=='QE'):
        return _quadratic_loss(y,t)
    else:
        raise NotImplementedError()

def get_normalized_vector(v):
    v = v / (1e-20 + T.max(T.abs_(v), axis=1, keepdims=True))
    v_2 = T.sum(v**2,axis=1,keepdims=True)
    return v / T.sqrt(1e-6+v_2)


def get_perturbation(dir, epsilon,norm_constraint):
        if (norm_constraint == 'max'):
            print 'perturb:max'
            return epsilon * T.sgn(dir)
        elif (norm_constraint == 'L2'):
            print 'perturb:L2'
            dir = get_normalized_vector(dir)
            dir = epsilon * dir
            return dir
        else:
            raise NotImplementedError()

def adversarial_training(x,t,forward_func,
                         main_obj_type,
                         epsilon,
                         lamb = numpy.asarray(1.0,theano.config.floatX),
                         norm_constraint='max',
                         forward_func_for_generating_adversarial_examples=None):
    print "costs/adversarial_training"
    print "### HyperParameters ###"
    print "epsilon:", str(epsilon)
    print "lambda:", str(lamb)
    print "norm_constraint:", str(norm_constraint)
    print "#######################"
    ret = 0
    nll_cost = get_main_obj(forward_func(x),t,main_obj_type)
    ret += nll_cost
    if(forward_func_for_generating_adversarial_examples!=None):
        forward_func = forward_func_for_generating_adversarial_examples
        y = forward_func(x)
        nll_cost = get_main_obj(y,t,main_obj_type)
    dL_dx = theano.gradient.disconnected_grad(T.grad(nll_cost,wrt=x))
    r_adv = get_perturbation(dL_dx,theano.shared(numpy.array(epsilon).astype(theano.config.floatX)),norm_constraint)
    adv_cost = get_main_obj(forward_func(x+r_adv),t,main_obj_type)
    ret += lamb*adv_cost

    return ret

