import numpy
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

import mlp


class MLP_SS(mlp.MLP):
    def __init__(self, rng, input, ul_input, layer_sizes, activations,
                 epsilon, lamb, m_batch_size, m_ul_batch_size, num_power_iter=1, norm_constraint='L2'):

        self.m_ul_batch_size = m_ul_batch_size
        mlp.MLP.__init__(self,rng=rng,input=input,layer_sizes=layer_sizes,activations=activations,
                         epsilon=epsilon,lamb=lamb,m_batch_size=m_batch_size,num_power_iter=num_power_iter,norm_constraint=norm_constraint)

        self.ul_input = ul_input
        ## re-define network outputs for semi-supervised learning
        self.p_y_given_x_for_train, means, vars_ = self.forward_for_train(self.input)
        self.ul_p_y_given_x_for_train, means_ul, vars_ul = self.forward_for_train(self.ul_input)
        self.p_y_given_x = self.forward(self.input)
        self.ul_p_y_given_x = self.forward(self.ul_input)

        self.m_v_updates_during_training = self.updates_mean_and_var(means_ul, vars_ul, self.m_ul_batch_size)
        self.m_v_updates_for_finetuning = self.updates_mean_and_var(means_ul, vars_ul, self.m_ul_batch_size,finetune=True)


    def reg_vat(self):
        p = self.ul_p_y_given_x_for_train
        d = self.srng.normal(size=self.ul_input.shape, dtype=theano.config.floatX)
        for power_iter in xrange(self.num_power_iter):
            d = self.xi * self.get_normalized_vector(d)
            p_d, ms, vs = self.forward_for_train(self.ul_input + d)
            kl = -T.mean(T.sum(p * T.log(p_d), axis=1))
            Hd = T.grad(kl, wrt=d) / self.xi
            Hd = theano.gradient.disconnected_grad(Hd)
            d = Hd
        r_vadv = self.get_perturbation(d, self.epsilon)
        ul_p_y_given_x_vadv, ms, vs = self.forward_for_train(self.ul_input + r_vadv)
        p_hat = theano.gradient.disconnected_grad(p)
        return -T.mean(T.sum(p_hat * (T.log(ul_p_y_given_x_vadv)), axis=1))


