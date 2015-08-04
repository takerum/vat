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
        nmeans = list()
        nvars = list()
        for i in xrange(self.num_layer):
            ratio_l = self.m_batch_size / numpy.float(self.m_batch_size + self.m_ul_batch_size)
            ratio_ul = self.m_ul_batch_size / numpy.float(self.m_batch_size + self.m_ul_batch_size)
            nmeans.append((ratio_l * means[i] + ratio_ul * means_ul[i]))
            nvars.append((ratio_l * vars_[i] + ratio_ul * vars_ul[i]))
        self.m_v_updates = self.updates_mean_and_var(nmeans, nvars)

    def normalize(self, input, l_ind):
        mean = T.mean(self.mean_list[l_ind], axis=0, keepdims=True)
        s_batch_size = self.m_batch_size + self.m_ul_batch_size
        var = (s_batch_size / (s_batch_size - 1)) * T.mean(self.var_list[l_ind], axis=0, keepdims=True)
        normalized_input = self.gamma_list[l_ind] * (input - mean) / T.sqrt(1e-6 + var)  + self.beta_list[l_ind]
        return normalized_input

    def LDS_cost(self):
        p = self.ul_p_y_given_x_for_train
        v = self.srng.normal(size=self.ul_input.shape, dtype=theano.config.floatX)
        for power_iter in xrange(self.num_power_iter):
            v = self.xi * self.get_normalized_vector(v)
            p_v, ms, vs = self.forward_for_train(self.ul_input + v)
            kl = -T.mean(T.sum(p * T.log(p_v), axis=1))
            Hv = T.grad(kl, wrt=v) / self.xi
            Hv = theano.gradient.disconnected_grad(Hv)
            v = Hv
        r_vadv = self.get_perturbation(v, self.epsilon)
        ul_p_y_given_x_vadv, ms, vs = self.forward_for_train(self.ul_input + r_vadv)
        p_hat = theano.gradient.disconnected_grad(p)
        return -T.mean(T.sum(p_hat * (T.log(ul_p_y_given_x_vadv)), axis=1))


