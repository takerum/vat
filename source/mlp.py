from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

class MLP(object):

    def get_act_out(self,x, act):
        if (act == 'Softmax'):
            y = T.exp(x) / (T.exp(x).sum(axis=1, keepdims=True))
        elif (act == 'ReLU'):
            y = T.maximum(0.0, x)
        elif (act == 'Sigmoid'):
            y = T.nnet.sigmoid(x)
        elif (act == 'Tanh'):
            y = T.tanh(x)
        elif (act == 'Softplus'):
            y = T.log(1 + T.exp(x))
        else:
            y = x
        return y

    def get_normalized_vector(self,v):
        v = v / (1e-6 + T.max(T.abs_(v), axis=1, keepdims=True))
        v_2 = T.sum(v**2,axis=1,keepdims=True)
        return v / T.sqrt(1e-6 + v_2)

    def __init__(self, rng, input, layer_sizes, activations, epsilon, lamb, m_batch_size, num_power_iter=1, norm_constraint='L2'):
        self.input = input
        self.activations = activations
        self.layer_sizes = layer_sizes
        self.epsilon = epsilon
        self.lamb = lamb
        self.num_power_iter = num_power_iter
        self.norm_constraint = norm_constraint
        self.m_batch_size = m_batch_size
        self.num_layer = len(layer_sizes) - 1
        self.xi = numpy.float(1.0 * 10 ** -6)
        self.moving_ave_range = 100
        print '======hyperparameters======='
        print 'layer_sizes:' + str(layer_sizes)
        print 'activations' + str(self.activations)
        print 'epsilon:' + str(self.epsilon)
        print 'lambda:' + str(self.lamb)
        print 'num_power_iter:' + str(self.num_power_iter)
        print 'norm_constraint:' + str(norm_constraint)
        print '============================'

        self.W_list = list()
        for i in xrange(self.num_layer):
            if (i == self.num_layer - 1):
                W_values = numpy.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(0.001 * rng.standard_normal(
                    size=(self.layer_sizes[i], self.layer_sizes[i + 1])), dtype=theano.config.floatX)
            self.W_list.append(theano.shared(value=W_values, name='W'))

        # for batch normalization ( inferece with moving average )
        self.gamma_list = list()
        self.beta_list = list()
        self.mean_list = list()
        self.var_list = list()
        for i in xrange(self.num_layer):
            gamma_values = numpy.ones((self.layer_sizes[i+1],), dtype=theano.config.floatX)
            beta_values = numpy.zeros((self.layer_sizes[i+1],), dtype=theano.config.floatX)
            self.gamma_list.append(theano.shared(gamma_values))
            self.beta_list.append(theano.shared(beta_values))
            var_values = numpy.ones((self.moving_ave_range, self.layer_sizes[i+1]), dtype=theano.config.floatX)
            mean_values = numpy.zeros((self.moving_ave_range, self.layer_sizes[i+1]), dtype=theano.config.floatX)
            self.var_list.append(theano.shared(var_values))
            self.mean_list.append(theano.shared(mean_values))

        ## set random stream
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(123456))

        ## define network outputs
        self.p_y_given_x_for_train, means, vars_ = self.forward_for_train(self.input)
        self.p_y_given_x = self.forward(self.input)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.m_v_updates = self.updates_mean_and_var(means, vars_)

        self.params = self.W_list + self.gamma_list + self.beta_list

    def updates_mean_and_var(self, means, vars):
        updates = OrderedDict()
        for i in xrange(self.num_layer):
            n_mean = self.mean_list[i]
            updates[n_mean] = T.concatenate((means[i], n_mean[0:-1]), axis=0)
            n_var = self.var_list[i]
            updates[n_var] = T.concatenate((vars[i], n_var[0:-1]), axis=0)
        return updates

    def normalize_for_train(self, input, l_ind):
        mean = T.mean(input, axis=0, keepdims=True)
        var = T.mean((input - mean) ** 2, axis=0, keepdims=True)
        normalized_input = self.gamma_list[l_ind] * (input - mean) / T.sqrt(1e-6+var) + self.beta_list[l_ind]
        return normalized_input, mean, var

    def forward_for_train(self, input):
        next_input = input
        means = list()
        vars_ = list()
        for i in xrange(self.num_layer):
            next_input = T.dot(next_input, self.W_list[i])
            next_input, mean, var = self.normalize_for_train(next_input, i)
            means.append(mean)
            vars_.append(var)
            next_input = self.get_act_out(next_input, self.activations[i])
        return next_input, means, vars_

    def normalize(self, input, l_ind):
        mean = T.mean(self.mean_list[l_ind], axis=0, keepdims=True)
        var = (self.m_batch_size / (self.m_batch_size - 1)) * T.mean(self.var_list[l_ind], axis=0, keepdims=True)
        normalized_input = self.gamma_list[l_ind] * (input - mean) / T.sqrt(1e-6+var) + self.beta_list[l_ind]
        return normalized_input

    def forward(self, input):
        next_input = input
        for i in xrange(self.num_layer):
            next_input = T.dot(next_input, self.W_list[i])
            next_input = self.normalize(next_input, i)
            next_input = self.get_act_out(next_input, self.activations[i])
        return next_input

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def LDS(self,num_power_iter):
        p = self.p_y_given_x
        d = self.srng.normal(size=self.input.shape, dtype=theano.config.floatX)
        for power_iter in xrange(num_power_iter):
            d = self.xi * self.get_normalized_vector(d)
            p_d = self.forward(self.input + d)
            kl = -T.mean(T.sum(p * T.log(p_d), axis=1))
            Hd = T.grad(kl, wrt=d) / self.xi
            Hd = theano.gradient.disconnected_grad(Hd)
            d = Hd
        ##train with virtural adversarial examples##
        r_vadv = self.get_perturbation(d, self.epsilon)
        p_r_vadv = self.forward(self.input + r_vadv)
        p_hat = theano.gradient.disconnected_grad(p)
        return -T.mean(T.sum(p_hat * (T.log(p_hat) - T.log(p_r_vadv)), axis=1))

    ## define costs ##

    def cost(self, y):
        print "cost of MLE"
        sup_cost = self.ll_cost(y)
        return sup_cost

    def cost_vat(self, y):
        print "cost of virtual adversarial training"
        sup_cost = self.ll_cost(y)
        vat_cost = self.reg_vat()
        return sup_cost + self.lamb * vat_cost

    def cost_at(self, y):
        print "cost of adversarial training"
        sup_cost = self.ll_cost(y)
        at_cost = self.reg_at(y)
        return sup_cost + self.lamb * at_cost

    def ll_cost(self, y):
        return -T.mean(T.log(self.p_y_given_x_for_train)[T.arange(y.shape[0]), y])

    def reg_vat(self):
        p = self.p_y_given_x_for_train
        d = self.srng.normal(size=self.input.shape, dtype=theano.config.floatX)
        for power_iter in xrange(self.num_power_iter):
            d = self.xi * self.get_normalized_vector(d)
            p_d, ms, vs = self.forward_for_train(self.input + d)
            kl = -T.mean(T.sum(p * T.log(p_d), axis=1))
            Hd = T.grad(kl, wrt=d) / self.xi
            Hd = theano.gradient.disconnected_grad(Hd)
            d = Hd
        r_vadv = self.get_perturbation(d, self.epsilon)
        p_y_given_x_vadv, ms, vs = self.forward_for_train(self.input + r_vadv)
        p_hat = theano.gradient.disconnected_grad(p)
        return -T.mean(T.sum(p_hat * (T.log(p_y_given_x_vadv)), axis=1))

    def reg_at(self, y):
        dL_dx = theano.gradient.disconnected_grad(T.grad(self.ll_cost(y),wrt=self.input))
        r_adv = self.get_perturbation(dL_dx,self.epsilon)
        self.p_y_given_x_adv,mn,vr = self.forward_for_train(self.input+r_adv)
        return -T.mean(T.log(self.p_y_given_x_adv)[T.arange(y.shape[0]),y])

    def get_perturbation(self, dir, epsilon):
        if (self.norm_constraint == 'max'):
            print 'perturb:max'
            return epsilon * T.sgn(dir)
        if (self.norm_constraint == 'L2'):
            print 'perturb:L2'
            dir = self.get_normalized_vector(dir)
            dir = epsilon * numpy.float(numpy.sqrt(self.layer_sizes[0])) * dir
            return dir

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
