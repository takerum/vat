import numpy
import cPickle
import sys
from collections import OrderedDict

import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from load_data import load_mnist_for_test, load_mnist_for_validation

import mlp
import mlp_ss


def ADAM(classifier, cost, lr, updates):
    t = theano.shared(numpy.int(1))
    alpha = lr
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1.0 * 10 ** -8.0
    lam = 1.0 - 1.0 * 10 ** -8.0
    g_model_params = []
    models_m = []
    models_v = []
    for param in classifier.params:
        gparam = T.grad(cost, wrt=param)
        g_model_params.append(gparam)
        m = theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
        v = theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX))
        models_m.append(m)
        models_v.append(v)
    for param, gparam, m, v in zip(classifier.params, g_model_params, models_m, models_v):
        beta_1_t = T.cast(beta_1 * lam ** (t - 1), theano.config.floatX)
        updates[m] = beta_1_t * m + (1.0 - beta_1_t) * gparam
        updates[v] = beta_2 * v + (1 - beta_2) * (gparam * gparam)
        m_hat = updates[m] / (1.0 - T.cast(beta_1 ** t, theano.config.floatX))
        v_hat = updates[v] / (1.0 - T.cast(beta_2 ** t, theano.config.floatX))
        updates[param] = param - alpha * m_hat / (T.sqrt(v_hat) + epsilon)
    updates[t] = t + 1
    return updates


def train_mlp(
        n_l, # Number of labeled samples.
        layer_sizes, # Layer sizes of neural network. For example, layer_sizes = [784,1200,1200,10] indicates 784 input nodes and 2 hidden layers and 1200 hidden nodes and 10 output nodes.
        activations, # Specification of activation functions.
        initial_model_learning_rate, # Initial learning rate of ADAM.
        learning_rate_decay, # Learning rate decay of ADAM.
        n_epochs, # Number of training epochs,
        n_it_batches, # Number of parameter update of mini-batch stochastic gradient in each epoch.
        m_batch_size=100, # Number of mini-batch size.
        m_ul_batch_size=250, # Number of mini-batch size for calculation of LDS (semi-supervised learning only)
        cost_type='vat', # Cost type. 'mle' is no regularization, 'at' is Adversarial training, 'vat' is Virtual Adversarial training (ours)
        lamb=1.0, # Balance parameter.
        epsilon=0.05, # Norm constraint parameter.
        num_power_iter=1, # Number of iterations of power method.
        norm_constraint='L2', # Specification of norm constraint. 'max' is [L-infinity norm] and 'L2' is [L2 norm].
        random_seed=1, # Random seed.
        semi_supervised=False, # Experiment on semi-supervised learning or not.
        n_v=10000, # Number of validation samples.
        full_train=False, # Training with all of training samples ( and evaluation on test samples )
        monitoring_cost_during_training=False # Monitoring transitions of cost during training.
):



    sys.setrecursionlimit(10000)

    # set random stream
    rng = numpy.random.RandomState(random_seed)

    # load mnist dataset
    if (full_train and (not semi_supervised)):
        dataset = load_mnist_for_test()
        train_set_x, train_set_y = dataset[0]
    else:
        dataset = load_mnist_for_validation(n_l=n_l, n_v=n_v, rng=rng)
        train_set_x, train_set_y, ul_train_set_x = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    n_train_batches = numpy.ceil((train_set_x.get_value(borrow=True).shape[0]) / numpy.float(m_batch_size))
    n_valid_batches = numpy.ceil((valid_set_x.get_value(borrow=True).shape[0]) / numpy.float(m_batch_size))

    print '... building the model'
    # define a classifier
    x = T.matrix('x')
    y = T.ivector('y')
    if (semi_supervised):
        n_ul_train_batches = numpy.ceil((ul_train_set_x.get_value(borrow=True).shape[0]) / numpy.float(m_ul_batch_size))
        ul_x = T.matrix('ul_x')
        classifier = mlp_ss.MLP_SS(rng=rng, input=x, ul_input=ul_x, layer_sizes=layer_sizes, activations=activations,
                                   epsilon=epsilon, lamb=lamb,
                                   m_batch_size=m_batch_size, m_ul_batch_size=m_ul_batch_size,
                                   num_power_iter=num_power_iter, norm_constraint=norm_constraint)
    else:
        classifier = mlp.MLP(rng=rng, input=x, layer_sizes=layer_sizes, activations=activations, epsilon=epsilon,
                             lamb=lamb,
                             m_batch_size=m_batch_size, num_power_iter=num_power_iter, norm_constraint=norm_constraint)

    # define a training_cost
    if (cost_type == 'mle'):
        cost = classifier.cost(y)
    elif (cost_type == 'vat'):
        cost = classifier.cost_vat(y)
    elif (cost_type == 'at'):
        cost = classifier.cost_at(y)
    else:
        raise ValueError('cost_type:' + cost_type + ' is not defined')

    # define a schedule of learning rate
    model_learning_rate = theano.shared(numpy.asarray(initial_model_learning_rate, dtype=theano.config.floatX))
    decay_model_learning_rate = theano.function(inputs=[],
                                                outputs=model_learning_rate,
                                                updates={
                                                    model_learning_rate: model_learning_rate * learning_rate_decay})
    updates = OrderedDict()
    updates = ADAM(classifier, cost, model_learning_rate, updates)
    updates.update(classifier.m_v_updates_during_training)

    # define permutation of train set
    def update_train_ind(x, y, ind):
        upd = OrderedDict()
        upd[x] = x[ind]
        if (y != None):
            upd[y] = y[ind]
        return upd, upd[x][0]
    ind = T.ivector()
    upd_tr_ind, n_x_0 = update_train_ind(train_set_x, train_set_y, ind)
    permute_train_set = theano.function(inputs=[ind], outputs=n_x_0, updates=upd_tr_ind)

    # compile optimization function
    index = T.lscalar()
    if (semi_supervised):
        upd_ul_tr_ind, n_ul_x_0 = update_train_ind(ul_train_set_x, None, ind)
        permute_ul_train_set = theano.function(inputs=[ind], outputs=n_ul_x_0, updates=upd_ul_tr_ind)
        ul_index = T.lscalar()
        optimize = theano.function(inputs=[index, ul_index], outputs=cost,
                                   updates=updates,
                                   givens={
                                       x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                       y: train_set_y[m_batch_size * index:m_batch_size * (index + 1)],
                                       ul_x: ul_train_set_x[m_ul_batch_size * ul_index:m_ul_batch_size * (ul_index + 1)]},
                                   on_unused_input='warn'
                                   )
    else:
        optimize = theano.function(inputs=[index], outputs=cost,
                                   updates=updates,
                                   givens={
                                       x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                       y: train_set_y[m_batch_size * index:m_batch_size * (index + 1)]},
                                   on_unused_input='warn'
                                   )

    # compile functions for monitoring error and cost
    training_error = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                         y: train_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                     )
    validation_error = theano.function(inputs=[index],
                                       outputs=classifier.errors(y),
                                       givens={
                                           x: valid_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                           y: valid_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                       )
    train_nll = theano.function(inputs=[index],
                                outputs=classifier.neg_log_likelihood(y),
                                givens={
                                    x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                    y: train_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                )
    valid_nll = theano.function(inputs=[index],
                                outputs=classifier.neg_log_likelihood(y),
                                givens={
                                    x: valid_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                    y: valid_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                )
    num_power_iter_for_evalueation_LDS = 10   # num power iter for evaluation LDS
    train_LDS = theano.function(inputs=[index],
                                outputs=classifier.LDS(num_power_iter=num_power_iter_for_evalueation_LDS),
                                givens={
                                    x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)]}
                                )
    valid_LDS = theano.function(inputs=[index],
                                outputs=classifier.LDS(num_power_iter=num_power_iter_for_evalueation_LDS),
                                givens={
                                    x: valid_set_x[m_batch_size * index:m_batch_size * (index + 1)]}
                                )

    print '... training'
    epoch_counter = 0
    l_index = 0
    ul_index = 0

    train_errors = list()
    valid_errors = list()
    train_nlls = list()
    valid_nlls = list()
    train_LDSs = list()
    valid_LDSs = list()
    train_LDSs_std = list()
    valid_LDSs_std = list()

    def monitor_error():
        training_errors = [training_error(i) for i in xrange(numpy.int(numpy.ceil(n_train_batches)))]
        validation_errors = [validation_error(i) for i in xrange(numpy.int(numpy.ceil(n_valid_batches)))]
        this_training_errors = numpy.sum(training_errors)
        this_validation_errors = numpy.sum(validation_errors)
        train_errors.append(this_training_errors)
        valid_errors.append(this_validation_errors)
        print 'epoch:{}, train error {}, valid error {}, learning_rate={}'.format(
            epoch_counter, this_training_errors, this_validation_errors,
            model_learning_rate.get_value(borrow=True))

    def monitor_cost():
        training_LDSs = [train_LDS(i) for i in xrange(numpy.int(numpy.ceil(n_train_batches)))]
        validation_LDSs = [valid_LDS(i) for i in xrange(numpy.int(numpy.ceil(n_valid_batches)))]
        train_LDSs.append(numpy.mean(training_LDSs))
        valid_LDSs.append(numpy.mean(validation_LDSs))
        train_LDSs_std.append(numpy.std(training_LDSs))
        valid_LDSs_std.append(numpy.std(validation_LDSs))
        print 'epoch:' + str(epoch_counter) + ' train_LDS:' + str(train_LDSs[-1]) + ' std:' + str(
            train_LDSs_std[-1]) + ' valid_LDS:' + str(
            valid_LDSs[-1]) + ' std:' + str(valid_LDSs_std[-1])
        train_losses = [numpy.mean(train_nll(i)) for i in xrange(numpy.int(numpy.ceil(n_train_batches)))]
        train_nlls.append(numpy.mean(train_losses))
        valid_losses = [numpy.mean(valid_nll(i)) for i in xrange(numpy.int(numpy.ceil(n_valid_batches)))]
        valid_nlls.append(numpy.mean(valid_losses))
        print 'epoch:' + str(epoch_counter) + ' train neg ll:' + str(
            train_nlls[-1]) + ' valid neg ll:' + str(valid_nlls[-1])

    while epoch_counter < n_epochs:
        # monitoring error and cost in middle of training
        monitor_error()
        monitor_cost() if monitoring_cost_during_training or epoch_counter == 0 else None

        epoch_counter = epoch_counter + 1

        # parameters update
        for it in xrange(n_it_batches):
            if (semi_supervised):
                optimize(l_index, ul_index)
                ul_index = (ul_index + 1) if ((ul_index + 1) < numpy.int(n_ul_train_batches)) else 0
            else:
                optimize(l_index)
            l_index = (l_index + 1) if ((l_index + 1) < numpy.int(n_train_batches)) else 0

        # permute train set
        rand_ind = numpy.asarray(rng.permutation(train_set_x.get_value().shape[0]), dtype='int32')
        permute_train_set(rand_ind)
        if (semi_supervised):
            ul_rand_ind = numpy.asarray(rng.permutation(ul_train_set_x.get_value().shape[0]), dtype='int32')
            permute_ul_train_set(ul_rand_ind)

        decay_model_learning_rate()

    print "finished training!"

    # finetune batch mean and var for batch normalization
    print "finetuning batch mean and var for batch normalization..."
    if(semi_supervised):
        finetune_batch_mean_and_var = theano.function(inputs=[index],
                                                      outputs=classifier.finetuning_N,
                                                      updates=classifier.m_v_updates_for_finetuning,
                                                      givens={
                                                          ul_x: ul_train_set_x[m_ul_batch_size * index:m_ul_batch_size * (index + 1)],
                                                      })
        [finetune_batch_mean_and_var(i) for i in xrange(numpy.int(numpy.ceil(n_ul_train_batches)))]
    else:
        finetune_batch_mean_and_var = theano.function(inputs=[index],
                                                      outputs=classifier.finetuning_N,
                                                      updates=classifier.m_v_updates_for_finetuning,
                                                      givens={
                                                          x: train_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                                      })
        [finetune_batch_mean_and_var(i) for i in xrange(numpy.int(numpy.ceil(n_train_batches)))]
    print "final errors and costs:"
    monitor_error()
    monitor_cost()

    classifier.train_errors = train_errors
    classifier.valid_errors = valid_errors
    classifier.train_LDSs = train_LDSs
    classifier.valid_LDSs = valid_LDSs
    classifier.train_LDSs_std = train_LDSs_std
    classifier.valid_LDSs_std = valid_LDSs_std
    classifier.train_nlls = train_nlls
    classifier.valid_nlls = valid_nlls

    return classifier


if __name__ == '__main__':

    if(len(sys.argv)>1):
        save_filename = sys.argv[1]
    else:
        save_filename = 'trained_classifier.pkl'

    # supervised learning for MNIST dataset
    classifier = train_mlp(n_l=60000,layer_sizes=[28 ** 2, 1200, 600, 300, 150, 10],activations=['ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax'],
              initial_model_learning_rate=0.002,learning_rate_decay=0.9,n_epochs=100,n_it_batches=600, m_batch_size=100,
              cost_type='vat',lamb=1.0,epsilon=0.075,num_power_iter=1,norm_constraint='L2',
             random_seed=1,full_train=True,monitoring_cost_during_training=False)

    # semi-supervised learning for MNIST dataset
    #classifier = train_mlp(n_l=100, layer_sizes=[28 ** 2, 1200, 1200, 10], activations=['ReLU', 'ReLU', 'Softmax'],
    #          initial_model_learning_rate=0.002, learning_rate_decay=0.9, n_epochs=100, n_it_batches=500,
    #          m_batch_size=100, m_ul_batch_size=250,
    #          cost_type='vat', lamb=1.0, epsilon=0.01, num_power_iter=1, norm_constraint='L2',
    #          random_seed=1, semi_supervised=True, n_v=1000, full_train=False, monitoring_cost_during_training=False)

    #save trained classifier
    cPickle.dump(classifier,open(save_filename,'wb'),cPickle.HIGHEST_PROTOCOL)
