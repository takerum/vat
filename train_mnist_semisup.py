"""
Usage:
  train.py [--save_filename=<str>] \
  [--num_epochs=<N>] [--batch_size==<N>] [--ul_batch_size=<N>] [--num_batch_it=<N>] \
  [--initial_learning_rate=<float>] [--learning_rate_decay=<float>] \
  [--layer_sizes=<str>] \
  [--cost_type=<str>] \
  [--dropout_rate=<float>] [--lamb=<float>] [--epsilon=<float>] [--norm_constraint=<str>] [--num_power_iter=<N>] \
  [--num_labeled_samples=<N>] [--num_validation_samples=<N>] \
  [--seed=<N>]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --save_filename=<str>                     [default: trained_model]
  --num_epochs=<N>                          num_epochs [default: 100].
  --batch_size=<N>                          batch_size [default: 100].
  --ul_batch_size=<N>                       ul_batch_size [default: 250].
  --num_batch_it=<N>                        num_batch_iteration [default: 500].
  --initial_learning_rate=<float>           initial_learning_rate [default: 0.002].
  --learning_rate_decay=<float>             learning_rate_decay [default: 0.9].
  --layer_sizes=<str>                       layer_sizes [default: 784-1200-1200-10]
  --cost_type=<str>                         cost_type [default: MLE].
  --lamb=<float>                            [default: 1.0].
  --epsilon=<float>                         [default: 2.0].
  --norm_constraint=<str>                   [default: L2].
  --num_power_iter=<N>                      [default: 1].
  --num_labeled_samples=<N>                 [default: 100].
  --num_validation_samples=<N>              [default: 1000].
  --seed=<N>                                [default: 1].
"""

from docopt import docopt
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from source import optimizers
from source import costs
from models.fnn_mnist_semisup import FNN_MNIST
from collections import OrderedDict
import load_data

import os
import errno


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def train(args):
    print args

    numpy.random.seed(int(args['--seed']))

    dataset = load_data.load_mnist_for_semi_sup(n_l=int(args['--num_labeled_samples']),
                                                n_v=int(args['--num_validation_samples']))

    x_train, t_train, ul_x_train = dataset[0]
    x_test, t_test = dataset[1]

    layer_sizes = [int(layer_size) for layer_size in args['--layer_sizes'].split('-')]
    model = FNN_MNIST(layer_sizes=layer_sizes)

    x = T.matrix()
    ul_x = T.matrix()
    t = T.ivector()

    if (args['--cost_type'] == 'MLE'):
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train)
    elif (args['--cost_type'] == 'L2'):
        cost = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params, coeff=float(args['--lamb']))
    elif (args['--cost_type'] == 'AT'):
        cost = costs.adversarial_training(x, t, model.forward_train,
                                          'CE',
                                          epsilon=float(args['--epsilon']),
                                          lamb=float(args['--lamb']),
                                          norm_constraint=args['--norm_constraint'],
                                          forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif (args['--cost_type'] == 'VAT'):
        cost = costs.virtual_adversarial_training(x, t, model.forward_train,
                                                  'CE',
                                                  epsilon=float(args['--epsilon']),
                                                  norm_constraint=args['--norm_constraint'],
                                                  num_power_iter=int(args['--num_power_iter']),
                                                  x_for_generating_adversarial_examples=ul_x,
                                                  forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif (args['--cost_type'] == 'VAT_finite_diff'):
        cost = costs.virtual_adversarial_training_finite_diff(x, t, model.forward_train,
                                                              'CE',
                                                              xi=1e-6,
                                                              epsilon=float(args['--epsilon']),
                                                              norm_constraint=args['--norm_constraint'],
                                                              num_power_iter=int(args['--num_power_iter']),
                                                              x_for_generating_adversarial_examples=ul_x,
                                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    nll = costs.cross_entropy_loss(x=x, t=t, forward_func=model.forward_test)
    error = costs.error(x=x, t=t, forward_func=model.forward_test)

    optimizer = optimizers.ADAM(cost=cost, params=model.params, alpha=float(args['--initial_learning_rate']))

    index = T.iscalar()
    ul_index = T.iscalar()
    batch_size = int(args['--batch_size'])
    ul_batch_size = int(args['--ul_batch_size'])

    f_train = theano.function(inputs=[index, ul_index], outputs=cost, updates=optimizer.updates,
                              givens={
                                  x: x_train[batch_size * index:batch_size * (index + 1)],
                                  t: t_train[batch_size * index:batch_size * (index + 1)],
                                  ul_x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]},
                              on_unused_input='warn')
    f_nll_train = theano.function(inputs=[index], outputs=nll,
                                  givens={
                                      x: x_train[batch_size * index:batch_size * (index + 1)],
                                      t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_nll_test = theano.function(inputs=[index], outputs=nll,
                                 givens={
                                     x: x_test[batch_size * index:batch_size * (index + 1)],
                                     t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_error_train = theano.function(inputs=[index], outputs=error,
                                    givens={
                                        x: x_train[batch_size * index:batch_size * (index + 1)],
                                        t: t_train[batch_size * index:batch_size * (index + 1)]})
    f_error_test = theano.function(inputs=[index], outputs=error,
                                   givens={
                                       x: x_test[batch_size * index:batch_size * (index + 1)],
                                       t: t_test[batch_size * index:batch_size * (index + 1)]})

    f_lr_decay = theano.function(inputs=[], outputs=optimizer.alpha,
                                 updates={optimizer.alpha: theano.shared(
                                     numpy.array(args['--learning_rate_decay']).astype(
                                         theano.config.floatX)) * optimizer.alpha})

    # Shuffle training set
    randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=x_train.shape[0])
    update_permutation = OrderedDict()
    update_permutation[x_train] = x_train[randix]
    update_permutation[t_train] = t_train[randix]
    f_permute_train_set = theano.function(inputs=[], outputs=x_train, updates=update_permutation)

    # Shuffle unlabeled training set
    ul_randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=ul_x_train.shape[0])
    update_ul_permutation = OrderedDict()
    update_ul_permutation[ul_x_train] = ul_x_train[ul_randix]
    f_permute_ul_train_set = theano.function(inputs=[], outputs=ul_x_train, updates=update_ul_permutation)

    statuses = {}
    statuses['nll_train'] = []
    statuses['error_train'] = []
    statuses['nll_test'] = []
    statuses['error_test'] = []

    n_train = x_train.get_value().shape[0]
    n_test = x_test.get_value().shape[0]
    n_ul_train = ul_x_train.get_value().shape[0]

    sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train / batch_size)])) * batch_size
    sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train / batch_size)]))
    sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test / batch_size)])) * batch_size
    sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test / batch_size)]))
    statuses['nll_train'].append(sum_nll_train / n_train)
    statuses['error_train'].append(sum_error_train)
    statuses['nll_test'].append(sum_nll_test / n_test)
    statuses['error_test'].append(sum_error_test)
    print "[Epoch]", str(-1)
    print  "nll_train : ", statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
        "nll_test : ", statuses['nll_test'][-1], "error_test : ", statuses['error_test'][-1]

    print "training..."

    make_sure_path_exists("./trained_model")

    l_i = 0
    ul_i = 0
    for epoch in xrange(int(args['--num_epochs'])):
        cPickle.dump((statuses, args), open('./trained_model/' + 'tmp-' + args['--save_filename'], 'wb'),
                     cPickle.HIGHEST_PROTOCOL)
        f_permute_train_set()
        f_permute_ul_train_set()
        for it in xrange(int(args['--num_batch_it'])):
            f_train(l_i, ul_i)
            l_i = 0 if l_i >= n_train / batch_size - 1 else l_i + 1
            ul_i = 0 if ul_i >= n_ul_train / ul_batch_size - 1 else ul_i + 1

        sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train / batch_size)])) * batch_size
        sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train / batch_size)]))
        sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test / batch_size)])) * batch_size
        sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test / batch_size)]))
        statuses['nll_train'].append(sum_nll_train / n_train)
        statuses['error_train'].append(sum_error_train)
        statuses['nll_test'].append(sum_nll_test / n_test)
        statuses['error_test'].append(sum_error_test)
        print "[Epoch]", str(epoch)
        print  "nll_train : ", statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
            "nll_test : ", statuses['nll_test'][-1], "error_test : ", statuses['error_test'][-1]
        f_lr_decay()
    ### finetune batch stat ###
    f_finetune = theano.function(inputs=[ul_index], outputs=model.forward_for_finetuning_batch_stat(x),
                                 givens={x: ul_x_train[ul_batch_size * ul_index:ul_batch_size * (ul_index + 1)]})
    [f_finetune(i) for i in xrange(n_ul_train / ul_batch_size)]

    sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train / batch_size)])) * batch_size
    sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train / batch_size)]))
    sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test / batch_size)])) * batch_size
    sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test / batch_size)]))
    statuses['nll_train'].append(sum_nll_train / n_train)
    statuses['error_train'].append(sum_error_train)
    statuses['nll_test'].append(sum_nll_test / n_test)
    statuses['error_test'].append(sum_error_test)
    print "[after finetuning]"
    print  "nll_train : ", statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
        "nll_test : ", statuses['nll_test'][-1], "error_test : ", statuses['error_test'][-1]

    ###########################

    make_sure_path_exists("./trained_model")
    cPickle.dump((model, statuses, args), open('./trained_model/' + args['--save_filename'], 'wb'),
                 cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = docopt(__doc__)
    train(args)
