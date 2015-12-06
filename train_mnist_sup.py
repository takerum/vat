
"""
Usage:
  train.py [--save_filename=<name>] \
  [--num_epochs=<n_epoch>] [--batch_size==<N>] \
  [--initial_learning_rate=<lr>] [--learning_rate_decay=<lr_decay>]\
  [--layer_sizes=<str>] \
  [--cost_type=<ctype>] \
  [--dropout_rate=<rate>] [--lamb=<lamb>] [--epsilon=<ep>][--norm_constraint=<nc>][--num_power_iter=<npi>] \
  [--monitoring_LDS] [--num_power_iter_for_monitoring_LDS=<npi>]\
  [--validation] [--num_validation_samples=<N>] \
  [--seed=<N>]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --save_filename=<name>                    [default: trained_model]
  --num_epochs=<n_ep>                       num_epochs [default: 100].
  --batch_size=<N>                          batch_size [default: 100].
  --initial_learning_rate=<lr>              initial_learning_rate [default: 0.002].
  --learning_rate_decay=<lr_decay>          learning_rate_decay [default: 0.95].
  --layer_sizes=<str>                       layer_sizes [default: 784-1000-500-500-500-500-10]
  --cost_type=<ctype>                       cost_type [default: MLE].
  --lamb=<lamb>                             [default: 1.0].
  --epsilon=<ep>                            [default: 2.1].
  --norm_constraint=<nc>                    [default: L2].
  --num_power_iter=<npi>                    [default: 1].
  --monitoring_LDS
  --num_power_iter_for_monitoring_LDS=<npi>    [default: 10].
  --validation
  --num_validation_samples=<N>              [default: 10000]
  --seed=<N>                                [default: 1]
"""

from docopt import docopt



import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from source import optimizers
from source import costs
from fnn_mnist_sup import FNN_MNIST


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

    if(args['--validation']):
        dataset = utils.load_mnist_for_validation(n_v=int(args['--num_validation_samples']))
    else:
        dataset = utils.load_mnist_for_test()
    x_train, t_train = dataset[0]
    x_test, t_test = dataset[1]


    layer_sizes = [int(layer_size) for layer_size in args['--layer_sizes'].split('-')] 
    model = FNN_MNIST(layer_sizes=layer_sizes)

    x = T.matrix()
    t = T.ivector()


    if(args['--cost_type']=='MLE'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train)
    elif(args['--cost_type']=='L2'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params,coeff=float(args['--lamb']))
    elif(args['--cost_type']=='AT'):
        cost = costs.adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              lamb=float(args['--lamb']),
                                              norm_constraint = args['--norm_constraint'],
                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    elif(args['--cost_type']=='VAT_finite_diff'):
        cost = costs.virtual_adversarial_training_finite_diff(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              norm_constraint = args['--norm_constraint'],
                                              num_power_iter = int(args['--num_power_iter']),
                                              forward_func_for_generating_adversarial_examples=model.forward_no_update_batch_stat)
    nll = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_test)
    error = costs.error(x=x,t=t,forward_func=model.forward_test)

    optimizer = optimizers.ADAM(cost=cost,params=model.params,alpha=float(args['--initial_learning_rate']))


    index = T.iscalar()
    batch_size = int(args['--batch_size'])
    f_train = theano.function(inputs=[index], outputs=cost, updates=optimizer.updates,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)],
                                  t:t_train[batch_size*index:batch_size*(index+1)]})
    f_nll_train = theano.function(inputs=[index], outputs=nll,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)],
                                  t:t_train[batch_size*index:batch_size*(index+1)]})
    f_nll_test = theano.function(inputs=[index], outputs=nll,
                              givens={
                                  x:x_test[batch_size*index:batch_size*(index+1)],
                                  t:t_test[batch_size*index:batch_size*(index+1)]})

    f_error_train = theano.function(inputs=[index], outputs=error,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)],
                                  t:t_train[batch_size*index:batch_size*(index+1)]})
    f_error_test = theano.function(inputs=[index], outputs=error,
                              givens={
                                  x:x_test[batch_size*index:batch_size*(index+1)],
                                  t:t_test[batch_size*index:batch_size*(index+1)]})
    if(args['--monitoring_LDS'] == True):
        LDS = costs.average_LDS_finite_diff(x,
                        model.forward_test,
                        main_obj_type='CE',
                        epsilon=float(args['--epsilon']),
                        norm_constraint = args['--norm_constraint'],
                        num_power_iter = int(args['--num_power_iter_for_monitoring_LDS']))
        f_LDS_train = theano.function(inputs=[index], outputs=LDS,
                              givens={
                                  x:x_train[batch_size*index:batch_size*(index+1)]})
        f_LDS_test = theano.function(inputs=[index], outputs=LDS,
                              givens={
                                  x:x_test[batch_size*index:batch_size*(index+1)]})
    f_lr_decay = theano.function(inputs=[],outputs=optimizer.alpha,
                                 updates={optimizer.alpha:theano.shared(numpy.array(args['--learning_rate_decay']).astype(theano.config.floatX))*optimizer.alpha})
    randix = RandomStreams(seed=numpy.random.randint(1234)).permutation(n=x_train.shape[0])
    f_permute_train_set = theano.function(inputs=[],outputs=x_train,updates={x_train:x_train[randix],t_train:t_train[randix]})

    statuses = {}
    statuses['nll_train'] = []
    statuses['error_train'] = []
    statuses['nll_test'] = []
    statuses['error_test'] = []
    if(args['--monitoring_LDS']==True):
        statuses['LDS_train'] = []
        statuses['LDS_test'] = []


    n_train = x_train.get_value().shape[0]
    n_test = x_test.get_value().shape[0]

    sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train/batch_size)]))*batch_size
    sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train/batch_size)]))
    sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test/batch_size)]))*batch_size
    sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test/batch_size)]))
    statuses['nll_train'].append(sum_nll_train/n_train)
    statuses['error_train'].append(sum_error_train)
    statuses['nll_test'].append(sum_nll_test/n_test)
    statuses['error_test'].append(sum_error_test)
    print "[Epoch]",str(-1)
    print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
            "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]
    if(args['--monitoring_LDS']):
        sum_LDS_train = numpy.sum(numpy.array([f_LDS_train(i) for i in xrange(n_train/batch_size)]))*batch_size
        sum_LDS_test = numpy.sum(numpy.array([f_LDS_test(i) for i in xrange(n_test/batch_size)]))*batch_size
        statuses['LDS_train'].append(sum_LDS_train/n_train)
        statuses['LDS_test'].append(sum_LDS_test/n_test)
        print "LDS_train : ", statuses['LDS_train'][-1], "LDS_test : " , statuses['LDS_test'][-1]

    print "training..."

    make_sure_path_exists("./trained_model")

    for epoch in xrange(int(args['--num_epochs'])):
        cPickle.dump((statuses,args),open('./trained_model/'+'tmp-' + args['--save_filename'],'wb'),cPickle.HIGHEST_PROTOCOL)
        
        f_permute_train_set()

        ### update parameters ###
        [f_train(i) for i in xrange(n_train/batch_size)]
        #########################

        sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train/batch_size)]))*batch_size
        sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train/batch_size)]))
        sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test/batch_size)]))*batch_size
        sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test/batch_size)]))
        statuses['nll_train'].append(sum_nll_train/n_train)
        statuses['error_train'].append(sum_error_train)
        statuses['nll_test'].append(sum_nll_test/n_test)
        statuses['error_test'].append(sum_error_test)
        print "[Epoch]",str(epoch)
        print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
                "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]
        if(args['--monitoring_LDS']):
            sum_LDS_train = numpy.sum(numpy.array([f_LDS_train(i) for i in xrange(n_train/batch_size)]))*batch_size
            sum_LDS_test = numpy.sum(numpy.array([f_LDS_test(i) for i in xrange(n_test/batch_size)]))*batch_size
            statuses['LDS_train'].append(sum_LDS_train/n_train)
            statuses['LDS_test'].append(sum_LDS_test/n_test)
            print "LDS_train : ", statuses['LDS_train'][-1], "LDS_test : " , statuses['LDS_test'][-1]

        f_lr_decay()

    ### finetune batch stat ###
    f_finetune = theano.function(inputs=[index],outputs=model.forward_for_finetuning_batch_stat(x),
                                 givens={x:x_train[batch_size*index:batch_size*(index+1)]})
    [f_finetune(i) for i in xrange(n_train/batch_size)]

    sum_nll_train = numpy.sum(numpy.array([f_nll_train(i) for i in xrange(n_train/batch_size)]))*batch_size
    sum_error_train = numpy.sum(numpy.array([f_error_train(i) for i in xrange(n_train/batch_size)]))
    sum_nll_test = numpy.sum(numpy.array([f_nll_test(i) for i in xrange(n_test/batch_size)]))*batch_size
    sum_error_test = numpy.sum(numpy.array([f_error_test(i) for i in xrange(n_test/batch_size)]))
    statuses['nll_train'].append(sum_nll_train/n_train)
    statuses['error_train'].append(sum_error_train)
    statuses['nll_test'].append(sum_nll_test/n_test)
    statuses['error_test'].append(sum_error_test)
    print "[after finetuning]"
    print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
        "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]
    if(args['--monitoring_LDS']):
        sum_LDS_train = numpy.sum(numpy.array([f_LDS_train(i) for i in xrange(n_train/batch_size)]))*batch_size
        sum_LDS_test = numpy.sum(numpy.array([f_LDS_test(i) for i in xrange(n_test/batch_size)]))*batch_size
        statuses['LDS_train'].append(sum_LDS_train/n_train)
        statuses['LDS_test'].append(sum_LDS_test/n_test)
        print "LDS_train : ", statuses['LDS_train'][-1], "LDS_test : " , statuses['LDS_test'][-1]
    ###########################

    make_sure_path_exists("./trained_model")
    cPickle.dump((model,statuses,args),open('./trained_model/'+args['--save_filename'],'wb'),cPickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    args = docopt(__doc__)
    train(args)
