
"""
Usage:
  train.py [--dataset_filename=<name>] [--save_filename=<name>] \
  [--num_epochs=<n_epoch>] [--initial_learning_rate=<lr>] [--learning_rate_decay=<lr_decay>] [--momentum_ratio=<ratio>]\
  [--cost_type=<ctype>] \
  [--dropout_rate=<rate>] [--lamb=<lamb>][--epsilon=<ep>][--norm_constraint=<nc>][--num_power_iter=<npi>] \
  [--monitoring_LDS] [--num_power_iter_for_monitoring_LDS=<npi>]
  train.py -h | --help

Options:
  -h --help                                 Show this screen.
  --dataset_filename=<name>                 [default: syndata_1.pkl]
  --save_filename=<name>                    [default: trained_model.pkl]
  --num_epochs=<n_ep>                       num_epochs [default: 1000].
  --initial_learning_rate=<lr>              initial_learning_rate [default: 1.0].
  --learning_rate_decay=<lr_decay>          learning_rate_decay [default: 0.995].
  --momentum_ratio=<ratio>                  [default: 0.9].
  --cost_type=<ctype>                       cost_type [default: MLE].
  --dropout_rate=<rate>                     [default: 0.0].
  --lamb=<lamb>                             [default: 1.0].
  --epsilon=<ep>                            [default: 0.5].
  --norm_constraint=<nc>                    [default: L2].
  --num_power_iter=<npi>                    [default: 1].
  --monitoring_LDS
  --num_power_iter_for_monitoring_LDS=<npi>    [default: 5].
"""

from docopt import docopt


import numpy
import theano
import theano.tensor as T
import cPickle

from source import optimizers
from source import costs
from models.fnn_syn import FNN_syn
from models.fnn_syn_dropout import FNN_syn_dropout

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

    numpy.random.seed(1)

    dataset = cPickle.load(open('dataset/' + args['--dataset_filename']))
    x_train = theano.shared(numpy.asarray(dataset[0][0][0],dtype=theano.config.floatX))
    t_train =  theano.shared(numpy.asarray(dataset[0][0][1],dtype='int32'))
    x_test =  theano.shared(numpy.asarray(dataset[0][1][0],dtype=theano.config.floatX))
    t_test =  theano.shared(numpy.asarray(dataset[0][1][1],dtype='int32'))

    if(args['--cost_type']=='dropout'):
        model = FNN_syn_dropout(drate=float(args['--dropout_rate']))
    else:
        model = FNN_syn()
    x = T.matrix()
    t = T.ivector()

    if(args['--cost_type']=='MLE' or args['--cost_type']=='dropout'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train)
    elif(args['--cost_type']=='L2'):
        cost = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_train) \
               + costs.weight_decay(params=model.params,coeff=float(args['--lamb']))
    elif(args['--cost_type']=='AT'):
        cost = costs.adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              lamb=float(args['--lamb']),
                                              norm_constraint = args['--norm_constraint'])
    elif(args['--cost_type']=='VAT'):
        cost = costs.virtual_adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              norm_constraint = args['--norm_constraint'],
                                              num_power_iter = int(args['--num_power_iter']))
    elif(args['--cost_type']=='VAT_finite_diff'):
        cost = costs.virtual_adversarial_training(x,t,model.forward_train,
                                              'CE',
                                              epsilon=float(args['--epsilon']),
                                              norm_constraint = args['--norm_constraint'],
                                              num_power_iter = int(args['--num_power_iter']))
    nll = costs.cross_entropy_loss(x=x,t=t,forward_func=model.forward_test)
    error = costs.error(x=x,t=t,forward_func=model.forward_test)

    optimizer = optimizers.MomentumSGD(cost=cost,params=model.params,lr=float(args['--initial_learning_rate']),
                                       momentum_ratio=float(args['--momentum_ratio']))

    f_train = theano.function(inputs=[], outputs=cost, updates=optimizer.updates,
                              givens={
                                  x:x_train,
                                  t:t_train})
    f_nll_train = theano.function(inputs=[], outputs=nll,
                              givens={
                                  x:x_train,
                                  t:t_train})
    f_nll_test = theano.function(inputs=[], outputs=nll,
                              givens={
                                  x:x_test,
                                  t:t_test})

    f_error_train = theano.function(inputs=[], outputs=error,
                              givens={
                                  x:x_train,
                                  t:t_train})
    f_error_test = theano.function(inputs=[], outputs=error,
                              givens={
                                  x:x_test,
                                  t:t_test})
    if(args['--monitoring_LDS']):
        LDS = costs.average_LDS_finite_diff(x,
                        model.forward_test,
                        main_obj_type='CE',
                        epsilon=float(args['--epsilon']),
                        norm_constraint = args['--norm_constraint'],
                        num_power_iter = int(args['--num_power_iter_for_monitoring_LDS']))
        f_LDS_train = theano.function(inputs=[], outputs=LDS,
                              givens={
                                  x:x_train})
        f_LDS_test = theano.function(inputs=[], outputs=LDS,
                              givens={
                                  x:x_test})
    f_lr_decay = theano.function(inputs=[],outputs=optimizer.lr,
                                 updates={optimizer.lr:theano.shared(numpy.array(args['--learning_rate_decay']).astype(theano.config.floatX))*optimizer.lr})


    statuses = {}
    statuses['nll_train'] = []
    statuses['error_train'] = []
    statuses['nll_test'] = []
    statuses['error_test'] = []
    if(args['--monitoring_LDS']==True):
        statuses['LDS_train'] = []
        statuses['LDS_test'] = []

    statuses['nll_train'].append(f_nll_train())
    statuses['error_train'].append(f_error_train())
    statuses['nll_test'].append(f_nll_test())
    statuses['error_test'].append(f_error_test())
    print "[Epoch]",str(0)
    print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
            "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]
    if(args['--monitoring_LDS']):
        statuses['LDS_train'].append(f_LDS_train())
        statuses['LDS_test'].append(f_LDS_test())
        print "LDS_train : ", statuses['LDS_train'][-1], "LDS_test : " , statuses['LDS_test'][-1]

    print "training..."

    for epoch in xrange(int(args['--num_epochs'])):
        train_cost = f_train()
        if((epoch+1)%20==0):
            statuses['nll_train'].append(f_nll_train())
            statuses['error_train'].append(f_error_train())
            statuses['nll_test'].append(f_nll_test())
            statuses['error_test'].append(f_error_test())
            print "[Epoch]",str(epoch)
            print  "nll_train : " , statuses['nll_train'][-1], "error_train : ", statuses['error_train'][-1], \
                    "nll_test : " , statuses['nll_test'][-1],  "error_test : ", statuses['error_test'][-1]
            if(args['--monitoring_LDS']):
                statuses['LDS_train'].append(f_LDS_train())
                statuses['LDS_test'].append(f_LDS_test())
                print "LDS_train : ", statuses['LDS_train'][-1], "LDS_test : " , statuses['LDS_test'][-1]

        f_lr_decay()
    make_sure_path_exists("./trained_model")
    cPickle.dump((model,statuses,args),open('./trained_model/'+args['--save_filename'],'wb'),cPickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    args = docopt(__doc__)
    train(args)
