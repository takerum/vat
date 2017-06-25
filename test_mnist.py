"""
Usage:
  test_mnist.py [--load_filename=<name>] \
  test_mnist.py -h | --help

Options:
  -h --help                                 Show this screen.
  --load_filename=<name>                    [default: trained_model]
"""

from docopt import docopt

import numpy
import cPickle

from load_data import load_mnist_full
import theano
import theano.tensor as T

from source.costs import error

if __name__ == '__main__':
    args = docopt(__doc__)

    m_batch_size = 100
    dataset = load_mnist_full()
    test_set_x, test_set_y = dataset[1]
    n_test_batches = numpy.ceil((test_set_x.get_value(borrow=True).shape[0]) / numpy.float(m_batch_size))

    trained_model = cPickle.load(open("trained_model/" + args['--load_filename'], 'rb'))[0]

    index = T.iscalar()
    x = T.matrix()
    t = T.ivector()
    test_error = theano.function(inputs=[index],
                                 outputs=error(x=x, t=t, forward_func=trained_model.forward_test),
                                 givens={
                                     x: test_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                     t: test_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                 )

    test_errors = [test_error(i) for i in xrange(numpy.int(numpy.ceil(n_test_batches)))]
    print "the number of misclassified examples on test set:" + str(
        numpy.sum(test_errors)) + ", and test error rate(%):" + str(
        100 * numpy.sum(test_errors) / numpy.float(test_set_x.get_value(borrow=True).shape[0]))
