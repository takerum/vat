import numpy
import cPickle
import sys

from load_data import load_mnist_for_test
import theano
import theano.tensor as T


if __name__ == '__main__':

    if(len(sys.argv)>1):
        load_filename = sys.argv[1]
    else:
        load_filename = 'trained_classifier.pkl'

    m_batch_size = 100
    dataset = load_mnist_for_test()
    test_set_x,test_set_y = dataset[1]
    n_test_batches = numpy.ceil((test_set_x.get_value(borrow=True).shape[0]) / numpy.float(m_batch_size))

    trained_classifier = cPickle.load(open(load_filename,'rb'))

    index = T.iscalar()
    y = T.ivector()
    test_error = theano.function(inputs=[index],
                                       outputs=trained_classifier.errors(y),
                                       givens={
                                           trained_classifier.input: test_set_x[m_batch_size * index:m_batch_size * (index + 1)],
                                           y: test_set_y[m_batch_size * index:m_batch_size * (index + 1)]}
                                       )

    test_errors = [test_error(i) for i in xrange(numpy.int(numpy.ceil(n_test_batches)))]
    print "the number of misclassified examples:" + str(numpy.sum(test_errors)) + ", and test error rate(%):" + str(100*numpy.sum(test_errors)/numpy.float(test_set_x.get_value(borrow=True).shape[0]))
