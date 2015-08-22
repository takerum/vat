import numpy
import cPickle
import os
import theano
import gzip

def load_original_mnist_dataset():
    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    filename = '../dataset/mnist.pkl.gz'
    data_dir, data_file = os.path.split(filename)
    if (not os.path.isfile(filename)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, filename)
    print '... loading data'

    # Load the dataset
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set,valid_set,test_set)


def _shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype='int32'), borrow=True)
    return shared_x, shared_y


def load_mnist_for_test():
    dataset = load_original_mnist_dataset()

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    train_set_x = numpy.concatenate((train_set_x, valid_set_x), axis=0)
    train_set_y = numpy.concatenate((train_set_y, valid_set_y), axis=0)

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    test_set_x, test_set_y = _shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = (test_set_x,test_set_y)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x,test_set_y)]

def load_mnist_for_validation(rng, n_l, n_v=1000):
    dataset = load_original_mnist_dataset()

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]

    _train_set_x = numpy.concatenate((train_set_x, valid_set_x), axis=0)
    _train_set_y = numpy.concatenate((train_set_y, valid_set_y), axis=0)
    rand_ind = rng.permutation(_train_set_x.shape[0])
    _train_set_x = _train_set_x[rand_ind]
    _train_set_y = _train_set_y[rand_ind]

    s_c = n_l / 10.0
    train_set_x = numpy.zeros((n_l, 28 ** 2))
    train_set_y = numpy.zeros(n_l)
    for i in xrange(10):
        ind = numpy.where(_train_set_y == i)[0]
        train_set_x[i * s_c:(i + 1) * s_c, :] = _train_set_x[ind[0:s_c], :]
        train_set_y[i * s_c:(i + 1) * s_c] = _train_set_y[ind[0:s_c]]
        _train_set_x = numpy.delete(_train_set_x, ind[0:s_c], 0)
        _train_set_y = numpy.delete(_train_set_y, ind[0:s_c])

    print rand_ind
    rand_ind = rng.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[rand_ind]
    train_set_y = train_set_y[rand_ind]
    valid_set_x = _train_set_x[:n_v]
    valid_set_y = _train_set_y[:n_v]
    # ul_train_set_x = _train_set_x[n_v:]
    train_set_ul_x = numpy.concatenate((train_set_x, _train_set_x[n_v:]), axis=0)
    train_set_ul_x = train_set_ul_x[rng.permutation(train_set_ul_x.shape[0])]
    ul_train_set_y = _train_set_y[n_v:]  # dummy

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    train_set_ul_x, ul_train_set_y = _shared_dataset((train_set_ul_x, ul_train_set_y))
    valid_set_x, valid_set_y = _shared_dataset((valid_set_x, valid_set_y))

    return [(train_set_x, train_set_y, train_set_ul_x), (valid_set_x, valid_set_y)]
