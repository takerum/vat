import numpy
import cPickle
import os
import theano


def load_mnist_dataset():
    dataset = cPickle.load(open('dataset/mnist.pkl', 'rb'))
    train_set_x = numpy.concatenate((dataset[0][0], dataset[1][0]), axis=0)
    train_set_y = numpy.concatenate((dataset[0][1], dataset[1][1]), axis=0)
    return ((train_set_x, train_set_y), (dataset[2][0], dataset[2][1]))


def _shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype='int32'), borrow=True)
    return shared_x, shared_y


def load_mnist_full():
    dataset = load_mnist_dataset()

    train_set_x, train_set_y = dataset[0]
    test_set_x, test_set_y = dataset[1]

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    test_set_x, test_set_y = _shared_dataset((test_set_x, test_set_y))

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_mnist_for_validation(n_v=10000):
    dataset = load_mnist_dataset()

    train_set_x, train_set_y = dataset[0]

    randix = numpy.random.permutation(train_set_x.shape[0])

    valid_set_x = train_set_x[randix[:n_v]]
    valid_set_y = train_set_y[randix[:n_v]]
    train_set_x = train_set_x[randix[n_v:]]
    train_set_y = train_set_y[randix[n_v:]]

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    valid_set_x, valid_set_y = _shared_dataset((valid_set_x, valid_set_y))

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]


def load_mnist_for_semi_sup(n_l=1000, n_v=1000):
    dataset = load_mnist_dataset()

    _train_set_x, _train_set_y = dataset[0]

    rand_ind = numpy.random.permutation(_train_set_x.shape[0])
    _train_set_x = _train_set_x[rand_ind]
    _train_set_y = _train_set_y[rand_ind]

    s_c = int(n_l / 10.0)
    train_set_x = numpy.zeros((n_l, 28 ** 2))
    train_set_y = numpy.zeros(n_l)
    for i in xrange(10):
        ind = numpy.where(_train_set_y == i)[0]
        train_set_x[i * s_c:(i + 1) * s_c, :] = _train_set_x[ind[0:s_c], :]
        train_set_y[i * s_c:(i + 1) * s_c] = _train_set_y[ind[0:s_c]]
        _train_set_x = numpy.delete(_train_set_x, ind[0:s_c], 0)
        _train_set_y = numpy.delete(_train_set_y, ind[0:s_c])

    print rand_ind
    rand_ind = numpy.random.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[rand_ind]
    train_set_y = train_set_y[rand_ind]
    valid_set_x = _train_set_x[:n_v]
    valid_set_y = _train_set_y[:n_v]
    print valid_set_y
    # ul_train_set_x = _train_set_x[n_v:]
    train_set_ul_x = numpy.concatenate((train_set_x, _train_set_x[n_v:]), axis=0)
    train_set_ul_x = train_set_ul_x[numpy.random.permutation(train_set_ul_x.shape[0])]
    ul_train_set_y = _train_set_y[n_v:]  # dummy

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    train_set_ul_x, ul_train_set_y = _shared_dataset((train_set_ul_x, ul_train_set_y))
    valid_set_x, valid_set_y = _shared_dataset((valid_set_x, valid_set_y))

    return [(train_set_x, train_set_y, train_set_ul_x), (valid_set_x, valid_set_y)]
