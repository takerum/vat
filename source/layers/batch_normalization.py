import theano
import theano.tensor as T
import numpy

from layer import LearnableLayer

class BatchNormalization(LearnableLayer):

    def __init__(self,size,moving_avg_ratio=0.9,initial_gamma=None,initial_beta=None):

        self.params = []
        self.moving_avg_ratio = theano.shared(numpy.array(moving_avg_ratio).astype(theano.config.floatX))
        self.finetune_N = theano.shared(0)

        if(initial_gamma != None):
            assert initial_gamma.shape == (size,) or initial_gamma.shape == (1,size,1)
            gamma_values = initial_gamma.reshape((1,size,1))
        else:
            gamma_values = numpy.ones(shape=(1,size,1),dtype=theano.config.floatX)
        self.gamma = theano.shared(gamma_values)
        self.params.append(self.gamma)

        if(initial_beta != None):
            assert initial_beta.shape == (size,) or initial_beta.shape == (1,size,1)
            beta_values = initial_beta.reshape((1,size,1))
        else:
            beta_values = numpy.zeros(shape=(1,size,1),dtype=theano.config.floatX)
        self.beta = theano.shared(beta_values)
        self.params.append(self.beta)

        est_var_values = numpy.ones((1,size,1),dtype=theano.config.floatX)
        est_mean_values = numpy.zeros((1,size,1),dtype=theano.config.floatX)
        self.est_var = theano.shared(est_var_values)
        self.est_mean = theano.shared(est_mean_values)

    def __call__(self,inputs,train=True,update_batch_stat=True,finetune=False):
        return self.forward(inputs,train=train,update_batch_stat=update_batch_stat,finetune=finetune)

    def forward(self,input_org,train=True,update_batch_stat=True,finetune=False):
        print "Layer/BatchNormalization"
        ldim,cdim,rdim = self._internal_shape(input_org)
        input = input_org.reshape((ldim,cdim,rdim))
        if (train):
            mean = T.mean(input, axis=(0, 2), keepdims=True )
            var = T.mean((input-mean)**2, axis=(0, 2), keepdims=True)

            if(update_batch_stat):
                finetune_N = theano.clone(self.finetune_N, share_inputs=False)
                if(finetune):
                    finetune_N.default_update = finetune_N+1
                    ratio = T.cast(1-1.0/(finetune_N+1),theano.config.floatX)
                else:
                    finetune_N.default_update = 0
                    ratio = self.moving_avg_ratio
                m = ldim*rdim
                scale = T.cast(m/(m-1.0),theano.config.floatX)
                est_mean = theano.clone(self.est_mean, share_inputs=False)
                est_var = theano.clone(self.est_var, share_inputs=False)
                est_mean.default_update = T.cast(ratio*self.est_mean + (1-ratio)*mean,theano.config.floatX)
                est_var.default_update = T.cast(ratio*self.est_var + (1-ratio)*scale*var,theano.config.floatX)
                mean += 0 * est_mean
                var += 0 * est_var
            output = self._pbc(self.gamma) * (input - self._pbc(mean)) \
                     / T.sqrt(1e-6+self._pbc(var)) + self._pbc(self.beta)

        else:
            output = self._pbc(self.gamma) * (input - self._pbc(self.est_mean)) \
                     / T.sqrt(1e-6+self._pbc(self.est_var)) + self._pbc(self.beta)

        return output.reshape(input_org.shape)

    def _pbc(self,x):
        return T.patternbroadcast(x,(True,False,True))

    def _internal_shape(self, x):
        ldim = x.shape[0]
        cdim = self.gamma.size
        rdim = x.size // (ldim * cdim)
        return ldim, cdim, rdim