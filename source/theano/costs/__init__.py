from source.theano.costs import cross_entropy_loss
from source.theano.costs import quadratic_loss
from source.theano.costs import weight_decay
from source.theano.costs import adversarial_training
from source.theano.costs import virtual_adversarial_training as vat
from source.theano.costs import virtual_adversarial_training_finite_diff as vat_finite_diff

from source.theano.costs import error

cross_entropy_loss = cross_entropy_loss.cross_entropy_loss
quadratic_loss = quadratic_loss.quadratic_loss
weight_decay = weight_decay.weight_decay
adversarial_training = adversarial_training.adversarial_training

LDS = vat.LDS
average_LDS = vat.average_LDS
virtual_adversarial_training = vat.virtual_adversarial_training

LDS_finite_diff = vat_finite_diff.LDS_finite_diff
average_LDS_finite_diff = vat_finite_diff.average_LDS_finite_diff
virtual_adversarial_training_finite_diff = vat_finite_diff.virtual_adversarial_training_finite_diff
error = error.error