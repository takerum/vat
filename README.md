# Virtual Adversarial training implemented with Theano
Python codes for reproducing the results on the MNIST dataset in the paper "Distributional Smoothing with Virtual Adversarial Training" http://arxiv.org/abs/1507.00677 

You can run an example code in command line:
```
python train.py <filename>
```
Then, the trained classifer will be saved with the name of `<filename> `.

You can obtain test accuracy of trained classifier by the following command:
```
python test.py <filename>
```
If you find bug or problem, please report it! 

## Required libraries
python 2.7, numpy 1.9, theano 0.7.0
 

I also implemented virtual adversarial training with Chainer(http://chainer.org/).
The codes are also available on github https://github.com/takerum/vat_chainer/.


