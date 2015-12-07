# Virtual Adversarial training (VAT) implemented with Theano
Python codes for reproducing the results in the paper "Distributional Smoothing with Virtual Adversarial Training" http://arxiv.org/abs/1507.00677v6 

## Required libraries
python 2.7, numpy 1.9, theano 0.7.0, docopt 0.6.2


## Examples for synthetic dataset
### Model's contours on synthetic datasets with different regularization methods (Fig.3,4 in our paper)
```
./vis_model_contours.sh
```
The coutour images will be saved in ` ./figure `.

## Examples for MNIST dataset

#### Download mnist.pkl
```
cd dataset
./download_mnist.sh
```

###VAT in supervised learning for MNIST dataset 
```
python train_mnist_sup.py --cost_type=VAT_finite_diff --epsilon=2.1 --layer_sizes=784-1200-600-300-150-10 --save_filename=<filename>
```
###VAT in semi-supervised learning for MNIST dataset (with 100 labeled samples)
```
python train_mnist_semisup.py --cost_type=VAT_finite_diff --epsilon=0.3 --layer_sizes=784-1200-1200-10 --save_filename=<filename>
```
After training, the trained classifer will be saved with the name of `<filename> ` in ` ./trained_model `.

You can obtain the test error of the trained classifier saved with the name of `<filename> ` by the following command:
```
python test_mnist.py --load_filename=<filename>
```
.

If you find bug or problem, please report it! 

I also implemented virtual adversarial training with Chainer(http://chainer.org/).
The codes are also available on github https://github.com/takerum/vat_chainer/.

