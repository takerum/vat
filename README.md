# Virtual Adversarial training (VAT) implemented with Theano
Python code for reproducing the results showed in the paper:"Distributional Smoothing with Virtual Adversarial Training" http://arxiv.org/abs/1507.00677

## Required libraries
python 2.7, numpy 1.9, theano 0.7.0, docopt 0.6.2


## Examples on synthetic dataset
### Model's contours on synthetic datasets with different regularization methods (Fig.3,4 in our paper)
```
./vis_model_contours.sh
```
The coutour images will be saved in ` ./figure `.

## Examples on MNIST dataset

#### Download mnist.pkl
```
cd dataset
./download_mnist.sh
```

###VAT for supervised learning on MNIST dataset 
```
python train_mnist_sup.py --cost_type=VAT_finite_diff --epsilon=2.1 --layer_sizes=784-1200-600-300-150-10 --save_filename=<filename>
```
###VAT for semi-supervised learning on MNIST dataset (with 100 labeled samples)
```
python train_mnist_semisup.py --cost_type=VAT_finite_diff --epsilon=0.3 --layer_sizes=784-1200-1200-10 --num_labeled_samples=100 --save_filename=<filename>
```
After finish training, the trained classifer will be saved with `<filename> ` in ` ./trained_model `.

You can obtain a test error of the trained classifier saved with `<filename> ` by the following command:
```
python test_mnist.py --load_filename=<filename>
```
.

If you find bug or problem, please report it! 

