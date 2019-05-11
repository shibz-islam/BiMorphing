"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import Utils_2
import config
import math

class LeNetConvPoolLayer_3_nlp(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def run_convNet(datasets, n_visible, n_out, batch_size, n_hidden, learning_rate,  n_epochs, n_words, vector_size,
                kernsDim, poolSize, nkerns=[20, 50]):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    #layer0_input = x.reshape((batch_size, 1, 28, 28)) # *

    #imageDim = int(math.sqrt(n_visible))
    imageDimR = n_words
    imageDimC = vector_size
    layer0_input = x.reshape((batch_size, 1, imageDimR, imageDimC))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

    kern0Dims = kernsDim[0] # [2,300]
    layer0 = LeNetConvPoolLayer_3_nlp(
        rng,
        input=layer0_input,
        #image_shape=(batch_size, 1, 28, 28), # *
        image_shape=(batch_size, 1, imageDimR, imageDimC),
        #filter_shape=(nkerns[0], 1, 5, 5), # *
        filter_shape=(nkerns[0], 1, kern0Dims[0], kern0Dims[1]),
        #poolsize=(2, 2) # *
        poolsize = (poolSize[0], poolSize[1]) # (2, 1)
    )
    imageDim2R = imageDimR - kern0Dims[0] + 1 # #words - 2 + 1
    imageDim2R = imageDim2R / poolSize[0]

    imageDim2C = imageDimC - kern0Dims[1] + 1 # 300 - 300 + 1 which is one column
    imageDim2C = imageDim2C / poolSize[1] # 1 / 1 = 1 still :)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    '''
    layer1 = LeNetConvPoolLayer_3_nlp(
        rng,
        input=layer0.output,
        #image_shape=(batch_size, nkerns[0], 12, 12), # *
        image_shape=(batch_size, nkerns[0], imageDim2R, imageDim2C),
        filter_shape=(nkerns[1], nkerns[0], 5, 5), # *
        #poolsize=(2, 2) # *
        poolsize=(poolSize[0], poolSize[1])  # (2, 1)
    )
    '''

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.

    #layer2_input = layer1.output.flatten(2)
    layer2_input = layer0.output.flatten(2)

    #imageDim3 = imageDim2 - 5 + 1
    #imageDim3 = imageDim3 / 2

    imageDim3R = imageDim2R
    imageDim3C = imageDim2C

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        #n_in=nkerns[1] * 4 * 4, # *
        #n_in=nkerns[1] * imageDim3 * imageDim3,
        #n_in=nkerns[1] * imageDim3R * imageDim3C,
        n_in=nkerns[0] * imageDim3R * imageDim3C,
        #n_out=500, # *
        n_out=batch_size,
        activation=T.tanh # *
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10) # *
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=n_out)


    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    #params = layer3.params + layer2.params + layer1.params + layer0.params # *
    params = layer3.params + layer2.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


    accuracy = 100.0 - (test_score * 100)
    return [accuracy, []]

def runDL(files, vector_size):
    print("Running CNN")
    print("###################################")
    ratio_valid = 1/5.0 # ratio of training from validation set
    #[train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses] = Utils_2.getDataIntLabelsClipValuesCNN2D(files, ratio_valid) # [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64).
    #[train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses] = Utils_2.getDataIntLabelsCNN2D(files, ratio_valid)

    #vector_size = 300


    [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses, n_words] = Utils_2.getDataIntLabelsCNN2D_nlp(files,
                                                                                                         ratio_valid, vector_size)

    train_set_x, train_set_y = Utils_2.shared_dataset(train_Xy) # theano shared datasets
    valid_set_x, valid_set_y = Utils_2.shared_dataset(valid_Xy)  # theano shared datasets
    test_set_x, test_set_y = Utils_2.shared_dataset(test_Xy) # theano shared datasets
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]

    n_visible = n_dimensions
    n_hidden = int(n_dimensions * float(config.DEEP_LEARNING_PARAMETERS['n_hidden_ratio'])) #n_dimensions / 3 #n_dimensions * (3/2) #n_dimensions * 2 #n_dimensions / 3
    #hidden_layers_sizes = [n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden, n_hidden] #, n_hidden, n_hidden, n_hidden] #, n_hidden] # hidden_layers_list
    hidden_layers_sizes = config.DEEP_LEARNING_PARAMETERS['layers'].split(',')
    hidden_layers_sizes = map(float, hidden_layers_sizes)

    for i in range(len(hidden_layers_sizes)):
        hidden_layers_sizes[i] = hidden_layers_sizes[i] * n_dimensions
    hidden_layers_sizes = map(int, hidden_layers_sizes)

    #print("Number of hidden layers: " + str(len(hidden_layers_sizes)))

    n_out = numClasses

    learning_rate = float(config.DEEP_LEARNING_PARAMETERS['learning_rate']) #0.01
    L1_reg = 0.00
    L2_reg = 0.0001
    n_epochs = int(config.DEEP_LEARNING_PARAMETERS['training_epochs']) #200 #100 #500 #100 #20 #16 #1000
    batch_size = int(config.DEEP_LEARNING_PARAMETERS['batch_size']) #20
    #n_hidden=500

    print ('n_dimensions: ' + str(n_dimensions) + ', n_hidden: ' + str(n_hidden) +
           ', #layers: ' + str(hidden_layers_sizes) +
           ', learning_rate: ' + str(learning_rate) + ', n_epochs: ' + str(n_epochs) +
           ', L1_reg: ' + str(L1_reg) + ', L2_reg: ' + str(L2_reg) +
           ', batch_size' + str(batch_size))

    n_words = n_words # see above for future change
    vector_size = vector_size # 300 see above for future change
    kernsDim = {0:[2,vector_size]} # key is layer, value is a list of kernel dimentions
    poolSize = [2,1] # for all layers
    return run_convNet(datasets, n_visible, n_out, batch_size, n_hidden, learning_rate, n_epochs, n_words, vector_size,
                       kernsDim, poolSize)

#if __name__ == '__main__':
    #evaluate_lenet5()


#def experiment(state, channel):
    #evaluate_lenet5(state.learning_rate, dataset=state.dataset)