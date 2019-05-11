"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


from Utils import Utils
import scipy
import random
import config

class dA_2(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784, # number of original features ( d ). * means change here
        n_hidden=500, # number of abstract features ( d'). *
        W=None,
        bhid=None, # b hidden ( to output layer which is the same dimensions as the input x )
        bvis=None # b_visible ( d' )
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """

        return self.theano_rng.normal(size=input.shape, avg=0.0, std=corruption_level, dtype=theano.config.floatX) + input

        '''
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
        '''

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0] # * send X and y here

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA_2(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28, # number of original features ( d ). * means change here
        n_hidden=500 # number of abstract features ( d'). *
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    #kld_start
    #write learned weights to a file
    #numpy.savetxt('dA_W_corrupted0.txt', da.W.get_value())
    # kld_end

    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA_2(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28, # number of original features ( d ). * means change here
        n_hidden=500 # number of abstract features ( d'). *
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The 30% corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
    # end-snippet-3

    # start-snippet-4
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
    # end-snippet-4

    #kld_start
    #write learned weights to a file
    #numpy.savetxt('dA_W_corrupted30.txt', da.W.get_value())
    # kld_end

    os.chdir('../')


#@staticmethod
def getData(files,ratio):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64).

    #trainList = Utils.readFile(files[0])
    #testList = Utils.readFile(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    train_y = []
    test_y = []

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2]
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)

    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)

    # clip
    train_X = numpy.clip(train_X, 0, 1)

    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # preprocessing, normalizing
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstnces = int(len(train_X) * ratio)
    train_Xy = (train_X[:numInstnces], train_y[:numInstnces])  # train_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)

    # clip
    test_X = numpy.clip(test_X, 0, 1)

    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, test_Xy, n_dimensions, classes]


# shuffle instances by class label
@staticmethod
def readFileShuffleInstances(fileName):
    fileLines = [line.strip() for line in open(fileName)]
    fileList = []
    instancesList = []

    for fileLine in fileLines:
        if fileLine.startswith("@"):
            fileList.append(fileLine)
        else:
            lineList = fileLine.split(",")
            instancesList.append(lineList)  # list of lists

    random.shuffle(instancesList)  # important for the training and validation sets to train a better DL model
    # random.shuffle(instancesList)

    for instance in instancesList:
        fileList.append(",".join(instance))

    return fileList

#@staticmethod
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    #print (shared_y.get_value()) # ok
    #print (T.cast(shared_y, 'int32')).get_value() # error after casting. AttributeError: 'NoneType' object has no attribute 'get_value'
    return shared_x, T.cast(shared_y, 'int32')

def run_dA(datasets, n_visible, n_hidden, corruption_level, learning_rate, training_epochs, batch_size):
            #dataset='mnist.pkl.gz',
            #batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    #datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0] # * send X and y here

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    '''
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA_2(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=n_visible, # number of original features ( d ). * means change here
        n_hidden=n_hidden # number of abstract features ( d'). *
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)


    #print(('The no corruption code for file ' +
    #       os.path.split(__file__)[1] +
    #       ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    #image = Image.fromarray(
    #    tile_raster_images(X=da.W.get_value(borrow=True).T,
    #                       img_shape=(28, 28), tile_shape=(10, 10),
    #                       tile_spacing=(1, 1)))
    #image.save('filters_corruption_0.png')

    # kld_start
    W_no_corruption = da.W.get_value()
    #write learned weights to a file
    #numpy.savetxt('dA_W_not_corrupted.txt', da.W.get_value())
    # kld_end
    '''
    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA_2(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=n_visible, # number of original features ( d ). * means change here
        n_hidden=n_hidden # number of abstract features ( d'). *
    )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    '''
    print(('The 30% corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
    # end-snippet-3

    # start-snippet-4
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
    # end-snippet-4
    '''

    #kld_start
    W_corruption = da.W.get_value()
    #write learned weights to a file
    #numpy.savetxt('dA_W_corrupted.txt', da.W.get_value())
    # kld_end

    #os.chdir('../')

    #return [W_no_corruption, W_corruption]
    return [[], W_corruption]


#@staticmethod
def constructNewArff(files, classes, new_train_X, train_y, new_test_X, test_y):
    newTrainList = []
    newTestList = []

    newTrainList.append('@RELATION sites')
    newTestList.append('@RELATION sites')

    new_dimensions = len(new_train_X[0]) # the number of hiddern layer neurons

    for i in range(0, new_dimensions):
        newTrainList.append('@ATTRIBUTE k'+ str(i+1) +' real')
        newTestList.append('@ATTRIBUTE k'+ str(i+1) +' real')

    newTrainList.append('@ATTRIBUTE class '+classes)
    newTestList.append('@ATTRIBUTE class '+classes)

    newTrainList.append('@DATA')
    newTestList.append('@DATA')

    for i in xrange(len(new_train_X)):
        newTrainList.append(','.join([str(k) for k in new_train_X[i].tolist()]) + ',webpage' + str(train_y[i]))

    for i in xrange(len(new_test_X)):
        newTestList.append(','.join([str(k) for k in new_test_X[i].tolist()]) + ',webpage' + str(test_y[i]))



    # writing the new training file (with lower dimensions)
    fnewTrainName = files[0][:-5]+'_dA.arff'
    fnewTrain = open(fnewTrainName, 'w')
    for item in newTrainList:
        fnewTrain.write(item+'\n')

    fnewTrain.close()

    # writing the new testing file (with lower dimensions)
    fnewTestName = files[1][:-5]+'_dA.arff'
    fnewTest = open(fnewTestName, 'w')
    for item in newTestList:
        fnewTest.write(item+'\n')

    fnewTest.close()

    return [fnewTrainName,fnewTestName]


#@staticmethod
#def calcAE(files):
def runDL(files):
    print('Running dA_2')
    print('###########')
    ratio = 1 # 1/5.0 # ratio of instnces from training for AE
    [train_Xy, test_Xy, n_dimensions, classes] = getData(files, ratio) # [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64).


    train_set_x, train_set_y = shared_dataset(train_Xy) # theano shared datasets
    test_set_x, test_set_y = shared_dataset(test_Xy) # theano shared datasets
    datasets = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    #print (train_set_x.get_value())
    #print (train_set_y.get_value()) # AttributeError: 'NoneType' object has no attribute 'get_value'
    n_visible = n_dimensions
    n_hidden = int(n_dimensions * float(config.DEEP_LEARNING_PARAMETERS['n_hidden_ratio'])) #n_dimensions / 3
    corruption_level = float(config.DEEP_LEARNING_PARAMETERS['corruption_level']) #0.01 #0.3 #def: 0.3

    learning_rate = float(config.DEEP_LEARNING_PARAMETERS['learning_rate']) #0.1 #0.01 #0.1
    training_epochs = int(config.DEEP_LEARNING_PARAMETERS['training_epochs']) #200 #100 #500 #100 #15
    batch_size = int(config.DEEP_LEARNING_PARAMETERS['batch_size']) #20

    print ('n_dimensions: ' + str(n_dimensions) + ', n_hidden: ' + str(n_hidden) +
           ', corruption_level: ' + str(corruption_level) +
           ', learning_rate: ' + str(learning_rate) + ', training_epochs: ' + str(training_epochs) +
           ', batch_size' + str(batch_size))

    W_noCorr_corr = run_dA(datasets, n_visible, n_hidden, corruption_level, learning_rate, training_epochs, batch_size) # list of theano shared matrices

    #W_new = W_noCorr_corr[0] # W_noCorr_corr[0] no corruption, W_noCorr_corr[1] with corruption
    W_new = W_noCorr_corr[1]  # W_noCorr_corr[0] no corruption, W_noCorr_corr[1] with corruption

    new_train_X = numpy.dot(train_Xy[0], W_new)
    #new_train_X = scipy.special.expit(new_train_X) # sigmoid

    new_test_X = numpy.dot(test_Xy[0], W_new)
    #new_test_X = scipy.special.expit(new_test_X) # sigmoid

    [trainingFile, testingFile] = constructNewArff(files, classes, new_train_X, train_Xy[1], new_test_X, test_Xy[1]) # train_Xy[1] for train y

    return [trainingFile, testingFile]


if __name__ == '__main__':
    #test_dA()
    pass