
import numpy

import theano
import theano.tensor as T

from Utils import Utils
import random
import math
import sys

def getData(files,ratio_valid):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64).

    trainList = Utils.readFile(files[0])
    testList = Utils.readFile(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
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
    train_y = numpy.array(train_y, dtype=numpy.int64)
    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)
    test_y = numpy.array(test_y, dtype=numpy.int64)
    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

def getDataIntLabels(files,ratio_valid):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])


    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])


    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)
    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing
    print ('normalizing data ...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)
    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

def getDataIntLabelsNormalize(files,ratio_valid):
    # normalize using train only
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])


    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)
    train_y = numpy.array(train_y, dtype=numpy.int64)


    # START preprocessing, normalizing
    print ('normalizing data using train data only...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing


    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)
    test_y = numpy.array(test_y, dtype=numpy.int64)


    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std


    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]


def getDataIntLabelsNormalize2(files,ratio_valid):
    # normalize using both train and test
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])


    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)
    train_y = numpy.array(train_y, dtype=numpy.int64)


    # START preprocessing, normalizing
    print ('normalizing data using both train and test data...')
    train_test_X = numpy.concatenate((train_X,test_X), axis=0)
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)

    train_test_X_mean = numpy.mean(train_test_X, 0)
    train_test_X_std = numpy.std(train_test_X, 0)

    train_X = (train_X - train_test_X_mean) / train_test_X_std # scale data
    # END preprocessing, normalizing


    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)
    test_y = numpy.array(test_y, dtype=numpy.int64)


    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_test_X_mean) / train_test_X_std # scale data
    # preprocessing, normalizing, using train data mean and std


    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

def getDataIntLabelsClipValues(files,ratio_valid):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])


    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)

    # clip
    train_X = numpy.clip(train_X, 0, 1)

    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing
    print ('normalizing data ...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)

    # clip
    test_X = numpy.clip(test_X, 0, 1)

    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

def getDataIntLabelsCNN2D(files,ratio_valid):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])

    # For CNN, find the new dimension that ensures an int square root for the 2D image
    newSqrdDim = calcNewSqrdDim(len(train_X[0]))

    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)

    # for CNN, reducing features to make it as an image with an int sqrt
    train_X = train_X[:,:newSqrdDim] # reducing the features

    # clip
    #train_X = numpy.clip(train_X, 0, 1)

    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing
    print ('normalizing data ...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)

    # for CNN, reducing features to make it as an image with an int sqrt
    test_X = test_X[:, :newSqrdDim]  # reducing the features

    # clip
    #test_X = numpy.clip(test_X, 0, 1)

    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

def getDataIntLabelsCNN2D_nlp(files, ratio_valid, vector_size):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])

    # For CNN, find the new dimension that ensures an int square root for the 2D image
    #newSqrdDim = calcNewSqrdDim(len(train_X[0]))
    min_n_words = calcMinNumWordsInTrace(train_X, test_X, vector_size)

    # for CNN with glove, as #words varies in traces
    min_n_features = min_n_words * vector_size

    #train_X = train_X[:,:min_n_features] # all instances, some columns(feastures). vector_size=300

    train_X = getNewData(train_X, min_n_features)

    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)


    # clip
    #train_X = numpy.clip(train_X, 0, 1)

    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing
    print ('normalizing data ...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # for CNN, reducing features to make it as an image with an int sqrt
    #test_X = test_X[:, :min_n_features] # all instances, some columns(feastures). vector_size=300

    test_X = getNewData(test_X, min_n_features)

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)

    # clip
    #test_X = numpy.clip(test_X, 0, 1)

    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses, min_n_words]

def getDataIntLabelsClipValuesCNN2D(files,ratio_valid):
    # returns [train_Xy, test_Xy]; train_Xy and test_Xy are tuples of (X: ndarray(float32), y: ndarray(int64)), where y is int labels starting from 0.

    #trainList = Utils.readFileShuffleInstances(files[0])
    #testList = Utils.readFileShuffleInstances(files[1])

    trainList = Utils.readFileShuffleInstancesBalanced(files[0])
    testList = Utils.readFileShuffleInstancesBalanced(files[1])

    arffHeaders = []
    train_X = []
    test_X = []
    classes = ""
    classesList = []
    numClasses = 0
    train_y = []
    test_y = []

    valid_X = []
    valid_y = []

    classesMap = {}
    classesMap = prepareClassIDs(trainList)

    for line in trainList:
        if line[0] == '@':
            arffHeaders.append(line)
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                numClasses = len(classesList)
        else:
            # instancesList.append(float(line.split(",")[:-1]))
            train_X.append([float(i) for i in line.split(",")[:-1]])
            #train_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            train_y.append(classesMap[classLabel])

    for line in testList:
        if line[0] != '@':
            test_X.append([float(i) for i in line.split(",")[:-1]])
            #test_y.append(line.split(",")[-1].split("webpage")[1]) # take the id of the webpage (int)
            classLabel = line.split(",")[-1]
            test_y.append(classesMap[classLabel])

    # For CNN, find the new dimension that ensures an int square root for the 2D image
    newSqrdDim = calcNewSqrdDim(len(train_X[0]))

    # train tuple
    train_X = numpy.array(train_X, dtype=numpy.float32)

    # for CNN, reducing features to make it as an image with an int sqrt
    train_X = train_X[:,:newSqrdDim] # reducing the features

    # clip
    train_X = numpy.clip(train_X, 0, 1)

    train_y = numpy.array(train_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing
    print ('normalizing data ...')
    train_X_mean = numpy.mean(train_X, 0)
    train_X_std = numpy.std(train_X, 0)
    train_X = (train_X - train_X_mean) / train_X_std # scale data
    # END preprocessing, normalizing
    '''

    #train_Xy = (train_X, train_y) # train_Xy is a touple
    numInstancesValid = int(len(train_X) * ratio_valid)
    numInstancesTrain = len(train_X) - numInstancesValid
    train_Xy = (train_X[:numInstancesTrain], train_y[:numInstancesTrain])  # train_Xy is a touple

    valid_Xy = (train_X[numInstancesTrain:], train_y[numInstancesTrain:])  # valid_Xy is a touple

    # test tuple
    test_X = numpy.array(test_X, dtype=numpy.float32)

    # for CNN, reducing features to make it as an image with an int sqrt
    test_X = test_X[:, :newSqrdDim]  # reducing the features

    # clip
    test_X = numpy.clip(test_X, 0, 1)

    test_y = numpy.array(test_y, dtype=numpy.int64)

    '''
    # START preprocessing, normalizing, using train data mean and std
    test_X = (test_X - train_X_mean) / train_X_std # scale data
    # preprocessing, normalizing, using train data mean and std
    '''

    test_Xy = (test_X, test_y)  # test_Xy is a touple

    n_dimensions = len(train_X[0])
    return [train_Xy, valid_Xy, test_Xy, n_dimensions, classes, numClasses]

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

def shared_dataset_svm(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    '''
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
    '''

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

    # one-hot encoded labels as {-1, 1}
    n_classes = len(numpy.unique(data_y))  # dangerous?
    y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
    y1[numpy.arange(data_y.shape[0]), data_y] = 1
    shared_y1 = theano.shared(numpy.asarray(y1,
                                         dtype=theano.config.floatX),
                              borrow=borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y1, 'int32')

def prepareClassIDs(trainList):
    """

    :param trainList:
    :return: classesMap<className, classID> where class IDs start from zero
    """
    for line in trainList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2] # {webpage1,webpage2,...}
                classesList = classes.split("{")[1].split("}")[0].split(",")
                break

    classesMap = {}
    classID = 0
    for label in classesList:
        classesMap[label] = classID
        classID += 1

    return classesMap

def calcNewSqrdDim(origDim):
    # For CNN, find the new dimension that ensures an int square root for the 2D image
    return int(math.floor(math.sqrt(origDim))**2)
    #return 784

def calcMinNumWordsInTrace(train_X, test_X, vector_size):
    minNumOfWords = sys.maxint
    for instance in train_X:
        instLen = len(instance)
        if instLen < minNumOfWords:
            minNumOfWords = instLen
    for instance in test_X:
        instLen = len(instance)
        if instLen < minNumOfWords:
            minNumOfWords = instLen

    return minNumOfWords // vector_size # example: 900 / 300 = 3 words

def getNewData(data, num_features):
    newData = []
    for instance in data:
        newData.append(instance[:num_features])

    return newData