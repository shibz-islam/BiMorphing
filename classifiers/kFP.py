"""
Attack based on the following paper "k-fingerprinting: a Robust Scalable Website Fingerprinting Technique" by Jamie Hayes & George Danezis
"""

import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math
import config
from Utils import Utils
import sys, os, glob, itertools, numpy, random, shutil, errno
import kFP_FeatureExtraction
from sklearn.ensemble import RandomForestClassifier

class kFP:

    @staticmethod
    def traceToInstance(trace):
        instance = kFP_FeatureExtraction.TOTAL_FEATURES(trace_data=trace, max_size=200)
        instance['class'] = 'webpage' + str(trace.getId())
        print "instance: ", instance['class'], len(instance)
        return instance


    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)

        if config.n_components_PCA != 0:
            [trainingFile, testingFile] = Utils.calcPCA2([trainingFile, testingFile])

        if config.n_components_LDA != 0:
            [trainingFile, testingFile] = Utils.calcLDA6([trainingFile, testingFile])

        if config.n_components_QDA != 0:
            [trainingFile, testingFile] = Utils.calcQDA([trainingFile, testingFile])

        classifier = "RF"
        kwargs = {}
        kwargs['n_estimators'] = 500   #number of trees/ length of the fingerprint
        kwargs['criterion'] = "gini"
        kwargs['oob_score'] = True
        kwargs['n_jobs'] = 3

        if config.NUM_MONITORED_SITES == -1 and config.NUM_NON_MONITORED_SITES == -1:
            # closed world
            if config.CROSS_VALIDATION == 0:
                return wekaAPI.executeSklearn(trainingFile, testingFile, classifier, **kwargs)
            else:
                file = Utils.joinTrainingTestingFiles(trainingFile, testingFile)  # join and shuffle
                return wekaAPI.executeSklearnCrossValidation(file, classifier, config.CROSS_VALIDATION, **kwargs) # CV with normalization
        else:
            # open world
            trainList = wekaAPI.readFile(trainingFile)
            testList = wekaAPI.readFile(testingFile)
            trainInstancesList = []
            testInstancesList = []
            classes = ""
            yTrain = []
            yTest = []

            for line in trainList:
                if line[0] == '@':
                    if line.lower().startswith("@attribute class"):
                        classes = line.split(" ")[2]
                else:
                    # instancesList.append(float(line.split(",")[:-1]))
                    trainInstancesList.append([float(i) for i in line.split(",")[:-1]])
                    yTrain.append(line.split(",")[-1])

            for line in testList:
                if line[0] != '@':
                    testInstancesList.append([float(i) for i in line.split(",")[:-1]])
                    yTest.append(line.split(",")[-1])

            XTr = numpy.array(trainInstancesList)
            yTr = numpy.array(yTrain)
            XTe = numpy.array(testInstancesList)
            yTe = numpy.array(yTest)

            clf = RandomForestClassifier(**kwargs)

            print('Generating Leaves...')
            training_leaves = clf.apply(XTr)
            test_leaves = clf.apply(XTe)
            training_leaves = [numpy.array(training_leaf, dtype=int) for training_leaf in training_leaves]
            test_leaves = [numpy.array(test_leaf, dtype=int) for test_leaf in test_leaves]
            true_positive = 0
            false_positive = 0
            knn = 3     # k value
            debugInfo = []

            print('Calculating Distances...')
            for test_leaf_idx in range(len(test_leaves)):
                test_leaf = test_leaves[test_leaf_idx]      #array of leaf values
                true_label = yTe[test_leaf_idx]
                dist_predicted_labels = []  # List of (distance, predicated_label) pairs

                for training_leaf_idx in range(len(training_leaves)):
                    training_leaf = training_leaves[training_leaf_idx] #array of leaf values
                    predicted_label = yTr[training_leaf_idx]

                    distance = numpy.sum(training_leaf != test_leaf) / float(training_leaf.size)
                    if distance == 1.0:
                        continue
                    dist_predicted_labels.append((distance, predicted_label)) # tuple(distance, predicted_label)

                closest_distances_labels = sorted(dist_predicted_labels)[:knn] #array of tuples (distance, predicted_label)
                # vote function
                labels = [label for _, label in closest_distances_labels]
                if len(set(labels)) == 1:
                    classified_label = labels[0]
                else:
                    classified_label =  config.binaryLabels[1] #webpageNonMon

                debugInfo.append([true_label, classified_label]) # for debug purposes

                if true_label != config.binaryLabels[1] and true_label == classified_label:
                    true_positive += 1

                if true_label == config.binaryLabels[1] and true_label != classified_label:
                    false_positive += 1

            num_unmonitored_test_instances = yTe.count(config.binaryLabels[1])
            num_monitored_test_instances = len(yTe) - num_unmonitored_test_instances

            true_positive_rate = true_positive / float(num_monitored_test_instances)
            false_positive_rate = false_positive / float(num_unmonitored_test_instances)

            print("True Positive Count = %d / %d" % (true_positive, num_monitored_test_instances))
            print("False Positive Count = %d / %d" % (false_positive, num_unmonitored_test_instances))
            print("True Positive Rate: ", true_positive_rate)
            print("False Positive Rate: ", false_positive_rate)

            result = [true_positive_rate, false_positive_rate]

            return [result, debugInfo]





