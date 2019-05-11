# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

"""
Attack based on the following paper "Website Fingerprinting at Internet Scale" by Panchenko et al.
"""

import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math
import config
from Utils import Utils
import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2
import theano_dir.logistic_sgd_2 as logistic_sgd_2
import theano_dir.mlp_2 as mlp_2
import sys, os, glob, itertools, numpy, random, shutil, errno
from sklearn.preprocessing import MinMaxScaler

class CUMUL:

    @staticmethod
    def traceToInstance(trace):
        instance = {}

        if trace.getPacketCount() == 0:
            instance = {}
            instance['class'] = 'webpage' + str(trace.getId())
            return instance


        features = []

        total = []
        cum = []
        pos = []
        neg = []
        inSize = 0
        outSize = 0
        inCount = 0
        outCount = 0

        # Process trace
        for item in trace.getPackets():
        # for item in itertools.islice(instance.packets, None):
            packetsize = int(item.getLength())
            packetDirection = item.getDirection()
            # print  packetsize, packetDirection

            # incoming packets
            if  packetDirection == Packet.DOWN:
                inSize += packetsize
                inCount += 1
                # cumulated packetsizes
                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(packetsize)
                    pos.append(packetsize)
                    neg.append(0)
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + packetsize)
                    neg.append(neg[-1] + 0)

            # outgoing packets
            if packetDirection == Packet.UP:
                packetsize = packetsize * (-1)
                outSize += abs(packetsize)
                outCount += 1
                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(abs(packetsize))
                    pos.append(0)
                    neg.append(abs(packetsize))
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + 0)
                    neg.append(neg[-1] + abs(packetsize))

        # add feature
        features.append(inCount)
        features.append(outCount)
        features.append(outSize)
        features.append(inSize)

        separateClassifier = False
        featureCount = 100

        if separateClassifier:
            # cumulative in and out
            posFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount / 2), total, pos)
            negFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount / 2), total, neg)
            for el in itertools.islice(posFeatures, None):
                features.append(el)
            for el in itertools.islice(negFeatures, None):
                features.append(el)
        else:
            # cumulative in one
            cumFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount + 1), total, cum)
            for el in itertools.islice(cumFeatures, 1, None):
                features.append(el)

        for i in range(len(features)):
            instance[i+1] = features[i]
        instance['class'] = 'webpage' + str(trace.getId())

        print "instance: ",instance['class'], len(instance)
        # print instance

        return instance



    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA6([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        classifier = "svm"
        kwargs = {}
        kwargs['C'] = 2**11
        kwargs['kernel'] = 'rbf'
        kwargs['gamma'] = 2

        if config.CROSS_VALIDATION == 0:
            return wekaAPI.executeSklearn(trainingFile, testingFile, classifier,
                                          **kwargs
                                          )
        else:
            file = Utils.joinTrainingTestingFiles(trainingFile, testingFile)  # join and shuffle
            return wekaAPI.executeSklearnCrossValidationScaleWithRange(file, classifier, config.CROSS_VALIDATION, (-1,1),
                                                         **kwargs
                                                         ) # CV with normalization



