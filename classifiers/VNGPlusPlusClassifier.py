# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math
from Utils import Utils
import config

import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2
import theano_dir.logistic_sgd_2 as logistic_sgd_2
import theano_dir.mlp_2 as mlp_2

class VNGPlusPlusClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance( trace ):
        instance = {}

        # Size/Number Markers
        directionCursor = None
        dataCursor      = 0
        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                dataKey = 'S'+str(directionCursor)+'-'+str( VNGPlusPlusClassifier.roundArbitrary(dataCursor, 600) )
                if not instance.get( dataKey ):
                    instance[dataKey] = 0
                instance[dataKey] += 1

                directionCursor = packet.getDirection()
                dataCursor      = 0

            dataCursor += packet.getLength()

        if dataCursor>0:
            key = 'S'+str(directionCursor)+'-'+str( VNGPlusPlusClassifier.roundArbitrary(dataCursor, 600) )
            if not instance.get( key ):
                instance[key] = 0
            instance[key] += 1

        instance['bandwidthUp'] = trace.getBandwidth( Packet.UP )
        instance['bandwidthDown'] = trace.getBandwidth( Packet.DOWN )

        maxTime = 0
        for packet in trace.getPackets():
             if packet.getTime() > maxTime:
                 maxTime = packet.getTime()
        instance['time'] = maxTime

        instance['class'] = 'webpage'+str(trace.getId())
        return instance
    
    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        # return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )

        # deep learning
        if config.DEEP_LEARNING_METHOD != -1:
            #DLMethod = Utils.intToDL(config.DEEP_LEARNING_METHOD)
            #print 'Deep Learning Method: ' + DLMethod
            #[trainingFile, testingFile] = DLMethod.runDL([trainingFile, testingFile])
            #[trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile])
            #SdA_2.calcSdA([trainingFile, testingFile])
            #logistic_sgd_2.calcLog_sgd([trainingFile, testingFile])
            if config.DEEP_LEARNING_METHOD == 1:
                logistic_sgd_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 2:
                [trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 3:
                mlp_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 4:
                SdA_2.runDL([trainingFile, testingFile])

        if config.CROSS_VALIDATION == 0:
            return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )
        else:
            file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
            return wekaAPI.executeCrossValidation( file,
                                        "weka.classifiers.bayes.NaiveBayes",
                                        ['-x',str(config.CROSS_VALIDATION), # number of folds
                                         '-K'] )