# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import wekaAPI
from Packet import Packet
import arffWriter

import config
import theano_dir.dA_2 as dA_2

class BandwidthClassifier:
    @staticmethod
    def traceToInstance( trace ):
        instance = {}
        instance['bandwidthUp'] = trace.getBandwidth( Packet.UP )
        instance['bandwidthDown'] = trace.getBandwidth( Packet.DOWN )
        instance['class'] = 'webpage'+str(trace.getId())
        return instance
    
    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        # deep learning (AE)
        if config.AE != -1:
            [trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile])
        return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )
