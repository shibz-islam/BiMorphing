# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import wekaAPI
import arffWriter

from statlib import stats

from Trace import Trace
from Packet import Packet
import math

import numpy as np
from sklearn.decomposition import PCA

import config
from Utils import Utils

import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2
import theano_dir.logistic_sgd_2 as logistic_sgd_2
import theano_dir.mlp_2 as mlp_2
import theano_dir.mlp_3 as mlp_3
import theano_dir.SdA_3 as SdA_3
import theano_dir.LeNetConvPoolLayer_2 as LeNetConvPoolLayer_2

class AdversarialClassifierTriDirectionFeaturesOnly:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def roundNumberMarker(n):
        if n==4 or n==5: return 3
        elif n==7 or n==8: return 6
        elif n==10 or n==11 or n==12 or n==13: return 9
        else: return n

    @staticmethod
    def traceToInstance( trace ):

        instance = {}

        if trace.getPacketCount()==0:
            instance = {}
            instance['class'] = 'webpage'+str(trace.getId())
            #instance['class'] = 'yes'
            '''
            if (config.DATA_SOURCE==64 or config.DATA_SOURCE==65) and trace.getId() < config.NUM_BENIGN_CLASSES:
                    instance['class'] = 'webpage0' # all benign have the same label
                    #instance['class'] = 'yes'
            else:
                instance['class'] = 'webpage'+str(trace.getId())
            '''
            return instance

        if config.GLOVE_OPTIONS['packetSize'] == 1:
            instance = trace.getHistogram()
        '''
            if config.DATA_SOURCE == 5: # WANG14 dataset
                #Unique packet lengths
                for i in range(-1500, 1501): # as in Wang's kNN paper
                    key = "0-" # uplink
                    if i < 0:
                        key = "1-" # downlink
                    key += str(i)
                    if not instance.get( key ):
                        instance[key] = 0
        '''
        # Size/Number Markers
        directionCursor = None
        dataCursor      = 0
        numberCursor    = 0
        timeCursor = 0
        burstTimeRef = 0
        timeBase = 1
        sizeBase = config.bucket_Size

        if (config.DATA_SOURCE == 5): timeBase = config.bucket_Time # works well with Wang Tor

        for packet in trace.getPackets():
            #print packet.getLength()
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:

                if config.GLOVE_OPTIONS['burstSize'] == 1:
                    #dataKey = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
                    dataKey = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(dataCursor, config.bucket_Size) )

                    if not instance.get( dataKey ):
                        instance[dataKey] = 0
                    instance[dataKey] += 1

                if config.GLOVE_OPTIONS['burstNumber'] == 1:
                    numberKey = 'N'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundNumberMarker(numberCursor) )
                    if not instance.get( numberKey ):
                        instance[numberKey] = 0
                    instance[numberKey] += 1

                if config.GLOVE_OPTIONS['burstTime'] == 1:
                    timeKey = 'T'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase) )
                    if not instance.get( timeKey ):
                        instance[timeKey] = 0
                    instance[timeKey] += 1

                directionCursor = packet.getDirection()
                dataCursor      = 0
                numberCursor    = 0
                burstTimeRef    = packet.getTime()


            dataCursor   += packet.getLength()
            numberCursor += 1
            timeCursor    = packet.getTime() - burstTimeRef

        if dataCursor>0:

            if config.GLOVE_OPTIONS['burstSize'] == 1:
                #key = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(dataCursor, 600) )
                key = 'S'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(dataCursor, config.bucket_Size) )

                if not instance.get( key ):
                    instance[key] = 0
                instance[key] += 1

            if config.GLOVE_OPTIONS['burstNumber'] == 1:
                numberKey = 'N'+str(directionCursor)+'-'+str( AdversarialClassifierTriDirectionFeaturesOnly.roundNumberMarker(numberCursor) )
                if not instance.get( numberKey ):
                    instance[numberKey] = 0
                instance[numberKey] += 1

            if config.GLOVE_OPTIONS['burstTime'] == 1:
                timeKey = 'T' + str(directionCursor) + '-' + str(
                    AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(timeCursor, timeBase))
                if not instance.get(timeKey):
                    instance[timeKey] = 0
                instance[timeKey] += 1

        burstsList = trace.getBurstsList()

        if len(burstsList) > 3:
            currBurst = burstsList[0] # first tuple
            nextBurst = burstsList[1]

            for burst in burstsList[2:]:
                # tuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
                nextNextBurst = burst
                #add features
                if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                    triBurstDataKey = 'triSize-' + currBurst[0] + '-' + nextBurst[0] + '-' + nextNextBurst[0] + '-' + \
                                       str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                           int(currBurst[1]), sizeBase)) + '-' + \
                                       str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                           int(nextBurst[1]), sizeBase)) + '-' + \
                                       str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                           int(nextNextBurst[1]), sizeBase))
                    if not instance.get(triBurstDataKey):
                        instance[triBurstDataKey] = 0
                    instance[triBurstDataKey] += 1

                # time
                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    triBurstTimeKey = 'triTime-' + currBurst[0] + '-' + nextBurst[0] + '-' + nextNextBurst[0] + '-' + \
                                     str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                         int(currBurst[3]), timeBase)) + '-' + \
                                     str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                         int(nextBurst[3]), timeBase)) + '-' + \
                                      str(AdversarialClassifierTriDirectionFeaturesOnly.roundArbitrary(
                                          int(nextNextBurst[3]), timeBase))

                    if not instance.get(triBurstTimeKey):
                        instance[triBurstTimeKey] = 0
                    instance[triBurstTimeKey] += 1


                currBurst = nextBurst
                nextBurst = nextNextBurst

        instance['class'] = 'webpage'+str(trace.getId())

        return instance


    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        [trainingFileOrig, testingFileOrig] = [trainingFile,testingFile]

        if config.NUM_MONITORED_SITES != -1: #no need to classify as this is for generating openworld datasets. See the line above (arffWriter)
            [accuracy,debugInfo] = ['NA', []]
            return [accuracy,debugInfo]

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA4([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        if config.lasso != 0:
            #[trainingFile,testingFile] = Utils.calcLasso3([trainingFile,testingFile])
            #[trainingFile,testingFile] = Utils.calcLogisticRegression([trainingFile,testingFile])
            Utils.calcLogisticRegression([trainingFile,testingFile])

        # deep learning
        if config.DEEP_LEARNING_METHOD != -1:
            #[trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile]) # one layer dA
            #[trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile]) # two layers dA
            #[trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile])
            #SdA_2.calcSdA([trainingFile, testingFile])
            if config.DEEP_LEARNING_METHOD == 1:
                [trainingFile, testingFile] = logistic_sgd_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 2:
                [trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
                [trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
                #[trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
                #[trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
                #[trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 3:
                # DL classifier
                return mlp_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 4:
                return SdA_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 5:
                return mlp_3.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 6:
                return SdA_3.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 7:
                return LeNetConvPoolLayer_2.runDL([trainingFile, testingFile])

        #Utils.plotDensity([trainingFile,testingFile])
        #Utils.plot([trainingFile,testingFile])

        if config.OC_SVM == 0: # multi-class svm
            if config.CROSS_VALIDATION == 0:
                #print 'WARNING: NB classifier with Bi-Di. ###########///////////XXXXXX???????? '
                #return wekaAPI.execute(trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'])

                return wekaAPI.execute( trainingFile,
                                 testingFile,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072', # Cost
                                  #'-S','2', # one-class svm
                                  '-B'] )  # confidence

            else:
                file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
                return wekaAPI.executeCrossValidation( file,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-x',str(config.CROSS_VALIDATION), # number of folds
                                  '-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072', # Cost
                                  '-B'] ) # confidence
        else: # one-class svm
            if config.CROSS_VALIDATION == 0:
                print str(config.SVM_KERNEL)
                print str(config.OC_SVM_Nu)
                return wekaAPI.executeOneClassSVM( trainingFile,
                                 testingFile,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-K',str(config.SVM_KERNEL),
                                  #'-K','2', # RBF kernel
                                  #'-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  #'-C','131072', # Cost
                                  #'-N','0.001', # nu
                                  '-N',str(config.OC_SVM_Nu), # nu
                                  '-S','2'])#, # one-class svm
                                  #'-B'] )  # confidence
            else:
                file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
                return wekaAPI.executeCrossValidation( file,
                                 "weka.Run weka.classifiers.functions.LibSVM",
                                 ['-x',str(config.CROSS_VALIDATION), # number of folds
                                  '-K','2', # RBF kernel
                                  '-G','0.0000019073486328125', # Gamma
                                  ##May20 '-Z', # normalization 18 May 2015
                                  '-C','131072', # Cost
                                  '-B'] ) # confidence


    '''
    #one class svm
    if config.CROSS_VALIDATION == 0:
        return wekaAPI.executeOneClassSVM( trainingFile,
                         testingFile,
                         "weka.Run weka.classifiers.functions.LibSVM",
                         ['-K','2', # RBF kernel
                          '-G','0.0000019073486328125', # Gamma
                          ##May20 '-Z', # normalization 18 May 2015
                          '-C','131072', # Cost
                          #'-N','0.2', # nu, def: 0.5
                          '-S','2'])#, # one-class svm
                          #'-B'] )  # confidence
    else:
        file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
        return wekaAPI.executeCrossValidation( file,
                         "weka.Run weka.classifiers.functions.LibSVM",
                         ['-x',str(config.CROSS_VALIDATION), # number of folds
                          '-K','2', # RBF kernel
                          '-G','0.0000019073486328125', # Gamma
                          ##May20 '-Z', # normalization 18 May 2015
                          '-C','131072', # Cost
                          '-B'] ) # confidence



    @staticmethod
    def classify(runID, trainingSet, testingSet):
        print 'DT'
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile,
                               testingFile,
                               "weka.classifiers.trees.J48",
                               ['-C', '0.25',
                                '-M', '2'])

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA4([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             [#'-K','0', # Linear kernel
                              '-K','2', # RBF kernel
                              #'-G','0.0000019073486328125', # Gamma
                              '-G','0.000030518',
                              ##May20 '-Z', # normalization 18 May 2015
                              #'-C','131072',
                              '-C','8'] ) # Cost



    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        if config.n_components_PCA != 0:
            [trainingFile,testingFile] = Utils.calcPCA2([trainingFile,testingFile])

        if config.n_components_LDA != 0:
            [trainingFile,testingFile] = Utils.calcLDA6([trainingFile,testingFile])

        if config.n_components_QDA != 0:
            [trainingFile,testingFile] = Utils.calcQDA([trainingFile,testingFile])

        return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )

    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072'] ) # Cost


    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'] )


    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.classifiers.trees.RandomForest",
                             ['-I','10', #
                              '-K','0', #
                              '-S','1'] ) #

    '''