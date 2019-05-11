# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import arffWriter
import wekaAPI
from glove import Glove
import numpy as np
import config

from Utils import Utils

import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2
import theano_dir.logistic_sgd_2 as logistic_sgd_2
import theano_dir.mlp_2 as mlp_2
import theano_dir.mlp_3 as mlp_3
import theano_dir.SdA_3 as SdA_3
import theano_dir.LeNetConvPoolLayer_2 as LeNetConvPoolLayer_2
import theano_dir.LeNetConvPoolLayer_3_nlp as LeNetConvPoolLayer_3_nlp

class GloveClassifier2:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance(trace):
        modelFile = "model/"+config.RUN_ID+"mygloveModel"
        myglove = Glove.load(modelFile)
        return GloveClassifier2.generateInstances2(myglove,trace)

    @staticmethod
    def generateInstances(myglove,trace):
        myVectors = []
        vectorDict = {}
        paragraph = []
        for packet in trace.getPackets():
            key = str(packet.getLength()) + "_" + str(packet.getDirection())
            paragraph.append(key)

        directionCursor = None
        dataCursor      = 0
        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                dataKey = 'S'+str(directionCursor)+'-'+str( GloveClassifier2.roundArbitrary(dataCursor, 600) )
                paragraph.append(dataKey)
                directionCursor = packet.getDirection()
                dataCursor      = 0

            dataCursor += packet.getLength()

        if dataCursor>0:
            key = 'S'+str(directionCursor)+'-'+str( GloveClassifier2.roundArbitrary(dataCursor, 600) )
            paragraph.append(key)

        for key in paragraph:
            if key in myglove.dictionary:
                word_idx = myglove.dictionary[str(key)]
                myVectors.append(list(myglove.word_vectors[word_idx]))
        # for each packet len get the vectors and sum it up by colum to get a 100 dim vector to represent a trace therefor an instance
        #myVectors = myglove.transform_paragraph(paragraph, epochs=90, ignore_missing=True)
        if len(myVectors) == 0:
            return None
        mymeanVector = np.mean(myVectors, axis=0)
        #print mymeanVector.shape
        count = 0
        for l in mymeanVector:
            vectorDict["v" + str(count)] = l;
            count = count + 1;
        instance = {}  # trace.getHistogram()
        # print instance
        instance['class'] = 'webpage' + str(trace.getId())
        newinstances = dict(instance.items() + vectorDict.items())
        # some instances just contain nan values that should be discarded
        if np.isnan(vectorDict["v1"]):
            return None
        return newinstances

    @staticmethod
    def generateInstances2(myglove,trace):
        myVectors = []
        vectorDict = {}
        paragraph = []
        for packet in trace.getPackets():
            key = str(packet.getLength()) + "_" + str(packet.getDirection())
            if config.GLOVE_OPTIONS['packetSize'] == 1:
                paragraph.append(key)

        directionCursor = None
        dataCursor      = 0
        timeCursor = 0
        burstTimeRef = 0
        numberCursor = 0
        prevTimeCursor = 0
        secondBurstAndUp = False
        prevDataCursor = 0
        prevDirectionCursor = None

        for packet in trace.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                dataKey = 'S'+str(directionCursor)+'-'+str( GloveClassifier2.roundArbitrary(dataCursor, 600) )
                #dataKey = 'S'+str(directionCursor)+'-'+str(dataCursor )
                if config.GLOVE_OPTIONS['burstSize'] == 1:
                    paragraph.append(dataKey)
                #directionCursor = packet.getDirection()
                #dataCursor      = 0

                timeKey = 'T'+str(directionCursor)+'-'+str( timeCursor  )
                #timeCursor = 0
                if config.GLOVE_OPTIONS['burstTime'] == 1:
                    paragraph.append(timeKey)
                burstTimeRef = packet.getTime()

                # number marker
                numberKey = 'N'+str(directionCursor)+'-'+str( numberCursor)
                if config.GLOVE_OPTIONS['burstNumber'] == 1:
                    paragraph.append(numberKey)
                numberCursor    = 0

                # BiBurst
                if secondBurstAndUp:
                    #biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                    #                 str( prevDataCursor )+'-'+ \
                    #                 str( dataCursor )
                    biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( GloveClassifier2.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                     str( GloveClassifier2.roundArbitrary(dataCursor, 600) )

                    if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                        paragraph.append(biBurstDataKey)


                    biBurstTimeKey = 'BiTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                     str( prevTimeCursor )+'-'+ \
                                     str( timeCursor )

                    if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                        paragraph.append(biBurstTimeKey)


                prevTimeCursor = timeCursor
                timeCursor = 0
                secondBurstAndUp = True
                prevDataCursor = dataCursor
                dataCursor      = 0
                prevDirectionCursor = directionCursor
                directionCursor = packet.getDirection()

            dataCursor += packet.getLength()
            timeCursor = packet.getTime() - burstTimeRef
            numberCursor += 1

        if dataCursor>0:
            #key = 'S'+str(directionCursor)+'-'+str( dataCursor )
            key = 'S'+str(directionCursor)+'-'+str( GloveClassifier2.roundArbitrary(dataCursor, 600) )
            if config.GLOVE_OPTIONS['burstSize'] == 1:
                paragraph.append(key)

            timeKey = 'T'+str(directionCursor)+'-'+str( timeCursor  )
            if config.GLOVE_OPTIONS['burstTime'] == 1:
                paragraph.append(timeKey)

            numberKey = 'N'+str(directionCursor)+'-'+str( numberCursor)
            if config.GLOVE_OPTIONS['burstNumber'] == 1:
                paragraph.append(numberKey)

            # BiBurst
            if secondBurstAndUp:
                #biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                #                 str( prevDataCursor )+'-'+ \
                #                 str( dataCursor )
                biBurstDataKey = 'Bi-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                 str( GloveClassifier2.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                 str( GloveClassifier2.roundArbitrary(dataCursor, 600) )

                if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                    paragraph.append(biBurstDataKey)


                biBurstTimeKey = 'BiTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                 str( prevTimeCursor )+'-'+ \
                                 str( timeCursor )

                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    paragraph.append(biBurstTimeKey)

        #for key in paragraph:
        #   if key in myglove.dictionary:
        #        word_idx = myglove.dictionary[str(key)]
        #       myVectors.append(list(myglove.word_vectors[word_idx]))
        # for each packet len get the vectors and sum it up by colum to get a 100 dim vector to represent a trace therefor an instance
        #commented out 4/12/2017##myVectors = myglove.transform_paragraph(paragraph, epochs=90, ignore_missing=True)

        myVectors = np.array([])

        for token in paragraph:
            #print token
            tokenList = [token]
            paragraphWordVec = myglove.transform_paragraph(tokenList, epochs=90, ignore_missing=True) # the word vector for each token(word)

            if not np.isnan(paragraphWordVec[0]):
                myVectors = np.concatenate((myVectors, paragraphWordVec), axis=0)

            #print paragraphWordVec
            #print myVectors

        if len(myVectors) == 0:
            return None
        #mymeanVector = np.mean(myVectors, axis=0)
        #print mymeanVector.shape
        count = 0
        for l in myVectors:
            #vectorDict["v" + str(count)] = l;
            vectorDict[str(count)] = l  # for sorting purposes in the arff file
            count = count + 1
        instance = {}  # trace.getHistogram()
        # print instance
        instance['class'] = 'webpage' + str(trace.getId())
        newinstances = dict(instance.items() + vectorDict.items())
        # some instances just contain nan values that should be discarded
        #if np.isnan(vectorDict["v1"]):
        if np.isnan(vectorDict["1"]):
            return None
        return newinstances



    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles_variable_instances(runID, trainingSet, testingSet)

        # deep learning
        if config.DEEP_LEARNING_METHOD != -1:

            if config.DEEP_LEARNING_METHOD == 8:
                return LeNetConvPoolLayer_3_nlp.runDL([trainingFile, testingFile],
                                                      config.GLOVE_PARAMETERS['no_components']) # passing vector size
            else:
                return ['-C 116 is for -z 8 (cnn_nlp)!',[]]

        #return wekaAPI.execute(trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'])
        return ['-C 116 is for -z 8 (cnn_nlp)!', []]


'''@staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile,
                               testingFile,
                               "weka.Run weka.classifiers.functions.LibSVM",
                               ['-K', '2',  # RBF kernel
                                '-G', '0.0000019073486328125',  # Gamma
                                '-C', '131072'])  # Cost'''



