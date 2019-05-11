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

class GloveClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))

    @staticmethod
    def traceToInstance(trace):
        modelFile = "model/"+config.RUN_ID+"mygloveModel"
        myglove = Glove.load(modelFile)
        return GloveClassifier.generateInstances2(myglove,trace)

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
                dataKey = 'S'+str(directionCursor)+'-'+str( GloveClassifier.roundArbitrary(dataCursor, 600) )
                paragraph.append(dataKey)
                directionCursor = packet.getDirection()
                dataCursor      = 0

            dataCursor += packet.getLength()

        if dataCursor>0:
            key = 'S'+str(directionCursor)+'-'+str( GloveClassifier.roundArbitrary(dataCursor, 600) )
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
                dataKey = 'S'+str(directionCursor)+'-'+str( GloveClassifier.roundArbitrary(dataCursor, 600) )
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
                                     str( GloveClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                     str( GloveClassifier.roundArbitrary(dataCursor, 600) )

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
            key = 'S'+str(directionCursor)+'-'+str( GloveClassifier.roundArbitrary(dataCursor, 600) )
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
                                 str( GloveClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                 str( GloveClassifier.roundArbitrary(dataCursor, 600) )

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
        myVectors = myglove.transform_paragraph(paragraph, epochs=90, ignore_missing=True)


        # testing
        '''
        for token in paragraph:
            print token
            tokenList = [token]
            paragraphWordVec = myglove.transform_paragraph(tokenList, epochs=90, ignore_missing=True)
            if np.isnan(paragraphWordVec[0]):
                print 'naaaaaaaaaaaan'
            print paragraphWordVec
        '''
        # end testing

        if len(myVectors) == 0:
            return None
        #mymeanVector = np.mean(myVectors, axis=0)
        #print mymeanVector.shape
        count = 0
        for l in myVectors:
            #vectorDict["v" + str(count)] = l
            vectorDict[str(count)] = l # for sorting purposes in the arff file
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
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)

        # deep learning (AE)
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


        return wekaAPI.execute(trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'])


'''@staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile,
                               testingFile,
                               "weka.Run weka.classifiers.functions.LibSVM",
                               ['-K', '2',  # RBF kernel
                                '-G', '0.0000019073486328125',  # Gamma
                                '-C', '131072'])  # Cost'''



