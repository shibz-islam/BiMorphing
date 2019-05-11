# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import arffWriter
import wekaAPI
from glove import Glove
import numpy as np
import config
from sklearn.feature_extraction.text import TfidfVectorizer
import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2

class TfidfClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x)/base))


    @staticmethod
    def traceToInstance(trace):
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
                dataKey = 'S'+str(directionCursor)+'-'+str( TfidfClassifier.roundArbitrary(dataCursor, 600) )
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
                                     str( TfidfClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                     str( TfidfClassifier.roundArbitrary(dataCursor, 600) )

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
            key = 'S'+str(directionCursor)+'-'+str( TfidfClassifier.roundArbitrary(dataCursor, 600) )
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
                                 str( TfidfClassifier.roundArbitrary(prevDataCursor, 600) )+'-'+ \
                                 str( TfidfClassifier.roundArbitrary(dataCursor, 600) )

                if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                    paragraph.append(biBurstDataKey)


                biBurstTimeKey = 'BiTime-'+str(prevDirectionCursor)+'-'+str(directionCursor)+'-'+ \
                                 str( prevTimeCursor )+'-'+ \
                                 str( timeCursor )

                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    paragraph.append(biBurstTimeKey)


        paragraph.append('webpage' + str(trace.getId()))

        return paragraph



    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [docs, classesTrain, classesTest] = TfidfClassifier.prepareDocs(trainingSet, testingSet)
        vectorizer = TfidfVectorizer(max_features=1000, use_idf='true')
        X = vectorizer.fit_transform(docs)
        trainingSetTfidf = []
        testingSetTfidf = []

        classTrainCtr = 0
        for row in X.toarray()[:len(classesTrain)]:
            instance = {}
            count = 0
            for w in row:
                instance['word' + str(count)] = w
                count += 1
            instance['class'] = classesTrain[classTrainCtr]
            trainingSetTfidf.append(instance)
            classTrainCtr += 1

        classTestCtr = 0
        for row in X.toarray()[len(classesTrain):]:
            instance = {}
            count = 0
            for w in row:
                instance['word' + str(count)] = w
                count += 1
            instance['class'] = classesTest[classTestCtr]
            testingSetTfidf.append(instance)
            classTestCtr += 1

        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSetTfidf, testingSetTfidf)
        #return wekaAPI.execute(trainingFile, testingFile, "weka.classifiers.bayes.NaiveBayes", ['-K'])

        # deep learning (AE)
        if config.AE != -1:
            [trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile]) # one layer dA
            [trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile]) # two layers dA
            #[trainingFile, testingFile] = dA_2.calcAE([trainingFile, testingFile])
            #SdA_2.calcSdA([trainingFile, testingFile])

        return wekaAPI.execute(trainingFile,
                           testingFile,
                           "weka.Run weka.classifiers.functions.LibSVM",
                           ['-K', '2',  # RBF kernel
                            '-G', '0.0000019073486328125',  # Gamma
                            '-C', '131072'])  # Cost'''


    @staticmethod
    def prepareDocs(trainingSet, testingSet):
        docs = []
        docsTrain = [] # list of strings, each string is a trace of words (packets size, burst size, biburst size, ...).
        classesTrain = []
        docsTest = []
        classesTest = []
        for l in trainingSet: # l = ['word1', 'word2', ..., 'webpage1']
            docsTrain.append(','.join(l[:-1])) # list to features as words
            classesTrain.append(l[-1]) # class

        for l in testingSet: # l = ['word1', 'word2', ..., 'webpage1']
            docsTest.append(','.join(l[:-1])) # list to features as words
            classesTest.append(l[-1]) # class

        docs = docsTrain + docsTest

        return [docs, classesTrain, classesTest]



'''@staticmethod
    def classify(runID, trainingSet, testingSet):
        [trainingFile, testingFile] = arffWriter.writeArffFiles(runID, trainingSet, testingSet)
        return wekaAPI.execute(trainingFile,
                               testingFile,
                               "weka.Run weka.classifiers.functions.LibSVM",
                               ['-K', '2',  # RBF kernel
                                '-G', '0.0000019073486328125',  # Gamma
                                '-C', '131072'])  # Cost'''



