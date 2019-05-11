

import arffWriter
import wekaAPI
from glove import Glove
import numpy as np
import config
from sklearn.feature_extraction.text import TfidfVectorizer
from Utils import Utils

import theano_dir.dA_2 as dA_2
import theano_dir.SdA_2 as SdA_2
import theano_dir.logistic_sgd_2 as logistic_sgd_2
import theano_dir.mlp_2 as mlp_2
import theano_dir.mlp_3 as mlp_3
import theano_dir.SdA_3 as SdA_3
import theano_dir.LeNetConvPoolLayer_2 as LeNetConvPoolLayer_2

class TfidfClassifierHpNGram:

    @staticmethod
    def traceToInstance( eventTrace ):

        paragraph = []

        if eventTrace.getEventCount()==0:
            paragraph.append('webpage' + str(eventTrace.getId()))
            return paragraph

        for event in eventTrace.getEvents():
            paragraph.append(str(event.getDirection()) + '-' + str(event.getSystemcallName()))
            #paragraph.append(str(event.getSystemcallName()))

        paragraph.append('webpage' + str(eventTrace.getId()))

        return paragraph

    @staticmethod
    def classify(runID, trainingSet, testingSet):
        [docs, classesTrain, classesTest] = TfidfClassifierHpNGram.prepareDocs(trainingSet, testingSet)
        vectorizer = TfidfVectorizer(max_features=1000, use_idf='true', ngram_range=(1,3))
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

        if config.NUM_FEATURES_RF != 0:
            [trainingFile,testingFile] = Utils.calcTreeBaseRF([trainingFile,testingFile], config.NUM_FEATURES_RF)

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



