
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


class KddClassifier:
    @staticmethod
    def traceToInstance( trace ):
        maxTime = 0
        for packet in trace.getPackets():
             if packet.getTime() > maxTime:
                 maxTime = packet.getTime()

        instance = {}

        #1: Basic features of individual TCP connections.
        instance['duration'] = maxTime
        instance['protocol_type'] = 'tcp'
        instance['service'] = 'http_443'
        instance['flag'] = 'SF'
        instance['src_bytes'] = trace.getBandwidth( Packet.UP )
        instance['dst_bytes'] = trace.getBandwidth( Packet.DOWN )
        instance['land'] = '0'
        instance['wrong_fragment'] = '0'
        instance['urgent'] = '0'

        #2: Content features within a connection suggested by domain knowledge.
        instance['hot'] = '0'
        instance['num_failed_logins'] = '0'
        instance['logged_in'] = '0'
        instance['lnum_compromised'] = '0'
        instance['lroot_shell'] = '0'
        instance['lsu_attempted'] = '0'
        instance['lnum_root'] = '0'
        instance['lnum_file_creations'] = '0'
        instance['lnum_shells'] = '0'
        instance['lnum_access_files'] = '0'
        instance['lnum_outbound_cmds'] = '0'
        instance['is_host_login'] = '0'
        instance['is_guest_login'] = '0'

        #3: Traffic features computed using a two-second time window.
        instance['count'] = '8' # * 8 in normal
        instance['srv_count'] = '8' # * 8 in normal
        instance['serror_rate'] = '0'
        instance['srv_serror_rate'] = '0'
        instance['rerror_rate'] = '0'
        instance['srv_rerror_rate'] = '0'
        instance['same_srv_rate'] = '1' # * 1 in normal
        instance['diff_srv_rate'] = '0'
        instance['srv_diff_host_rate'] = '0'
        instance['dst_host_count'] = '9' # * 9, 19, ... in normal
        instance['dst_host_srv_count'] = '9' # * 9, 19, ... in normal
        instance['dst_host_same_srv_rate'] = '1' # * 1 in normal
        instance['dst_host_diff_srv_rate'] = '0'
        instance['dst_host_same_src_port_rate'] = '0.11' # * 0.11, 0.05, ... in normal
        instance['dst_host_srv_diff_host_rate'] = '0'
        instance['dst_host_serror_rate'] = '0'
        instance['dst_host_srv_serror_rate'] = '0'
        instance['dst_host_rerror_rate'] = '0'
        instance['dst_host_srv_rerror_rate'] = '0'

        instance['class'] = 'webpage'+str(trace.getId())
        return instance
    
    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

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
