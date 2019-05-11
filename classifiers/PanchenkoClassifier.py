# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

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

class PanchenkoClassifier:
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
        if trace.getPacketCount()==0:
            instance = {}
            instance['class'] = 'webpage'+str(trace.getId())
            return instance

        instance = trace.getHistogram()

        # Size/Number Markers
        directionCursor = None
        dataCursor      = 0
        numberCursor    = 0
        for packet in trace.getPackets():
            #print packet.getLength()
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                #dataKey = 'S'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundArbitrary(dataCursor, 600) )
                dataKey = 'S'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundArbitrary(dataCursor, config.bucket_Size) )

                if not instance.get( dataKey ):
                    instance[dataKey] = 0
                instance[dataKey] += 1

                numberKey = 'N'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundNumberMarker(numberCursor) )
                if not instance.get( numberKey ):
                    instance[numberKey] = 0
                instance[numberKey] += 1

                directionCursor = packet.getDirection()
                dataCursor      = 0
                numberCursor    = 0

            dataCursor += packet.getLength()
            numberCursor += 1

        if dataCursor>0:
            #key = 'S'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundArbitrary(dataCursor, 600) )
            key = 'S'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundArbitrary(dataCursor, config.bucket_Size) )
            if not instance.get( key ):
                instance[key] = 0
            instance[key] += 1

        if numberCursor>0:
            numberKey = 'N'+str(directionCursor)+'-'+str( PanchenkoClassifier.roundNumberMarker(numberCursor) )
            if not instance.get( numberKey ):
                instance[numberKey] = 0
            instance[numberKey] += 1

        # HTML Markers
        counterUP = 0
        counterDOWN = 0
        htmlMarker = 0
        for packet in trace.getPackets():
            if packet.getDirection() == Packet.UP:
                counterUP += 1
                if counterUP>1 and counterDOWN>0: break
            elif packet.getDirection() == Packet.DOWN:
                counterDOWN += 1
                htmlMarker += packet.getLength()

        #htmlMarker = PanchenkoClassifier.roundArbitrary( htmlMarker, 600 )
        htmlMarker = PanchenkoClassifier.roundArbitrary( htmlMarker, config.bucket_Size )
        instance['H'+str(htmlMarker)] = 1

        # Ocurring Packet Sizes
        packetsUp = []
        packetsDown = []
        for packet in trace.getPackets():
            if packet.getDirection()==Packet.UP and packet.getLength() not in packetsUp:
                packetsUp.append( packet.getLength() )
            if packet.getDirection()==Packet.DOWN and packet.getLength() not in packetsDown:
                packetsDown.append( packet.getLength() )
        instance['uniquePacketSizesUp'] = PanchenkoClassifier.roundArbitrary( len( packetsUp ), 2)
        instance['uniquePacketSizesDown'] = PanchenkoClassifier.roundArbitrary( len( packetsDown ), 2)

        # Percentage Incoming Packets
        instance['percentageUp']   = PanchenkoClassifier.roundArbitrary( 100.0 * trace.getPacketCount( Packet.UP ) / trace.getPacketCount(), 5)
        instance['percentageDown'] = PanchenkoClassifier.roundArbitrary( 100.0 * trace.getPacketCount( Packet.DOWN ) / trace.getPacketCount(), 5)

        # Number of Packets
        instance['numberUp']   = PanchenkoClassifier.roundArbitrary( trace.getPacketCount( Packet.UP ), 15)
        instance['numberDown'] = PanchenkoClassifier.roundArbitrary( trace.getPacketCount( Packet.DOWN ), 15)

        # Total Bytes Transmitted
        bandwidthUp   = PanchenkoClassifier.roundArbitrary( trace.getBandwidth( Packet.UP ),   10000)
        bandwidthDown = PanchenkoClassifier.roundArbitrary( trace.getBandwidth( Packet.DOWN ), 10000)
        instance['0-B'+str(bandwidthUp)] = 1
        instance['1-B'+str(bandwidthDown)] = 1

        # Label
        instance['class'] = 'webpage'+str(trace.getId())

        return instance
    
    @staticmethod
    def classify( runID, trainingSet, testingSet ):
        [trainingFile,testingFile] = arffWriter.writeArffFiles( runID, trainingSet, testingSet )

        # deep learning
        if config.DEEP_LEARNING_METHOD != -1:
            if config.DEEP_LEARNING_METHOD == 1:
                logistic_sgd_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 2:
                [trainingFile, testingFile] = dA_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 3:
                mlp_2.runDL([trainingFile, testingFile])
            elif config.DEEP_LEARNING_METHOD == 4:
                SdA_2.runDL([trainingFile, testingFile])


        if config.NUM_MONITORED_SITES != -1:
            [accuracy,debugInfo] = ['NA', []]
            return [accuracy,debugInfo]
        '''
        return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              '-C','131072'] ) # Cost
        '''

        if config.CROSS_VALIDATION == 0:
            return wekaAPI.execute( trainingFile,
                             testingFile,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072'] ) # Cost
        else:
            file = Utils.joinTrainingTestingFiles(trainingFile, testingFile) # join and shuffle
            return wekaAPI.executeCrossValidation( file,
                             "weka.Run weka.classifiers.functions.LibSVM",
                             ['-x',str(config.CROSS_VALIDATION), # number of folds
                              '-K','2', # RBF kernel
                              '-G','0.0000019073486328125', # Gamma
                              ##May20 '-Z', # normalization 18 May 2015
                              '-C','131072'] ) # Cost