

from Trace import Trace
from Packet import Packet
import config
import random
import operator

import numpy as np
#from pylab import *
import matplotlib.pyplot as plt
import math

import sys
import heapq

from Datastore import Datastore
from Webpage import Webpage

class BiDiMorphing:

    @staticmethod
    def buildMetadata(webpageTrain, targetWebpage, webpageIds, unMonitoredWebpageIdsObj, seed):

        # dynamic morphing: pick a different target (closest) for each source
        if config.DATA_SOURCE == 5:
            #targetWebpage = BiDiMorphing.getClosestTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            #targetWebpage = BiDiMorphing.getRandomTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            targetWebpage = BiDiMorphing.getLargestTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            #targetWebpage = BiDiMorphing.getLargestTargets(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)

        # dim based on target
        up_dn_col = 0
        up_dn_row = 0
        dn_up_col = 0
        dn_up_row = 0

        #BI_BURST_HISTO_UP_DOWN_TARGET = {}
        #BI_BURST_HISTO_DOWN_UP_TARGET = {}

        if config.BUILD_TARGET_INFO:
            # start with target to bui ld W, X
            for trace in targetWebpage.getTraces():

                burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

                prevBurst = None
                for burst in burstsList:
                    if prevBurst == None:
                        prevBurst = burst
                        continue

                    currBurst = burst

                    key = prevBurst[1] + "_" + currBurst[1]

                    if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                        if int(prevBurst[1]) > up_dn_col:
                            up_dn_col = int(prevBurst[1])
                        if int(currBurst[1]) > up_dn_row:
                            up_dn_row = int(currBurst[1])

                        if not config.BI_BURST_HISTO_UP_DOWN_TARGET.get(key):
                            config.BI_BURST_HISTO_UP_DOWN_TARGET[key] = 0
                        config.BI_BURST_HISTO_UP_DOWN_TARGET[key] += 1

                    elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                        if int(prevBurst[1]) > dn_up_col:
                            dn_up_col = int(prevBurst[1])
                        if int(currBurst[1]) > dn_up_row:
                            dn_up_row = int(currBurst[1])

                        if not config.BI_BURST_HISTO_DOWN_UP_TARGET.get(key):
                            config.BI_BURST_HISTO_DOWN_UP_TARGET[key] = 0
                        config.BI_BURST_HISTO_DOWN_UP_TARGET[key] += 1

                    prevBurst = currBurst


                ############ start extra dummy
                # if needed, dummy burst padding at the end of newTrace
                # target burst list
                # config.targetBurstsList = []  # [ [(tuple), (tuple), ...],
                #                   [(tuple), (tuple), ...],
                #                   ...]
                # burst tuple = (direction, size, number, timeDiff, burstPackets)
                # [(tuple), (tuple), ...] is one trace

                config.targetBurstsList.append(trace.getBurstsListWithPackets())

                ############ end extra dummy

            config.BUILD_TARGET_INFO = False # build target info just once

            config.dim = 0 # dimension of the matrices in the Objective function
            if up_dn_col > config.dim:
                config.dim = up_dn_col
            if up_dn_row > config.dim:
                config.dim = up_dn_row
            if dn_up_col > config.dim:
                config.dim = dn_up_col
            if dn_up_row > config.dim:
                config.dim = dn_up_row

            config.dim += 1 # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)

        # X in the Obj Function, counts of target
        # send col, row and it will be handled in the buildMatrix func
        X_up_dn = BiDiMorphing.buildMatrix(config.BI_BURST_HISTO_UP_DOWN_TARGET, config.dim, config.dim)
        X_dn_up = BiDiMorphing.buildMatrix(config.BI_BURST_HISTO_DOWN_UP_TARGET, config.dim, config.dim)

        # W to learn (to adjust X of target, to reduce effect of frequent bi-bursts so sampling is not biased)
        W_up_dn = np.ones( (config.dim,config.dim) )
        W_dn_up = np.ones( (config.dim,config.dim) )


        # source webpage, to build Prob (P_ij) of the obj function
        BI_BURST_HISTO_UP_DOWN_SOURCE = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE = {}

        for trace in webpageTrain.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not BI_BURST_HISTO_UP_DOWN_SOURCE.get(key):
                        BI_BURST_HISTO_UP_DOWN_SOURCE[key] = 0
                    BI_BURST_HISTO_UP_DOWN_SOURCE[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not BI_BURST_HISTO_DOWN_UP_SOURCE.get(key):
                        BI_BURST_HISTO_DOWN_UP_SOURCE[key] = 0
                    BI_BURST_HISTO_DOWN_UP_SOURCE[key] += 1

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, config.dim, config.dim)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, config.dim, config.dim)

        """Shihab: commented out calcGradientDescent calculation """
        itr = 100
        stepSize = 0.001
        #new_W_up_dn = BiDiMorphing.calcGradientDescent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, config.dim, config.dim, itr, stepSize)
        #new_W_dn_up = BiDiMorphing.calcGradientDescent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, config.dim, config.dim, itr, stepSize)

        # these are adjusted pmfs of target (to be used in sampling)
        """Shihab: changed new_W_up_dn to W_up_dn and new_W_dn_up to W_dn_up"""
        config.PMF_UP_DN = BiDiMorphing.buildPMF_with_W(config.BI_BURST_HISTO_UP_DOWN_TARGET, W_up_dn)
        config.PMF_DN_UP = BiDiMorphing.buildPMF_with_W(config.BI_BURST_HISTO_DOWN_UP_TARGET, W_dn_up)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)

        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)
        #BiDiMorphing.buildMetadataIAT_Ws(webpageTrain, targetWebpage) # max obj

        config.buildMetadata = False


        return [targetBurstDistribution, config.targetBurstsList]

    @staticmethod
    def buildMetadataOld4(webpageTrain, targetWebpage, webpageIds, unMonitoredWebpageIdsObj, seed):

        # dynamic morphing: pick a different target (closest) for each source
        if config.DATA_SOURCE == 5:
            #targetWebpage = BiDiMorphing.getClosestTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            #targetWebpage = BiDiMorphing.getRandomTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            targetWebpage = BiDiMorphing.getLargestTarget(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)
            #targetWebpage = BiDiMorphing.getLargestTargets(webpageTrain, webpageIds, unMonitoredWebpageIdsObj, seed)

        # dim based on target
        up_dn_col = 0
        up_dn_row = 0
        dn_up_col = 0
        dn_up_row = 0

        BI_BURST_HISTO_UP_DOWN_TARGET = {}
        BI_BURST_HISTO_DOWN_UP_TARGET = {}

        # start with target to build W, X
        for trace in targetWebpage.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if int(prevBurst[1]) > up_dn_col:
                        up_dn_col = int(prevBurst[1])
                    if int(currBurst[1]) > up_dn_row:
                        up_dn_row = int(currBurst[1])

                    if not BI_BURST_HISTO_UP_DOWN_TARGET.get(key):
                        BI_BURST_HISTO_UP_DOWN_TARGET[key] = 0
                    BI_BURST_HISTO_UP_DOWN_TARGET[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if int(prevBurst[1]) > dn_up_col:
                        dn_up_col = int(prevBurst[1])
                    if int(currBurst[1]) > dn_up_row:
                        dn_up_row = int(currBurst[1])

                    if not BI_BURST_HISTO_DOWN_UP_TARGET.get(key):
                        BI_BURST_HISTO_DOWN_UP_TARGET[key] = 0
                    BI_BURST_HISTO_DOWN_UP_TARGET[key] += 1

                prevBurst = currBurst

        dim = 0 # dimension of the matrices in the Objective function
        if up_dn_col > dim:
            dim = up_dn_col
        if up_dn_row > dim:
            dim = up_dn_row
        if dn_up_col > dim:
            dim = dn_up_col
        if dn_up_row > dim:
            dim = dn_up_row

        dim += 1 # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)

        # X in the Obj Function, counts of target
        # send col, row and it will be handled in the buildMatrix func
        X_up_dn = BiDiMorphing.buildMatrix(BI_BURST_HISTO_UP_DOWN_TARGET, dim, dim)
        X_dn_up = BiDiMorphing.buildMatrix(BI_BURST_HISTO_DOWN_UP_TARGET, dim, dim)

        # W to learn (to adjust X of target, to reduce effect of frequent bi-bursts so sampling is not biased)
        W_up_dn = np.ones( (dim,dim) )
        W_dn_up = np.ones( (dim,dim) )


        # source webpage, to build Prob (P_ij) of the obj function
        BI_BURST_HISTO_UP_DOWN_SOURCE = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE = {}

        for trace in webpageTrain.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not BI_BURST_HISTO_UP_DOWN_SOURCE.get(key):
                        BI_BURST_HISTO_UP_DOWN_SOURCE[key] = 0
                    BI_BURST_HISTO_UP_DOWN_SOURCE[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not BI_BURST_HISTO_DOWN_UP_SOURCE.get(key):
                        BI_BURST_HISTO_DOWN_UP_SOURCE[key] = 0
                    BI_BURST_HISTO_DOWN_UP_SOURCE[key] += 1

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, dim, dim)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, dim, dim)

        itr = 100
        stepSize = 0.001
        new_W_up_dn = BiDiMorphing.calcGradientDescent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, dim, dim, itr, stepSize)
        new_W_dn_up = BiDiMorphing.calcGradientDescent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, dim, dim, itr, stepSize)

        # these are adjusted pmfs of target (to be used in sampling)
        config.PMF_UP_DN = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_UP_DOWN_TARGET, new_W_up_dn)
        config.PMF_DN_UP = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_DOWN_UP_TARGET, new_W_dn_up)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)

        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)
        #BiDiMorphing.buildMetadataIAT_Ws(webpageTrain, targetWebpage) # max obj

        config.buildMetadata = False


        # if needed, dummy burst padding at the end of newTrace
        # target burst list
        burstsList = [] # [ [(tuple), (tuple), ...],
        #                   [(tuple), (tuple), ...],
        #                   ...]
        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # [(tuple), (tuple), ...] is one trace

        for trace in targetWebpage.getTraces():
            burstsList.append(trace.getBurstsListWithPackets())


        return [targetBurstDistribution, burstsList]

    @staticmethod
    def buildMetadataOld3(webpageTrain, targetWebpage):

        # dynamic morphing: pick a different target for each source

        # dim based on target
        up_dn_col = 0
        up_dn_row = 0
        dn_up_col = 0
        dn_up_row = 0

        BI_BURST_HISTO_UP_DOWN_TARGET = {}
        BI_BURST_HISTO_DOWN_UP_TARGET = {}

        # start with target to build W, X
        for trace in targetWebpage.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if int(prevBurst[1]) > up_dn_col:
                        up_dn_col = int(prevBurst[1])
                    if int(currBurst[1]) > up_dn_row:
                        up_dn_row = int(currBurst[1])

                    if not BI_BURST_HISTO_UP_DOWN_TARGET.get(key):
                        BI_BURST_HISTO_UP_DOWN_TARGET[key] = 0
                    BI_BURST_HISTO_UP_DOWN_TARGET[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if int(prevBurst[1]) > dn_up_col:
                        dn_up_col = int(prevBurst[1])
                    if int(currBurst[1]) > dn_up_row:
                        dn_up_row = int(currBurst[1])

                    if not BI_BURST_HISTO_DOWN_UP_TARGET.get(key):
                        BI_BURST_HISTO_DOWN_UP_TARGET[key] = 0
                    BI_BURST_HISTO_DOWN_UP_TARGET[key] += 1

                prevBurst = currBurst

        dim = 0 # dimension of the matrices in the Objective function
        if up_dn_col > dim:
            dim = up_dn_col
        if up_dn_row > dim:
            dim = up_dn_row
        if dn_up_col > dim:
            dim = dn_up_col
        if dn_up_row > dim:
            dim = dn_up_row

        dim += 1 # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)

        # X in the Obj Function, counts of target
        # send col, row and it will be handled in the buildMatrix func
        X_up_dn = BiDiMorphing.buildMatrix(BI_BURST_HISTO_UP_DOWN_TARGET, dim, dim)
        X_dn_up = BiDiMorphing.buildMatrix(BI_BURST_HISTO_DOWN_UP_TARGET, dim, dim)

        # W to learn (to adjust X of target, to reduce effect of frequent bi-bursts so sampling is not biased)
        W_up_dn = np.ones( (dim,dim) )
        W_dn_up = np.ones( (dim,dim) )


        # source webpage, to build Prob (P_ij) of the obj function
        BI_BURST_HISTO_UP_DOWN_SOURCE = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE = {}

        for trace in webpageTrain.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not BI_BURST_HISTO_UP_DOWN_SOURCE.get(key):
                        BI_BURST_HISTO_UP_DOWN_SOURCE[key] = 0
                    BI_BURST_HISTO_UP_DOWN_SOURCE[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not BI_BURST_HISTO_DOWN_UP_SOURCE.get(key):
                        BI_BURST_HISTO_DOWN_UP_SOURCE[key] = 0
                    BI_BURST_HISTO_DOWN_UP_SOURCE[key] += 1

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, dim, dim)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, dim, dim)

        itr = 100
        stepSize = 0.001
        new_W_up_dn = BiDiMorphing.calcGradientDescent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, dim, dim, itr, stepSize)
        new_W_dn_up = BiDiMorphing.calcGradientDescent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, dim, dim, itr, stepSize)

        # these are adjusted pmfs of target (to be used in sampling)
        config.PMF_UP_DN = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_UP_DOWN_TARGET, new_W_up_dn)
        config.PMF_DN_UP = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_DOWN_UP_TARGET, new_W_dn_up)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)

        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)
        #BiDiMorphing.buildMetadataIAT_Ws(webpageTrain, targetWebpage) # max obj

        config.buildMetadata = False


        # if needed, dummy burst padding at the end of newTrace
        # target burst list
        burstsList = [] # [ [(tuple), (tuple), ...],
        #                   [(tuple), (tuple), ...],
        #                   ...]
        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # [(tuple), (tuple), ...] is one trace

        for trace in targetWebpage.getTraces():
            burstsList.append(trace.getBurstsListWithPackets())


        return [targetBurstDistribution, burstsList]

    @staticmethod
    def buildMetadataOld2(webpageTrain, targetWebpage):

        # dim based on target
        up_dn_col = 0
        up_dn_row = 0
        dn_up_col = 0
        dn_up_row = 0

        BI_BURST_HISTO_UP_DOWN_TARGET = {}
        BI_BURST_HISTO_DOWN_UP_TARGET = {}

        # start with target to build W, X
        for trace in targetWebpage.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if int(prevBurst[1]) > up_dn_col:
                        up_dn_col = int(prevBurst[1])
                    if int(currBurst[1]) > up_dn_row:
                        up_dn_row = int(currBurst[1])

                    if not BI_BURST_HISTO_UP_DOWN_TARGET.get(key):
                        BI_BURST_HISTO_UP_DOWN_TARGET[key] = 0
                    BI_BURST_HISTO_UP_DOWN_TARGET[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if int(prevBurst[1]) > dn_up_col:
                        dn_up_col = int(prevBurst[1])
                    if int(currBurst[1]) > dn_up_row:
                        dn_up_row = int(currBurst[1])

                    if not BI_BURST_HISTO_DOWN_UP_TARGET.get(key):
                        BI_BURST_HISTO_DOWN_UP_TARGET[key] = 0
                    BI_BURST_HISTO_DOWN_UP_TARGET[key] += 1

                prevBurst = currBurst

        dim = 0 # dimension of the matrices in the Objective function
        if up_dn_col > dim:
            dim = up_dn_col
        if up_dn_row > dim:
            dim = up_dn_row
        if dn_up_col > dim:
            dim = dn_up_col
        if dn_up_row > dim:
            dim = dn_up_row

        dim += 1 # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)

        # X in the Obj Function, counts of target
        # send col, row and it will be handled in the buildMatrix func
        X_up_dn = BiDiMorphing.buildMatrix(BI_BURST_HISTO_UP_DOWN_TARGET, dim, dim)
        X_dn_up = BiDiMorphing.buildMatrix(BI_BURST_HISTO_DOWN_UP_TARGET, dim, dim)

        # W to learn (to adjust X of target, to reduce effect of frequent bi-bursts so sampling is not biased)
        W_up_dn = np.ones( (dim,dim) )
        W_dn_up = np.ones( (dim,dim) )


        # source webpage, to build Prob (P_ij) of the obj function
        BI_BURST_HISTO_UP_DOWN_SOURCE = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE = {}

        for trace in webpageTrain.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not BI_BURST_HISTO_UP_DOWN_SOURCE.get(key):
                        BI_BURST_HISTO_UP_DOWN_SOURCE[key] = 0
                    BI_BURST_HISTO_UP_DOWN_SOURCE[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not BI_BURST_HISTO_DOWN_UP_SOURCE.get(key):
                        BI_BURST_HISTO_DOWN_UP_SOURCE[key] = 0
                    BI_BURST_HISTO_DOWN_UP_SOURCE[key] += 1

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, dim, dim)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, dim, dim)

        itr = 100
        stepSize = 0.001
        new_W_up_dn = BiDiMorphing.calcGradientDescent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, dim, dim, itr, stepSize)
        new_W_dn_up = BiDiMorphing.calcGradientDescent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, dim, dim, itr, stepSize)

        # these are adjusted pmfs of target (to be used in sampling)
        config.PMF_UP_DN = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_UP_DOWN_TARGET, new_W_up_dn)
        config.PMF_DN_UP = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_DOWN_UP_TARGET, new_W_dn_up)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)

        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)
        #BiDiMorphing.buildMetadataIAT_Ws(webpageTrain, targetWebpage) # max obj

        config.buildMetadata = False


        # if needed, dummy burst padding at the end of newTrace
        # target burst list
        burstsList = [] # [ [(tuple), (tuple), ...],
        #                   [(tuple), (tuple), ...],
        #                   ...]
        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # [(tuple), (tuple), ...] is one trace

        for trace in targetWebpage.getTraces():
            burstsList.append(trace.getBurstsListWithPackets())


        return [targetBurstDistribution, burstsList]

    @staticmethod
    def buildMetadataOld(webpageTrain, targetWebpage):

        # dim based on target
        up_dn_col = 0
        up_dn_row = 0
        dn_up_col = 0
        dn_up_row = 0

        BI_BURST_HISTO_UP_DOWN_TARGET = {}
        BI_BURST_HISTO_DOWN_UP_TARGET = {}

        # start with target to build W, X
        for trace in targetWebpage.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if int(prevBurst[1]) > up_dn_col:
                        up_dn_col = int(prevBurst[1])
                    if int(currBurst[1]) > up_dn_row:
                        up_dn_row = int(currBurst[1])

                    if not BI_BURST_HISTO_UP_DOWN_TARGET.get(key):
                        BI_BURST_HISTO_UP_DOWN_TARGET[key] = 0
                    BI_BURST_HISTO_UP_DOWN_TARGET[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if int(prevBurst[1]) > dn_up_col:
                        dn_up_col = int(prevBurst[1])
                    if int(currBurst[1]) > dn_up_row:
                        dn_up_row = int(currBurst[1])

                    if not BI_BURST_HISTO_DOWN_UP_TARGET.get(key):
                        BI_BURST_HISTO_DOWN_UP_TARGET[key] = 0
                    BI_BURST_HISTO_DOWN_UP_TARGET[key] += 1

                prevBurst = currBurst

        dim = 0
        if up_dn_col > dim:
            dim = up_dn_col
        if up_dn_row > dim:
            dim = up_dn_row
        if dn_up_col > dim:
            dim = dn_up_col
        if dn_up_row > dim:
            dim = dn_up_row

        dim += 1 # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)

        X_up_dn = BiDiMorphing.buildMatrix(BI_BURST_HISTO_UP_DOWN_TARGET, dim, dim)
        X_dn_up = BiDiMorphing.buildMatrix(BI_BURST_HISTO_DOWN_UP_TARGET, dim, dim)

        W_up_dn = np.ones( (dim,dim) )
        W_dn_up = np.ones( (dim,dim) )



        # source webpage
        BI_BURST_HISTO_UP_DOWN_SOURCE = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE = {}

        for trace in webpageTrain.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not BI_BURST_HISTO_UP_DOWN_SOURCE.get(key):
                        BI_BURST_HISTO_UP_DOWN_SOURCE[key] = 0
                    BI_BURST_HISTO_UP_DOWN_SOURCE[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not BI_BURST_HISTO_DOWN_UP_SOURCE.get(key):
                        BI_BURST_HISTO_DOWN_UP_SOURCE[key] = 0
                    BI_BURST_HISTO_DOWN_UP_SOURCE[key] += 1

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, dim, dim)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, dim, dim)

        itr = 100
        stepSize = 0.001
        new_W_up_dn = BiDiMorphing.calcGradientDescent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, dim, dim, itr, stepSize)
        new_W_dn_up = BiDiMorphing.calcGradientDescent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, dim, dim, itr, stepSize)

        config.PMF_UP_DN = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_UP_DOWN_TARGET, new_W_up_dn)
        config.PMF_DN_UP = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_DOWN_UP_TARGET, new_W_dn_up)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)

        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)

        config.buildMetadata = False

        return [targetBurstDistribution]

    @staticmethod
    def buildMatrix(BI_BURST_HISTO, col, row):


        '''
                    up_down:
                    up (col)
               -----------------------------
               |
               |
         down  |
        (row)  |
               |


                     down_up:
                     down (col)
               -----------------------------
               |
               |
         up    |
        (row)  |
               |


        '''

        X = np.zeros((row, col))  # if up_down hisot, then col = up, and row = down

        #fileLines = [line.strip() for line in open(histoFile)]
        histoTuples = sorted(BI_BURST_HISTO.items(), key=lambda x: x[1], reverse=True)
        #histoTuples = BI_BURST_HISTO.items()

        for tu in histoTuples:
            # e.g., histoTuples = [('1_1', 4), ('3_4', 3), ('1_2', 2)]

            upOrDown = tu[0].split('_')
            upOrDownLeft = int(upOrDown[0]) # col
            upOrDownRight = int(upOrDown[1]) # row

            X[upOrDownRight][upOrDownLeft] = int(tu[1]) # X[row][col]

        return X

    @staticmethod
    def buildProbMatrix(pmf, col, row):

        '''
                    up_down:
                    up (col)
               -----------------------------
               |
               |
         down  |
        (row)  |
               |


                     down_up:
                     down (col)
               -----------------------------
               |
               |
         up    |
        (row)  |
               |


        '''

        # pmf, e.g. {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

        P = np.zeros((row, col))  # if up_down , then col = up, and row = down

        for key1 in pmf.keys(): # key1 is the col in the matrix
            for key2 in pmf[key1].keys(): # key2 is the row in the matrix
                if key2 < row and key1 < col: # as some websites may have higher dimentions than the target website
                    P[key2][key1] = pmf[key1][key2] # key1 is col and key2 is row

        return P

    @staticmethod
    def calcGradientDescentAscent(W, X, Prob_b, col, row, itr, stepSize, descent):
        new_W = W
        Xmax = 100 # from the GloVe paper
        alpha = 0.75 # from the GloVe paper
        for t in range(0,itr):
            for i in range(1,row): # as col is prev burst so it is i in the obj function
                for j in range(1,col):
                    if X[i][j] < Xmax:
                        fx = math.pow((X[i][j] / Xmax), alpha)
                    else:
                        fx = 1
                    #print 'Xij: ' + str(X[i][j])
                    #print 'fx: ' + str(fx)
                    #new_W[i][j] = new_W[i][j] - (stepSize * Prob_b[i][j] * X[i][j] * math.pow((i-j), 2) * 2 * new_W[i][j])
                    #print 'before: ' + str(new_W[i][j])
                    if descent:
                        new_W[i][j] = new_W[i][j] - (stepSize * Prob_b[i][j] * fx * math.pow((i - j), 2) * 2 * new_W[i][j])
                    else:
                        new_W[i][j] = new_W[i][j] + (stepSize * Prob_b[i][j] * fx * math.pow((i - j), 2) * 2 * new_W[i][j])
                    #print 'after:  ' + str(new_W[i][j])
                    #print '-------'


        return new_W

    @staticmethod
    def calcGradientDescent(W, X, Prob_b, col, row, itr, stepSize):
        new_W = W
        Xmax = 100 # from the GloVe paper
        alpha = 0.75 # from the GloVe paper
        for t in range(0,itr):
            for i in range(1,col): # as col is prev burst so it is i in the obj function
                for j in range(1,row):
                    if X[i][j] < Xmax:
                        fx = math.pow((X[i][j] / Xmax), alpha)
                    else:
                        fx = 1
                    #print 'Xij: ' + str(X[i][j])
                    #print 'fx: ' + str(fx)
                    #new_W[i][j] = new_W[i][j] - (stepSize * Prob_b[i][j] * X[i][j] * math.pow((i-j), 2) * 2 * new_W[i][j])
                    #print 'before: ' + str(new_W[i][j])
                    new_W[i][j] = new_W[i][j] - (stepSize * Prob_b[i][j] * fx * math.pow((i - j), 2) * 2 * new_W[i][j])
                    #print 'after:  ' + str(new_W[i][j])
                                 #print '-------'


        return new_W

    @staticmethod
    def buildMetadataOld(webpageTrain, targetWebpage):

        for trace in targetWebpage.getTraces():

            burstsList = trace.getBurstsList()  # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not config.BI_BURST_HISTO_UP_DOWN.get(key):
                        config.BI_BURST_HISTO_UP_DOWN[key] = 0
                    config.BI_BURST_HISTO_UP_DOWN[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not config.BI_BURST_HISTO_DOWN_UP.get(key):
                        config.BI_BURST_HISTO_DOWN_UP[key] = 0
                    config.BI_BURST_HISTO_DOWN_UP[key] += 1

                prevBurst = currBurst


        print 'target webpage id: ' + str(targetWebpage.getId())
        config.PMF_UP_DN = BiDiMorphing.buildPMF2(config.BI_BURST_HISTO_UP_DOWN)
        config.PMF_DN_UP = BiDiMorphing.buildPMF2(config.BI_BURST_HISTO_DOWN_UP)

        #print "10 websites"
        #config.PMF_UP_DN = BiDiMorphing.buildPMF('biburst_up_down_histo_10sites')
        #config.PMF_DN_UP = BiDiMorphing.buildPMF('biburst_down_up_histo_10sites')

        thr = 20 # num packets
        #config.PMF_UP_DN = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_10sites', thr)
        #config.PMF_DN_UP = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_10sites', thr)

        #config.PMF_UP_DN = BiDiMorphing.buildPMF('biburst_up_down_histo_allSites')
        #config.PMF_DN_UP = BiDiMorphing.buildPMF('biburst_down_up_histo_allSites')

        #config.PMF_UP_DN = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_allSites', thr)
        #config.PMF_DN_UP = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_allSites', thr)

        config.CDF_UP_DN = BiDiMorphing.buildCDF(config.PMF_UP_DN)
        config.CDF_DN_UP = BiDiMorphing.buildCDF(config.PMF_DN_UP)

        ## building metadata
        # to be used in L1 distance between traces (based on burst histgrams not packet histograms as
        # packet histograms in tor is just 0_1 for uplink and 1_1 for downlink)
        targetBurstDistribution = targetWebpage.getBurstHistogram(True)


        # bi-IAT hitograms
        BiDiMorphing.buildMetadataIAT(targetWebpage)

        config.buildMetadata = False

        return [targetBurstDistribution]

    @staticmethod
    def buildMetadataIAT_Ws(webpageTrain, targetWebpage):

        # dim based on target
        up_dn_col = 0 # prev burst size
        up_dn_row = 0 # next burst packets IAT
        dn_up_col = 0 # prev burst size
        dn_up_row = 0 # next burst packets IAT

        BI_BURST_HISTO_UP_DOWN_TARGET_IAT = {}
        BI_BURST_HISTO_DOWN_UP_TARGET_IAT = {}
        for trace in targetWebpage.getTraces():
            burstsList = trace.getBurstsListWithPackets() # (direction, size, number, timeDiff, currBurstPackets)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                prevPktTime = None
                for packet in currBurst[4]: # loop over burst packets list
                    if prevPktTime == None:
                        prevPktTime = packet.getTime()
                        continue

                    currPktTime = packet.getTime()
                    IAT = currPktTime - prevPktTime # InterArrival Time

                    key = prevBurst[1] + "_" + str(IAT)

                    if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                        if int(prevBurst[1]) > up_dn_col:
                            up_dn_col = int(prevBurst[1])
                        if IAT > up_dn_row:
                            up_dn_row = IAT

                        if not BI_BURST_HISTO_UP_DOWN_TARGET_IAT.get(key):
                            BI_BURST_HISTO_UP_DOWN_TARGET_IAT[key] = 0
                            BI_BURST_HISTO_UP_DOWN_TARGET_IAT[key] += 1

                    elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                        if int(prevBurst[1]) > dn_up_col:
                            dn_up_col = int(prevBurst[1])
                        if IAT > dn_up_row:
                            dn_up_row = IAT

                        if not BI_BURST_HISTO_DOWN_UP_TARGET_IAT.get(key):
                            BI_BURST_HISTO_DOWN_UP_TARGET_IAT[key] = 0
                            BI_BURST_HISTO_DOWN_UP_TARGET_IAT[key] += 1

                    prevPktTime = currPktTime

                prevBurst = currBurst

        ''' for target page 47
        print up_dn_row # 3745 # IAT
        print up_dn_col # 69   # burst
        print dn_up_row # 1855 # UAT
        print dn_up_col # 150
        '''

        # unlike histo sizes, this won't be a square matrix
        dim_row = 0 # IAT
        dim_col = 0 # prev burst size
        if up_dn_col > dim_col:
            dim_col = up_dn_col
        if dn_up_col > dim_col:
            dim_col = dn_up_col

        if up_dn_row > dim_row:
            dim_row = up_dn_row
        if dn_up_row > dim_row:
            dim_row = dn_up_row

        # to deal with zero indexing (no zero index will be used, i.e., [0][anythin] = 0 and [anything][0] = 0)
        dim_row += 1
        dim_col += 1

        # X in the Obj Function, counts of target
        # send col, row and it will be handled in the buildMatrix func
        X_up_dn = BiDiMorphing.buildMatrix(BI_BURST_HISTO_UP_DOWN_TARGET_IAT, dim_col, dim_row)
        X_dn_up = BiDiMorphing.buildMatrix(BI_BURST_HISTO_DOWN_UP_TARGET_IAT, dim_col, dim_row)

        # W to learn (to adjust X of target, to reduce effect of frequent bi-bursts so sampling is not biased)
        W_up_dn = np.ones( (dim_row, dim_col) )
        W_dn_up = np.ones( (dim_row, dim_col) )

        # source webpage, to build Prob (P_ij) of the obj function
        BI_BURST_HISTO_UP_DOWN_SOURCE_IAT = {}
        BI_BURST_HISTO_DOWN_UP_SOURCE_IAT = {}
        for trace in webpageTrain.getTraces():
            burstsList = trace.getBurstsListWithPackets() # (direction, size, number, timeDiff, currBurstPackets)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                prevPktTime = None
                for packet in currBurst[4]: # loop over burst packets list
                    if prevPktTime == None:
                        prevPktTime = packet.getTime()
                        continue

                    currPktTime = packet.getTime()
                    IAT = currPktTime - prevPktTime # InterArrival Time

                    key = prevBurst[1] + "_" + str(IAT)

                    if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                        if not BI_BURST_HISTO_UP_DOWN_SOURCE_IAT.get(key):
                            BI_BURST_HISTO_UP_DOWN_SOURCE_IAT[key] = 0
                            BI_BURST_HISTO_UP_DOWN_SOURCE_IAT[key] += 1

                    elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                        if not BI_BURST_HISTO_DOWN_UP_SOURCE_IAT.get(key):
                            BI_BURST_HISTO_DOWN_UP_SOURCE_IAT[key] = 0
                            BI_BURST_HISTO_DOWN_UP_SOURCE_IAT[key] += 1

                    prevPktTime = currPktTime

                prevBurst = currBurst

        PMF_UP_DN_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_SOURCE_IAT)
        PMF_DN_UP_SOURCE = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_SOURCE_IAT)

        Prob_b_UP_DOWN = BiDiMorphing.buildProbMatrix(PMF_UP_DN_SOURCE, dim_col, dim_row)
        Prob_b_DOWN_UP = BiDiMorphing.buildProbMatrix(PMF_DN_UP_SOURCE, dim_col, dim_row)

        itr = 100
        stepSize = 0.001
        descent = False
        new_W_up_dn = BiDiMorphing.calcGradientDescentAscent(W_up_dn, X_up_dn, Prob_b_UP_DOWN, dim_col, dim_row, itr, stepSize, descent=False)
        new_W_dn_up = BiDiMorphing.calcGradientDescentAscent(W_dn_up, X_dn_up, Prob_b_DOWN_UP, dim_col, dim_row, itr, stepSize, descent=False)

        # these are adjusted pmfs of target (to be used in sampling)
        config.PMF_UP_DN_IAT = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_UP_DOWN_TARGET_IAT, new_W_up_dn)
        config.PMF_DN_UP_IAT = BiDiMorphing.buildPMF_with_W(BI_BURST_HISTO_DOWN_UP_TARGET_IAT, new_W_dn_up)


        #config.PMF_UP_DN_IAT = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_IAT)
        #config.PMF_DN_UP_IAT = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_IAT)

        config.CDF_UP_DN_IAT = BiDiMorphing.buildCDF(config.PMF_UP_DN_IAT)
        config.CDF_DN_UP_IAT = BiDiMorphing.buildCDF(config.PMF_DN_UP_IAT)

    @staticmethod
    def buildMetadataIAT(targetWebpage):
        BI_BURST_HISTO_UP_DOWN_IAT = {}
        BI_BURST_HISTO_DOWN_UP_IAT = {}
        for trace in targetWebpage.getTraces():
            burstsList = trace.getBurstsListWithPackets() # (direction, size, number, timeDiff, currBurstPackets)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                prevPktTime = None
                for packet in currBurst[4]: # loop over burst packets list
                    if prevPktTime == None:
                        prevPktTime = packet.getTime()
                        continue

                    currPktTime = packet.getTime()
                    IAT = currPktTime - prevPktTime # InterArrival Time

                    key = prevBurst[1] + "_" + str(IAT)

                    if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                        if not BI_BURST_HISTO_UP_DOWN_IAT.get(key):
                            BI_BURST_HISTO_UP_DOWN_IAT[key] = 0
                        BI_BURST_HISTO_UP_DOWN_IAT[key] += 1

                    elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                        if not BI_BURST_HISTO_DOWN_UP_IAT.get(key):
                            BI_BURST_HISTO_DOWN_UP_IAT[key] = 0
                        BI_BURST_HISTO_DOWN_UP_IAT[key] += 1

                    prevPktTime = currPktTime

                prevBurst = currBurst

        config.PMF_UP_DN_IAT = BiDiMorphing.buildPMF2(BI_BURST_HISTO_UP_DOWN_IAT)
        config.PMF_DN_UP_IAT = BiDiMorphing.buildPMF2(BI_BURST_HISTO_DOWN_UP_IAT)

        config.CDF_UP_DN_IAT = BiDiMorphing.buildCDF(config.PMF_UP_DN_IAT)
        config.CDF_DN_UP_IAT = BiDiMorphing.buildCDF(config.PMF_DN_UP_IAT)

    @staticmethod
    def buildMetadataIATold(targetWebpage):
        for trace in targetWebpage.getTraces():
            burstsList = trace.getBurstsListWithPackets() # (direction, size, number, timeDiff, currBurstPackets)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                prevPktTime = None
                for packet in currBurst[4]: # loop over burst packets list
                    if prevPktTime == None:
                        prevPktTime = packet.getTime()
                        continue

                    currPktTime = packet.getTime()
                    IAT = currPktTime - prevPktTime # InterArrival Time

                    key = prevBurst[1] + "_" + str(IAT)

                    if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                        if not config.BI_BURST_HISTO_UP_DOWN_IAT.get(key):
                            config.BI_BURST_HISTO_UP_DOWN_IAT[key] = 0
                        config.BI_BURST_HISTO_UP_DOWN_IAT[key] += 1

                    elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                        if not config.BI_BURST_HISTO_DOWN_UP_IAT.get(key):
                            config.BI_BURST_HISTO_DOWN_UP_IAT[key] = 0
                        config.BI_BURST_HISTO_DOWN_UP_IAT[key] += 1

                    prevPktTime = currPktTime

                prevBurst = currBurst

        config.PMF_UP_DN_IAT = BiDiMorphing.buildPMF2(config.BI_BURST_HISTO_UP_DOWN_IAT)
        config.PMF_DN_UP_IAT = BiDiMorphing.buildPMF2(config.BI_BURST_HISTO_DOWN_UP_IAT)

        config.CDF_UP_DN_IAT = BiDiMorphing.buildCDF(config.PMF_UP_DN_IAT)
        config.CDF_DN_UP_IAT = BiDiMorphing.buildCDF(config.PMF_DN_UP_IAT)

    @staticmethod
    def applyCountermeasure(trace, metadata):

        # one time call
        # BiDiMorphing.buildHistos(trace)
        #BiDiMorphing.buildHistos2(trace)
        #newTrace = trace


        # {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

        #pmf_up_dn = BiDiMorphing.buildPMF('biburst_up_down_histo_allSites')
        #pmf_dn_up = BiDiMorphing.buildPMF('biburst_down_up_histo_allSites')

        #pmf_up_dn = BiDiMorphing.buildPMF('biburst_up_down_histo_10sites')
        #pmf_dn_up = BiDiMorphing.buildPMF('biburst_down_up_histo_10sites')



        #pmf_up_dn = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_allSites')
        #pmf_dn_up = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_allSites')

        ##pmf_up_dn = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_10sites')
        ##pmf_dn_up = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_10sites')



        ##cdf_up_dn = BiDiMorphing.buildCDF(pmf_up_dn)
        ##cdf_dn_up = BiDiMorphing.buildCDF(pmf_dn_up)


        newTrace = Trace(trace.getId())

        burstsList = trace.getBurstsListWithPackets()

        # a new list of burst tuples (direction, size, number, timeDiff, burstPacketsList)
        # since this is assuming tor cells, burst size = burst number
        # direction of the new burst = direction of the old burst
        # size and number will be determined by the sampling from the up and down cdf
        # timeDiff is the same as we assume injecting dummy packets between real packets (real packets are sent
        # immediately)
        newBurstsList = []

        newBurstPacketList = []
        prevSampledBurstPacketList = []

        prevBurst = None
        currBurst = None

        for index, burst in enumerate(burstsList): # burst tuple (direction, size, number, timeDiff, burstPacketsList)
            if prevBurst == None:
                prevBurst = burst
                #newBurstsList.append(burst)
                #take the first burst as is
                for packet in burst[4]:
                    newTrace.addPacket(packet)
                continue

            currBurst = burst

            if index < len(burstsList) - 1:
                nextBurst = burstsList[index + 1]
            else:
                nextBurst = currBurst


            """ shihab """
            # sample the current burst based on the previous burst
            if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                 # if prev burst is up and curr burst is down, sample from cdf_up (go to cdf up)
                 #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_UP_DN )
                 #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_UP_DN, prevSampledBurstPacketList)
                 newBurstPacketList = BiDiMorphing.sampleWithIAT(currBurst, prevBurst, config.CDF_UP_DN, config.CDF_UP_DN_IAT, nextBurst)
                 #newBurstPacketList = BiDiMorphing.sampleWithIATandOppositeIAT(currBurst, prevBurst, config.CDF_UP_DN, config.CDF_UP_DN_IAT, config.CDF_DN_UP_IAT, nextBurst)
                 for packet in newBurstPacketList:
                     newTrace.addPacket(packet)

                 prevSampledBurstPacketList = newBurstPacketList
            elif int(prevBurst[0]) == 1 and int (currBurst[0]) == 0:
                 # if prev burst is down and curr burst is up, sample from cdf_dn (go to cdf dn)
                 #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_DN_UP)
                 #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_DN_UP, prevSampledBurstPacketList)
                 newBurstPacketList = BiDiMorphing.sampleWithIAT(currBurst, prevBurst, config.CDF_DN_UP, config.CDF_DN_UP_IAT, nextBurst)
                 #newBurstPacketList = BiDiMorphing.sampleWithIATandOppositeIAT(currBurst, prevBurst, config.CDF_DN_UP, config.CDF_DN_UP_IAT, config.CDF_UP_DN_IAT, nextBurst)
                 for packet in newBurstPacketList:
                     newTrace.addPacket(packet)

                 prevSampledBurstPacketList = newBurstPacketList

            prevBurst = currBurst


        # important think of how to continue sampling untile L1 distance is ... (like dts and tm)
        #[targeBursttDistributionBi] = config.metadata
        [targeBursttDistributionBi, targetBursts] = metadata


        '''
        # secondary sampling
        # no IAT sampling here as all the packets are not real
        # first prevBurst is the last burst sampled above
        while True:
            l1Distance = newTrace.calcBurstL1Distance(targeBursttDistributionBi)
            #print str(l1Distance)
            if l1Distance < 0.75:
                break

            # get the opposite direction of the previous burst: 0 --> 1 and 1 --> 0
            currDirection = 1 - int(prevBurst[0]) # prevBurst: burst tuple (direction, size, number, timeDiff, burstPacketsList)

            # sample the current burst based on the previous burst
            if int(prevBurst[0]) == 0:
                # if prev burst is up and curr burst is down, sample from cdf_up (go to cdf up)
                #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_UP_DN )
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_UP_DN, prevSampledBurstPacketList)
                newBurstPacketList = BiDiMorphing.sampleSecondary(currDirection, prevBurst, config.CDF_UP_DN)


            elif int(prevBurst[0]) == 1:
                # if prev burst is down and curr burst is up, sample from cdf_dn (go to cdf dn)
                #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_DN_UP)
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_DN_UP, prevSampledBurstPacketList)
                newBurstPacketList = BiDiMorphing.sampleSecondary(currDirection, prevBurst, config.CDF_DN_UP)

            for packet in newBurstPacketList:
                newTrace.addPacket(packet)

            prevBurst = BiDiMorphing.toBurstTuple(newBurstPacketList)

        '''


        # modify if using l1 (burstsList to be newTrace, lastSrcPktTime from newTrace, etc)
        # if needed, extra dummy burst padding
        rndmIndx = random.randint(0,len(targetBursts)-1) # pick a trace from target at random

        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # pick one trace randomly [(tuple), (tuple), ...]
        targetBurstsList = targetBursts[rndmIndx]

        srcLastBurstPktTimes = []
        for packet in currBurst[4]: # list of packets for the last burst to get the time of last burst
            srcLastBurstPktTimes.append(packet.getTime())

        lastSrcPktTime = srcLastBurstPktTimes[-1]  # for extra burst padding down below, before pop

        if len(targetBurstsList) > len(burstsList):
            pointer = len(burstsList)
            while pointer < len(targetBurstsList):
                targetBurstPackets = targetBurstsList[pointer][4] # list of packets for that burst
                for packet in targetBurstPackets:
                    #newTrace.addPacket(packet)
                    lastSrcPktTime += 1
                    newPkt = Packet(packet.getDirection(), lastSrcPktTime, packet.getLength())
                    newTrace.addPacket(newPkt)
                pointer += 1


        #print str(trace.getId())
        #print str(trace.getPacketCount())
        #print str(newTrace.getPacketCount())
        #print "--------------"
        return newTrace

    @staticmethod
    def applyCountermeasureOld(trace, metadata):

        # one time call
        # BiDiMorphing.buildHistos(trace)
        #BiDiMorphing.buildHistos2(trace)
        #newTrace = trace


        # {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

        #pmf_up_dn = BiDiMorphing.buildPMF('biburst_up_down_histo_allSites')
        #pmf_dn_up = BiDiMorphing.buildPMF('biburst_down_up_histo_allSites')

        #pmf_up_dn = BiDiMorphing.buildPMF('biburst_up_down_histo_10sites')
        #pmf_dn_up = BiDiMorphing.buildPMF('biburst_down_up_histo_10sites')



        #pmf_up_dn = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_allSites')
        #pmf_dn_up = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_allSites')

        ##pmf_up_dn = BiDiMorphing.buildPMF_Thr('biburst_up_down_histo_10sites')
        ##pmf_dn_up = BiDiMorphing.buildPMF_Thr('biburst_down_up_histo_10sites')



        ##cdf_up_dn = BiDiMorphing.buildCDF(pmf_up_dn)
        ##cdf_dn_up = BiDiMorphing.buildCDF(pmf_dn_up)


        newTrace = Trace(trace.getId())

        burstsList = trace.getBurstsListWithPackets()

        # a new list of burst tuples (direction, size, number, timeDiff, burstPacketsList)
        # since this is assuming tor cells, burst size = burst number
        # direction of the new burst = direction of the old burst
        # size and number will be determined by the sampling from the up and down cdf
        # timeDiff is the same as we assume injecting dummy packets between real packets (real packets are sent
        # immediately)
        newBurstsList = []

        newBurstPacketList = []
        prevSampledBurstPacketList = []

        prevBurst = None
        currBurst = None

        for index, burst in enumerate(burstsList): # burst tuple (direction, size, number, timeDiff, burstPacketsList)
            if prevBurst == None:
                prevBurst = burst
                #newBurstsList.append(burst)
                #take the first burst as is
                for packet in burst[4]:
                    newTrace.addPacket(packet)
                continue

            currBurst = burst

            if index < len(burstsList) - 1:
                nextBurst = burstsList[index + 1]
            else:
                nextBurst = currBurst

            # sample the current burst based on the previous burst
            if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                # if prev burst is up and curr burst is down, sample from cdf_up (go to cdf up)
                #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_UP_DN )
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_UP_DN, prevSampledBurstPacketList)
                newBurstPacketList = BiDiMorphing.sampleWithIAT(currBurst, prevBurst, config.CDF_UP_DN, config.CDF_UP_DN_IAT, nextBurst)
                for packet in newBurstPacketList:
                    newTrace.addPacket(packet)

                prevSampledBurstPacketList = newBurstPacketList
            elif int(prevBurst[0]) == 1 and int (currBurst[0]) == 0:
                # if prev burst is down and curr burst is up, sample from cdf_dn (go to cdf dn)
                #newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_DN_UP)
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_DN_UP, prevSampledBurstPacketList)
                newBurstPacketList = BiDiMorphing.sampleWithIAT(currBurst, prevBurst, config.CDF_DN_UP, config.CDF_DN_UP_IAT, nextBurst)
                for packet in newBurstPacketList:
                    newTrace.addPacket(packet)

                prevSampledBurstPacketList = newBurstPacketList

            prevBurst = currBurst


        # important think of how to continue sampling untile L1 distance is ... (like dts and tm)
        #[targeBursttDistributionBi] = config.metadata
        [targeBursttDistributionBi] = metadata

        #l1DistanceOrig = trace.calcBurstL1Distance(targeBursttDistributionBi)

        #l1Distance = newTrace.calcBurstL1Distance(targeBursttDistributionBi)

        #print str(l1DistanceOrig) + ', ' + str(l1Distance) # sample of output: 0.534030497571, 0.087438428983


        # secondary sampling
        '''
        while True:
            l1Distance = newTrace.calcBurstL1Distance(targeBursttDistributionBi)
            if l1Distance < 0.9:
                break

            # sample the current burst based on the previous burst
            if int(prevBurst[0]) == 0:
                # if prev burst is up and curr burst is down, sample from cdf_up (go to cdf up)
                newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_UP_DN )
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_UP_DN, prevSampledBurstPacketList)

            elif int(prevBurst[0]) == 1:
                # if prev burst is down and curr burst is up, sample from cdf_dn (go to cdf dn)
                newBurstPacketList = BiDiMorphing.sample(currBurst, prevBurst, config.CDF_DN_UP)
                #newBurstPacketList = BiDiMorphing.sample2(currBurst, prevBurst, config.CDF_DN_UP, prevSampledBurstPacketList)

            for packet in newBurstPacketList:
                newTrace.addPacket(packet)

            prevSampledBurstPacketList = newBurstPacketList

            prevBurst = currBurst
            currBurst = BiDiMorphing.toBurstTuple(newBurstPacketList)
        '''

        return newTrace

    @staticmethod
    def buildHistos(trace):
        burstsList = trace.getBurstsList() # # list of burst tuples (direction, size, number, timeDiff)

        prevBurst = None
        for burst in burstsList:
            if prevBurst == None:
                prevBurst = burst
                continue

            currBurst = burst

            key = prevBurst[1] + "_" + currBurst[1]

            if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                if not config.BI_BURST_HISTO_UP_DOWN.get(key):
                    config.BI_BURST_HISTO_UP_DOWN[key] = 0
                config.BI_BURST_HISTO_UP_DOWN[key] += 1

            elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                if not config.BI_BURST_HISTO_DOWN_UP.get(key):
                    config.BI_BURST_HISTO_DOWN_UP[key] = 0
                config.BI_BURST_HISTO_DOWN_UP[key] += 1

            prevBurst = currBurst

    @staticmethod
    def buildHistos2(trace):

        if trace.getId() % 10 == 0:
            burstsList = trace.getBurstsList() # # list of burst tuples (direction, size, number, timeDiff)

            prevBurst = None
            for burst in burstsList:
                if prevBurst == None:
                    prevBurst = burst
                    continue

                currBurst = burst

                key = prevBurst[1] + "_" + currBurst[1]

                if int(prevBurst[0]) == 0 and int(currBurst[0]) == 1:
                    if not config.BI_BURST_HISTO_UP_DOWN.get(key):
                        config.BI_BURST_HISTO_UP_DOWN[key] = 0
                    config.BI_BURST_HISTO_UP_DOWN[key] += 1

                elif int(prevBurst[0]) == 1 and int(currBurst[0]) == 0:
                    if not config.BI_BURST_HISTO_DOWN_UP.get(key):
                        config.BI_BURST_HISTO_DOWN_UP[key] = 0
                    config.BI_BURST_HISTO_DOWN_UP[key] += 1

                prevBurst = currBurst

    @staticmethod
    def buildPMF(histoFile):

        pmfCount = {} # {up, {down: count, ...}}
        fileLines = [line.strip() for line in open(histoFile)]
        for li in fileLines:
            lineList = li.split(' ') # 1_1 count
            upOrDown = lineList[0].split('_')
            upOrDownLeft = int(upOrDown[0])
            upOrDownRight = int(upOrDown[1])

            if not pmfCount.get(upOrDownLeft):
                pmfCount[upOrDownLeft] = {}

            pmfCount[upOrDownLeft][upOrDownRight] = int(lineList[1])

        pmfProb = {}

        for key1 in pmfCount.keys():
            pmfProb[key1] = {}
            sum = 0.0
            for key2 in pmfCount[key1].keys():
                sum += pmfCount[key1][key2]
            for key2 in pmfCount[key1].keys():
                pmfProb[key1][key2] = pmfCount[key1][key2] / sum

        return pmfProb # e.g. {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

    @staticmethod
    def buildPMF_Thr(histoFile, thr):

        pmfCount = {} # {up, {down: count, ...}}
        fileLines = [line.strip() for line in open(histoFile)]
        for li in fileLines:
            lineList = li.split(' ') # 1_1 count
            upOrDown = lineList[0].split('_')
            upOrDownLeft = int(upOrDown[0])
            upOrDownRight = int(upOrDown[1])

            if not pmfCount.get(upOrDownLeft):
                pmfCount[upOrDownLeft] = {}

            # threshold setting
            if upOrDownRight > thr:
                pmfCount[upOrDownLeft][upOrDownRight] = int(lineList[1])

        pmfProb = {}

        for key1 in pmfCount.keys():
            pmfProb[key1] = {}
            sum = 0.0
            for key2 in pmfCount[key1].keys():
                sum += pmfCount[key1][key2]
            for key2 in pmfCount[key1].keys():
                pmfProb[key1][key2] = pmfCount[key1][key2] / sum

        return pmfProb # e.g. {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}


    @staticmethod
    def buildPMF2(BI_BURST_HISTO):

        pmfCount = {} # {up, {down: count, ...}} hashmap of hashmaps
        #fileLines = [line.strip() for line in open(histoFile)]
        histoTuples = sorted(BI_BURST_HISTO.items(), key=lambda x: x[1], reverse=True)
        #histoTuples = BI_BURST_HISTO.items()

        for tu in histoTuples:
            # e.g., histoTuples = [('1_1', 4), ('3_4', 3), ('1_2', 2)]

            upOrDown = tu[0].split('_')
            upOrDownLeft = int(upOrDown[0])
            upOrDownRight = int(upOrDown[1])

            if not pmfCount.get(upOrDownLeft):
                pmfCount[upOrDownLeft] = {}

            pmfCount[upOrDownLeft][upOrDownRight] = int(tu[1])

        pmfProb = {}

        for key1 in pmfCount.keys():
            pmfProb[key1] = {}
            sum = 0.0
            for key2 in pmfCount[key1].keys():
                sum += pmfCount[key1][key2]
            for key2 in pmfCount[key1].keys():
                pmfProb[key1][key2] = pmfCount[key1][key2] / sum

        return pmfProb # e.g. {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

    @staticmethod
    def buildPMF_with_W(BI_BURST_HISTO, new_W_up_dn):

        pmfCount = {} # {up, {down: count, ...}} hashmap of hashmaps
        #fileLines = [line.strip() for line in open(histoFile)]
        histoTuples = sorted(BI_BURST_HISTO.items(), key=lambda x: x[1], reverse=True)
        #histoTuples = BI_BURST_HISTO.items()

        for tu in histoTuples:
            # e.g., histoTuples = [('1_1', 4), ('3_4', 3), ('1_2', 2)]

            upOrDown = tu[0].split('_')
            upOrDownLeft = int(upOrDown[0])
            upOrDownRight = int(upOrDown[1])

            if not pmfCount.get(upOrDownLeft):
                pmfCount[upOrDownLeft] = {}

            pmfCount[upOrDownLeft][upOrDownRight] = new_W_up_dn[upOrDownRight][upOrDownLeft] * int(tu[1])

        pmfProb = {}

        for key1 in pmfCount.keys():
            pmfProb[key1] = {}
            sum = 0.0
            for key2 in pmfCount[key1].keys():
                sum += pmfCount[key1][key2]
            for key2 in pmfCount[key1].keys():
                pmfProb[key1][key2] = pmfCount[key1][key2] / sum

        return pmfProb # e.g. {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}

    @staticmethod
    def buildCDF(pmf):

        cdf = {} # hashmap of hashmaps

        for key1 in pmf.keys():
            if len(pmf[key1].keys()) > 1: # discarding empty pmfs
                cdf[key1] = {}
                cumProb = 0.0
                for key2 in pmf[key1].keys():
                    cumProb += pmf[key1][key2]
                    cdf[key1][key2] = cumProb

        # e.g. pmf {1: {1: 0.4, 2: 0.6}, 2: {1: 0.3, 2: 0.4, 3: 0.3}, 3: {10: 0.1, 20: 0.3, 5: 0.2, 15: 0.4}}\
        # cdf      {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        return cdf




    @staticmethod
    def sample(currBurst, prevBurst, cdf):
        # cdf  {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        sizePkts = int(prevBurst[1])

        if not cdf.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdf.keys())

        cdf_ = cdf[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1)) # sorted tuples (by value)
                                                                       # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        #BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0,1)
        sampledSize = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledSize = int(tu[0])
                break

        if sampledSize == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        newBurstPacketList = []
        lastPacketTime = 0.0
        for pkt in currBurst[4]:
            lastPacketTime = pkt.getTime() # temp: to be used for new padded packets so burst time diff doesn't change
                                           # as we assume padding in the packet gaps without affecting real packets
            newBurstPacketList.append(pkt)

        #print 'sample size     = ' + str(sampledSize)
        #print 'curr burst size = ' + str(int(currBurst[1]))
        #print '----------------'


        i = 1
        currDirection = int(currBurst[0])
        while (sampledSize > int(currBurst[1])):
            lastPacketTime += 10
            newBurstPacketList.append(Packet(currDirection, lastPacketTime, 1)) # Packet(direction, time, size)
            sampledSize -= 1

            '''
            i += 1
            if i % 4 == 0:
                oppDirection = 0
                if currDirection == 0:
                    oppDirection = 1

                #print str(oppDirection) + ', ' + str(int(currBurst[0]))
                newBurstPacketList.append(Packet(oppDirection, lastPacketTime, 1)) # Packet(direction, time, size)
            '''


        #diff = sampledSize - int(currBurst[1])
        #if diff > 0 and diff <= 10:
        #    while diff > 0:
        #        newBurstPacketList.append(Packet(currDirection, lastPacketTime, 1))  # Packet(direction, time, size)
        #        diff -= 1

        return newBurstPacketList

    @staticmethod
    def sample2(currBurst, prevBurst, cdf, prevSampledBurstPacketList):
        # cdf  {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        #sizePkts = int(prevBurst[1])
        sizePkts = len(prevSampledBurstPacketList)

        if not cdf.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdf.keys())

        cdf_ = cdf[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1)) # sorted tuples (by value)
                                                                       # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        #BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0,1)
        sampledSize = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledSize = int(tu[0])
                break

        if sampledSize == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        newBurstPacketList = []
        lastPacketTime = 0.0
        for pkt in currBurst[4]:
            lastPacketTime = pkt.getTime() # temp: to be used for new padded packets so burst time diff doesn't change
                                           # as we assume padding in the packet gaps without affecting real packets
            newBurstPacketList.append(pkt)

        #print 'sample size     = ' + str(sampledSize)
        #print 'curr burst size = ' + str(int(currBurst[1]))
        #print '----------------'

        currDirection = int(currBurst[0])
        while (sampledSize > int(currBurst[1])):
            lastPacketTime += 10
            newBurstPacketList.append(Packet(currDirection, lastPacketTime, 1)) # Packet(direction, time, size)
            sampledSize -= 1

        #diff = sampledSize - int(currBurst[1])
        #if diff > 0 and diff <= 10:
        #    while diff > 0:
        #        newBurstPacketList.append(Packet(currDirection, lastPacketTime, 1))  # Packet(direction, size, time)
        #        diff -= 1

        return newBurstPacketList

    @staticmethod
    def sampleWithIAT(currBurst, prevBurst, cdf, cdfIAT, nextBurst):
        # cdf  {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        sizePkts = int(prevBurst[1])

        if not cdf.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdf.keys())

        cdf_ = cdf[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1)) # sorted tuples (by value)
                                                                       # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        #BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0,1)
        sampledSize = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledSize = int(tu[0])
                break

        if sampledSize == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        newBurstPacketList = []

        prevPktTime = None
        i = 1
        currDirection = int(currBurst[0])
        for pkt in currBurst[4]:
            if prevPktTime == None:
                prevPktTime = pkt.getTime()
                newBurstPacketList.append(pkt) # send/rcv first real packet
                continue

            currPktTime = pkt.getTime()
            actualIAT = currPktTime - prevPktTime
            sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

            while (sampledIAT < actualIAT) and (sampledSize > int(currBurst[1])):
                # send dummy packet
                time = prevPktTime + sampledIAT
                newBurstPacketList.append(Packet(currDirection, time, 1))  # Packet(direction, time, size)
                sampledSize -= 1
                actualIAT = currPktTime - time
                sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

                prevPktTime = time  # added 8/24/17

            newBurstPacketList.append(pkt) # send/rcv real packet
            prevPktTime = currPktTime

        # START. residuals, send/rcv until next burst FIRST real packet
        nextBurstFirstPacket = nextBurst[4][0]
        currPktTime = nextBurstFirstPacket.getTime()
        actualIAT = currPktTime - prevPktTime
        sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

        while (sampledIAT < actualIAT) and (sampledSize > int(currBurst[1])):
            # send dummy packet
            time = prevPktTime + sampledIAT
            newBurstPacketList.append(Packet(currDirection, time, 1))  # Packet(direction, time, size)
            sampledSize -= 1
            actualIAT = currPktTime - time
            sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

            prevPktTime = time  # added 8/24/17


        # END. residuals


        return newBurstPacketList

    @staticmethod
    def sampleWithIATandOppositeIAT(currBurst, prevBurst, cdf, cdfIAT, cdfIAT_Opposite, nextBurst):
        # cdf  {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        sizePkts = int(prevBurst[1])

        if not cdf.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdf.keys())

        cdf_ = cdf[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1)) # sorted tuples (by value)
                                                                       # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        #BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0,1)
        sampledSize = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledSize = int(tu[0])
                break

        if sampledSize == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        newBurstPacketList = []

        prevPktTime = None
        i = 1
        currDirection = int(currBurst[0])

        # opposite direction dummy
        oppDirection = (currDirection + 1) % 2
        oneOppDummyPkt = True

        for pkt in currBurst[4]:
            if prevPktTime == None:
                prevPktTime = pkt.getTime()
                newBurstPacketList.append(pkt) # send/rcv first real packet
                continue

            currPktTime = pkt.getTime()
            actualIAT = currPktTime - prevPktTime
            sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

            while (sampledIAT < actualIAT) and (sampledSize > int(currBurst[1])):
                # send dummy packet
                time = prevPktTime + sampledIAT
                newBurstPacketList.append(Packet(currDirection, time, 1))  # Packet(direction, time, size)

                # opposite direction dummy
                oppTime = time + 1
                if oppTime < currPktTime and oneOppDummyPkt == True:
                    print "adding dummy opp"
                    newBurstPacketList.append(Packet(oppDirection, oppTime, 1))
                    oneOppDummyPkt = False

                sampledSize -= 1
                actualIAT = currPktTime - time
                sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

                prevPktTime = time # added 8/24/17

            newBurstPacketList.append(pkt) # send/rcv real packet
            prevPktTime = currPktTime

        # START. residuals, send/rcv until next burst FIRST real packet
        nextBurstFirstPacket = nextBurst[4][0]
        currPktTime = nextBurstFirstPacket.getTime()
        actualIAT = currPktTime - prevPktTime
        sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

        while (sampledIAT < actualIAT) and (sampledSize > int(currBurst[1])):
            # send dummy packet
            time = prevPktTime + sampledIAT
            newBurstPacketList.append(Packet(currDirection, time, 1))  # Packet(direction, time, size)
            sampledSize -= 1
            actualIAT = currPktTime - time
            sampledIAT = BiDiMorphing.sampleIAT(currBurst, prevBurst, cdfIAT)

            prevPktTime = time  # added 8/24/17

        # END. residuals


        return newBurstPacketList

    @staticmethod
    def sampleIAT(currBurst, prevBurst, cdfIAT):
        # cdfIAT  {1: {1:   0.4, 2:   1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # where   {1: {IAT: 0.4, IAT: 1.0}, 2: ...}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        sizePkts = int(prevBurst[1])

        if not cdfIAT.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdfIAT.keys())

        cdf_ = cdfIAT[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1)) # sorted tuples (by value)
                                                                       # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        #BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0,1)
        sampledIAT = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledIAT = int(tu[0])
                break

        if sampledIAT == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        return sampledIAT

    @staticmethod
    def drawCDF(sorted_cdf_):
        # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]
        x = []
        x_ind = []
        y = []
        for i in range(0, len(sorted_cdf_)):
            print sorted_cdf_[i]
            x.append(sorted_cdf_[i][0])
            x_ind.append(i+1)
            y.append(sorted_cdf_[i][1])

        plt.plot(x_ind, y)

        plt.xlabel('index \n' + str(x))
        plt.ylabel('cumulative probability')

        plt.show()



    @staticmethod
    def toBurstTuple(newBurstPacketList):

        directionCursor = None
        dataCursor      = 0
        numberCursor    = 0
        timeCursor      = 0
        burstTimeRef    = newBurstPacketList[0].getTime()

        for packet in newBurstPacketList:
            if directionCursor == None:
                directionCursor = packet.getDirection()

            dataCursor += packet.getLength()
            numberCursor += 1
            timeCursor = packet.getTime() - burstTimeRef

        tuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor), newBurstPacketList)

        return tuple



    @staticmethod
    def getClosestTarget(webpageTrainSource, webpageIds, unMonitoredWebpageIdsObj, seed):
        if config.BUILD_ALL_WEBPAGES_HISTOS:
            for webpageId in webpageIds:
                if not unMonitoredWebpageIdsObj.__contains__(webpageId):
                    # monitored webpage so we take instances for training and testing as we do regularly
                    webpageTrain = Datastore.getWebpagesWangTor([webpageId], seed - config.NUM_TRAINING_TRACES, seed)
                    #webpageTest = Datastore.getWebpagesWangTor([webpageId], seed, seed + config.NUM_TESTING_TRACES)
                else:
                    # unmonitored so we take just one testing trace
                    # webpageTrain = Datastore.getDummyWebpages(webpageId)
                    webpageTest = Datastore.getWebpagesWangTor([webpageId], 1, 2)
                    webpageTrain = webpageTest  # just to overcome assigning targetWebpage for c8 and c9 defenses, but it will not be appended to the training set

                webpageTrain = webpageTrain[0]
                #histo = webpageTrain.getHistogram(None, True)
                histo = webpageTrain.getBurstHistogram(True)
                config.ALL_WEBPAGES_HISTOS[webpageTrain] = histo


            config.BUILD_ALL_WEBPAGES_HISTOS = False


        #l1Distance = sys.float_info.max
        l1Distance = sys.float_info.min
        closestTargetWebpage = None
        for targetWebpage in config.ALL_WEBPAGES_HISTOS:
            if targetWebpage.getId() != webpageTrainSource.getId():
                #currL1Distance = webpageTrainSource.calcL1Distance(config.ALL_WEBPAGES_HISTOS[targetWebpage])
                currL1Distance = webpageTrainSource.calcL1DistanceBurst(config.ALL_WEBPAGES_HISTOS[targetWebpage])
                #if l1Distance > currL1Distance:
                if l1Distance < currL1Distance:
                    l1Distance = currL1Distance
                    closestTargetWebpage = targetWebpage


        print "src id: " + str(webpageTrainSource.getId())
        print "closest target id: " + str(closestTargetWebpage.getId())
        print "---"
        return closestTargetWebpage


    @staticmethod
    def getRandomTarget(webpageTrainSource, webpageIds, unMonitoredWebpageIdsObj, seed):
        if config.BUILD_ALL_WEBPAGES:
            for webpageId in webpageIds:
                if not unMonitoredWebpageIdsObj.__contains__(webpageId):
                    # monitored webpage so we take instances for training and testing as we do regularly
                    webpageTrain = Datastore.getWebpagesWangTor([webpageId], seed - config.NUM_TRAINING_TRACES, seed)
                    # webpageTest = Datastore.getWebpagesWangTor([webpageId], seed, seed + config.NUM_TESTING_TRACES)
                else:
                    # unmonitored so we take just one testing trace
                    # webpageTrain = Datastore.getDummyWebpages(webpageId)
                    webpageTest = Datastore.getWebpagesWangTor([webpageId], 1, 2)
                    webpageTrain = webpageTest  # just to overcome assigning targetWebpage for c8 and c9 defenses, but it will not be appended to the training set

                webpageTrain = webpageTrain[0]
                config.ALL_WEBPAGES.append(webpageTrain)

            config.BUILD_ALL_WEBPAGES = False

        randPageId = random.choice(webpageIds)
        randTargetWebpage = None
        for targetWebpage in config.ALL_WEBPAGES:
            if targetWebpage.getId() == randPageId:
                randTargetWebpage = targetWebpage

        print "src id: " + str(webpageTrainSource.getId())
        print "random target id: " + str(randTargetWebpage.getId())
        print "---"
        return randTargetWebpage



    @staticmethod
    def getRandomTarget(webpageTrainSource, webpageIds, unMonitoredWebpageIdsObj, seed):
        if config.BUILD_ALL_WEBPAGES:
            for webpageId in webpageIds:
                if not unMonitoredWebpageIdsObj.__contains__(webpageId):
                    # monitored webpage so we take instances for training and testing as we do regularly
                    webpageTrain = Datastore.getWebpagesWangTor([webpageId], seed - config.NUM_TRAINING_TRACES, seed)
                    # webpageTest = Datastore.getWebpagesWangTor([webpageId], seed, seed + config.NUM_TESTING_TRACES)
                else:
                    # unmonitored so we take just one testing trace
                    # webpageTrain = Datastore.getDummyWebpages(webpageId)
                    webpageTest = Datastore.getWebpagesWangTor([webpageId], 1, 2)
                    webpageTrain = webpageTest  # just to overcome assigning targetWebpage for c8 and c9 defenses, but it will not be appended to the training set

                webpageTrain = webpageTrain[0]
                config.ALL_WEBPAGES.append(webpageTrain)

            config.BUILD_ALL_WEBPAGES = False

        randPageId = random.choice(webpageIds)
        randTargetWebpage = None
        for targetWebpage in config.ALL_WEBPAGES:
            if targetWebpage.getId() == randPageId:
                randTargetWebpage = targetWebpage

        print "src id: " + str(webpageTrainSource.getId())
        print "random target id: " + str(randTargetWebpage.getId())
        print "---"
        return randTargetWebpage


    @staticmethod
    def getLargestTarget(webpageTrainSource, webpageIds, unMonitoredWebpageIdsObj, seed):
        if config.BUILD_ALL_WEBPAGES:
            for webpageId in webpageIds:
                if not unMonitoredWebpageIdsObj.__contains__(webpageId):
                    # monitored webpage so we take instances for training and testing as we do regularly
                    webpageTrain = Datastore.getWebpagesWangTor([webpageId], seed - config.NUM_TRAINING_TRACES, seed)
                    # webpageTest = Datastore.getWebpagesWangTor([webpageId], seed, seed + config.NUM_TESTING_TRACES)
                else:
                    # unmonitored so we take just one testing trace
                    # webpageTrain = Datastore.getDummyWebpages(webpageId)
                    webpageTest = Datastore.getWebpagesWangTor([webpageId], 1, 2)
                    webpageTrain = webpageTest  # just to overcome assigning targetWebpage for c8 and c9 defenses, but it will not be appended to the training set

                webpageTrain = webpageTrain[0]
                config.ALL_WEBPAGES.append(webpageTrain)

            config.BUILD_ALL_WEBPAGES = False

            maxNumPackets = sys.float_info.min
            largestTargetWebpage = None
            for targetWebpage in config.ALL_WEBPAGES:
                bw = targetWebpage.getBandwidth()
                if bw > maxNumPackets: # for -d 5, BW is #packets
                    maxNumPackets = bw
                    largestTargetWebpage = targetWebpage

            config.LARGEST_WEBPAGE = largestTargetWebpage

        print "src id: " + str(webpageTrainSource.getId())
        print "largest target id: " + str(config.LARGEST_WEBPAGE.getId())
        print "---"
        return config.LARGEST_WEBPAGE


    @staticmethod
    def getLargestTargets(webpageTrainSource, webpageIds, unMonitoredWebpageIdsObj, seed):
        # this method combines several webpages as target
        # result: not good results :)
        if config.BUILD_ALL_WEBPAGES:
            for webpageId in webpageIds:
                if not unMonitoredWebpageIdsObj.__contains__(webpageId):
                    # monitored webpage so we take instances for training and testing as we do regularly
                    webpageTrain = Datastore.getWebpagesWangTor([webpageId], seed - config.NUM_TRAINING_TRACES, seed)
                    # webpageTest = Datastore.getWebpagesWangTor([webpageId], seed, seed + config.NUM_TESTING_TRACES)
                else:
                    # unmonitored so we take just one testing trace
                    # webpageTrain = Datastore.getDummyWebpages(webpageId)
                    webpageTest = Datastore.getWebpagesWangTor([webpageId], 1, 2)
                    webpageTrain = webpageTest  # just to overcome assigning targetWebpage for c8 and c9 defenses, but it will not be appended to the training set

                webpageTrain = webpageTrain[0]
                #config.ALL_WEBPAGES.append(webpageTrain)
                bw = webpageTrain.getBandwidth() # for the priority queue based on bandwidth (min)
                bw = -1 * bw # to make it a priority queue based on max
                heapq.heappush(config.ALL_WEBPAGES, (bw, webpageTrain)) # https://www.youtube.com/watch?v=l1JHLnKFHaQ

            config.BUILD_ALL_WEBPAGES = False

            config.LARGEST_WEBPAGE = Webpage(-1) # unique webpage ID
            #config.LARGEST_WEBPAGE = heapq.heappop(config.ALL_WEBPAGES) # (bw, webpage)
            #config.LARGEST_WEBPAGE = config.LARGEST_WEBPAGE[1]

            for i in range(0, 2): # largest websites
                webpage = heapq.heappop(config.ALL_WEBPAGES) # (bw, webpage)
                webpage = webpage[1] # get the webpage
                print "top webpage id: " + str(webpage.getId())
                print "top webpage bw: " + str(webpage.getBandwidth())
                for trace in webpage.getTraces():
                    config.LARGEST_WEBPAGE.addTrace(trace)

                print "Updated BW: " + str(config.LARGEST_WEBPAGE.getBandwidth())

        print "src id: " + str(webpageTrainSource.getId())
        print "target id : " + str(config.LARGEST_WEBPAGE.getId())
        print "BW: " + str(config.LARGEST_WEBPAGE.getBandwidth())
        print "---"
        return config.LARGEST_WEBPAGE


    @staticmethod
    def sampleSecondary(currDirection, prevBurst, cdf):
        # cdf  {1: {1: 0.4, 2: 1.0}, 2: {1: 0.3, 2: 0.7, 3: 1.0}, 3: {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}}
        # burst tuple (direction, size, number, timeDiff, burstPacketsList)
        sizePkts = int(prevBurst[1])

        if not cdf.keys().__contains__(sizePkts):
            sizePkts = random.choice(cdf.keys())

        cdf_ = cdf[sizePkts]  # go to the specific cdf for the prev burst {10: 0.1, 20: 0.4, 5: 0.6, 15: 1.0}
        sorted_cdf_ = sorted(cdf_.items(), key=operator.itemgetter(1))  # sorted tuples (by value)
        # [(10, 0.1), (20, 0.4), (5, 0.6), (15, 1.0)]

        # BiDiMorphing.drawCDF(sorted_cdf_)

        rnd = random.uniform(0, 1)
        sampledSize = None
        for tu in sorted_cdf_:
            # check values and pick the key with a value < rnd
            # pad curr burst to that burst
            if rnd <= tu[1]:
                sampledSize = int(tu[0])
                break

        if sampledSize == None:
            raise Exception("Sampling problem at BiDiMorphing.py")

        newBurstPacketList = []

        # prevBurst[4]: list of packets
        # [-1]: get last packet
        lastPacketTime = (prevBurst[4][-1]).getTime()

        while (sampledSize > 0):
            lastPacketTime += 1
            newBurstPacketList.append(Packet(currDirection, lastPacketTime, 1))  # Packet(direction, time, size)
            sampledSize -= 1

        return newBurstPacketList