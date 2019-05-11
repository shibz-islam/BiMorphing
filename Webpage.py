# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import random
from EventTrace import EventTrace
import math

class Webpage:
    def __init__( self, id ):
        self.__id = int(id)
        self.__traceSet = []

    def addTrace( self, trace ):
        self.__traceSet.append( trace )

    def getTrace( self, n ):
        return self.__traceSet[n]

    def getTraces( self ):
        return self.__traceSet

    def getId( self ):
        return self.__id

    def getBandwidth(self):
        totalBandwidth = 0
        try:
            for trace in self.getTraces():
                #if isinstance(trace, EventTrace):
                #    print 'Event trace: ' + str(trace.getTraceIndex())
                totalBandwidth += trace.getBandwidth()
        except:
            print 'Error getting bandwidth in webpage: ' + str(self.getId())
        return totalBandwidth

    def getHistogram( self, direction = None, normalize = False ):
        histogram    = {}
        totalPackets = 0
        for trace in self.getTraces():
            traceHistogram = trace.getHistogram( direction, False )
            for key in traceHistogram.keys():
                if not histogram.get( key ):
                    histogram[key] = 0
                histogram[key] += traceHistogram[key]
                totalPackets   += traceHistogram[key]

        if normalize:
            for key in histogram:
                histogram[key] = (histogram[key] * 1.0) / totalPackets
       
        return histogram


    def getBurstHistogram(self, normalize=False):
        histogram = {}
        totalBursts = 0
        for trace in self.getTraces():
            traceHistogram = trace.getBurstHistogram(False) # passing false as normalizing is done below
            for key in traceHistogram.keys():
                if not histogram.get(key):
                    histogram[key] = 0
                histogram[key] += traceHistogram[key]
                totalBursts += traceHistogram[key]

        if normalize==True:
            for key in histogram:
                histogram[key] = (histogram[key] * 1.0) / totalBursts


        return histogram


    def calcL1Distance(self, targetDistribution, filterDirection=None):
        localDistribution = self.getHistogram(filterDirection, True)

        keys = localDistribution.keys()
        for key in targetDistribution:
            if key not in keys:
                keys.append(key)

        distance = 0
        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None and r == None: continue
            if l == None: l = 0
            if r == None: r = 0

            distance += math.fabs(l - r)

        return distance

    def calcL1DistanceBurst(self, targetDistribution, filterDirection=None):
        #localDistribution = self.getHistogram(filterDirection, True)
        localDistribution = self.getBurstHistogram(True)

        keys = localDistribution.keys()
        for key in targetDistribution:
            if key not in keys:
                keys.append(key)

        distance = 0
        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None and r == None: continue
            if l == None: l = 0
            if r == None: r = 0

            distance += math.fabs(l - r)

        return distance