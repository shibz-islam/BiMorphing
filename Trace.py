# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

# Expanded by Khaled Al-Naami

import math
import config
from Packet import Packet
from Utils import Utils

class Trace:
    def __init__(self,id):
        self.__packetArray   = []
        self.__id            = id
        self.__histogramUp   = {}
        self.__histogramDown = {}
        self.__packetsUp     = 0
        self.__packetsDown   = 0
        self.__filePath      = None
        self.__year          = 0
        self.__month         = 0
        self.__day           = 0
        self.__hour          = 0

    def getId(self): return self.__id

    def setId(self,id):
        self.__id = id

    def getPacketCount( self, direction = None ):
        return len(self.getPackets(direction))

    def getPackets( self, direction = None ):
        retArray = []
        for packet in self.__packetArray:
            if direction == None or packet.getDirection() == direction:
                retArray.append( packet )
        return retArray

    def addPacket( self, packet ):
        # type: (object) -> object
        # Completely ignore ACK packet
        if config.IGNORE_ACK and packet.getLength() == Packet.HEADER_LENGTH:
            return self.__packetArray

        # Classifiers other than Bi-Di ignore Ack Feb 14, 2016 Khaled
        if not config.IGNORE_ACK and config.CLASSIFIER != config.ADVERSARIAL_CLASSIFIER_BiDirection_Only and packet.getLength() == Packet.HEADER_LENGTH:
            return self.__packetArray

        '''
        # No Ack for Wang (C 102), temp, to be removed
        if config.CLASSIFIER == config.TO_WANG_FILES_OPEN_WORLD and packet.getLength() == Packet.HEADER_LENGTH:
            #print "here"
            #print str(config.CLASSIFIER)
            return self.__packetArray
        '''

        key = str(packet.getDirection())+'-'+str(packet.getLength())

        if packet.getDirection()==Packet.UP:
            self.__packetsUp += 1
            if not self.__histogramUp.get( key ):
                self.__histogramUp[key] = 0
            self.__histogramUp[key] += 1
        elif packet.getDirection()==Packet.DOWN:
            self.__packetsDown += 1
            if not self.__histogramDown.get( key ):
                self.__histogramDown[key] = 0
            self.__histogramDown[key] += 1

        return self.__packetArray.append( packet )

    def getBandwidth( self, direction = None ):
        totalBandwidth = 0
        for packet in self.getPackets():
            if (direction == None or direction == packet.getDirection()) and packet.getLength() != Packet.HEADER_LENGTH:
                totalBandwidth += packet.getLength()

        return totalBandwidth

    def getTime( self, direction = None ):
        timeCursor = 0
        for packet in self.getPackets():
            if direction == None or direction == packet.getDirection():
                timeCursor = packet.getTime()

        return timeCursor

    def getHistogram( self, direction = None, normalize = False ):
        if direction == Packet.UP:
            histogram = dict(self.__histogramUp)
            totalPackets = self.__packetsUp
        elif direction == Packet.DOWN:
            histogram = dict(self.__histogramDown)
            totalPackets = self.__packetsDown
        else:
            histogram = dict(self.__histogramUp)
            for key in self.__histogramDown:
                histogram[key] = self.__histogramDown[key]
            totalPackets = self.__packetsDown + self.__packetsUp

        if normalize==True:
            for key in histogram:
                histogram[key] = (histogram[key] * 1.0) / totalPackets

        return histogram

    def calcL1Distance( self, targetDistribution, filterDirection=None ):
        localDistribution  = self.getHistogram( filterDirection, True )

        keys = localDistribution.keys()
        for key in targetDistribution:
            if key not in keys:
                keys.append( key )

        distance = 0
        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None and r == None: continue
            if l == None: l = 0
            if r == None: r = 0

            distance += math.fabs( l - r )

        return distance

    def getMostSkewedDimension( self, targetDistribution ):
        localDistribution  = self.getHistogram( None, True )

        keys = targetDistribution.keys()

        worstKey = None
        worstKeyDistance = 0

        for key in keys:
            l = localDistribution.get(key)
            r = targetDistribution.get(key)

            if l == None: l = 0
            if r == None: r = 0

            if worstKey==None or (r - l) > worstKeyDistance:
                worstKeyDistance = r - l
                worstKey = key

        bits = worstKey.split('-')

        return [int(bits[0]),int(bits[1])]

    def getBurstsList(self):
        directionCursor = None
        dataCursor      = 0
        numberCursor    = 0
        timeCursor      = 0
        burstTimeRef    = 0
        burstsList = [] # list of burst tuples (direction, size, number, timeDiff)

        for packet in self.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection()!=directionCursor:
                currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
                burstsList.append(currTuple)

                directionCursor = packet.getDirection()
                dataCursor      = 0
                numberCursor    = 0
                burstTimeRef    = packet.getTime() # time ref = new burst first packet time

            dataCursor += packet.getLength()
            numberCursor += 1
            timeCursor = packet.getTime() - burstTimeRef

        # last burst
        if dataCursor>0:
            currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
            burstsList.append(currTuple)


        return burstsList


    def getBurstsList2(self):
        directionCursor = None
        dataCursor = 0
        numberCursor = 0
        timeCursor = 0
        burstTimeRef = 0
        burstsList = []  # list of tuples (direction, size, number, timeDiff)

        #for packet in self.getPackets():
        #for packet in self.getNewPackets():
        #for packet in self.getNewPackets2():
        for packet in self.getNewPackets3():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection() != directionCursor:
                currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
                #if numberCursor > 1: # removing astray packets, consider last burst as well
                #    burstsList.append(currTuple)

                burstsList.append(currTuple)

                directionCursor = packet.getDirection()
                dataCursor = 0
                numberCursor = 0
                burstTimeRef = packet.getTime()  # time ref = new burst first packet time

            dataCursor += packet.getLength()
            numberCursor += 1
            timeCursor = packet.getTime() - burstTimeRef

        # last burst
        if dataCursor > 0:
            currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
            #if numberCursor > 1:  # # removing astray packets, consider last burst as well
            #    burstsList.append(currTuple)

            burstsList.append(currTuple)

        return burstsList

    # removing astray packets
    def getHistogram2(self, direction=None, normalize=False):
        histogramUp = {}
        histogramDown = {}
        packetsUp = 0
        packetsDown= 0
        directionCursor = None
        numberCursor = 0
        burstPackets = [] # list of burst pkts, where each pkt is a tuple of (dir, length, time)
        newTracePackets = [] # list of new trace pkts, where each pkt is a tuple of (dir, length, time)

        #for packet in self.getPackets(): # busrt = tuple (direction, size, number, timeDiff)
        #for packet in self.getNewPackets():
        #for packet in self.getNewPackets2():
        for packet in self.getNewPackets3():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection() != directionCursor:
                #if numberCursor > 1: # removing astray packets,
                #    for burstPacket in burstPackets:
                #        newTracePackets.append(burstPacket)

                for burstPacket in burstPackets:
                    newTracePackets.append(burstPacket)

                directionCursor = packet.getDirection()

                numberCursor = 0
                burstPackets = []

            numberCursor += 1
            pktTuple = (packet.getDirection(), packet.getLength(), packet.getTime())
            burstPackets.append(pktTuple)

        for pktTuple in newTracePackets:
            pktDir = pktTuple[0]
            pktLength = pktTuple[1]
            key = str(pktDir) + '-' + str(pktLength)

            if pktDir == Packet.UP:
                packetsUp += 1
                if not histogramUp.get(key):
                    histogramUp[key] = 0
                histogramUp[key] += 1
            elif pktDir == Packet.DOWN:
                packetsDown += 1
                if not histogramDown.get(key):
                    histogramDown[key] = 0
                histogramDown[key] += 1

        if direction == Packet.UP:
            histogram = dict(histogramUp)
            totalPackets = packetsUp
        elif direction == Packet.DOWN:
            histogram = dict(histogramDown)
            totalPackets = packetsDown
        else:
            histogram = dict(histogramUp)
            for key in histogramDown:
                histogram[key] = histogramDown[key]
            totalPackets = packetsDown + packetsUp

        if normalize == True:
            for key in histogram:
                histogram[key] = (histogram[key] * 1.0) / totalPackets

        return histogram


    def getNewPackets(self):

        directionCursor = None
        numberCursor = 0
        burstPackets = []  # list of burst pkts, where each pkt is a tuple of (dir, length, time)
        newTracePackets = []  # list of new trace pkts

        for packet in self.getPackets():  # busrt = tuple (direction, size, number, timeDiff)
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection() != directionCursor:
                if numberCursor > 1:  # removing astray packets,
                    for burstPacket in burstPackets:
                        newTracePackets.append(burstPacket)

                directionCursor = packet.getDirection()

                numberCursor = 0
                burstPackets = []

            numberCursor += 1
            pkt = Packet(packet.getDirection(), packet.getLength(), packet.getTime())
            burstPackets.append(pkt)


        return newTracePackets


    # Jul 10, 2017 --> remove pkts with high ranking interarrival times
    def getNewPackets2(self):

        fileLines = [line.strip() for line in open('interArrvlTimes')]
        fileList = []
        ctr = 0
        intArrTimesSelected = []
        for fileLine in fileLines:
            fileList.append(fileLine)
            t = fileLine.split(" ")[0]
            intArrTimesSelected.append(float(t))

            ctr += 1
            if ctr == 100: # first high ranking inter arrival times
                break


        prev = None
        direction = None
        newTracePackets = []  # list of new trace pkts

        for packet in self.getPackets():

            if prev == None or direction == None: # first packet
                prev = packet
                direction = packet.getDirection()
                continue

            curr = packet

            timeInt = float(curr.timeStr) - float(prev.timeStr)

            # if interarrival time is in from the high ranking ones, then don't add the current packet
            #if not timeInt in intArrTimesSelected:
            #if timeInt != 0:
            if timeInt <= 0.00064:
                newTracePackets.append(packet)

            prev = curr

        return newTracePackets


    # Jul 11, 2017 --> seperating uplink and downlink IAT
    def getNewPackets3(self):

        thr = 100

        '''
        fileLines = [line.strip() for line in open('interArrvlTimes')]
        fileList = []
        ctr = 0
        intArrTimesSelected = []

        # up and downlink
        for fileLine in fileLines:
            fileList.append(fileLine)
            t = fileLine.split(" ")[0]
            intArrTimesSelected.append(float(t))

            ctr += 1
            if ctr == thr:  # first high ranking inter arrival times
                break
        '''

        # uplink only
        fileLines = [line.strip() for line in open('interArrvlTimesUP_Protected')]
        fileList = []
        ctr = 0
        intArrTimesSelectedUP = []
        for fileLine in fileLines:
            fileList.append(fileLine)
            t = fileLine.split(" ")[0]
            intArrTimesSelectedUP.append(float(t))

            ctr += 1
            if ctr == thr:  # first high ranking inter arrival times
                break

        # downlink only
        fileLines = [line.strip() for line in open('interArrvlTimesDN_Protected')]
        fileList = []
        ctr = 0
        intArrTimesSelectedDN = []
        for fileLine in fileLines:
            fileList.append(fileLine)
            t = fileLine.split(" ")[0]
            intArrTimesSelectedDN.append(float(t))

            ctr += 1
            if ctr == thr:  # first high ranking inter arrival times
                break

        prevUp = None
        prevDn = None
        direction = None
        newTracePackets = []  # list of new trace pkts

        for packet in self.getPackets():

            if prevUp == None and packet.getDirection() == Packet.UP:  # first uplink packet
                prevUp = packet
                newTracePackets.append(packet) # Adaptive Padding sends always the first packet
                continue

            if prevDn == None and packet.getDirection() == Packet.DOWN:  # first downlink packet
                prevDn = packet
                newTracePackets.append(packet) # Adaptive Padding sends always the first packet
                continue

            if packet.getDirection() == Packet.UP:
                currUp = packet

                timeInt = float(currUp.timeStr) - float(prevUp.timeStr)

                # if interarrival time is in from the high ranking ones, then don't add the current packet
                #if not timeInt in intArrTimesSelectedUP:
                #if timeInt != 0:
                #if timeInt <= 0.00064:
                if timeInt < 0.02048:
                    newTracePackets.append(packet)

                prevUp = currUp
            else:
                currDn = packet

                timeInt = float(currDn.timeStr) - float(prevDn.timeStr)

                # if interarrival time is in from the high ranking ones, then don't add the current packet
                #if not timeInt in intArrTimesSelectedDN:
                #if timeInt != 0:
                #if timeInt <= 0.00064:
                if timeInt < 0.02048:
                    newTracePackets.append(packet)

                prevDn = currDn

        return newTracePackets

    # jul 25, 2017
    def getBurstsListWithPackets(self):
        directionCursor = None
        dataCursor = 0
        numberCursor = 0
        timeCursor = 0
        burstTimeRef = 0
        burstsList = []  # list of burst tuples (direction, size, number, timeDiff, currBurstPackets)
        currBurstPackets = []

        for packet in self.getPackets():
            if directionCursor == None:
                directionCursor = packet.getDirection()

            if packet.getDirection() != directionCursor:
                currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor),
                             currBurstPackets)
                burstsList.append(currTuple)

                directionCursor = packet.getDirection()
                dataCursor = 0
                numberCursor = 0
                burstTimeRef = packet.getTime()  # time ref = new burst first packet time
                currBurstPackets = []

            dataCursor += packet.getLength()
            numberCursor += 1
            timeCursor = packet.getTime() - burstTimeRef
            currBurstPackets.append(packet)

        # last burst
        if dataCursor > 0:
            currTuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor),
                         currBurstPackets)
            burstsList.append(currTuple)

        return burstsList



    def getBurstHistogram(self, normalize=False):
        instance = {}

        if self.getPacketCount() == 0:
            instance = {}
            return instance

        burstsList = self.getBurstsList()

        #if config.GLOVE_OPTIONS['packetSize'] == 1:
        #    instance = self.getHistogram()

        timeBase = 1
        sizeBase = config.bucket_Size

        if (config.DATA_SOURCE == 5): timeBase = config.bucket_Time  # works well with Wang Tor

        # uni burst
        for burst in burstsList:
            # tuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
            key = 'S' + burst[0] + '-' + \
                  str(Utils.roundArbitrary(
                      int(burst[1]), config.bucket_Size))

            if not instance.get(key):
                instance[key] = 0
            instance[key] += 1

        if normalize==True:
            for key in instance.keys():
                instance[key] = (instance[key] * 1.0) / len(burstsList)


        '''
        # uni burst
        for burst in burstsList:
            # tuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
            if config.GLOVE_OPTIONS['burstSize'] == 1:
                key = 'S' + burst[0] + '-' + \
                      str(Utils.roundArbitrary(
                          int(burst[1]), config.bucket_Size))

                if not instance.get(key):
                    instance[key] = 0
                instance[key] += 1

            if config.GLOVE_OPTIONS['burstNumber'] == 1:
                numberKey = 'N' + burst[0] + \
                            '-' + str(Utils.roundNumberMarker(
                    int(burst[2])))
                if not instance.get(numberKey):
                    instance[numberKey] = 0
                instance[numberKey] += 1

            if config.GLOVE_OPTIONS['burstTime'] == 1:
                timeKey = 'T' + burst[0] + '-' + str(
                    Utils.roundArbitrary(
                        int(burst[3]), timeBase))
                if not instance.get(timeKey):
                    instance[timeKey] = 0
                instance[timeKey] += 1

        # bi burst
        if len(burstsList) > 2:
            currBurst = burstsList[0]  # first tuple
            # nextBurst = burstsList[1]

            for burst in burstsList[1:]:
                # tuple = (str(directionCursor), str(dataCursor), str(numberCursor), str(timeCursor))
                nextBurst = burst
                # add features
                if config.GLOVE_OPTIONS['biBurstSize'] == 1:
                    biBurstDataKey = 'biSize-' + currBurst[0] + '-' + nextBurst[0] + '-' + \
                                     str(Utils.roundArbitrary(
                                         int(currBurst[1]), sizeBase)) + '-' + \
                                     str(Utils.roundArbitrary(
                                         int(nextBurst[1]), sizeBase))
                    if not instance.get(biBurstDataKey):
                        instance[biBurstDataKey] = 0
                    instance[biBurstDataKey] += 1

                # time
                if config.GLOVE_OPTIONS['biBurstTime'] == 1:
                    biBurstTimeKey = 'biTime-' + currBurst[0] + '-' + nextBurst[0] + '-' + \
                                     str(Utils.roundArbitrary(
                                         int(currBurst[3]), timeBase)) + '-' + \
                                     str(Utils.roundArbitrary(
                                         int(nextBurst[3]), timeBase))

                    if not instance.get(biBurstTimeKey):
                        instance[biBurstTimeKey] = 0
                    instance[biBurstTimeKey] += 1

                currBurst = nextBurst
                # nextBurst = nextNextBurst

        '''

        return instance


    def calcBurstL1Distance(self, targetDistribution):
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











