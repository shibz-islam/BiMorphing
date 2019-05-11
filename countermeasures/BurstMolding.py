
import config
import random
import sys

from Trace import Trace
from Packet import Packet
from Datastore import Datastore

class BurstMolding:

    @staticmethod
    def buildMetadata(webpageTrain, targetWebpage):

        burstsList = [] # [ [(tuple), (tuple), ...],
        #                   [(tuple), (tuple), ...],
        #                   ...]
        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # [(tuple), (tuple), ...] is one trace

        # open-world, get non_monitored as target
        if config.NUM_NON_MONITORED_SITES != -1:
            targetWebpage = BurstMolding.getNonMonitoredTarget(webpageTrain)

        for trace in targetWebpage.getTraces():
            burstsList.append(trace.getBurstsListWithPackets())


        return [burstsList]

    @staticmethod
    def buildMetadataOld(webpageTrain, targetWebpage):

        burstsList = [] # [ [(tuple), (tuple), ...],
        #                   [(tuple), (tuple), ...],
        #                   ...]
        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # [(tuple), (tuple), ...] is one trace

        for trace in targetWebpage.getTraces():
            burstsList.append(trace.getBurstsListWithPackets())


        return [burstsList]



    @staticmethod
    def applyCountermeasure(trace, metadata):
        [targetBursts] = metadata

        rndmIndx = random.randint(0, len(targetBursts) - 1)  # pick a trace from target at random

        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # pick one trace randomly [(tuple), (tuple), ...]
        targetBurstsList = targetBursts[rndmIndx]

        sourceBurstsList = trace.getBurstsListWithPackets()

        newTrace = Trace(trace.getId())

        pointer = 0  # to loop over burst tuples in src and trgt in order until one is exhausted

        # first packet time
        newPktTime = sourceBurstsList[pointer][4][0].getTime()
        lastSrcPktTime = 0

        while True:
            if pointer >= len(sourceBurstsList) or pointer >= len(targetBurstsList):
                break

            sourceBurstPackets = sourceBurstsList[pointer][4]  # list of packets for that burst
            targetBurstPackets = targetBurstsList[pointer][4]

            srcPktTimes = []
            for packet in sourceBurstPackets:
                srcPktTimes.append(packet.getTime())

            lastSrcPktTime = srcPktTimes[-1]  # for extra burst padding down below, before pop

            if len(sourceBurstPackets) < len(targetBurstPackets):
                for packet in targetBurstPackets:
                    #-#if len(srcPktTimes) != 0:
                        #-#newPktTime = srcPktTimes.pop(0)  # remove first from the queue
                        # eventually last src packet time is repeated
                        # to make sure we don't exceed original burst transimission time

                    newPktTime += 1
                    newPkt = Packet(packet.getDirection(), newPktTime, packet.getLength())
                    newTrace.addPacket(newPkt)
            else:
                for packet in sourceBurstPackets:
                    newPktTime += 1
                    newPkt = Packet(packet.getDirection(), newPktTime, packet.getLength())
                    newTrace.addPacket(newPkt)

            pointer += 1

        # for calculating time overhead
        config.TIME_ORIGINAL += lastSrcPktTime # last
        config.TIME_EXTRA += newPktTime

        #print 'source id:' + str(trace.getId())
        #print 'orig. end time:' + str(config.TIME_ORIGINAL)
        #print 'extra end time:' + str(config.TIME_EXTRA)
        #print "---------"
        # end: for calculating time overhead

        while pointer < len(sourceBurstsList):
            sourceBurstPackets = sourceBurstsList[pointer][4]
            for packet in sourceBurstPackets:
                newPktTime += 1
                newPkt = Packet(packet.getDirection(), newPktTime, packet.getLength())
                newTrace.addPacket(newPkt)
            pointer += 1

        # extra bursts tail
        # '''
        while pointer < len(targetBurstsList):
            targetBurstPackets = targetBurstsList[pointer][4]
            for packet in targetBurstPackets:
                lastSrcPktTime += 1
                newPkt = Packet(packet.getDirection(), lastSrcPktTime, packet.getLength())
                newTrace.addPacket(newPkt)
            pointer += 1
        # '''

        #print str(trace.getId())
        #print str(trace.getPacketCount())
        #print str(newTrace.getPacketCount())
        #print "--------------"
        return newTrace


    @staticmethod
    def applyCountermeasureOld(trace, metadata):
        [targetBursts] = metadata

        rndmIndx = random.randint(0, len(targetBursts) - 1)  # pick a trace from target at random

        # burst tuple = (direction, size, number, timeDiff, burstPackets)
        # pick one trace randomly [(tuple), (tuple), ...]
        targetBurstsList = targetBursts[rndmIndx]

        sourceBurstsList = trace.getBurstsListWithPackets()

        newTrace = Trace(trace.getId())

        pointer = 0  # to loop over burst tuples in src and trgt in order until one is exhausted

        while True:
            if pointer >= len(sourceBurstsList) or pointer >= len(targetBurstsList):
                break

            sourceBurstPackets = sourceBurstsList[pointer][4]  # list of packets for that burst
            targetBurstPackets = targetBurstsList[pointer][4]

            srcPktTimes = []
            for packet in sourceBurstPackets:
                srcPktTimes.append(packet.getTime())

            lastSrcPktTime = srcPktTimes[-1]  # for extra burst padding down below, before pop

            if len(sourceBurstPackets) < len(targetBurstPackets):
                for packet in targetBurstPackets:
                    if len(srcPktTimes) != 0:
                        newPktTime = srcPktTimes.pop(0)  # remove first from the queue
                        # eventually last src packet time is repeated
                        # to make sure we don't exceed original burst transimission time

                    newPkt = Packet(packet.getDirection(), newPktTime, packet.getLength())
                    newTrace.addPacket(newPkt)
            else:
                for packet in sourceBurstPackets:
                    newTrace.addPacket(packet)

            pointer += 1

        while pointer < len(sourceBurstsList):
            sourceBurstPackets = sourceBurstsList[pointer][4]
            for packet in sourceBurstPackets:
                newTrace.addPacket(packet)
            pointer += 1

        # extra bursts tail
        # '''
        while pointer < len(targetBurstsList):
            targetBurstPackets = targetBurstsList[pointer][4]
            for packet in targetBurstPackets:
                lastSrcPktTime += 1
                newPkt = Packet(packet.getDirection(), lastSrcPktTime, packet.getLength())
                newTrace.addPacket(newPkt)
            pointer += 1
        # '''

        print str(trace.getId())
        print str(trace.getPacketCount())
        print str(newTrace.getPacketCount())
        print "--------------"
        return newTrace


    @staticmethod
    def getNonMonitoredTarget(webpageTrainSource):

        if config.BUILD_ALL_WEBPAGES:
            webpageTrain = Datastore.getWebpagesWangTor([101], 1, 2)

            webpageTrain = webpageTrain[0]
            config.ALL_WEBPAGES.append(webpageTrain)

            config.BUILD_ALL_WEBPAGES = False

            config.LARGEST_WEBPAGE = webpageTrain

        #print "src id: " + str(webpageTrainSource.getId())
        #print "largest target id: " + str(config.LARGEST_WEBPAGE.getId())
        #print "---"
        return config.LARGEST_WEBPAGE
