import os, math, random
from Trace import Trace
from Packet import Packet

DATASIZE = 750

class Tamaraw:

    @staticmethod
    def applyCountermeasure(trace):
        packets = []
        for packet in trace.getPackets():
            if packet.getDirection() == Packet.UP:
                x = 1 # outgoing
            else:
                x = -1 # incoming
            packets.append([packet.getTime(), x])

        list2 = []
        parameters = [""]

        Tamaraw.Anoa(packets, list2, parameters)
        list2 = sorted(list2, key=lambda list2: list2[0])

        list3 = []

        Tamaraw.AnoaPad(list2, list3, 100, 0)

        newTrace = Trace(trace.getId())
        for item in list3:
            pDirection = Packet.UP
            if (item[1] < 0):
                pDirection = Packet.DOWN
            newTrace.addPacket(Packet(direction=pDirection, time=item[0], length=abs(item[1])))

        return newTrace


    @staticmethod
    def fsign(num):
        if num > 0:
            return 0
        else:
            return 1

    @staticmethod
    def rsign(num):
        if num == 0:
            return 1
        else:
            return abs(num) / num

    @staticmethod
    def AnoaTime(parameters):
        direction = parameters[0]  # 0 out, 1 in
        method = parameters[1]
        if (method == 0):
            if direction == 0:
                return 0.02
            if direction == 1:
                return 0.006

    @staticmethod
    def AnoaPad(list1, list2, padL, method):
        lengths = [0, 0]
        times = [0, 0]
        for x in list1:
            if (x[1] > 0):
                lengths[0] += 1
                times[0] = x[0]
            else:
                lengths[1] += 1
                times[1] = x[0]
            list2.append(x)
        for j in range(0, 2):
            curtime = times[j]
            topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1)  # 1/2 1, 1/4 2, 1/8 3, ... #check this
            if (method == 0):
                topad = (lengths[j] / padL + topad) * padL
            while (lengths[j] < topad):
                curtime += Tamaraw.AnoaTime([j, 0])
                if j == 0:
                    list2.append([curtime, DATASIZE])
                else:
                    list2.append([curtime, -DATASIZE])
                lengths[j] += 1

    @staticmethod
    def Anoa(list1, list2, parameters):  # inputpacket, outputpacket, parameters
        # Does NOT do padding, because ambiguity set analysis.
        # list1 WILL be modified! if necessary rewrite to tempify list1.
        starttime = list1[0][0]
        times = [starttime, starttime]  # lastpostime, lastnegtime
        curtime = starttime
        lengths = [0, 0]
        datasize = DATASIZE
        method = 0
        if (method == 0):
            parameters[0] = "Constant packet rate: " + str(Tamaraw.AnoaTime([0, 0])) + ", " + str(Tamaraw.AnoaTime([1, 0])) + ". "
            parameters[0] += "Data size: " + str(datasize) + ". "
        if (method == 1):
            parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
            parameters[0] += "Tolerance: 2x."
        listind = 0  # marks the next packet to send
        while (listind < len(list1)):
            # decide which packet to send
            if times[0] + Tamaraw.AnoaTime([0, method, times[0] - starttime]) < times[1] + Tamaraw.AnoaTime(
                    [1, method, times[1] - starttime]):
                cursign = 0
            else:
                cursign = 1
            times[cursign] += Tamaraw.AnoaTime([cursign, method, times[cursign] - starttime])
            curtime = times[cursign]

            tosend = datasize
            while (list1[listind][0] <= curtime and Tamaraw.fsign(list1[listind][1]) == cursign and tosend > 0):
                if (tosend >= abs(list1[listind][1])):
                    tosend -= abs(list1[listind][1])
                    listind += 1
                else:
                    list1[listind][1] = (abs(list1[listind][1]) - tosend) * Tamaraw.rsign(list1[listind][1])
                    tosend = 0
                if (listind >= len(list1)):
                    break
            if cursign == 0:
                list2.append([curtime, datasize])
            else:
                list2.append([curtime, -datasize])
            lengths[cursign] += 1