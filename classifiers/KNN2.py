"""
Attack based on the following paper "Effective Attacks and Provable Defenses for Website Fingerprinting" by T. Wang et al.
"""
import arffWriter
import wekaAPI
import config
from Utils import Utils
import numpy
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sys import stdout
import os

OPEN_WORLD = True
CV = True

class kNN():
    """
    k Neirest Neighbour classifier described in the paper outlined above.
    Essentially it calculates the distance of different instances by using a set of weights described below:
    $$ d(P, P') = \sum_{1 \leq i \leq |F|} w_i |f_i(P) - f_i(P')| $$
    Attributes:
        - is_multiclass does not make a difference. It is just part of the interface used for models
        - K_CLOSEST_NEIGHBORS is the amount of neighbours it uses for a majority vote
        - weights is a list of `AMOUNT_FEATURES` length that is used to signify how 'important' features are
        - K_RECO is the number of closest neighbours used for weight learning
    """

    def __init__(self, is_multiclass=True, K_CLOSEST_NEIGHBORS=2):
        # Constants
        self.K_RECO = 5.0 # Num of neighbors for weight learning
        self.K_CLOSEST_NEIGHBORS = K_CLOSEST_NEIGHBORS

        self.weights = None

        self.kNN_finder = NearestNeighbors(
            n_neighbors=K_CLOSEST_NEIGHBORS,
            metric=self._calculate_dist,
            metric_params=None, # Dict otherwise
            n_jobs=3
        )

    def _init_weights(self):
        """
        Randomly assign the weights a value between 0.5 and 1.5
        """
        from random import uniform

        self.weights = [uniform(0.5, 1.5) for _ in range(self.AMOUNT_FEATURES)]

    def _calculate_dist(self, point1, point2):
        """
        Calculates the distance between 2 points as follows:
        $$ d(P, P') = \sum_{1 \leq i \leq |F|} w_i |f_i(P) - f_i(P')| $$
        @param point1, point2 are two lists, with each item in the list being a feature.
        """
        if isinstance(point1, dict):
            point1 = point1['point']

        if isinstance(point2, dict):
            point2 = point2['point']

        # print self.AMOUNT_FEATURES
        # print len(self.weights)
        # print point1
        # print point2

        dist = 0
        for i, f1 in enumerate(point1):
            # print i,f1
            f2 = point2[i]
            w = self.weights[i]

            if f1 != -1 and f2 != -1:
                dist += w * abs(f1 - f2)

        return dist

    def _calculate_all_dist_for_feature(self, point, data, feature_index, index=-1):
        """
        Calculates the distance to all other points for a specific feature
        @param point is the point from where we are calculating the distance
        @param data
        @param feature_index is the index of the feature we are examining
        @param index is the index of the point. So we can ignore it (since the distance would be 0 anyway)
        @return a list of distances
        """
        distances = []
        for i, row in enumerate(data):
            if i == index:
                distances.append(float("inf"))

            else:
                dist = self.weights[index] * abs(row['point'][feature_index] - point[feature_index])
                distances.append(dist)

        return distances

    def _find_closest_reco_points(self, label, data):
        """
        Find the `K_RECO`-closest members that either have the same label and the ones that have another
        @param label is the label of the point we are examining
        @param data is a list as follows: [{'point': data, 'distance': distance, 'label': label}]
        @return a tuple of (same_label, different_label) where each a list of length K_RECO
        """
        # Sort on distance
        data = sorted(data, key=lambda x: x['distance'])

        same_label, different_label = [], []

        for val in data:
            if len(same_label) == self.K_RECO and len(different_label) == self.K_RECO:
                break

            if val['label'] == label and len(same_label) < self.K_RECO:
                same_label.append(val)

            elif val['label'] != label and len(different_label) < self.K_RECO:
                different_label.append(val)

        return (same_label, different_label)

    def _get_point_baddness(self, same_label, different_label):
        """
        Calculates the fraction of different label points that are closer than the maximum
        in the `same_label` list.
        @param same_label is a list of the `K_RECO` closest points to the current point we are examining with the same label as that point
        @param different_label is a list of the `K_RECO` closest points to the current point we are examining with a different label as that point
        @return a measure of how bad the feature is
        """
        max_good = max(same_label, key=lambda x: x['distance'])
        point_badness = len([x for x in different_label if x['distance'] < max_good['distance']])

        point_badness /= self.K_RECO # Calculate fraction

        return point_badness

    def _update_weights(self, point_badness, features_badness):
        """
        The features_badness gives us a measure of how bad certain features are.
        For every `i, w in enumerate(self.weights)` (except for the weight with the minimum feature badness), we decrease the weight by `0.01 * w`
        Then we increase all the weights by `min(features_badness)`
        See paper for extra steps
        @param point_badness is the general measure of how bad a point is classified
        @param features_badness is a measure of how bad a measure is
        """
        min_badness = min(features_badness)

        # Make all weights smaller
        for i, w in enumerate(self.weights):

            # Skip the minimum
            if features_badness[i] == min_badness:
                continue

            subtract = w * 0.01 * (features_badness[i] / self.K_RECO) * (0.2 + (point_badness / self.K_RECO))

            self.weights[i] -= subtract

        # Increase weights to maintain $d(P_{train}, S_{bad})$
        for i, w in enumerate(self.weights):
            self.weights[i] += min_badness

    def _learn_weights(self, data, labels):
        """
        In the training process, learns a set of weights
        @param data is a 2D matrix with the data samples and features
        @param labels is a 1D list where `class_of(data[i]) == labels[i]` for all i in `range(len(data))`
        """
        print ("Learning Weights...")
        training = min(len(data), 8000)

        for i in range(training):
            try:
                # update_progess(training, i)
                print ("Training instance: ", i)

                row = data[i]
                distances, indexes = self.kNN_finder.kneighbors([row], n_neighbors=int(self.K_RECO * 3), return_distance=True)
                new_data = [{'point': data[y], 'distance': x, 'label': labels[y]} for (x, y) in zip(distances[0], indexes[0])]

                same_label, different_label = self._find_closest_reco_points(labels[i], new_data)

                features_badness = []

                # Go over all features
                for j, feature in enumerate(row):
                    diff_label_distances = self._calculate_all_dist_for_feature(row, different_label, j)
                    same_label_distances = self._calculate_all_dist_for_feature(row, same_label, j)

                    diff_label_distances = [{'distance': x} for x in diff_label_distances]
                    same_label_distances = [{'distance': x} for x in same_label_distances]

                    point_badness_feature = self._get_point_baddness(same_label_distances, diff_label_distances)

                    features_badness.append(point_badness_feature)

                point_badness = self._get_point_baddness(same_label, different_label)
                self._update_weights(point_badness, features_badness)
            except:
                import pdb; pdb.set_trace()

    def _majority_vote(self, points):
        #modified on 27 Mar, 2018
        votes = {}
        for point in points:
            if point['label'] not in votes:
                votes[point['label']] = 0
            votes[point['label']] += 1

        if OPEN_WORLD == False:
            #closed-world: return the class with majority of votes
            className = max(votes, key=votes.get)
        else:
            # open-world: return 'monitored' class if all the votes are 'monitored', else return 'non-monitored' class
            if len(votes.keys()) == 1:
                className = config.binaryLabels[0]  # webpageMon
            else:
                className = config.binaryLabels[1]  # webpageNonMon

        return className


    def fit(self, X, y):
        """
        Trains the model
        """

        self.AMOUNT_FEATURES = len(X[0])
        self._init_weights()
        # print X
        # print y
        self.kNN_finder.fit(X, y)

        self._learn_weights(X, y)

        self.data = [{'point': X[i], 'label': y[i]} for i in range(len(X))]


    def predict(self, X):
        """
        Predicts using a majority vote
        @param X is an array-like object
        """
        print ("Started Prediction...")
        if self.data is None or len(self.data) == 0:
            raise Exception("Train model first!")

        elif len(X) == 0:
            raise Exception("Cannot predict on empty array")

        elif len(X[0]) != self.AMOUNT_FEATURES:
            raise Exception("Does not match the shape with {} features".format(self.AMOUNT_FEATURES))

        else:
            predicted = []
            distances, indexes = self.kNN_finder.kneighbors(X, n_neighbors=self.K_CLOSEST_NEIGHBORS, return_distance=True)
            count = 0
            for distance, index in zip(distances, indexes):
                count += 1
                print ("Test instance: ",count)
                points = [{'distance': x, 'label': self.data[y]['label']} for (x, y) in zip(distance, index)]
                points = sorted(points, key=lambda x: x['distance'])

                predicted.append(self._majority_vote(points))

            return predicted

    @staticmethod
    def traceToInstance(trace):
        instance = {}

        if trace.getPacketCount() == 0:
            instance = {}
            instance['class'] = 'webpage' + str(trace.getId())
            return instance

        ## General Features
        # transmissionSize = 0
        # numIncoming = 0
        # numOutgoing = 0
        # for packet in trace.getPackets():
        #     transmissionSize += packet.getLength()
        #     if packet.getDirection() == packet.UP:
        #         numOutgoing += 1
        #     else:
        #         numIncoming += 1
        # instance['transmissionSize'] = transmissionSize
        # instance['numIncoming'] = numIncoming
        # instance['numOutgoing'] = numOutgoing
        # #warning: total transmission time missing
        #
        # ## Unique Packet Lengths
        # packet_length_range = (1, 1500)
        # for size in range(1, 1500):
        #     keyOut = 'S' + str(size) + '-' + 'D' + str(Packet.UP)
        #     keyIn = 'S' + str(size) + '-' + 'D' + str(Packet.DOWN)
        #     instance[keyOut] = 0
        #     instance[keyIn] = 0
        #
        # for packet in trace.getPackets():
        #     key = 'S' + str(packet.getLength()) + '-' + 'D' + str(packet.getDirection())
        #     if instance.get(key):
        #         instance[key] = 1

        times = []
        sizes = []
        features = []
        padValue = -1

        for packet in trace.getPackets():
            direction = 0
            if packet.getDirection() == packet.UP:
                direction = 1
            else:
                direction = -1
            sizes.append(packet.getLength() * direction)
            times.append(packet.getTime())
        '''
        General Features
        '''
        features.append(len(sizes)) # Transmission size features

        count = 0
        for x in sizes:
            if x > 0:
                count += 1
        features.append(count)  # numIncoming
        features.append(len(times) - count)  # numOutgoing

        features.append(times[-1] - times[0])  # total transmission time

        '''
        Unique packet lengths
        Note: No need for Tor Dataset
        '''
        # Unique packet lengths
        ##    for i in range(-1500, 1501):
        ##        if i in sizes:
        ##            features.append(1)
        ##        else:
        ##            features.append(0)

        '''
        Packet Ordering
        '''
        # Number of packets before it in the sequence
        count = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                features.append(i)
            if count == 500:
                break
        for i in range(count, 500):
            features.append(padValue)

        # Number of incoming packets between outgoing packets
        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                features.append(i - prevloc)
                prevloc = i
            if count == 500:
                break
        for i in range(count, 500):
            features.append(padValue)

        '''
        Concentration of outgoing packets
        '''
        count = 0
        for i in range(0, min(len(sizes), 3000)):
            if i % 30 != 29:
                if sizes[i] > 0:
                    count += 1
            else:
                features.append(count)
                count = 0
        for i in range(len(sizes) / 30, 100):
            features.append(padValue)

        '''
        Bursts
        '''
        bursts = []
        curburst = 0
        stopped = 0
        for x in sizes:
            if x < 0:
                curburst -= x
            if x > 0:
                if len(bursts) > 0:
                    if bursts[-1] != curburst:
                        bursts.append(curburst)
                else:
                    bursts.append(curburst)

        # Maximum burst length, Mean burst length, Number of bursts
        if (len(bursts) > 0):
            features.append(max(bursts))
            features.append(numpy.mean(bursts))
            features.append(len(bursts))
        else:
            features.append(padValue)
            features.append(padValue)
            features.append(padValue)

        # Amount of bursts greater than 2, 5, 10, 15, 20, 50
        counts = [0, 0, 0, 0, 0, 0]
        for x in bursts:
            if x > 2:
                counts[0] += 1
            if x > 5:
                counts[1] += 1
            if x > 10:
                counts[2] += 1
            if x > 15:
                counts[3] += 1
            if x > 20:
                counts[4] += 1
            if x > 50:
                counts[5] += 1
        features.append(counts[0])
        features.append(counts[1])
        features.append(counts[2])
        features.append(counts[3])
        features.append(counts[4])
        features.append(counts[5])

        # Lengths of the first 100 bursts
        for i in range(0, 100):
            try:
                features.append(bursts[i])
            except:
                features.append(padValue)

        '''
        Initial packets
        '''
        for i in range(0, 20):
            try:
                features.append(sizes[i] + 1500)
            except:
                features.append(padValue)


        # itimes = [0] * (len(sizes) - 1)
        # for i in range(1, len(sizes)):
        #     itimes[i - 1] = times[i] - times[i - 1]
        # features.append(numpy.mean(itimes))
        # features.append(numpy.std(itimes))

        # print "Total features: " + str(len(features))

        '''
        Making a dictionary
        '''
        for i in range(len(features)):
            instance[i + 1] = features[i]
        instance['class'] = 'webpage' + str(trace.getId())

        print "instance: ", instance['class'], len(instance)

        return instance

    @staticmethod
    def classify(runID, trainingFile, testingFile):
        trainList = wekaAPI.readFile(trainingFile)
        testList = wekaAPI.readFile(testingFile)
        trainInstancesList = []
        testInstancesList = []
        classes = ""
        yTrain = []
        yTest = []

        for line in trainList:
            if line[0] == '@':
                if line.lower().startswith("@attribute class"):
                    classes = line.split(" ")[2]
            else:
                # instancesList.append(float(line.split(",")[:-1]))
                trainInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTrain.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                testInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTest.append(line.split(",")[-1])

        XTr = numpy.array(trainInstancesList)
        yTr = numpy.array(yTrain)
        XTe = numpy.array(testInstancesList)
        yTe = numpy.array(yTest)

        # print ("Scaling data...")
        # scaler = StandardScaler()
        # XTr = scaler.fit_transform(XTr)
        # XTe = scaler.fit_transform(XTe)
        if config.NUM_MONITORED_SITES == -1 and config.NUM_NON_MONITORED_SITES == -1:
            print "Closed-world"
        else:
            print "Open-world"

        print ("Classification...")
        classifier = kNN()

        classifier.fit(XTr, yTr)
        prediction = classifier.predict(XTe)

        totalPredictions = 0
        totalCorrectPredictions = 0
        debugInfo = []
        for i in range(0, len(yTe)):
            actualClass = yTe[i]
            predictedClass = prediction[i]
            probEstimate = 'NA'
            # debugInfo.append([actualClass,predictedClass])
            debugInfo.append([actualClass, predictedClass, probEstimate])
            totalPredictions += 1.0
            if actualClass == predictedClass:
                totalCorrectPredictions += 1.0

        accuracy = totalCorrectPredictions / totalPredictions * 100.0
        print ("Accuracy = ", accuracy)

        return [accuracy, debugInfo]


def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    print (current/total) * 100
    # stdout.write("{:.2f} %\r".format((current/total) * 100))
    # stdout.flush()


# Tests that have nothing to do with website fingerprinting
if __name__ == '__main__':
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import load_iris
    from functools import reduce

    # filename = "datafile-oypkntswk100.c0.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000"
    # filename = "datafile-uo4nmw0yk100.c100.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000"
    filename = "datafile-ytacbzg4k100.c200.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000"
    trainingFile = os.path.join("../", config.CACHE_DIR, filename + '-train.arff')
    testingFile = os.path.join("../", config.CACHE_DIR, filename + '-test.arff')

    trainInstancesList = []
    testInstancesList = []
    classes = ""
    yTrain = []
    yTest = []


    if CV == False:
        trainList = wekaAPI.readFile(trainingFile)
        testList = wekaAPI.readFile(testingFile)
        for line in trainList:
            if line[0] == '@':
                if line.lower().startswith("@attribute class"):
                    classes = line.split(" ")[2]
            else:
                # instancesList.append(float(line.split(",")[:-1]))
                trainInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTrain.append(line.split(",")[-1])

        for line in testList:
            if line[0] != '@':
                testInstancesList.append([float(i) for i in line.split(",")[:-1]])
                yTest.append(line.split(",")[-1])
    else:
        trainFileLines = [line.strip() for line in open(trainingFile)]
        testFileLines = [line.strip() for line in open(testingFile)]
        instancesList = []

        for line in trainFileLines:
            if line[0] != '@':
                instancesList.append(line)
        for line in testFileLines:
            if line[0] != '@':
                instancesList.append(line)

        fileInstancesList = []
        y = []
        for line in instancesList:
            if line[0] == '@':
                if line.lower().startswith("@attribute class"):
                    classes = line.split(" ")[2]
            else:
                fileInstancesList.append([float(i) for i in line.split(",")[:-1]])
                y.append(line.split(",")[-1])
        trainInstancesList, testInstancesList, yTrain, yTest = train_test_split(fileInstancesList, y, train_size=6000,
                                                                                random_state=42)

    print len(trainInstancesList), len(yTrain), len(testInstancesList), len(yTest)

    XTr = numpy.array(trainInstancesList)
    yTr = numpy.array(yTrain)
    XTe = numpy.array(testInstancesList)
    yTe = numpy.array(yTest)

    # print ("Scaling data...")
    # scaler = StandardScaler()
    # XTr = scaler.fit_transform(XTr)
    # XTe = scaler.fit_transform(XTe)
    if OPEN_WORLD == False:
        print "Closed-world"
    else:
        print "Open-world"

    print ("Classification...")
    classifier = kNN()

    classifier.fit(XTr, yTr)
    prediction = classifier.predict(XTe)

    totalPredictions = 0
    totalCorrectPredictions = 0
    debugInfo = []
    for i in range(0, len(yTe)):
        actualClass = yTe[i]
        predictedClass = prediction[i]
        probEstimate = 'NA'
        # debugInfo.append([actualClass,predictedClass])
        debugInfo.append([actualClass, predictedClass, probEstimate])
        totalPredictions += 1.0
        if actualClass == predictedClass:
            totalCorrectPredictions += 1.0

    accuracy = totalCorrectPredictions / totalPredictions * 100.0
    print ("Accuracy = ", accuracy)

    positive = []  # monitored
    negative = []  # non-monitored
    positive.append(config.binaryLabels[0])  # 'webpageMon'
    negative.append(config.binaryLabels[1])  # 'webpageNonMon'
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for entry in debugInfo:
        if entry[0] in positive:  # actual is positive
            if entry[1] in positive:  # predicted is positive too
                tp += 1
            else:  # predicted is negative
                fn += 1
        elif entry[0] in negative:  # actual is negative
            if entry[1] in positive:  # predicted is positive
                fp += 1
            else:  # predicted is negative too
                tn += 1

    tpr = str("%.4f" % (float(tp) / float(tp + fn)))
    fpr = str("%.4f" % (float(fp) / float(fp + tn)))
    Acc = str("%.4f" % (float(tp + tn) / float(tp + tn + fp + fn)))
    F1 = str("%.4f" % (float(2 * tp) / float((2 * tp) + (fn) + (fp))))
    F2 = str("%.4f" % (float(5 * tp) / float((5 * tp) + (4 * fn) + (fp))))  # beta = 2
    print "TPR, FPR, ACC, tp, tn, fp, fn, F1, F2"
    print tpr, fpr, Acc, tp, tn, fp, fn, F1, F2



