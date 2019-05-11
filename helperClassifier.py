
import classifiers.wekaAPI
import config
from Utils import Utils


# this code runs classification

#file = "/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-DL/cache/datafile-p2b5adhuk100.c100.d5.C23.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000.arff"
file = "/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-DL/cache/datafile-t3xdpn06k100.c200.d5.C23.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000.arff"

classifier = "svm"
kwargs = {}
kwargs['C'] = 131072
kwargs['kernel'] = 'rbf'
kwargs['gamma'] = 0.0000019073486328125

folds = 10

#outputFilename = "/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-DL/output/results.k100.c0.d5.C23.N101.t60.T30.D1.E1.F1.G1.H1.I1.A1.V0.P0.g0.l0.b600.u5000.cv10"
outputFilename = "/data/kld/temp/Website-Fingerprinting-Glove/WF-BiDirection-PCA-Glove-OSAD-DL/output/results.k100.c200.d5.C23.N101.t60.T30.D1.E1.F1.G1.H1.I1.A1.V0.P0.g0.l0.b1.u5000.cv10"

[accuracy,debugInfo] =  classifiers.wekaAPI.executeSklearnCrossValidation(file, classifier, folds, **kwargs)
print "acc" + str(accuracy)
print debugInfo

positive = []  # monitored
negative = []  # non-monitored

positive.append(config.binaryLabels[0]) # 'webpageMon'
negative.append(config.binaryLabels[1]) # 'webpageNonMon'

Utils.calcTPR_FPR(debugInfo, outputFilename, positive, negative)