import config
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import classifiers.wekaAPI as wekaAPI
from sklearn.neighbors import KNeighborsClassifier

cv = 10
scale_range = (-1,1)


def readFile(fileName):
    fileLines = [line.strip() for line in open(fileName)]
    fileList = []
    for fileLine in fileLines:
        fileList.append(fileLine)

    return fileList

def scaleDataInRange(XTr):
    print ("Scaling data using train data only to range: ", scale_range)
    scaler = MinMaxScaler(feature_range=scale_range)
    XTr = scaler.fit_transform(XTr)
    return XTr

def scaleData(XTr):
    print ("Scaling data using train data")
    scaler = StandardScaler()
    XTr = scaler.fit_transform(XTr)
    return XTr

def randomSearch(classifier, parameters, XTr, yTr, cv, n_iter):
    print("***** Random Search *****")
    print("Cross-Validation:{0} and number of iterations:{1}".format(cv, n_iter))

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_micro'

        clf = RandomizedSearchCV(classifier, param_distributions=parameters, cv=cv, scoring=scoring_method,
                                 n_iter=n_iter)
        clf.fit(XTr, yTr)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        return clf.best_params_


def gridSearch(classifier, parameters, XTr, yTr, cv):
    print("***** Grid Search *****")
    print("Cross-Validation: ",cv)

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        if (score == 'accuracy'):
            scoring_method = score
        else:
            scoring_method = score + '_micro'

        clf = GridSearchCV(classifier, param_grid=parameters, cv=cv, scoring=scoring_method)
        clf.fit(XTr, yTr)

        print("Best parameters and scores set found on development set:")
        # print(self.clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        # print("Grid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        # print()

        return clf.best_params_

def performParameterTuningFromSingleFile(filePath):
    fileList = readFile(filePath)

    fileInstancesList = []
    y = []
    for line in fileList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2]
        else:
            fileInstancesList.append([float(i) for i in line.split(",")[:-1]])
            y.append(line.split(",")[-1])

    XTr = np.array(fileInstancesList)
    yTr = np.array(y)

    # XTr = scaleDataInRange(XTr)  # Pre-processing: scale the data in a Range
    # tuneParameterWithSVM(XTr, yTr)

    XTr = scaleData(XTr)  # Pre-processing: scale the data
    # tuneParameterWithRF(XTr, yTr)
    tuneParameterWithKNN(XTr, yTr)

def performParameterTuningFromTrainTest(trainfile, testfile):
    trainList = readFile(trainfile)
    testList = readFile(testfile)

    fileList = []
    fileList.extend(trainList)
    fileList.extend(testList)

    fileInstancesList = []
    y = []
    for line in fileList:
        if line[0] == '@':
            if line.lower().startswith("@attribute class"):
                classes = line.split(" ")[2]
        else:
            fileInstancesList.append([float(i) for i in line.split(",")[:-1]])
            y.append(line.split(",")[-1])

    XTr = np.array(fileInstancesList)
    yTr = np.array(y)

    # XTr = scaleDataInRange(XTr)  # Pre-processing: scale the data in a Range
    # tuneParameterWithSVM(XTr, yTr)

    XTr = scaleData(XTr)  # Pre-processing: scale the data
    # tuneParameterWithRF(XTr, yTr)
    tuneParameterWithKNN(XTr, yTr)

def calcTPR_FPR(debugInfo, isOpenWorld):
    if isOpenWorld == False:
        print "Need open world for calculating tpr/fpr"
        exit()

    positive = []  # monitored
    negative = []  # non-monitored
    positive.append(config.binaryLabels[0])  # 'webpageMon'
    negative.append(config.binaryLabels[1])  # 'webpageNonMon'
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for entry in debugInfo:
        if entry[0] in positive: # actual is positive
            if entry[1] in positive: # predicted is positive too
                tp += 1
            else: # predicted is negative
                fn += 1
        elif entry[0] in negative: # actual is negative
            if entry[1] in positive: # predicted is positive
                fp += 1
            else: # predicted is negative too
                tn += 1

    tpr = str( "%.4f" % (float(tp)/float(tp+fn)) )
    fpr = str( "%.4f" % (float(fp)/float(fp+tn) ))
    Acc = str( "%.4f" % (float(tp+tn)/float(tp+tn+fp+fn) ))
    F1 = str( "%.4f" % (float(2*tp)/float((2*tp)+(fn)+(fp)) ))
    F2  = str( "%.4f" % (float(5*tp)/float((5*tp)+(4*fn)+(fp)) )) # beta = 2
    print "TPR, FPR, ACC, tp, tn, fp, fn, F1, F2"
    print tpr, fpr, Acc, tp, tn, fp, fn, F1, F2

def tuneParameterWithSVM(XTr, yTr):
    classifier = SVC()
    parameters = {'kernel': ['rbf'],
                  'gamma': [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3],
                  'C': [2**10 ,2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
                  }
    # best_params = randomSearch(classifier, parameters, XTr, yTr, cv, 50)
    best_params = gridSearch(classifier, parameters, XTr, yTr, cv)

    print("*** Classification ***")
    clf = SVC()
    clf.set_params(**best_params)
    print("Test Result for SVM")
    acc = cross_validation.cross_val_score(clf, XTr, yTr, cv=cv)
    print acc
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))


def tuneParameterWithRF(XTr, yTr):
    classifier = RandomForestClassifier()
    parameters = {'criterion': ['gini'],
                  'n_estimators': [50, 100, 200, 300, 400, 500, 700, 1000],
                  'oob_score': [True],
                  'n_jobs': [3]
                  }
    # best_params = randomSearch(classifier, parameters, XTr, yTr, cv, 50)
    best_params = gridSearch(classifier, parameters, XTr, yTr, 10)

    print("*** Classification ***")
    clf = RandomForestClassifier()
    clf.set_params(**best_params)
    print("Test Result for RandomForestClassifier")
    acc = cross_validation.cross_val_score(clf, XTr, yTr, cv=cv)
    print acc
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))

def tuneParameterWithKNN(XTr, yTr):
    classifier = KNeighborsClassifier()
    parameters = {'n_neighbors': [2, 3, 5, 7],
                  'weights': ["distance"],
                  'algorithm': ["auto", "kd_tree", "ball_tree"],
                  'leaf_size': [20, 30, 40, 50],
                  'n_jobs': [-1]
                  }
    # best_params = randomSearch(classifier, parameters, XTr, yTr, cv, 50)
    best_params = gridSearch(classifier, parameters, XTr, yTr, 10)

    print("*** Classification ***")
    clf = KNeighborsClassifier()
    clf.set_params(**best_params)
    print("Test Result for RandomForestClassifier")
    acc = cross_validation.cross_val_score(clf, XTr, yTr, cv=cv)
    print acc
    print("Accuracy: %0.2f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))

def performClassification(fp):
    #BIND
    trainingFile = os.path.join(config.CACHE_DIR, fp + '-train.arff')
    testingFile = os.path.join(config.CACHE_DIR, fp + '-test.arff')
    classifier = "svm"
    kwargs = {}
    kwargs['C'] = 131072
    kwargs['kernel'] = 'rbf'
    kwargs['gamma'] = 0.0000019073486328125
    [accuracy, debugInfo] = wekaAPI.executeSklearnFromTwoFilesWithCrossValidation(trainingFile, testingFile, classifier, folds=10, **kwargs)
    calcTPR_FPR(debugInfo, isOpenWorld=True)

    #SVM
    # filePath = os.path.join(config.CACHE_DIR, fp + '.arff')
    # classifier = "svm"
    # kwargs = {}
    # kwargs['C'] = 2 ** 11
    # kwargs['kernel'] = 'rbf'
    # kwargs['gamma'] = 2
    # wekaAPI.executeSklearnCrossValidationScaleWithRange(filePath, classifier, cv, scale_range, **kwargs)

    # RF
    # trainingFile = os.path.join(config.CACHE_DIR, fp + '-train.arff')
    # testingFile = os.path.join(config.CACHE_DIR, fp + '-test.arff')
    # classifier = "RF"
    # kwargs = {}
    # kwargs['n_estimators'] = 500  # number of trees/ length of the fingerprint
    # kwargs['criterion'] = "gini"
    # kwargs['oob_score'] = True
    # kwargs['n_jobs'] = 3
    # wekaAPI.executeSklearn(trainingFile, testingFile, classifier, **kwargs)
    # wekaAPI.executeSklearnFromTwoFilesWithCrossValidation(trainingFile, testingFile, classifier, folds=10, **kwargs)

    #KNN
    # trainingFile = os.path.join(config.CACHE_DIR, fp + '-train.arff')
    # testingFile = os.path.join(config.CACHE_DIR, fp + '-test.arff')
    # kwargs = {}
    # kwargs['n_neighbors'] = 2
    # kwargs['weights'] = "distance"
    # kwargs['algorithm'] = "auto"
    # kwargs['leaf_size'] = 20
    # kwargs['n_jobs'] = -1
    # wekaAPI.executeSklearnKNN(trainingFile, testingFile, 10, **kwargs)


def tuneParameter():
    # filename = 'datafile-ifs4zy0uk100.c200.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'
    # filename = 'datafile-8to4r3kuk100.c100.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'
    # filename = 'datafile-hvyg1z3kk100.c0.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'
    filename = 'datafile-ln24xobvk100.c0.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'

    trainPath = os.path.join(config.CACHE_DIR, filename + '-train.arff') # remove '-train' part accordingly
    testPath = os.path.join(config.CACHE_DIR, filename + '-test.arff')  # remove '-test' part accordingly
    # performParameterTuningFromSingleFile(trainPath)
    performParameterTuningFromTrainTest(trainPath, testPath)


def classification():
    filenames = [
        'datafile-x3nrfeq4k100.c100.d5.C23.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1.u5000'
        # 'datafile-ln24xobvk100.c0.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'
        # 'datafile-ifs4zy0uk100.c200.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1',
        # 'datafile-8to4r3kuk100.c100.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1',
        # 'datafile-hvyg1z3kk100.c0.d5.C601.N101.t60.T30.D1.E1.F1.G1.H1.I1.B16.J8.K300.L0.05.M100.A1.V0.P0.G0.l0.0.b1'
    ]

    for fp in filenames:
        print("************************************")
        print fp
        performClassification(fp)



if __name__ == '__main__':
    # tuneParameter()
    classification()




