# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.
# Extended by Khaled Alnaami

import os
import sys
from Packet import Packet
import random

# Set the following to a directory that contains
# * weka-X-Y-Z (see WEKA_ROOT to change the weka version)
# * pcap-logs (a diretory that contains all of the LL pcap files)
# * [optional] (a directory that contains custom/local python modules)

BASE_DIR2        = '/home/gbaduz/Downloads'
#BASE_DIR        = '/home/gbaduz/Downloads/pcalog'
#BASE_DIR        = '/home/WebsiteFingerprinting/'
#BASE_DIR        = '/home/WF/'
#BASE_DIR        = '/home/dml_utd/WF/'
BASE_DIR        = '/data/kld/papers/InferringSrcOfEncrHTTPCon/'

# Enviromental settings
#JVM_MEMORY_SIZE = '4192m'
JVM_MEMORY_SIZE = '8192m'

#WEKA_ROOT          = os.path.join(BASE_DIR2   ,'weka-3-7-12')
WEKA_ROOT          = os.path.join(BASE_DIR   ,'weka-3-7-11')
WEKA_JAR           = os.path.join(WEKA_ROOT  ,'weka.jar')
PCAP_ROOT          = os.path.join(BASE_DIR   ,'pcap-logs')
PYTHON_ROOT        = os.path.join(BASE_DIR   ,'python2.4')
PYTHONPATH         = os.path.join(PYTHON_ROOT,'lib/python')
CACHE_DIR          = './cache'
COUNTERMEASURE_DIR = './countermeasures'
CLASSIFIERS_DIR    = './classifiers'
OUTPUT_DIR         = './output'
#WANG              = './wang/output_setwise_openworld'
#WANG              = './wang/output_setwise_openworld_2'
#WANG              = './wang/output_setwise_openworld_cpd_https'
#WANG              = './wang/output_setwise_openworld_cpd_android'
WANG               = './wang'
#WANG_OSAD         = './wang/OSAD_CPD_closedworld'
WANG_OSAD          = './wang/OSAD_CPD_closedworld/Exp3and7_For_Ahmad_HTTPS_Andr_OSAD_CPD/OSAD_CPD_Andr_For_Ahmad/'
SYSDIG             = './sysdig'

RUN_ID = ""

#Specify options for Herrmann MySQL database
MYSQL_HOST = 'localhost'
MYSQL_DB = 'fingerprints'
MYSQL_USER = 'fingerprints'
MYSQL_PASSWD = 'fingerprints'

sys.path.append(PYTHONPATH)
sys.path.append(COUNTERMEASURE_DIR)
sys.path.append(CLASSIFIERS_DIR)

COUNTERMEASURE      = 0
CLASSIFIER          = 0
BUCKET_SIZE         = 2
DATA_SOURCE         = 0
NUM_TRAINING_TRACES = 16
NUM_TESTING_TRACES  = 4
NUM_TRIALS          = 1
TOP_N               = 775
PACKET_PENALTY      = 68

#IGNORE_ACK          = True
IGNORE_ACK          = False # June 23, 2015 Include ack packets

# Glove
#GLOVE_OPTIONS = {'packetSize':1,'burstSize':1,'burstTime':1,'burstNumber':1,'biBurstSize':1,'biBurstTime':1} # 1 means True
GLOVE_OPTIONS = {'packetSize':0,'burstSize':0,'burstTime':0,'burstNumber':0,'biBurstSize':0,'biBurstTime':0,'ModelTraceNum':16}
GLOVE_PARAMETERS = {'window':8,'no_components':300,'learning_rate':0.05,'epochs':100}

NUM_MONITORED_SITES = -1; # For open world July 24, 2015
NUM_NON_MONITORED_SITES = -1; # For open world Jan  15, 2016. Number of nonMonitored Websites for Open World (used in Wang Tor dataset.)

FIVE_NUM_SUM = 0 # five number summary for BiDirection

EXTRA = 0 # For OSAD files or setwise

n_components_PCA = 0 # Number of dimentions for PCA
n_components_LDA = 0 # Number of dimentions for LDA Linear Discriminant Analysis
n_components_QDA = 0 # Number of dimentions for QDA Linear Discriminant Analysis

lasso = 0 # Lasso

Num_Features_Selected = 0

bucket_Size = 600
bucket_Time = 600

CLASSIFIER_LIST = [0] # multiple classifier, same webpages, same traces, Dec 8, 2015
COUNTERMEASURE_LIST = [0] # multiple Defenses, same webpages, same traces, Mar 17, 2017
COVARIATE_SHIFT = 0 # Jan 27, 2016. For covariate Shift experiments. Training traces are collected at time t and testing traces are collected at time t+1. (Default is 0, no time constraint on training and testing instances)

START_SLIDE_TRAIN = 0
ACC_THRESHOLD = 85.0
COVARIATE_SHIFT_LIST = [0]

BAD_WEBSITES = []

CROSS_VALIDATION = 0 # no cross validation

NUM_BENIGN_CLASSES = -1 # For honeypatch dataset

NUM_TRACE_PACKETS = -1 # number of trace packets to be used

NUM_HP_DCOY_ATTACKS_TOTAL = -1 # Number of total attacks (collected between Honeypatch and Decoy) (default -1, no HP_DCOY Attacks). Honeypatch datasets.
NUM_HP_DCOY_ATTACKS_TRAIN = -1 # Number of attacks (collected between Honeypatch and Decoy) to be used in training (default -1, use all). Honeypatch datasets.
NUM_HP_DCOY_ATTACKS_TEST  = -1 # Number of attacks (collected between Honeypatch and Decoy) to be used in testing  (default -1, use all). Honeypatch datasets.

NUM_NEIGHBORS = 3 # Number of neighbors to be used in kNN

NUM_FEATURES_RF = 0 # Tree-based Random Forest Feature Selection

ENSEMBLE = 0 # using ensemble - ENSEMBLE = 23 means -C 23 BiDi is preferred in case of no majority vote

OC_SVM = 0 # one-class svm not enabled
SVM_KERNEL = 2 # RBF kernel for svm and oc_svm
OC_SVM_Nu = 0.5

USERNAME = "" # for Data Analytics in the GWT apps project

AE = -1 # auto encoder

#DEEP_LEARNING = {1: "logistic_sgd_2.py", 2: "dA_2.py"}
DEEP_LEARNING_METHOD = -1
DEEP_LEARNING_METHOD_LIST = [-1]
DEEP_LEARNING_PARAMETERS = {
        'n_hidden_ratio':'0.333','corruption_level':'0.3',
        'layers':'1000,1000,1000', 'corruption_levels':'0.1,0.2,0.3',
        'training_epochs':'20','pretraining_epochs':'20',
        'learning_rate':'0.1','pretrain_lr':'0.001','finetune_lr':'0.1',
        'batch_size':'20'
        }

# Liberatore and Levine Training and Testing configuration
DATA_SET = [
{'month':2,'day':10,'hour':13},
{'month':2,'day':11,'hour':11},
{'month':2,'day':13,'hour':8},
{'month':2,'day':13,'hour':19},
{'month':2,'day':14,'hour':9},
{'month':2,'day':14,'hour':23},
{'month':2,'day':15,'hour':8},
{'month':2,'day':16,'hour':12},
{'month':2,'day':20,'hour':10},
{'month':2,'day':20,'hour':16},
{'month':2,'day':20,'hour':22},
{'month':2,'day':21,'hour':4},
{'month':2,'day':21,'hour':10},
{'month':2,'day':21,'hour':16},
{'month':2,'day':21,'hour':22},
{'month':2,'day':22,'hour':4},
{'month':2,'day':22,'hour':10},
{'month':2,'day':22,'hour':16},
{'month':2,'day':22,'hour':22},
{'month':2,'day':23,'hour':4},
{'month':2,'day':23,'hour':10},
{'month':2,'day':20,'hour':10},
{'month':2,'day':20,'hour':16},
{'month':2,'day':20,'hour':22},
{'month':2,'day':21,'hour':4},
{'month':2,'day':21,'hour':10},
{'month':2,'day':21,'hour':16},
{'month':2,'day':21,'hour':22},
{'month':2,'day':22,'hour':4},
{'month':2,'day':22,'hour':10},
{'month':2,'day':22,'hour':16},
{'month':2,'day':22,'hour':22},
{'month':2,'day':23,'hour':4},
{'month':2,'day':23,'hour':10},
{'month':3,'day':6,'hour':16},
{'month':3,'day':6,'hour':22},
{'month':3,'day':7,'hour':4},
{'month':3,'day':7,'hour':10},
{'month':3,'day':7,'hour':16},
{'month':3,'day':7,'hour':22},
{'month':3,'day':8,'hour':4},
{'month':3,'day':8,'hour':10},
{'month':3,'day':8,'hour':16},
{'month':3,'day':8,'hour':22},
{'month':3,'day':9,'hour':4},
{'month':3,'day':9,'hour':16},
{'month':3,'day':9,'hour':22},
{'month':3,'day':10,'hour':4},
{'month':3,'day':10,'hour':10},
{'month':3,'day':10,'hour':16},
{'month':3,'day':10,'hour':22},
{'month':3,'day':11,'hour':4},
{'month':3,'day':11,'hour':10},
{'month':3,'day':11,'hour':16},
{'month':3,'day':11,'hour':22},
{'month':3,'day':12,'hour':4},
{'month':3,'day':12,'hour':10},
{'month':3,'day':12,'hour':16},
{'month':3,'day':12,'hour':22},
{'month':3,'day':13,'hour':16},
{'month':3,'day':13,'hour':22},
{'month':3,'day':14,'hour':4},
{'month':3,'day':14,'hour':10},
{'month':3,'day':14,'hour':16},
{'month':3,'day':14,'hour':22},
{'month':3,'day':15,'hour':4},
{'month':3,'day':15,'hour':10},
{'month':3,'day':15,'hour':16},
{'month':3,'day':15,'hour':22},
{'month':3,'day':16,'hour':4},
{'month':3,'day':16,'hour':10},
{'month':3,'day':16,'hour':16},
{'month':3,'day':16,'hour':22},
{'month':3,'day':17,'hour':4},
{'month':3,'day':17,'hour':10},
{'month':3,'day':17,'hour':16},
{'month':3,'day':17,'hour':22},
{'month':3,'day':20,'hour':10},
{'month':3,'day':20,'hour':16},
{'month':3,'day':20,'hour':22},
{'month':3,'day':21,'hour':4},
{'month':3,'day':21,'hour':10},
{'month':3,'day':21,'hour':16},
{'month':3,'day':21,'hour':22},
{'month':3,'day':22,'hour':4},
{'month':3,'day':22,'hour':10},
{'month':3,'day':22,'hour':16},
{'month':3,'day':22,'hour':22},
{'month':3,'day':23,'hour':4},
{'month':3,'day':23,'hour':10},
{'month':3,'day':23,'hour':16},
{'month':3,'day':23,'hour':22},
{'month':3,'day':24,'hour':10},
{'month':3,'day':24,'hour':16},
{'month':3,'day':24,'hour':22},
{'month':3,'day':25,'hour':4},
{'month':3,'day':25,'hour':10},
{'month':3,'day':25,'hour':16},
{'month':3,'day':25,'hour':22},
{'month':3,'day':26,'hour':4},
{'month':3,'day':26,'hour':10},
{'month':3,'day':26,'hour':16},
{'month':3,'day':26,'hour':22},
{'month':3,'day':27,'hour':4},
{'month':3,'day':27,'hour':10},
{'month':3,'day':27,'hour':16},
{'month':3,'day':28,'hour':16},
{'month':3,'day':28,'hour':22},
{'month':3,'day':29,'hour':4},
{'month':3,'day':29,'hour':10},
{'month':3,'day':29,'hour':16},
{'month':3,'day':29,'hour':22},
{'month':3,'day':30,'hour':4},
{'month':3,'day':30,'hour':10},
{'month':3,'day':30,'hour':16},
{'month':3,'day':30,'hour':22},
{'month':3,'day':31,'hour':4},
{'month':3,'day':31,'hour':10},
{'month':3,'day':31,'hour':16},
{'month':3,'day':31,'hour':22},
{'month':4,'day':1,'hour':4},
{'month':4,'day':1,'hour':10},
{'month':4,'day':1,'hour':16},
{'month':4,'day':1,'hour':22},
{'month':4,'day':2,'hour':4},
{'month':4,'day':2,'hour':10},
{'month':4,'day':2,'hour':16},
{'month':4,'day':2,'hour':22},
{'month':4,'day':3,'hour':4},
{'month':4,'day':3,'hour':10},
{'month':4,'day':3,'hour':16},
{'month':4,'day':3,'hour':22},
{'month':4,'day':4,'hour':4},
{'month':4,'day':4,'hour':10},
{'month':4,'day':4,'hour':16},
{'month':4,'day':4,'hour':22},
{'month':4,'day':5,'hour':4},
{'month':4,'day':5,'hour':10},
{'month':4,'day':5,'hour':16},
{'month':4,'day':5,'hour':22},
{'month':4,'day':6,'hour':4},
{'month':4,'day':6,'hour':10},
{'month':4,'day':6,'hour':16},
{'month':4,'day':6,'hour':22},
{'month':4,'day':7,'hour':4},
{'month':4,'day':7,'hour':10},
{'month':4,'day':7,'hour':16},
{'month':4,'day':7,'hour':22},
{'month':4,'day':8,'hour':4},
{'month':4,'day':8,'hour':10},
{'month':4,'day':8,'hour':16},
{'month':4,'day':13,'hour':22},
{'month':4,'day':14,'hour':4},
{'month':4,'day':14,'hour':10},
{'month':4,'day':14,'hour':16},
{'month':4,'day':14,'hour':22},
{'month':4,'day':15,'hour':4},
{'month':4,'day':15,'hour':10},
{'month':4,'day':15,'hour':16},
{'month':4,'day':15,'hour':22},
{'month':4,'day':16,'hour':4},
{'month':4,'day':16,'hour':16},
{'month':4,'day':16,'hour':22},
{'month':4,'day':18,'hour':16},
{'month':4,'day':18,'hour':22},
{'month':4,'day':19,'hour':4},
{'month':4,'day':19,'hour':10},
{'month':4,'day':19,'hour':16},
{'month':4,'day':19,'hour':22},
{'month':4,'day':20,'hour':4},
{'month':4,'day':20,'hour':10},
{'month':4,'day':20,'hour':16},
{'month':4,'day':20,'hour':22},
{'month':4,'day':21,'hour':4},
{'month':4,'day':21,'hour':10},
{'month':4,'day':21,'hour':16},
{'month':4,'day':21,'hour':22},
{'month':4,'day':22,'hour':4},
{'month':4,'day':22,'hour':16},
{'month':4,'day':22,'hour':22},
{'month':4,'day':23,'hour':4},
{'month':4,'day':23,'hour':10},
{'month':4,'day':23,'hour':16},
{'month':4,'day':23,'hour':22},
{'month':4,'day':24,'hour':4},
{'month':4,'day':24,'hour':16},
{'month':4,'day':24,'hour':22},
{'month':4,'day':25,'hour':4},
{'month':4,'day':25,'hour':10},
{'month':4,'day':25,'hour':16},
{'month':4,'day':25,'hour':22},
{'month':4,'day':26,'hour':4},
{'month':4,'day':26,'hour':10},
{'month':4,'day':26,'hour':16},
{'month':4,'day':26,'hour':22},
{'month':4,'day':27,'hour':4},
{'month':4,'day':27,'hour':10},
{'month':4,'day':27,'hour':16},
{'month':4,'day':27,'hour':22},
{'month':4,'day':28,'hour':4},
{'month':4,'day':28,'hour':10},
{'month':4,'day':28,'hour':16},
{'month':4,'day':28,'hour':22}
]

# Android Tor (device)
# ANDROID TOR Training and Testing configuration
DATA_SET_ANDROID_TOR = [
{'month':7,'day':31,'hour':17},
{'month':7,'day':31,'hour':22},

{'month':8,'day':1,'hour':3},
{'month':8,'day':1,'hour':9},
{'month':8,'day':1,'hour':14},
{'month':8,'day':1,'hour':19},

{'month':8,'day':2,'hour':0},
{'month':8,'day':2,'hour':5},
{'month':8,'day':2,'hour':11},
{'month':8,'day':2,'hour':16},
{'month':8,'day':2,'hour':21},

{'month':8,'day':3,'hour':12},
{'month':8,'day':3,'hour':18},

{'month':8,'day':4,'hour':18},

{'month':8,'day':5,'hour':0},
{'month':8,'day':5,'hour':5},
{'month':8,'day':5,'hour':10},
{'month':8,'day':5,'hour':15},
{'month':8,'day':5,'hour':20},

{'month':8,'day':6,'hour':1},
{'month':8,'day':6,'hour':6},
{'month':8,'day':6,'hour':11},
{'month':8,'day':6,'hour':16},
{'month':8,'day':6,'hour':22},

{'month':8,'day':7,'hour':3},
{'month':8,'day':7,'hour':8},
{'month':8,'day':7,'hour':13},
{'month':8,'day':7,'hour':18},

{'month':8,'day':8,'hour':0},
{'month':8,'day':8,'hour':5},
{'month':8,'day':8,'hour':10},
{'month':8,'day':8,'hour':16},
{'month':8,'day':8,'hour':21},

{'month':8,'day':9,'hour':2},
{'month':8,'day':9,'hour':7},
{'month':8,'day':9,'hour':13},
{'month':8,'day':9,'hour':18},

{'month':8,'day':10,'hour':0},
{'month':8,'day':10,'hour':5},
{'month':8,'day':10,'hour':10},
{'month':8,'day':10,'hour':16},
{'month':8,'day':10,'hour':21},

{'month':8,'day':11,'hour':2},
{'month':8,'day':11,'hour':7},
{'month':8,'day':11,'hour':13},
{'month':8,'day':11,'hour':18},
{'month':8,'day':11,'hour':23},

{'month':8,'day':12,'hour':4},
{'month':8,'day':12,'hour':10},
{'month':8,'day':12,'hour':15},
{'month':8,'day':12,'hour':20},

{'month':8,'day':13,'hour':2},
{'month':8,'day':13,'hour':7},
{'month':8,'day':13,'hour':12},
{'month':8,'day':13,'hour':18},
{'month':8,'day':13,'hour':23},

{'month':8,'day':14,'hour':4},
{'month':8,'day':14,'hour':10},
{'month':8,'day':14,'hour':15},
{'month':8,'day':14,'hour':20},

{'month':8,'day':15,'hour':1},
{'month':8,'day':15,'hour':7},
{'month':8,'day':15,'hour':12},
{'month':8,'day':15,'hour':18},
{'month':8,'day':15,'hour':23},

{'month':8,'day':16,'hour':5},
{'month':8,'day':16,'hour':10},
{'month':8,'day':16,'hour':16},
{'month':8,'day':16,'hour':21},

{'month':8,'day':17,'hour':2},
{'month':8,'day':17,'hour':8},
{'month':8,'day':17,'hour':13},
{'month':8,'day':17,'hour':18},

{'month':8,'day':18,'hour':0},
{'month':8,'day':18,'hour':6},

]

'''
# Android Tor (emulator)
# ANDROID TOR Training and Testing configuration
DATA_SET_ANDROID_TOR_Old = [
{'month':1,'day':10,'hour':15},
{'month':1,'day':10,'hour':21},

{'month':1,'day':11,'hour':2},
{'month':1,'day':11,'hour':7},
{'month':1,'day':11,'hour':13},
{'month':1,'day':11,'hour':18},

{'month':1,'day':12,'hour':0},
{'month':1,'day':12,'hour':5},
{'month':1,'day':12,'hour':14},
{'month':1,'day':12,'hour':19},

{'month':1,'day':13,'hour':19},

{'month':1,'day':14,'hour':0},
{'month':1,'day':14,'hour':4},
{'month':1,'day':14,'hour':16},
{'month':1,'day':14,'hour':21},

{'month':1,'day':15,'hour':2},
{'month':1,'day':15,'hour':8},
{'month':1,'day':15,'hour':21},

{'month':1,'day':16,'hour':3},
{'month':1,'day':16,'hour':10},
{'month':1,'day':16,'hour':17},
{'month':1,'day':16,'hour':23},

{'month':1,'day':17,'hour':5},
{'month':1,'day':17,'hour':13},
{'month':1,'day':17,'hour':19},

{'month':1,'day':18,'hour':1},
{'month':1,'day':18,'hour':8},
{'month':1,'day':18,'hour':14},

{'month':1,'day':19,'hour':13},
{'month':1,'day':19,'hour':19},

{'month':1,'day':20,'hour':1},
{'month':1,'day':20,'hour':8},
{'month':1,'day':20,'hour':14},
{'month':1,'day':20,'hour':21},

{'month':1,'day':21,'hour':3},
{'month':1,'day':21,'hour':9},
{'month':1,'day':21,'hour':15},
{'month':1,'day':21,'hour':22},

{'month':1,'day':22,'hour':4},
{'month':1,'day':22,'hour':11},
{'month':1,'day':22,'hour':17},

{'month':1,'day':23,'hour':0},
{'month':1,'day':23,'hour':7},
{'month':1,'day':23,'hour':15},
{'month':1,'day':23,'hour':22},

{'month':1,'day':24,'hour':4},

{'month':1,'day':26,'hour':12},
{'month':1,'day':26,'hour':18},

{'month':1,'day':27,'hour':0},
{'month':1,'day':27,'hour':8},
{'month':1,'day':27,'hour':19},

{'month':1,'day':28,'hour':1},
{'month':1,'day':28,'hour':7},
{'month':1,'day':28,'hour':15},
{'month':1,'day':28,'hour':22},

{'month':1,'day':29,'hour':4},
{'month':1,'day':29,'hour':12},
{'month':1,'day':29,'hour':19},

{'month':1,'day':30,'hour':1},
{'month':1,'day':30,'hour':9},
{'month':1,'day':30,'hour':15},

{'month':2,'day':1,'hour':17},
{'month':2,'day':1,'hour':23},

{'month':2,'day':2,'hour':5},
{'month':2,'day':2,'hour':18},

{'month':2,'day':3,'hour':15},

{'month':2,'day':4,'hour':15},
{'month':2,'day':4,'hour':21},

{'month':2,'day':5,'hour':3},
{'month':2,'day':5,'hour':9}
]
'''

DATA_SET_ANDROID_APPS = [
{'month':1,'day':1,'hour':2},
{'month':1,'day':1,'hour':5},
{'month':1,'day':2,'hour':15},
{'month':1,'day':4,'hour':0},
{'month':1,'day':5,'hour':9},
{'month':1,'day':6,'hour':18},
{'month':1,'day':8,'hour':4},
{'month':1,'day':9,'hour':13},
{'month':1,'day':10,'hour':22},
{'month':1,'day':12,'hour':7},
{'month':1,'day':22,'hour':11},
{'month':1,'day':24,'hour':4},
{'month':1,'day':25,'hour':21},
{'month':1,'day':27,'hour':13},
{'month':1,'day':29,'hour':7},
{'month':1,'day':30,'hour':23},
{'month':2,'day':1,'hour':17},
{'month':2,'day':3,'hour':10},
{'month':2,'day':5,'hour':3},
{'month':2,'day':6,'hour':20}
]

# Wang Tor dataset. 90 instances per every monitored website.
DATA_SET_WANG_TOR = [0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
31,
32,
33,
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
51,
52,
53,
54,
55,
56,
57,
58,
59,
60,
61,
62,
63,
64,
65,
66,
67,
68,
69,
70,
71,
72,
73,
74,
75,
76,
77,
78,
79,
80,
81,
82,
83,
84,
85,
86,
87,
88,
89]


# HoneyPatch Binary datasets (class 0 is benign and class 1 is attack)
DATA_SET_HONEYPATCH_BENIGN = range(0, 1573) # [0, 1, ..., 1572]
DATA_SET_HONEYPATCH_ATTACK = range(0, 458)  # [0, 1, ..., 457]

# HoneyPatch Multiclass (MC) datasets
DATA_SET_HONEYPATCH_MC_BENIGN = range(0, 960) # benign blog [0, 1, ..., 959], can be used for benign wordpress
#DATA_SET_HONEYPATCH_MC_BENIGN = range(0, 3001)  # benign wordpress
DATA_SET_HONEYPATCH_MC_ATTACK = range(0, 201)  # each attack class has 201 traces [0, 1, ..., 200]

# HoneyPatch Multiclass (MC) datasets, benattack dataset where attacks are embedded inside benign traffic
DATA_SET_HONEYPATCH_MC_BENATTACK_BENIGN = range(0, 200)
DATA_SET_HONEYPATCH_MC_BENATTACK_ATTACK = range(0, 200)
#DATA_SET_HONEYPATCH_MC_BENATTACK_BENIGN = range(0, 100) # for exp1, exp8 only
#DATA_SET_HONEYPATCH_MC_BENATTACK_ATTACK = range(0, 100) # for exp1, exp8 only


NUM_TRAINING_TRACES_BENIGN_FACTOR = 1
NUM_TESTING_TRACES_BENIGN_FACTOR = 1

LABEL_NOISE_RATIO = 10 # noise to data (mislabeling)

#For APT pairwise
#HP_OFFSET_BFRHP_AFTRHP = 7 # for pair wise, pick from attack 12 and 19 (12 + 7) for example
#HP_OFFSET_HP_ATTACK_STARTING_ID = 19

#For regular pairwise
HP_OFFSET_BFRHP_AFTRHP = 12 # for pair wise, pick from attack 12 and 19 (12 + 7) for example
HP_OFFSET_HP_ATTACK_STARTING_ID = 24

#excludedClasses = []
#excludedInstances=[]
excludedInst = {}

# For Outlier Detection. 1: kmeans++, 2: pca + kmeans++, 3: pca + kmeans++ + convex hull, 4: pca + kmeans++ + convex hull + knn
CLUSTERING = { 1: "kmeans", 2: "pca_kmeans", 3: "pca_kmeans_convexhull", 4: "pca_kmeans_convexhull_mixed", 5: "pca_kmeans_convexhull_knn"}
CLUSTERING_METHOD = 1
NUM_CLUSTERS = -1

ATTACK_LIST = [] # ordered attack list, mainly for IDS Zero-day experiments

# Honeypatch dataset index and packet/syscalls folders
HP_DATASETS = {1: ["honeypatckBenattackNewData", "honeypatckBenattackSysdigFilesNewData"],
                2: ["honeypatckBenattackNewDataUnpatched", "honeypatckBenattackSysdigFilesNewDataUnpatched"],
                3: ["honeypatckBenattackNewData-regularPatched", "honeypatckBenattackSysdigFilesNewData-regularPatched"],
                4: ["honeypatckBenattackNewDataZday", "honeypatckBenattackSysdigFilesNewDataZday"],
                5: ["honeypatckBenattackNewData_unpatchedServer_manyAttacks", "honeypatckBenattackSysdigFilesNewData_unpatchedServer_manyAttacks"]}

# Binary classification for honeypatching datsets
HP_BINARY = 0

# How much of the training to be morphed, see mainBiDirectionLatestEnsembleNewDataZdayDefenseAdaptiveMorphing.py
# if 1, then morph all traning traces, 2, morph half of the traning traces, 4, morph quarter of the training traces
TRAIN_MORPH_CONSTANT = 1

sysCallTable={
    'read':0,
    'write':1,
    'open':2,
    'close':3,
    'stat':4,
    'fstat':5,
    'lstat':6,
    'poll':7,
    'lseek':8,
    'mmap':9,
    'mprotect':10,
    'munmap':11,
    'brk':12,
    'rt_sigaction':13,
    'rt_sigprocmask':14,
    'rt_sigreturn':15,
    'ioctl':16,
    'pread64':17,
    'pwrite64':18,
    'readv':19,
    'writev':20,
    'access':21,
    'pipe':22,
    'select':23,
    'sched_yield':24,
    'mremap':25,
    'msync':26,
    'mincore':27,
    'madvise':28,
    'shmget':29,
    'shmat':30,
    'shmctl':31,
    'dup':32,
    'dup2':33,
    'pause':34,
    'nanosleep':35,
    'getitimer':36,
    'alarm':37,
    'setitimer':38,
    'getpid':39,
    'sendfile':40,
    'socket':41,
    'connect':42,
    'accept':43,
    'sendto':44,
    'recvfrom':45,
    'sendmsg':46,
    'recvmsg':47,
    'shutdown':48,
    'bind':49,
    'listen':50,
    'getsockname':51,
    'getpeername':52,
    'socketpair':53,
    'setsockopt':54,
    'getsockopt':55,
    'clone':56,
    'fork':57,
    'vfork':58,
    'execve':59,
    'exit':60,
    'wait4':61,
    'kill':62,
    'uname':63,
    'semget':64,
    'semop':65,
    'semctl':66,
    'shmdt':67,
    'msgget':68,
    'msgsnd':69,
    'msgrcv':70,
    'msgctl':71,
    'fcntl':72,
    'flock':73,
    'fsync':74,
    'fdatasync':75,
    'truncate':76,
    'ftruncate':77,
    'getdents':78,
    'getcwd':79,
    'chdir':80,
    'fchdir':81,
    'rename':82,
    'mkdir':83,
    'rmdir':84,
    'creat':85,
    'link':86,
    'unlink':87,
    'symlink':88,
    'readlink':89,
    'chmod':90,
    'fchmod':91,
    'chown':92,
    'fchown':93,
    'lchown':94,
    'umask':95,
    'gettimeofday':96,
    'getrlimit':97,
    'getrusage':98,
    'sysinfo':99,
    'times':100,
    'ptrace':101,
    'getuid':102,
    'syslog':103,
    'getgid':104,
    'setuid':105,
    'setgid':106,
    'geteuid':107,
    'getegid':108,
    'setpgid':109,
    'getppid':110,
    'getpgrp':111,
    'setsid':112,
    'setreuid':113,
    'setregid':114,
    'getgroups':115,
    'setgroups':116,
    'setresuid':117,
    'getresuid':118,
    'setresgid':119,
    'getresgid':120,
    'getpgid':121,
    'setfsuid':122,
    'setfsgid':123,
    'getsid':124,
    'capget':125,
    'capset':126,
    'rt_sigpending':127,
    'rt_sigtimedwait':128,
    'rt_sigqueueinfo':129,
    'rt_sigsuspend':130,
    'sigaltstack':131,
    'utime':132,
    'mknod':133,
    'uselib':134,
    'personality':135,
    'ustat':136,
    'statfs':137,
    'fstatfs':138,
    'sysfs':139,
    'getpriority':140,
    'setpriority':141,
    'sched_setparam':142,
    'sched_getparam':143,
    'sched_setscheduler':144,
    'sched_getscheduler':145,
    'sched_get_priority_max':146,
    'sched_get_priority_min':147,
    'sched_rr_get_interval':148,
    'mlock':149,
    'munlock':150,
    'mlockall':151,
    'munlockall':152,
    'vhangup':153,
    'modify_ldt':154,
    'pivot_root':155,
    '_sysctl':156,
    'prctl':157,
    'arch_prctl':158,
    'adjtimex':159,
    'setrlimit':160,
    'chroot':161,
    'sync':162,
    'acct':163,
    'settimeofday':164,
    'mount':165,
    'umount2':166,
    'swapon':167,
    'swapoff':168,
    'reboot':169,
    'sethostname':170,
    'setdomainname':171,
    'iopl':172,
    'ioperm':173,
    'create_module':174,
    'init_module':175,
    'delete_module':176,
    'get_kernel_syms':177,
    'query_module':178,
    'quotactl':179,
    'nfsservctl':180,
    'getpmsg':181,
    'putpmsg':182,
    'afs_syscall':183,
    'tuxcall':184,
    'security':185,
    'gettid':186,
    'readahead':187,
    'setxattr':188,
    'lsetxattr':189,
    'fsetxattr':190,
    'getxattr':191,
    'lgetxattr':192,
    'fgetxattr':193,
    'listxattr':194,
    'llistxattr':195,
    'flistxattr':196,
    'removexattr':197,
    'lremovexattr':198,
    'fremovexattr':199,
    'tkill':200,
    'time':201,
    'futex':202,
    'sched_setaffinity':203,
    'sched_getaffinity':204,
    'set_thread_area':205,
    'io_setup':206,
    'io_destroy':207,
    'io_getevents':208,
    'io_submit':209,
    'io_cancel':210,
    'get_thread_area':211,
    'lookup_dcookie':212,
    'epoll_create':213,
    'epoll_ctl_old':214,
    'epoll_wait_old':215,
    'remap_file_pages':216,
    'getdents64':217,
    'set_tid_address':218,
    'restart_syscall':219,
    'semtimedop':220,
    'fadvise64':221,
    'timer_create':222,
    'timer_settime':223,
    'timer_gettime':224,
    'timer_getoverrun':225,
    'timer_delete':226,
    'clock_settime':227,
    'clock_gettime':228,
    'clock_getres':229,
    'clock_nanosleep':230,
    'exit_group':231,
    'epoll_wait':232,
    'epoll_ctl':233,
    'tgkill':234,
    'utimes':235,
    'vserver':236,
    'mbind':237,
    'set_mempolicy':238,
    'get_mempolicy':239,
    'mq_open':240,
    'mq_unlink':241,
    'mq_timedsend':242,
    'mq_timedreceive':243,
    'mq_notify':244,
    'mq_getsetattr':245,
    'kexec_load':246,
    'waitid':247,
    'add_key':248,
    'request_key':249,
    'keyctl':250,
    'ioprio_set':251,
    'ioprio_get':252,
    'inotify_init':253,
    'inotify_add_watch':254,
    'inotify_rm_watch':255,
    'migrate_pages':256,
    'openat':257,
    'mkdirat':258,
    'mknodat':259,
    'fchownat':260,
    'futimesat':261,
    'newfstatat':262,
    'unlinkat':263,
    'renameat':264,
    'linkat':265,
    'symlinkat':266,
    'readlinkat':267,
    'fchmodat':268,
    'faccessat':269,
    'pselect6':270,
    'ppoll':271,
    'unshare':272,
    'set_robust_list':273,
    'get_robust_list':274,
    'splice':275,
    'tee':276,
    'sync_file_range':277,
    'vmsplice':278,
    'move_pages':279,
    'utimensat':280,
    'epoll_pwait':281,
    'signalfd':282,
    'timerfd_create':283,
    'eventfd':284,
    'fallocate':285,
    'timerfd_settime':286,
    'timerfd_gettime':287,
    'accept4':288,
    'signalfd4':289,
    'eventfd2':290,
    'epoll_create1':291,
    'dup3':292,
    'pipe2':293,
    'inotify_init1':294,
    'preadv':295,
    'pwritev':296,
    'rt_tgsigqueueinfo':297,
    'perf_event_open':298,
    'recvmmsg':299,
    'fanotify_init':300,
    'fanotify_mark':301,
    'prlimit64':302,
    'name_to_handle_at':303,
    'open_by_handle_at':304,
    'clock_adjtime':305,
    'syncfs':306,
    'sendmmsg':307,
    'setns':308,
    'getcpu':309,
    'process_vm_readv':310,
    'process_vm_writev':311,
    'kcmp':312,
    'finit_module':313
     }

# packet range (LL)
PACKET_RANGE = range(Packet.HEADER_LENGTH,Packet.MTU+1,8)
PACKET_RANGE2 = range(Packet.HEADER_LENGTH,Packet.MTU+1,4)


KDD_FEATURES = [
['duration', '@attribute duration numeric'],
['protocol_type', '@attribute protocol_type {tcp,udp,icmp}'],
['service', '@attribute service {vmnet,smtp,ntp_u,shell,kshell,aol,imap4,urh_i,netbios_ssn,tftp_u,mtp,uucp,nnsp,echo,tim_i,ssh,iso_tsap,time,netbios_ns,systat,hostnames,login,efs,supdup,http_8001,courier,ctf,finger,nntp,ftp_data,red_i,ldap,http,ftp,pm_dump,exec,klogin,auth,netbios_dgm,other,link,X11,discard,private,remote_job,IRC,daytime,pop_3,pop_2,gopher,sunrpc,name,rje,domain,uucp_path,http_2784,Z39_50,domain_u,csnet_ns,whois,eco_i,bgp,sql_net,printer,telnet,ecr_i,urp_i,netstat,http_443,harvest}'],
['flag', '@attribute flag {RSTR,S3,SF,RSTO,SH,OTH,S2,RSTOS0,S1,S0,REJ}'],
['src_bytes', '@attribute src_bytes numeric'],
['dst_bytes', '@attribute dst_bytes numeric'],
['land', '@attribute land {1,0}'],
['wrong_fragment', '@attribute wrong_fragment numeric'],
['urgent', '@attribute urgent numeric'],
['hot', '@attribute hot numeric'],
['num_failed_logins', '@attribute num_failed_logins numeric'],
['logged_in', '@attribute logged_in {1,0}'],
['lnum_compromised', '@attribute lnum_compromised numeric'],
['lroot_shell', '@attribute lroot_shell numeric'],
['lsu_attempted', '@attribute lsu_attempted numeric'],
['lnum_root', '@attribute lnum_root numeric'],
['lnum_file_creations', '@attribute lnum_file_creations numeric'],
['lnum_shells', '@attribute lnum_shells numeric'],
['lnum_access_files', '@attribute lnum_access_files numeric'],
['lnum_outbound_cmds', '@attribute lnum_outbound_cmds numeric'],
['is_host_login', '@attribute is_host_login {1,0}'],
['is_guest_login', '@attribute is_guest_login {1,0}'],
['count', '@attribute count numeric'],
['srv_count', '@attribute srv_count numeric'],
['serror_rate', '@attribute serror_rate numeric'],
['srv_serror_rate', '@attribute srv_serror_rate numeric'],
['rerror_rate', '@attribute rerror_rate numeric'],
['srv_rerror_rate', '@attribute srv_rerror_rate numeric'],
['same_srv_rate', '@attribute same_srv_rate numeric'],
['diff_srv_rate', '@attribute diff_srv_rate numeric'],
['srv_diff_host_rate', '@attribute srv_diff_host_rate numeric'],
['dst_host_count', '@attribute dst_host_count numeric'],
['dst_host_srv_count', '@attribute dst_host_srv_count numeric'],
['dst_host_same_srv_rate', '@attribute dst_host_same_srv_rate numeric'],
['dst_host_diff_srv_rate', '@attribute dst_host_diff_srv_rate numeric'],
['dst_host_same_src_port_rate', '@attribute dst_host_same_src_port_rate numeric'],
['dst_host_srv_diff_host_rate', '@attribute dst_host_srv_diff_host_rate numeric'],
['dst_host_serror_rate', '@attribute dst_host_serror_rate numeric'],
['dst_host_srv_serror_rate', '@attribute dst_host_srv_serror_rate numeric'],
['dst_host_rerror_rate', '@attribute dst_host_rerror_rate numeric'],
['dst_host_srv_rerror_rate', '@attribute dst_host_srv_rerror_rate numeric']
]

# Esorics 16 WRF-PAD, 100 websites, 40 instances each
DATA_SET_ESORICS16_TOR = range(0, 40)

# Wang Tor dataset. 100 instances per every monitored website.
DATA_SET_WANG_TOR_USENIX17 = range(0, 100)

INT_LIST = range(0,1000)
RANDOM_LIST = random.sample(INT_LIST,  100)
INTER_PACKET_ARRIVAL_HISTO = {}
INTER_PACKET_ARRIVAL_HISTO_UP = {}
INTER_PACKET_ARRIVAL_HISTO_DN = {}

BI_BURST_HISTO_UP_DOWN = {}
BI_BURST_HISTO_DOWN_UP = {}

PMF_UP_DN = {}
PMF_DN_UP = {}

CDF_UP_DN = {}
CDF_DN_UP = {}

# InterArrival Time IAT
BI_BURST_HISTO_UP_DOWN_IAT = {}
BI_BURST_HISTO_DOWN_UP_IAT = {}

PMF_UP_DN_IAT = {}
PMF_DN_UP_IAT = {}

CDF_UP_DN_IAT = {}
CDF_DN_UP_IAT = {}
####

BI_BURST_HISTO_UP_DOWN_TARGET = {}
BI_BURST_HISTO_DOWN_UP_TARGET = {}

BUILD_TARGET_INFO = True # build target info just once
dim = 0
targetBurstsList = []

binaryLabels = ['webpageMon','webpageNonMon'] # [positive, negative]


buildMetadata = True

metadata = []

COMMENTS = ""

BUILD_ALL_WEBPAGES_HISTOS = True
ALL_WEBPAGES_HISTOS = {} # <webpage, histogram>

BUILD_ALL_WEBPAGES = True
ALL_WEBPAGES = []

LARGEST_WEBPAGE = None

TIME_ORIGINAL = 0
TIME_EXTRA = 0

# hp chat dataset
DATA_SET_HP_CHAT_LEAD = range(0, 16879)
DATA_SET_HP_CHAT_OPPORTUNITY = range(0, 17044)

# packet range (H)

# Security Strategy Enum
NONE                     = 0
PAD_TO_MTU               = 1
RFC_COMPLIANT_FIXED_PAD  = 2
RFC_COMPLIANT_RANDOM_PAD = 3
RANDOM_PAD               = 4
PAD_ROUND_EXPONENTIAL    = 5
PAD_ROUND_LINEAR         = 6
MICE_ELEPHANTS           = 7
DIRECT_TARGET_SAMPLING   = 8
WRIGHT_STYLE_MORPHING    = 9
FIXED_PAD                = 10
DIRECT_TARGET_SAMPLING_IDS   = 81
BI_DI_MORPHING           = 100 # bi directional bursting morphing
BURST_MOLDING            = 200
TAMARAW                  = 300

# Classifier enum
LIBERATORE_CLASSIFIER    = 0
WRIGHT_CLASSIFIER        = 1
JACCARD_CLASSIFIER       = 2
PANCHENKO_CLASSIFIER     = 3
BANDWIDTH_CLASSIFIER     = 4
ESORICS_CLASSIFIER       = 5
HERRMANN_CLASSIFIER      = 6
TIME_CLASSIFIER          = 10
VNG_CLASSIFIER           = 14
VNG_PLUS_PLUS_CLASSIFIER = 15
GLOVE_CLASSIFIER = 16
W2V_CLASSIFIER = 17
ADVERSIAL_CLASSIFIER     = 21 # On VNG++
ADVERSARIAL_CLASSIFIER_ON_PANCHENKO = 22
ADVERSIAL_CLASSIFIER_TOR = 31
ADVERSARIAL_CLASSIFIER_BiDirection_Only= 23
TO_WANG_FILES_CLOSED_WORLD = 101
TO_WANG_FILES_OPEN_WORLD = 102
HP_KNN_LCS = 33 # HoneyPatch Syscalls Classifier - kNN with Longest Common Subsequence distance metric
HP_NGRAM = 43 # HoneyPatch sysdig syscalls classifier
BI_DI_CLUSTERING = 123
HP_NGRAM_CLUSTERING = 143
BI_DI_TFIDF = 223
GLOVE_CLASSIFIER2 = 116 # glove taking all word vectors concatenated
HP_NGRAM_TFIDF = 443
KDD_CLASSIFIER = 500
TIME_CUM = 501
ADVERSARIAL_CLASSIFIER_TriDirection_Only= 1023 # used in mainBiDirectionLatest2.py
ADVERSARIAL_CLASSIFIER_BiDirection_Only_Optimized= 2023 # used in mainBiDirectionLatest2.py
RAW_DATA_DL_CLASSIFIER = 1000
CUMUL = 502
kFP = 602
kNN = 601

### Sanity
def sanity():
    if not os.path.exists(WEKA_JAR):
        print 'Weka does not exist in path: '+str(WEKA_JAR)
        print 'Please install Weka properly.'
        #sys.exit()

    if BASE_DIR == '':
        print "!!!!"
        print "Please open config.py and set your BASE_DIR."
        #sys.exit()

sanity()
###
