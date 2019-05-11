import os
import fnmatch

numAttacks=17 # 5  attacks and no attack -w 0

#TPR
TPR = [None] * numAttacks

for i in range(len(TPR)):
    TPR[i] = [None] * 4

#FPR
FPR = [None] * numAttacks

for i in range(len(FPR)):
    FPR[i] = [None] * 4

#ACC
ACC = [None] * numAttacks

for i in range(len(ACC)):
    ACC[i] = [None] * 4

#F2
F2 = [None] * numAttacks

for i in range(len(F2)):
    F2[i] = [None] * 4


desc = os.path.dirname(os.path.realpath(__file__)).split(os.sep)[-2]

def __writeFile(resultsList, type):
    resultType = type
    dir = os.path.join('.', desc)
    if not os.path.exists(dir):
        os.mkdir(dir)

    f = open( os.path.join(dir, resultType), 'w' )
    f.write("#"+desc+"\n")
    for entry in resultsList:
        f.write( str(entry[0])+'\t'+str(entry[1])+'\t'+str(entry[2])+'\t'+str(entry[3])+"\n" )

    f.close()



def extract(exp,ttype):
	for (path, dirs, files) in os.walk('.'):
	    for myfile in files:
		if fnmatch.fnmatch(myfile, '*binary'):
		    print myfile
		    fileLines = [line.strip() for line in open(os.path.join('.', myfile))]

		    for fileLine in fileLines:
		        if not fileLine.startswith("tpr"):
		            lineResults = fileLine.split(",") # [tpr	, fpr	, Acc	, F2	, tp	, tn	, fp	, fn	, File ID]

		    C = int(myfile.split(".C")[1].split(".")[0])
		    w = int(myfile.split(".w")[1].split(".")[0]) 
		    exptype = int(myfile.split(".o")[1].split(".")[0])
                    train = int(myfile.split(".t")[1].split(".")[0])
                    if str(exptype) == str(exp) and str(ttype) == str(train):
			    print w
			    print len(TPR)
			    TPR[w][0] = w
			    FPR[w][0] = w
			    ACC[w][0] = w
			    F2[w][0]  = w

			    if C == 23 and "ensemble" not in myfile:
				TPR[w][1] = '%.2f' % (float(lineResults[0]) * 100)
				FPR[w][1] = '%.2f' % (float(lineResults[1]) * 100)
				ACC[w][1] = '%.2f' % (float(lineResults[2]) * 100)
				F2[w][1]  = '%.2f' % (float(lineResults[3]) * 100)
			    elif C == 43 and "ensemble" not in myfile:
				TPR[w][2] = '%.2f' % (float(lineResults[0]) * 100)
				FPR[w][2] = '%.2f' % (float(lineResults[1]) * 100)
				ACC[w][2] = '%.2f' % (float(lineResults[2]) * 100)
				F2[w][2]  = '%.2f' % (float(lineResults[3]) * 100)
			    elif C == 43 and "ensemble" in myfile:
				TPR[w][3] = '%.2f' % (float(lineResults[0]) * 100)
				FPR[w][3] = '%.2f' % (float(lineResults[1]) * 100)
				ACC[w][3] = '%.2f' % (float(lineResults[2]) * 100)
				F2[w][3]  = '%.2f' % (float(lineResults[3]) * 100)
			    else:
				print "Code shouldn't come to here!"	



	print 'tpr'
	for entry in TPR:
	    print entry
	    print '\n'
	print 'fpr'
	for entry in FPR:
	    print entry
	    print '\n'

	__writeFile(TPR, 'tpr'+str(exp)+"_"+str(ttype))
	__writeFile(FPR, 'fpr'+str(exp)+"_"+str(ttype))
	__writeFile(ACC, 'acc'+str(exp)+"_"+str(ttype))
	__writeFile(F2,  'f2'+str(exp)+"_"+str(ttype))

extract(1,50)

extract(1,100)
extract(2,100)
extract(3,100)
