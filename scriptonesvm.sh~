# For -d 64, 65 (ensemble):
# ----------
# -i: used in closed-world and open-world to decide number of benign classes
# -u: for the open world setting only, benign is the nonMonitored, -u is split between training and testing
# -p: some packets are used only for testing
# -Q: Total number of HP-DCOY attack classes
# -w: Number of HP-DCOY attack classes to include in training set (If -1 or not set, then include all. If 0, then don't include any. If x, then include x HP-DCOY attack classes only).
# -W: Number of HP-DCOY attack classes to include in testing set  (If -1 or not set, then include all. If 0, then don't include any. If x, then include x HP-DCOY attack classes only).
# -e: Ensemble is used, even if only two classifiers are there, -d 65 -C 23,43 so -C 23 can get -d 64 (see code, config.Ensemble)

hpDcoyTotalAttks=16
numFeaturesNgram=100




(
for tsize in 10 20 30 40 50 60 70 80 90 100
do

    

     	python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 28 -N 29 -t $tsize -T $tsize -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 13 -w 16 -W 16 -Q $hpDcoyTotalAttks -o 1 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37

	


done
)&

(
for tsize in 10 20 30 40 50 60 70 80 90 100
do

    

    	python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 43 -c 0 -k 28 -N 29 -t $tsize -T $tsize -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 13 -w 16 -W 16 -Q $hpDcoyTotalAttks -o 1 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37

	


done
)&


#-d 65 -C 23,43 -c 0 -k 24 -N 25 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 12 -w 12 -W 12 -Q 12 -f 100 -e 1
#-d 65 -C 23,43 -c 0 -k 24 -N 25 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 12 -w 12 -W 12 -Q 12 -f 100 -e 1


