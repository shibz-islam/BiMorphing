# For -d 64, 65 (ensemble):
# ----------
# -i: used in closed-world and open-world to decide number of benign classes
# -u: for the open world setting only, benign is the nonMonitored, -u is split between training and testing
# -p: some packets are used only for testing
# -Q: Total number of HP-DCOY attack classes
# -w: Number of HP-DCOY attack classes to include in training set (If -1 or not set, then include all. If 0, then don't include any. If x, then include x HP-DCOY attack classes only).
# -W: Number of HP-DCOY attack classes to include in testing set  (If -1 or not set, then include all. If 0, then don't include any. If x, then include x HP-DCOY attack classes only).
# -e: Ensemble is used, even if only two classifiers are there, -d 65 -C 23,43 so -C 23 can get -d 64 (see code, config.Ensemble)

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w 1 -W 16 -Q 16 -e 1 -o 3 -a 35,36,37,28,29,30,31,32,33,34,22,23,24,25,26,27

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w 7 -W 16 -Q 16 -e 1 -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37


hpDcoyTotalAttks=16
numFeaturesNgram=100

#for hpDcoyAttks in {0..16}
#do



#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 50 -T 50 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w 0 -W 16 -Q 16 -e 1 -o 1 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37


 #    	python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 50 -T 50 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 1 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37



#done



#for hpDcoyAttks in {0..16}
#do

    

 #  	python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 1 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37

	


#done

(
#remember to uncomment 2to4gram reg to unp only bidi
for hpDcoyAttks in {0..4}
do



    	

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

  
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

    	
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
	


done
)&

#remember to uncomment 2to4gram reg to unp only bidi
(
for hpDcoyAttks in {4..7}
do



    	

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

  
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

    	
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
	


done

)&


#remember to uncomment 2to4gram reg to unp only bidi
(
for hpDcoyAttks in {7..11}
do



    	

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

  
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

    	
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
	


done
)&

(
#remember to uncomment 2to4gram reg to unp only bidi
for hpDcoyAttks in {11..13}
do



    	

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

  
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

    	
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
	


done
)&

(
#remember to uncomment 2to4gram reg to unp only bidi
for hpDcoyAttks in {13..16}
do



    	

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

  
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

    	
python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

python mainBiDirectionLatestEnsembleNewDataZday.py -d 64 -C 23 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
	


done
)&

    
#remember to uncomment 2to4gram reg to unpatched
#for hpDcoyAttks in {0..16}
#do



    	

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 28,29,30,31,32,33,34,35,36,37,22,23,24,25,26,27

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 22,23,24,32,33,34,35,36,37,25,26,27,28,29,30,31

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 23,22,26,25,24,32,28,29,30,35,27,33,34,31,36,37

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 36,37,22,23,24,27,28,29,30,31,32,33,34,35,25,26

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 33,32,31,30,29,28,27,26,25,24,23,22,37,36,35,34

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 37,36,26,25,24,23,22,35,34,33,32,31,30,29,28,27

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 29,28,27,26,25,24,23,22,37,36,35,34,33,32,31,30

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 29,28,27,26,37,36,33,32,31,35,34,30,25,24,23,22

#python mainBiDirectionLatestEnsembleNewDataZday.py -d 65 -C 23,43 -c 0 -k 34 -N 35 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 18 -w $hpDcoyAttks -W $hpDcoyTotalAttks -Q $hpDcoyTotalAttks -e 1 -o 3 -a 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37

	


#done

#-d 65 -C 23,43 -c 0 -k 24 -N 25 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 12 -w 12 -W 12 -Q 12 -f 100 -e 1
#-d 65 -C 23,43 -c 0 -k 24 -N 25 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 12 -w 12 -W 12 -Q 12 -f 100 -e 1


