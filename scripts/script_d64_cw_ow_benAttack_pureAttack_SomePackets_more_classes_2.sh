# For -d 64:
# ----------
# -i: used in closed-world and open-world to decide number of benign classes
# -u: for the open world setting only, benign is the nonMonitored, -u is split between training and testing
# -p: some packets are used only for testing


for numPackets in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
	for var in 1 2 3 4 5
	do

	# cw
	python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 64 -C 3,15,23 -c 0 -k 18 -N 19 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -i 7 -p $numPackets

	python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 64 -C 3,15,23 -c 0 -k 18 -N 19 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -X 10 -i 7 -p $numPackets # -X cross validation


	# ow
	python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 64 -C 3,15,23 -c 0 -k 18 -N 19 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 7 -p $numPackets # not cv, benign: 200/2 = 100 train and 100 test

	python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 64 -C 3,15,23 -c 0 -k 18 -N 19 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -X 10 -i 7 -p $numPackets # -X cross validation

        done
 
done


