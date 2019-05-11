# For -d 63:
# ----------
# -i: used in closed-world and open-world to decide number of benign classes
# -u: for the open world setting only, benign is the nonMonitored, -u is split between training and testing
for var in 1 2 3 4 5
do

# cw
python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 63 -C 3,15,23 -c 0 -k 2 -N 3 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -i 1

python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 63 -C 3,15,23 -c 0 -k 2 -N 3 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -X 10 -i 1 # -X cross validation


# ow
python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 63 -C 3,15,23 -c 0 -k 2 -N 3 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -i 1 # not cv, benign: 200/2 = 100 train and 100 test

python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 63 -C 3,15,23 -c 0 -k 2 -N 3 -t 100 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 200 -X 10 -i 1 # -X cross validation


 
done


