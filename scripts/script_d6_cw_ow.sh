
for var in 1 2 3 4 5
do

# cw
python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 6 -C 3,15,23 -c 0 -k 2 -N 3 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600

python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 6 -C 3,15,23 -c 0 -k 2 -N 3 -t 16 -T 4 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -X 10 # -X cross validation


#ow
python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 6 -C 3,15,23 -c 0 -k 2 -N 3 -t 350 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 1500

python mainBiDirectionFeaturesMultipleClassifiersOpenWorldWangTor.py -d 6 -C 3,15,23 -c 0 -k 2 -N 3 -t 350 -T 100 -n 1 -D 1 -E 1 -F 1 -G 1 -H 1 -I 1 -A 1 -V 0 -b 600 -u 1500 -X 10 # -X cross validation



done


