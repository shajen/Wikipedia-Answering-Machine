#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)
THREADS=$(grep -c ^processor /proc/cpuinfo)

pushd $ROOT_PATH
METHOD="test"
NEIGHBORS="0,10,20"
MINIMAL_IDF="0,1.6094379124341003,3.2188758248682006,4.8283137373023015,6.437751649736401"
PF="1,2,3,4"

./manage.py runscript resolve --script-args="-t $THREADS -m $METHOD -N $NEIGHBORS -mwiw $MINIMAL_IDF -pf $PF -dti 3 -vv -n 1"
./manage.py runscript resolve --script-args="-t $THREADS -m $METHOD -N $NEIGHBORS -mwiw $MINIMAL_IDF -pf $PF -dti 3 -vv -n 2"
./manage.py runscript resolve --script-args="-t $THREADS -m $METHOD -N $NEIGHBORS -mwiw $MINIMAL_IDF -pf $PF -dti 3 -vv -n 1 -T"
./manage.py runscript resolve --script-args="-t $THREADS -m $METHOD -N $NEIGHBORS -mwiw $MINIMAL_IDF -pf $PF -dti 3 -vv -n 2 -T"
popd
