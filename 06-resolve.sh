#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)
THREADS=$(grep -c ^processor /proc/cpuinfo)

pushd $ROOT_PATH
./manage.py runscript resolve --script-args="-t $THREADS -m test -dti 3 -vv -n 1"
./manage.py runscript resolve --script-args="-t $THREADS -m test -dti 3 -vv -n 2"
./manage.py runscript resolve --script-args="-t $THREADS -m test -dti 3 -vv -n 1 -T"
./manage.py runscript resolve --script-args="-t $THREADS -m test -dti 3 -vv -n 2 -T"
popd
