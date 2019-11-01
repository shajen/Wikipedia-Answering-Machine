#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH
#./manage.py runscript resolve --script-args="-t 1 -m fast_001 -dti 3 -vvv -mwiw 0 -pf 3.0"
./manage.py runscript resolve --script-args="-t 8 -m full_001 -dti 3 -vv"
popd
