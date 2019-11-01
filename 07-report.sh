#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH
#./manage.py runscript report --script-args="$1 $2 $3 $4 $5 $6 $7 $8 $9"
./manage.py runscript report --script-args="-snf -qc 0 -vv"
#./manage.py runscript report --script-args="-snf -qc 0 -vv -mp test3"
#./manage.py runscript report --script-args="-snf -qc 0 -vv -mp test_004"
#./manage.py runscript report --script-args="-snf -qc 0 -vv -mp try"
popd
