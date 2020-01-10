#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH
./manage.py runscript report --script-args="-snf -qc 0 -vv -t 1,10,100 -hs -1.0 -a"
popd
