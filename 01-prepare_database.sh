#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH
./manage.py migrate
popd
