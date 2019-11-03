#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)
THREADS=$(grep -c ^processor /proc/cpuinfo)

JSON=$ROOT_PATH/files/plwiki-latest-pages-articles.json.gz
MORFOLOGIK=$ROOT_PATH/files/polimorfologik-2.1.txt
STOP_WORDS=$ROOT_PATH/files/stop_words.txt

pushd $ROOT_PATH
./manage.py runscript upload_json --script-args "$JSON $MORFOLOGIK $STOP_WORDS -t $THREADS -vv"
popd
