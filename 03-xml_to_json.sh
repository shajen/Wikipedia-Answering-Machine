#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)
THREADS=$(grep -c ^processor /proc/cpuinfo)

IN=$ROOT_PATH/files/plwiki-latest-pages-articles.xml.bz2
OUT=$ROOT_PATH/files/plwiki-latest-pages-articles.json.gz

pushd $ROOT_PATH
./manage.py runscript xml_to_json --script-args "$IN $OUT -t $THREADS -i -m 5 -vv"
popd
