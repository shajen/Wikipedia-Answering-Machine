#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH
./manage.py runscript upload_questions --script-args "$ROOT_PATH/files/czywieszki/pierwszy_zestaw_czywieszek.txt -m 20 -vvv"
./manage.py runscript upload_questions --script-args "$ROOT_PATH/files/czywieszki/drugi_zestaw_czywieszek.txt -m 20 -vvv"
./manage.py runscript upload_questions --script-args "$ROOT_PATH/files/czywieszki/czywieszki_wiki_main_2019_06_16.txt -m 20 -vvv"
popd
