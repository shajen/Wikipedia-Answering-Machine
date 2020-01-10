#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)
THREADS=$(grep -c ^processor /proc/cpuinfo)

pushd $ROOT_PATH
METHOD="final"
QUESTIONS="100000"

W2V="-w2vf /home/shajen/mgr/sources/files/word2vec_100_3_polish.bin -w2vs 100"
CACHE="-cd /home/shajen/mgr/sources/cache/"
DATA_LOADER="-lm_qwc 20 -lm_atwc 20 -lm_acwc 100 -cm_qwc 100 -cm_atwc 100 -cm_acwc 1000"
OTHER="-t $THREADS -m $METHOD -dti 100 -vv -q $QUESTIONS -dp 60:20:20"
COMMON="$W2V $CACHE $DATA_LOADER $OTHER"

MODELS_TF="-tm -vm_cos"
MODELS_W2V="-w2vm"
MODELS_NN="-nm_m 1"
MODELS_EA="-ea -ea_p 100 -ea_mp $METHOD"

./manage.py runscript resolve --script-args="$COMMON $MODELS_TF  -N 0,15,250 -n 1"
./manage.py runscript resolve --script-args="$COMMON $MODELS_TF  -N 0,15,250 -n 2"
./manage.py runscript resolve --script-args="$COMMON $MODELS_TF  -N 0,3 -n 1 -T"
./manage.py runscript resolve --script-args="$COMMON $MODELS_TF  -N 0,3 -n 2 -T"

./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 0"
./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 10"
./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 999"
./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 0 -T"
./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 10 -T"
./manage.py runscript resolve --script-args="$COMMON $MODELS_W2V -tn 999 -T"

./manage.py runscript resolve --script-args="$COMMON $MODELS_NN -e 50 -cnn"
./manage.py runscript resolve --script-args="$COMMON $MODELS_NN -e 200 -dan"

./manage.py runscript resolve --script-args="$COMMON $MODELS_EA -e 20"

popd
