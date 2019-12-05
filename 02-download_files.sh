#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH/files
wget -N https://github.com/morfologik/polimorfologik/releases/download/2.1/polimorfologik-2.1.zip
unzip -f polimorfologik-2.1.zip polimorfologik-2.1.txt
wget -N https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2
wget -N https://github.com/sdadas/polish-nlp-resources/releases/download/v1.0/word2vec.zip
wget -N https://clarin-pl.eu/dspace/bitstream/handle/11321/327/skipgram_v100.zip
unzip skipgram_v100.zip skipgram/skip_gram_v100m8.w2v.txt
unzip word2vec.zip
popd
