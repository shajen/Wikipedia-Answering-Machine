#!/bin/bash

ROOT_PATH=$(git -C $(dirname $0) rev-parse --show-toplevel)

pushd $ROOT_PATH/files
wget -N https://github.com/morfologik/polimorfologik/releases/download/2.1/polimorfologik-2.1.zip
unzip -f polimorfologik-2.1.zip polimorfologik-2.1.txt
wget -N https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2
popd
