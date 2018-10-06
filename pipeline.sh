#!/bin/bash
set -ueo pipefail
#===================================
# Config directory
#===================================
# Please specify the JAMR home here.
JAMR_HOME=/Users/yijialiu/work/projects/jamr/
# Please specify the TAMR home here.
TAMR_HOME=/Users/yijialiu/work/projects/tamr/

if [ -z "$TAMR_HOME" ]; then
    echo 'Error: please specify $TAMR_HOME'
    exit 1
fi

if [ -z "$JAMR_HOME" ]; then
    echo 'Error: please specify $JAMR_HOME'
    exit 1
fi

TAMR_DATA=${TAMR_HOME}/data
TAMR_LP_DATA=${TAMR_DATA}/little_prince
TAMR_ALIGNER=${TAMR_HOME}/amr_aligner
TAMR_PARSER=${TAMR_HOME}/amr_parser

#===================================
# Download data
#===================================
echo 'Downloading dataset (little prince) ...'
mkdir -p ${TAMR_LP_DATA}
wget -O ${TAMR_LP_DATA}/training.txt https://amr.isi.edu/download/amr-bank-struct-v1.6-training.txt
wget -O ${TAMR_LP_DATA}/dev.txt      https://amr.isi.edu/download/amr-bank-struct-v1.6-dev.txt
wget -O ${TAMR_LP_DATA}/test.txt     https://amr.isi.edu/download/amr-bank-struct-v1.6-test.txt

pushd "$JAMR_HOME" > /dev/null
set -x

#==================================
# Run JAMR baseline aligner
#==================================
. scripts/config.sh
for split in training dev test;
do
    echo 'Running JAMR aligner on '${split};
    #scripts/ALIGN.sh < ${TAMR_LP_DATA}/${split}.txt > ${TAMR_LP_DATA}/${split}.txt.aligned
done

pushd "$TAMR_ALIGNER" > /dev/null
#==================================
# Run TAMR aligner
#==================================
for split in training dev test;
do
    echo 'Running TAMR aligner on '${split};
    python rule_base_align.py \
        -verbose \
        -data \
        ${TAMR_LP_DATA}/${split}.txt.aligned \
        -output \
        ${TAMR_LP_DATA}/${split}.txt.alignment \
        -wordvec \
        ${TAMR_ALIGNER}/resources/word2vec/glove.840B.300d.w2v.ldc2014t12_filtered \
        -trials \
        10000 \
        -improve_perfect \
        -morpho_match \
        -semantic_match
done

#==================================
# Replace the alignments
#==================================
for split in training dev test;
do
    echo 'Replacing the alignments on '${split};
    python replace_comments.py \
        -key \
        alignments \
        -lexicon \
        ${TAMR_LP_DATA}/${split}.txt.alignment \
        -data \
        ${TAMR_LP_DATA}/${split}.txt.aligned \
        > ${TAMR_LP_DATA}/${split}.txt.new_aligned
done

#=================================
# Generate actions
#=================================
for split in training dev test;
do
    echo 'Generating actions on '${split};
    python eager_oracle.py \
        -mod \
        dump \
        -aligned \
        ${TAMR_LP_DATA}/${split}.txt.new_aligned \
        > ${TAMR_LP_DATA}/${split}.txt.actions
done

#================================
# Training and testing the parser
#================================
./amr_parser/bin/parser_l2r \
    --dynet-seed \
    1 \
    --train \
    --training_data \
    ./data/little_prince/training.txt.actions \
    --devel_data \
    ./data/little_prince/dev.txt.actions \
    --test_data \
    ./data/little_prince/test.txt.actions \
    --pretrained \
    ./amr_aligner/resources/word2vec/glove.840B.300d.w2v.ldc2014t12_filtered \
    --model \
    data/little_prince/model \
    --optimizer_enable_eta_decay \
    true \
    --optimizer_enable_clipping \
    true \
    --external_eval \
    ./amr_parser/scripts/eval_eager.sh \
    --devel_gold \
    ./data/little_prince/dev.txt.new_aligned \
    --test_gold \
    ./data/little_prince/test.txt.new_aligned
