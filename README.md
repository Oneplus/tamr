tamr
====

A transition-based AMR parser along with an aligner tuned by the parser.
Used in our EMNLP 2018 paper [An AMR Aligner Tuned by Transition-based Parser]().


## Notion

In the following sections, we will use the following notions:

- `${TAMR_HOME}`: the root directory of the project
- `${TAMR_ALIGNER}`: the directory of the AMR aligner, which equals
 to `${TAMR_HOME}/amr_aligner`
- `${TAMR_PARSER}`: the directory of the transition-based aligner, which equals
 to `${TAMR_HOME}/amr_parser`
 
## Aligner

The code for AMR aligner is under `${TAMR_ALIGNER}`.

### Pre-requisites

- python2.7
- JAMR
- nltk
- gensim
- penman
- Cython (optional, for fast_smatch.py)

### Prepare resource
We use `word2vec` for semantic matching. See the [README.md](https://github.com/Oneplus/tamr/tree/master/amr_aligner/resources/word2vec)
for more information about filtering wordvec.

### Prepare data
Our alignment is built on the JAMR alignment results.
You can get the input data with the following commends:
```
pushd "$JAMR_HOME" > /dev/null
. scripts/config.sh
scripts/ALIGN.sh < /path/to/your/input/data > /path/to/your/baseline/data
```

### Run the Aligner
Go into `${TAMR_ALIGNER}` and run the following commands:

```
python rule_base_align.py \
    -verbose \
    -data \
    /path/to/your/baseline/data \
    -output \
    /path/to/your/alignment/data \
    -wordvec \
    /path/to/your/wordvec/data \
    -trials \
    10000 \
    -improve_perfect \
    -morpho_match \
    -semantic_match
```

The quality of an alignment is evaluated by the smatch
score of the graph
it leads to. Here using `-improve_perfect` will
update the alignment even with the baseline alignment
achieve an smatch score of 1.0.

The output alignment is shown as blocks of results in the following format:
```
id
# ::alignment:
```

After getting the alignment, use the following commands to generate
new alignment:
```
python replace_comments.py \
    -key \
    alignments \
    -lexicon \
    /path/to/your/alignment/data \
    -data \
    /path/to/your/baseline/data \
    > /path/to/your/new/alignment/data
```

You can also use `replace_comments.py` to yield aligned AMR file
for LDC2014T12 with the alignment we release.

## Parser

### Pre-requisites

- cmake
- c++ supporting c++11
- eigen

### Build

Before compiling, you need to fetch the `dynet` and `dynet_layer` with
```
git submodule init
git submodule update
```
under `${TAMR_HOME}`.

After fetching the submodules, run the following commends.

```
cd amr_parser
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/your/eigen/
make
```

The compilation will generate an executable under `${TAMR_PARSER}/bin/`.

### Prepare data

After getting your data with alignment,
do run the `${TAMR_ALIGNER}/eager_oracle.py`
to generate training action file for the alignment as
```
python eager_oracle.py \
    -mod \
    dump \
    -aligned \
    /path/to/your/new/alignment/data \
    > /path/to/your/actions
```

### Training the Parser
With the following commands under `$TAMR_PARSER`:
```
./amr_parser/bin/parser_l2r \
    --dynet-seed \
    1 \
    --train \
    --training_data \
    /path/to/your/new/actions/training/data \
    --devel_data \
    /path/to/your/new/actions/dev/data \
    --test_data \
    /path/to/your/new/actions/test/data \
    --pretrained \
    /path/to/your/embedding/file \
    --model \
    data/little_prince/model \
    --optimizer_enable_eta_decay \
    true \
    --optimizer_enable_clipping \
    true \
    --external_eval \
    ./amr_parser/scripts/eval_eager.sh \
    --devel_gold \
    /path/to/your/new/alignment/dev/data \
    --test_gold \
    /path/to/your/new/alignment/test/data \
    --max_iter \
    1
```

## Released Alignments
 
### [LDC2014T12](https://catalog.ldc.upenn.edu/LDC2014T12)

You can find our alignment for LDC2014T12 under `${TAMR_HOME}/release`.
Since JAMR and CAMR use different tokenization, our release includes
the alignment processed with cdec tokenization and stanford tokenization.

### [LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)

WIP

## Pipeline Script

We demonstrate the process in the `pipeline.sh` script.

## Awesome AMR

Our alignment helps other AMR parser to achieve better performance.
We show how to hack into several open-source AMR parser and replace
their alignment with ours in the [awesome.md](https://github.com/Oneplus/tamr/blob/master/awesome.md).

## Contact

Yijia Liu <<yjliu@ir.hit.edu.cn>>
