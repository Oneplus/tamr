TAMR alignment for LDC2014T12
=============================

We release the alignment file (the output of `rule_base_aligner.py`).
It's in the zipped format. Extract the zip file to use it.

You can replace the JAMR alignment with ours using the following
commands:
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

Since JAMR and CAMR uses different tokenizer, we provide
alignment for cdec tokenizer (used by JAMR) and stanford tokenizer
(used by CAMR).

- for cdec tokenizer: see `amr-release-1.0-training_fix.txt.cdec_tok.tamr_alignment.bz2`
- for stanford tokenizer: see `amr-release-1.0-training_fix.txt.sd_tok.tamr_alignment.bz2`

To reproduce the alignment, you need to do a patch on the original ldc2014t12,
because there are illegal AMR graph in the original data (like two concepts
using the same variable). You can get the patched ldc2014t12 with the following
steps:

### Merge the Training Data

Go into the `amr_anno_1.0/data/split/training` folder of the original release of `ldc2014t12`,
and get a concatenated training data with the following commands:
```
cat amr-release-1.0-training-proxy.txt \
    amr-release-1.0-training-bolt.txt \
    amr-release-1.0-training-dfa.txt \
    amr-release-1.0-training-mt09sdl.txt \
    amr-release-1.0-training-xinhua.txt > amr-release-1.0-training.txt
```

### Patching
Do the patching with the following commands:
```
patch amr-release-1.0-training.txt \
    -i amr-release-1.0-training_fix.patch \
    -o amr-release-1.0-training_fix.txt
``` 

It's done!