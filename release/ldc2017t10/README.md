TAMR alignment for LDC2017T10
=============================

You can replace the JAMR alignment with ours using the following commands:

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

Similar to LDC2017T10, you need to do a little patching on the original data
to use this alignment. The patch file in under this folder with `.patch` suffix.
 
In addition to the patching, you will also need to remove the entity linking (`:wiki`).
We provide a python script `remove_wiki.py` and you can use it with as
```
python remove_wiki.py /path/to/your/input > /path/to/your/output
```