Awesome AMR Parsers
===================

As you may know, there are several open-source AMR parsers and our
aligner improves these parsers. I would like to share
some experiences with how to plugin in our alignments into the existing
AMR parsers, although running most of these parser requires
a certain amount of hacking.

## [JAMR](https://github.com/jflanigan/jamr)

"A Discriminative Graph-Based Parser for the Abstract Meaning Representation",
Jeffrey Flanigan, Sam Thomson, Jaime Carbonell, Chris Dyer, and Noah A. Smith.

### Alignment Hacking
The JAMR experiments are carried out with a pipeline of shell scripts.
This made plugining our alignments very easy and saved a lot of my life.
The hook for replacing the alignment is in the preprocessing script:

```
jamr/scripts/preprocessing/cmd.aligned
```

It takes an input AMR file with `# ::tok` header for each graph and adds
an additional `# ::alignments` header to each graph.

To replace the alignment, I commented this line and inject our alignment.
You can do the same by [TODO].

### Results on LDC2014T12

| JAMR parser     | Smatch |
|-----------------|--------|
| +JAMR alignment |        |
| +Our alignment  |        |

### Note
- JAMR uses the `cdec` tokenizer and our released alignments 
include the one preprocessed with `cdec`.

## [CAMR](https://github.com/c-amr/camr)
"A Transition-based Algorithm for AMR Parsing", Chuan Wang, Nianwen Xue, and Sameer Pradhan

### Alignment Hacking
The CAMR uses a single program entry `amr_parsing.py` in their project. 
You can replace the JAMR aligner generated training file with ours.

### Results on LDC2014T12

| CAMR parser     | Smatch |
|-----------------|--------|
| +JAMR alignment |        |
| +Our alignment  |        |

### Note
- CAMR uses StanfordCoreNLP as tokenizer. In our release,
we also includes an alignment on this tokenization.

## [CCG-AMR](https://github.com/clic-lab/amr)
"Broad-coverage CCG Semantic Parsing with AMR", Yoav Artzi, Kenton Lee, and Luke Zettlemoyer.

### Alignment Hacking
Alignment in the CCG-AMR is provided in a standalone file.

## [CacheTransition-Seq2Seq](https://github.com/xiaochang13/CacheTransition-Seq2seq)

