#!/usr/bin/env python
from __future__ import print_function
import sys


def main():
    lexicon = set()
    is_header = True
    for line in open(sys.argv[1], 'r'):
        if is_header:
            is_header = False
            continue
        tokens = line.strip().split(',')
        verb, noun = tokens[0], tokens[3]
        verb = verb.split('%')[0]
        noun = noun.split('%')[0]
        lexicon.add((verb, noun))
    for verb, noun in lexicon:
        print('{0},{1}'.format(verb, noun))


if __name__ == "__main__":
    main()
