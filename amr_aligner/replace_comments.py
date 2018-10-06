#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from amr.aligned import AlignmentReader, Alignment


def main():
    cmd = argparse.ArgumentParser('Get the block that contains certain amr graph.')
    cmd.add_argument('-lexicon', help='the path to the alignment file.')
    cmd.add_argument('-data', help='the path to the alignment file.')
    cmd.add_argument('-key', required=True, help='the key')
    cmd.add_argument('-remove_node_edge_and_root', default=False, action='store_true', help='')
    opt = cmd.parse_args()

    lexicon = {}
    for data in open(opt.lexicon, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        assert len(lines) == 2
        lexicon[lines[0].strip()] = lines[1].strip()

    signature = '# ::{0}'.format(opt.key)
    handler = AlignmentReader(opt.data)
    for block in handler:
        graph = Alignment(block)
        for line in block:
            if opt.remove_node_edge_and_root and\
                    (line.startswith('# ::node') or line.startswith('# ::edge') or line.startswith('# ::root')):
                continue
            if line.startswith('#'):
                if not line.startswith(signature):
                    print(line.encode('utf-8'))
                else:
                    print(lexicon[graph.n])
        print(graph.amr_graph.encode('utf-8'), end='\n\n')


if __name__ == "__main__":
    main()
