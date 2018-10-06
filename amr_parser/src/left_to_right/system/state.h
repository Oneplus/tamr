#ifndef RLPARSER_LEFT_TO_RIGHT_STATE_H
#define RLPARSER_LEFT_TO_RIGHT_STATE_H

#include <vector>
#include <set>

struct State {
  static const unsigned MAX_N_WORDS = 1024;

  std::vector<std::pair<unsigned, unsigned>> stack;
  std::vector<std::pair<unsigned, unsigned>> buffer;
  std::vector<std::pair<unsigned, unsigned>> deque;
  std::vector<unsigned> aux;

  std::set < std::vector<unsigned> > existing_edges;

  unsigned num_nodes;

  State(unsigned n);

  unsigned new_amr_node();

  bool terminated();
};


#endif  //  end for RLPARSER_LEFT_TO_RIGHT_STATE_H