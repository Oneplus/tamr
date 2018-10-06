#include "state.h"


State::State(unsigned n) : num_nodes(0) {
}

unsigned State::new_amr_node() {
  return num_nodes++;
}

bool State::terminated() {
  return stack.empty() && buffer.empty();
}
