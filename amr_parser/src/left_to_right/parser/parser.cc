#include "parser.h"
#include "dynet/expr.h"
#include "corpus.h"
#include "logging.h"
#include <vector>
#include <random>

std::pair<unsigned, float> Parser::get_best_action(const std::vector<float>& scores,
                                                   const std::vector<unsigned>& valid_actions) {
  unsigned best_a = valid_actions[0];
  float best_score = scores[best_a];
  //! should use next valid action.
  for (unsigned i = 1; i < valid_actions.size(); ++i) {
    unsigned a = valid_actions[i];
    if (best_score < scores[a]) {
      best_a = a;
      best_score = scores[a];
    }
  }
  return std::make_pair(best_a, best_score);
}

dynet::Expression Parser::get_scores() {
  return get_a_values();
}

void Parser::initialize(dynet::ComputationGraph & cg,
                        const InputUnits & input,
                        State & state) {
  initialize_state(input, state);
  initialize_parser(cg, input);
}

void Parser::initialize_state(const InputUnits & input, State & state) {
  unsigned len = input.size();
  state.buffer.resize(len);
  for (unsigned i = 0; i < len; ++i) { state.buffer[len - i - 1] = std::make_pair(input[i].wid, 0); }
  state.buffer[0].second = 2; //Corpus::ROOT;
}