#ifndef RLPARSER_LEFT_TO_RIGHT_SYSTEM_H
#define RLPARSER_LEFT_TO_RIGHT_SYSTEM_H

#include <vector>
#include "state.h"
#include "corpus.h"

struct TransitionSystem {
  enum REWARD { kLocal, kGlobal, kGlobalMaxout };
  REWARD reward_type;

  Alphabet action_map;
  Alphabet node_map;
  Alphabet rel_map;
  Alphabet entity_map;

  TransitionSystem(const Alphabet & action_map,
                   const Alphabet & node_map,
                   const Alphabet & rel_map,
                   const Alphabet & entity_map) :
    action_map(action_map), node_map(node_map), rel_map(rel_map), entity_map(entity_map) {}

  unsigned get_action_arg1(const Alphabet & map, const unsigned & action);

  virtual std::string name(unsigned id) const = 0;

  virtual unsigned num_actions() const = 0;

  virtual void perform_action(State& state, const unsigned& action) = 0;

  virtual bool is_valid_action(const State& state, const unsigned& act) const = 0; 

  virtual void get_valid_actions(const State& state, std::vector<unsigned>& valid_actions) = 0;
};

#endif  //  end for RLPARSER_LEFT_TO_RIGHT_SYSTEM_H
