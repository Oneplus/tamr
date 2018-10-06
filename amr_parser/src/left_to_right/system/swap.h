#ifndef RLPARSER_LEFT_TO_RIGHT_SWAP_H
#define RLPARSER_LEFT_TO_RIGHT_SWAP_H

#include "system.h"

struct Swap : public TransitionSystem {
  unsigned n_actions;

  Swap(const Alphabet & action_map, const Alphabet & node_map, const Alphabet & rel_map, const Alphabet & entity_map);

  std::string name(unsigned id) const override;

  unsigned num_actions() const override;

  void perform_action(State& state, const unsigned& action) override;

  void get_valid_actions(const State& state,
    std::vector<unsigned>& valid_actions) override;

  bool is_valid_action(const State& state, const unsigned& act) const override;

  void shift_unsafe(State& state) const;
  void confirm_unsafe(State & state) const;
  void reduce_unsafe(State& state) const;
  void merge_unsafe(State& state) const;
  void entity_unsafe(State & state) const;
  void newnode_unsafe(State& state, const unsigned & node) const;
  void swap_unsafe(State& state) const;
  void la_unsafe(State & state, const unsigned & rel) const;
  void ra_unsafe(State& state, const unsigned & rel) const;

  static std::string get_action_type(const unsigned& action, const Alphabet & action_map);

};

#endif  //  end for RLPARSER_LEFT_TO_RIGHT_SWAP_H