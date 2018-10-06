#include "eager.h"
#include "logging.h"
#include "corpus.h"
#include <boost/algorithm/string.hpp>
#include <iostream>

Eager::Eager(const Alphabet & action_map, const Alphabet & node_map, const Alphabet & rel_map, const Alphabet & entity_map) :
  TransitionSystem(action_map, node_map, rel_map, entity_map) {
  n_actions = action_map.size();
  _INFO << "TransitionSystem:: show action names:";
  for (const auto& x : action_map.str_to_id) {
    _INFO << "- " << x.first;
  }
}

std::string Eager::name(unsigned id) const {
  BOOST_ASSERT_MSG(id < action_map.size(), "id in illegal range");
  return action_map.get(id);
}

unsigned Eager::num_actions() const { return n_actions; }

void Eager::perform_action(State & state, const unsigned & action) {
  std::string action_type = get_action_type(action, action_map);
  if (action_type == "SHIFT") {
    shift_unsafe(state);
  } else if (action_type == "CONFIRM") {
    confirm_unsafe(state);
  } else if (action_type == "MERGE") {
    merge_unsafe(state);
  } else if (action_type == "ENTITY") {
    entity_unsafe(state);
  } else if (action_type == "NEWNODE") {
    unsigned nid = get_action_arg1(node_map, action);
    newnode_unsafe(state, nid);
  } else if (action_type == "REDUCE") {
    reduce_unsafe(state);
  } else if (action_type == "DROP") {
    drop_unsafe(state);
  } else if (action_type == "CACHE") {
    cache_unsafe(state);
  } else if (action_type == "LEFT") {
    unsigned rid = get_action_arg1(rel_map, action);
    la_unsafe(state, rid);
  } else if (action_type == "RIGHT") {
    unsigned rid = get_action_arg1(rel_map, action);
    ra_unsafe(state, rid);
  } else {
    BOOST_ASSERT_MSG(false, "Illegal Action");
  }
}

void Eager::get_valid_actions(const State & state,
  std::vector<unsigned>& valid_actions) {
  valid_actions.clear();
  for (unsigned a = 0; a < n_actions; ++a) {
    //if (!is_valid_action(state, action_names[a])) { continue; }
    if (!is_valid_action(state, a)) { continue; }
    valid_actions.push_back(a);
  }
  BOOST_ASSERT_MSG(valid_actions.size() > 0, "There should be one or more valid action.");
}

void Eager::shift_unsafe(State & state) const {
  while (state.deque.size() > 0) {
    state.stack.push_back(state.deque.back());
    state.deque.pop_back();
  }
  state.stack.push_back(state.buffer.back());
  state.buffer.pop_back();
}

void Eager::confirm_unsafe(State & state) const {
  state.buffer[state.buffer.size() - 1] = std::make_pair(state.new_amr_node(), 2);
}

void Eager::reduce_unsafe(State & state) const {
  state.stack.pop_back();
}

void Eager::merge_unsafe(State & state) const {
  state.buffer.pop_back();
  state.buffer[state.buffer.size() - 1].second = 1;
  state.buffer[state.buffer.size() - 1].first = 1;
  // state.buffer[state.buffer.size() - 1].first = -1;
}

void Eager::entity_unsafe(State & state) const {
  state.buffer[state.buffer.size() - 1] = std::make_pair(state.new_amr_node(), 2);
}

void Eager::newnode_unsafe(State & state, const unsigned & node) const {
  state.buffer.push_back(std::make_pair(state.new_amr_node(), state.buffer.back().second + 1));
  state.buffer[state.buffer.size() - 2].second += 5;
}

void Eager::drop_unsafe(State & state) const {
  state.buffer.pop_back();
}

void Eager::cache_unsafe(State & state) const {
  state.deque.push_back(state.stack.back());
  state.stack.pop_back();
}

void Eager::la_unsafe(State & state, const unsigned & rel) const {
  unsigned u = state.buffer.back().first;
  unsigned v = state.stack.back().first;
  state.existing_edges.insert({ u, rel });
}

void Eager::ra_unsafe(State& state, const unsigned & rel) const {
  unsigned u = state.stack.back().first;
  unsigned v = state.buffer.back().first;
  state.existing_edges.insert({ u, rel });
}


std::string Eager::get_action_type(const unsigned & idx, const Alphabet & action_map) {
  std::string action = action_map.get(idx);
  std::vector<std::string> terms;
  boost::algorithm::split(terms, action, boost::is_any_of(" \t"), boost::token_compress_on);
  return terms[0];
}

bool Eager::is_valid_action(const State& state, const unsigned& action) const {
  std::string action_type = get_action_type(action, action_map);
  if (action_type == "_UNK_") {
    return false;
  } else if (action_type == "SHIFT") {
    return state.buffer.size() > 0 && state.buffer.back().second > 1;
  } else if (action_type == "CONFIRM") {
    return state.buffer.size() > 0 && state.buffer.back().second < 2;
  } else if (action_type == "MERGE") {
    return state.buffer.size() > 1 && state.buffer.back().second < 2 && state.buffer[state.buffer.size() - 2].second == 0;
  } else if (action_type == "ENTITY") {
    return state.buffer.size() > 0 && state.buffer.back().second < 2;
  } else if (action_type == "REDUCE") {
    return state.stack.size() > 0 && state.stack.back().second > 1;
  } else if (action_type == "DROP") {
    return state.buffer.size() > 0 && state.buffer.back().second == 0;
  } else if (action_type == "CACHE") {
    return state.buffer.size() > 0 && state.stack.size() > 0;
  } else if (action_type == "NEWNODE") {
    return state.buffer.size() > 0 && state.buffer.back().second > 1 && state.buffer.back().second <= 5;
  } else if (action_type == "LEFT" || action_type == "RIGHT") {
    if (state.stack.size() < 1 || state.stack.back().second < 2 || state.buffer.size() < 1 || state.buffer.back().second < 2) {
      return false;
    }
    unsigned u = state.stack.back().first;
    unsigned v = state.buffer.back().first;
    if (action_type == "LEFT") {
      std::swap(u, v);
    }
    std::vector<std::string> terms;
    std::string a_str = action_map.get(action);
    boost::algorithm::split(terms, a_str, boost::is_any_of(" \t"), boost::token_compress_on);
    unsigned rid = rel_map.get(terms[1]);
    return state.existing_edges.find({ u, rid }) == state.existing_edges.end(); 
  } else {
    BOOST_ASSERT_MSG(false, "Illegal Action");
  }
  return true;
}
