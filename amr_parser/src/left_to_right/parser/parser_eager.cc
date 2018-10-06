#include "parser_eager.h"
#include "dynet/expr.h"
#include "logging.h"
#include "system/eager.h"
#include <vector>
#include <random>
#include <boost/algorithm/string.hpp>

dynet::Expression ParserEager::TransitionSystemFunction::get_arg_emb(const std::string & a_str,
                                                                         const Alphabet & arg_map,
                                                                         SymbolEmbedding & arg_emb) {
  std::vector<std::string> terms;
  boost::algorithm::split(terms, a_str, boost::is_any_of(" \t"), boost::token_compress_on);
  unsigned aid = arg_map.get(terms[1]);
  dynet::Expression arg_expr = arg_emb.embed(aid);
  return arg_expr;
}

void ParserEager::EagerFunction::perform_action(const unsigned & action,
                                                dynet::ComputationGraph & cg,
                                                std::vector<dynet::Expression>& stack,
                                                std::vector<dynet::Expression>& buffer,
                                                std::vector<dynet::Expression>& deque,
                                                dynet::RNNBuilder & a_lstm, dynet::RNNPointer & a_pointer,
                                                dynet::RNNBuilder & s_lstm, dynet::RNNPointer & s_pointer,
                                                dynet::RNNBuilder & q_lstm, dynet::RNNPointer & q_pointer,
                                                dynet::RNNBuilder & d_lstm, dynet::RNNPointer & d_pointer,
                                                dynet::Expression & act_expr,
                                                const Alphabet & action_map,
                                                const Alphabet & node_map,
                                                SymbolEmbedding & node_emb,
                                                const Alphabet & rel_map,
                                                SymbolEmbedding & rel_emb,
                                                const Alphabet & entity_map,
                                                SymbolEmbedding & entity_emb,
                                                DenseLayer & confirm_layer,
                                                Merge3Layer & merge_parent,
                                                Merge3Layer & merge_child,
                                                Merge2Layer & merge_token,
                                                Merge2Layer & merge_entity) {
  std::string action_type = Eager::get_action_type(action, action_map);
  
  a_lstm.add_input(a_pointer, act_expr);
  a_pointer = a_lstm.state();
  std::string a_str = action_map.get(action);

  if (action_type == "SHIFT") {
    while (deque.size() > 1) {
      stack.push_back(deque.back());
      s_lstm.add_input(deque.back());
      s_pointer = s_lstm.state();

      deque.pop_back();
      d_pointer = d_lstm.get_head(d_pointer);
    }
    
    stack.push_back(buffer.back());
    s_lstm.add_input(s_pointer, buffer.back());
    s_pointer = s_lstm.state();
    
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (action_type == "CONFIRM") {
    dynet::Expression concept_expr = dynet::rectify(confirm_layer.get_output(buffer.back()));
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
    buffer.push_back(concept_expr);
    q_lstm.add_input(q_pointer, concept_expr);
    q_pointer = q_lstm.state();
  } else if (action_type == "REDUCE") {
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
  } else if (action_type == "MERGE") {
    dynet::Expression token_A = buffer.back();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
    dynet::Expression token_B = buffer.back();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
    dynet::Expression token_AB = dynet::rectify(merge_token.get_output(token_A, token_B));
    buffer.push_back(token_AB);
    q_lstm.add_input(q_pointer, token_AB);
    q_pointer = q_lstm.state();
  } else if (action_type == "ENTITY") {
    dynet::Expression entity_expr = get_arg_emb(a_str, entity_map, entity_emb);
    entity_expr = dynet::rectify(merge_entity.get_output(buffer.back(), entity_expr));

    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);

    buffer.push_back(entity_expr);
    q_lstm.add_input(q_pointer, entity_expr);
    q_pointer = q_lstm.state();
  } else if (action_type == "NEWNODE") {
    dynet::Expression node_expr = get_arg_emb(a_str, node_map, node_emb);
    buffer.push_back(node_expr);
    q_lstm.add_input(q_pointer, node_expr);
    q_pointer = q_lstm.state();
  } else if (action_type == "DROP") {
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (action_type == "CACHE") {
    deque.push_back(stack.back());
    d_lstm.add_input(stack.back());
    d_pointer = d_lstm.state();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
  } else if (action_type == "LEFT") {
    dynet::Expression parent_expr = buffer.back();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
    dynet::Expression child_expr = stack.back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);

    dynet::Expression rel_expr = get_arg_emb(a_str, rel_map, rel_emb);
    dynet::Expression new_parent_expr = dynet::rectify(merge_parent.get_output(parent_expr, rel_expr, child_expr));
    dynet::Expression new_child_expr = dynet::rectify(merge_child.get_output(parent_expr, rel_expr, child_expr));

    buffer.push_back(new_parent_expr);
    q_lstm.add_input(q_pointer, new_parent_expr);
    q_pointer = q_lstm.state();

    stack.push_back(new_child_expr);
    s_lstm.add_input(s_pointer, new_child_expr);
    s_pointer = s_lstm.state();
  } else if (action_type == "RIGHT") {
    dynet::Expression child_expr = buffer.back();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
    dynet::Expression parent_expr = stack.back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);

    dynet::Expression rel_expr = get_arg_emb(a_str, rel_map, rel_emb);
    dynet::Expression new_parent_expr = dynet::rectify(merge_parent.get_output(parent_expr, rel_expr, child_expr));
    dynet::Expression new_child_expr = dynet::rectify(merge_child.get_output(parent_expr, rel_expr, child_expr));

    buffer.push_back(new_child_expr);
    q_lstm.add_input(q_pointer, new_child_expr);
    q_pointer = q_lstm.state();

    stack.push_back(new_parent_expr);
    s_lstm.add_input(s_pointer, new_parent_expr);
    s_pointer = s_lstm.state();
  } else {
    BOOST_ASSERT_MSG(false, "Illegal Action");
  }
}

ParserEager::ParserEager(dynet::ParameterCollection & m,
                         unsigned size_w,  //
                         unsigned dim_w,   // word size, word dim
                         unsigned size_p,  //
                         unsigned dim_p,   // pos size, pos dim
                         unsigned size_t,  //
                         unsigned dim_t,   // pword size, pword dim
                         unsigned size_c,  //
                         unsigned dim_c,   // char size, char dim
                         unsigned size_a,  //
                         unsigned dim_a,   // act size, act dim
                         unsigned size_n,  //
                         unsigned dim_n,   // newnode size, newnode dim
                         unsigned size_r,
                         unsigned dim_r,   // rel size, rel dim
                         unsigned size_e,
                         unsigned dim_e,   // entity size, entity dim
                         unsigned n_layers,
                         unsigned dim_lstm_in,
                         unsigned dim_hidden,
                         const std::string& system_name,
                         TransitionSystem& system,
                         const std::unordered_map<unsigned, std::vector<float>>& embedding,
                         const std::unordered_map<unsigned, Alphabet> & confirm_map,
                         const Alphabet & char_map):
  Parser(m, system, system_name),
  s_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  q_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  a_lstm(n_layers, dim_a, dim_hidden, m),
  d_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  c_lstm(1, dim_c, dim_c, m), 
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  preword_emb(m, size_t, dim_t, false),
  char_emb(m, size_c, dim_c),
  act_emb(m, size_a, dim_a),
  node_emb(m, size_n, dim_n),
  rel_emb(m, size_r, dim_r),
  entity_emb(m, size_e, dim_e),
  merge_input(m, dim_p, dim_t, dim_c * 2, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
  scorer(m, dim_hidden, size_a),
  confirm_layer(m, dim_lstm_in, dim_lstm_in),
  merge_parent(m, dim_lstm_in, dim_r, dim_lstm_in, dim_lstm_in),
  merge_child(m, dim_lstm_in, dim_r, dim_lstm_in, dim_lstm_in),
  merge_token(m, dim_lstm_in, dim_lstm_in, dim_lstm_in),
  merge_entity(m, dim_lstm_in, dim_e, dim_lstm_in),
  p_action_start(m.add_parameters({ dim_a })),
  p_buffer_guard(m.add_parameters({ dim_lstm_in })),
  p_stack_guard(m.add_parameters({ dim_lstm_in })),
  p_deque_guard(m.add_parameters({ dim_lstm_in })),
  sys_func(nullptr),
  pretrained(embedding),
  char_map(char_map),
  confirm_map(confirm_map),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  size_t(size_t), dim_t(dim_t),
  size_a(size_a), dim_a(dim_a), 
  size_n(size_n), dim_n(dim_n),
  size_r(size_r), dim_r(dim_r),
  size_e(size_e), dim_e(dim_e),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {

  for (auto & it : pretrained) {
    preword_emb.p_e.initialize(it.first, it.second);
  }

  _INFO << "Parser:: number of layers " << n_layers;

  for (auto & it : confirm_map) {
    if (it.second.size() > 1) {
      confirm_scorer[it.first] = new DenseLayer(m, dim_hidden, it.second.size());
    }
  }

  if (system_name == "eager") {
    sys_func = new EagerFunction();
  } else {
    _ERROR << "Main:: Unknown transition system: " << system_name;
    exit(1);
  }
}

Parser * ParserEager::copy_architecture(dynet::Model& new_model) {
  Parser * ret = new ParserEager(new_model,
                                    size_w, dim_w,
                                    size_p, dim_p,
                                    size_t, dim_t,
                                    size_c, dim_c,
                                    size_a, dim_a,
                                    size_n, dim_n,
                                    size_r, dim_r,
                                    size_e, dim_e,
                                    n_layers,
                                    dim_lstm_in,
                                    dim_hidden,
                                    system_name,
                                    sys,
                                    pretrained,
                                    confirm_map,
                                    char_map);
  return ret;
}

void ParserEager::activate_training() {
  trainable = true;
  s_lstm.active_training();
  q_lstm.active_training();
  a_lstm.active_training();
  d_lstm.active_training();
  c_lstm.active_training();
  word_emb.active_training();
  pos_emb.active_training();
  pos_emb.active_training();
  rel_emb.active_training();
  //! Don't active preword_emb;
  char_emb.active_training();
  node_emb.active_training();
  act_emb.active_training();
  merge_input.active_training();
  merge.active_training();
  scorer.active_training();
  confirm_layer.active_training();
  merge_parent.active_training();
  merge_child.active_training();
  merge_token.active_training();
  merge_entity.active_training();

  for (auto & it : confirm_scorer) {
    it.second->active_training();
  }
}

void ParserEager::inactivate_training() {
  trainable = false;
  s_lstm.inactive_training();
  q_lstm.inactive_training();
  a_lstm.inactive_training();
  d_lstm.inactive_training();
  c_lstm.inactive_training();
  word_emb.inactive_training();
  pos_emb.inactive_training();
  rel_emb.inactive_training();
  entity_emb.inactive_training();
  //! Don't active preword_emb;
  char_emb.active_training();
  node_emb.active_training();
  act_emb.inactive_training();
  merge_input.inactive_training();
  merge.inactive_training();
  scorer.inactive_training();
  confirm_layer.inactive_training();
  merge_parent.inactive_training();
  merge_child.inactive_training();
  merge_token.inactive_training();
  merge_entity.inactive_training();
  for (auto & it : confirm_scorer) {
    it.second->inactive_training();
  }
}

void ParserEager::perform_action(const unsigned& action,
                                    dynet::ComputationGraph& cg,
                                    State& state) {
  dynet::Expression act_repr = act_emb.embed(action);
  sys_func->perform_action(action, cg, stack, buffer, deque,
    a_lstm, a_pointer, s_lstm, s_pointer, q_lstm, q_pointer, d_lstm, d_pointer, 
    act_repr, sys.action_map, sys.node_map, node_emb, sys.rel_map, rel_emb, sys.entity_map, entity_emb,
    confirm_layer, merge_parent, merge_child, merge_token, merge_entity);
  sys.perform_action(state, action);
}
void ParserEager::new_graph(dynet::ComputationGraph& cg) {
  s_lstm.new_graph(cg);
  q_lstm.new_graph(cg);
  a_lstm.new_graph(cg);
  d_lstm.new_graph(cg);
  c_lstm.new_graph(cg);

  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  preword_emb.new_graph(cg);
  char_emb.new_graph(cg);
  node_emb.new_graph(cg);
  act_emb.new_graph(cg);
  rel_emb.new_graph(cg);
  entity_emb.new_graph(cg);
 
  merge_input.new_graph(cg);
  merge.new_graph(cg);
  scorer.new_graph(cg); 
  confirm_layer.new_graph(cg);
  merge_parent.new_graph(cg);
  merge_child.new_graph(cg);
  merge_token.new_graph(cg);
  merge_entity.new_graph(cg);

  for (auto & it : confirm_scorer) {
    it.second->new_graph(cg);
  }

  confirm_to_one = dynet::ones(cg, { 1 }); 

  if (trainable) {
    action_start = dynet::parameter(cg, p_action_start);
    buffer_guard = dynet::parameter(cg, p_buffer_guard);
    stack_guard = dynet::parameter(cg, p_stack_guard);
    deque_guard = dynet::parameter(cg, p_deque_guard);
  } else {
    action_start = dynet::const_parameter(cg, p_action_start);
    buffer_guard = dynet::const_parameter(cg, p_buffer_guard);
    stack_guard = dynet::const_parameter(cg, p_stack_guard);
    deque_guard = dynet::const_parameter(cg, p_deque_guard);
  }
}

std::vector<dynet::Expression> ParserEager::get_params() {
  std::vector<dynet::Expression> ret;
  for (auto & layer : s_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : q_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : a_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : d_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }

  for (auto & layer : c_lstm.fw_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : c_lstm.bw_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }

  for (auto & e : merge_input.get_params()) { ret.push_back(e); }
  for (auto & e : merge.get_params()) { ret.push_back(e); }
  for (auto & e : scorer.get_params()) { ret.push_back(e); }
  for (auto & e : confirm_layer.get_params()) { ret.push_back(e); }
  for (auto & e : merge_parent.get_params()) { ret.push_back(e); }
  for (auto & e : merge_child.get_params()) { ret.push_back(e); }
  for (auto & e : merge_token.get_params()) { ret.push_back(e); }
  for (auto & e : merge_entity.get_params()) { ret.push_back(e); }

  //for (auto & it : confirm_scorer) {
  //  for (auto & e : it.second->get_params()) {
  //    ret.push_back(e);
  //  }
  //}

  ret.push_back(action_start);
  ret.push_back(buffer_guard);
  ret.push_back(stack_guard);
  ret.push_back(deque_guard);
  ret.push_back(c_lstm.fw_guard);
  ret.push_back(c_lstm.bw_guard);
  return ret;
}

void ParserEager::initialize_parser(dynet::ComputationGraph & cg,
                                        const InputUnits & input) {
  s_lstm.start_new_sequence();
  q_lstm.start_new_sequence();
  a_lstm.start_new_sequence();
  d_lstm.start_new_sequence();

  a_lstm.add_input(action_start);

  unsigned len = input.size();
  stack.clear();
  buffer.resize(len + 1);

  // Pay attention to this, if the guard word is handled here, there is no need
  // to insert it when loading the data.
  buffer[0] = buffer_guard;
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;
    unsigned aux_wid = input[i].aux_wid;
    if (!pretrained.count(aux_wid)) { aux_wid = 0; }

    buffer[len - i] = dynet::rectify(merge_input.get_output(
      pos_emb.embed(pid), preword_emb.embed(aux_wid), c_lstm.get_h(char_emb, input[i].c_id)));
    //buffer[len - i] = dynet::rectify(merge_input.get_output(
    //  word_emb.embed(wid), pos_emb.embed(pid), preword_emb.embed(aux_wid), c_lstm.get_h(char_emb, input[i].c_id)
    //));
  }

  // push word into buffer in reverse order, pay attention to (i == len).
  for (unsigned i = 0; i <= len; ++i) {
    q_lstm.add_input(buffer[i]);
  }

  stack.push_back(stack_guard);
  s_lstm.add_input(stack.back());


  while (!deque.empty()) {
    deque.pop_back();
  }
  deque.push_back(deque_guard);
  d_lstm.add_input(deque.back());

  a_pointer = a_lstm.state();
  s_pointer = s_lstm.state();
  q_pointer = q_lstm.state();
  d_pointer = d_lstm.state();
}

dynet::Expression ParserEager::get_a_values() {
  return scorer.get_output(dynet::rectify(merge.get_output(
    s_lstm.get_h(s_pointer).back(),
    q_lstm.get_h(q_pointer).back(),
    a_lstm.get_h(a_pointer).back(),
    d_lstm.get_h(d_pointer).back())
    ));
}

dynet::Expression ParserEager::get_confirm_values(unsigned wid) {
  if (confirm_scorer.find(wid) == confirm_scorer.end()) {
    return confirm_to_one; //[1.0]
  } else {
    dynet::Expression tmp_expr = dynet::rectify(merge.get_output(
      s_lstm.get_h(s_pointer).back(),
      q_lstm.get_h(q_pointer).back(),
      a_lstm.get_h(a_pointer).back(),
      d_lstm.get_h(d_pointer).back()));
    return confirm_scorer[wid]->get_output(tmp_expr);
  }
}