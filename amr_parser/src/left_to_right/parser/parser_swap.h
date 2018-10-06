#ifndef PARSER_FULL_H
#define PARSER_FULL_H

#include "parser.h"
#include "lstm.h"
#include "dynet_layer/layer.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserSwap : public Parser {
  struct TransitionSystemFunction {
    virtual void perform_action(const unsigned& action,
                                dynet::ComputationGraph& cg,
                                std::vector<dynet::Expression>& stack,
                                std::vector<dynet::Expression>& buffer,
                                dynet::RNNBuilder& a_lstm, dynet::RNNPointer& a_pointer,
                                dynet::RNNBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                                dynet::RNNBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                                dynet::Expression& act_expr,
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
                                Merge2Layer & merge_entity) = 0;
    dynet::Expression get_arg_emb(const std::string & a_str, const Alphabet & arg_map, SymbolEmbedding & arg_emb);
  };

  struct SwapFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        std::vector<dynet::Expression>& stack,
                        std::vector<dynet::Expression>& buffer,
                        dynet::RNNBuilder& a_lstm, dynet::RNNPointer& a_pointer,
                        dynet::RNNBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                        dynet::RNNBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                        dynet::Expression& act_expr,
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
                        Merge2Layer & merge_entity) override;
  };

  LSTMBuilder s_lstm;
  LSTMBuilder q_lstm;
  LSTMBuilder a_lstm;
  BiLSTMBuilder c_lstm;

  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;
  SymbolEmbedding act_emb;
  SymbolEmbedding char_emb;
  SymbolEmbedding node_emb;
  SymbolEmbedding rel_emb;
  SymbolEmbedding entity_emb;

  Merge3Layer merge_input;    // merge (pos, pretained, char_emb)
  Merge3Layer merge;          // merge (s_lstm, q_lstm, a_lstm)
  Merge3Layer merge_parent;  // merge (parent, rel, child) -> parent
  Merge3Layer merge_child; // merge (parent, rel, child) -> child
  Merge2Layer merge_token; // merge (A, B) -> AB
  Merge2Layer merge_entity; // merge (AB, entity_label) -> X
  DenseLayer scorer;        // Q / A value scorer.
  DenseLayer confirm_layer;
  
  Alphabet char_map;
  std::unordered_map<unsigned, DenseLayer*> confirm_scorer; //confirm scorer.
  std::unordered_map<unsigned, Alphabet> confirm_map;

  dynet::Expression confirm_to_one;

  dynet::Parameter p_action_start;  // start of action
  dynet::Parameter p_buffer_guard;  // end of buffer
  dynet::Parameter p_stack_guard;   // end of stack
  dynet::Expression action_start;
  dynet::Expression buffer_guard;
  dynet::Expression stack_guard;

  /// state machine
  dynet::RNNPointer s_pointer;
  dynet::RNNPointer q_pointer;
  dynet::RNNPointer a_pointer;
  std::vector<dynet::Expression> stack;
  std::vector<dynet::Expression> buffer;

  bool trainable;
  /// The reference
  TransitionSystemFunction* sys_func;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  /// The Configurations: useful for other models.
  unsigned size_w, dim_w, size_p, dim_p, size_t, dim_t, size_c, dim_c, size_a, dim_a, size_n, dim_n, size_r, dim_r, size_e, dim_e;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  explicit ParserSwap(dynet::ParameterCollection& m,
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
                      const std::unordered_map<unsigned, std::vector<float>>& pretrained,
                      const std::unordered_map<unsigned, Alphabet> & confirm_map,
                      const Alphabet & char_map);

  Parser* copy_architecture(dynet::Model& new_model) override;
  void activate_training() override;
  void inactivate_training() override;
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::Expression> get_params() override;

  void initialize_parser(dynet::ComputationGraph& cg,
                         const InputUnits& input) override;

  void perform_action(const unsigned& action,
                      dynet::ComputationGraph& cg,
                      State& state) override;

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::Expression get_confirm_values(unsigned wid) override;
  dynet::Expression get_a_values() override;
};

#endif  //  end for PARSER_H
