#ifndef RLPARSER_LEFT_TO_RIGHT_S2A_PARSER_H
#define RLPARSER_LEFT_TO_RIGHT_S2A_PARSER_H

#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "corpus.h"
#include "system/state.h"
#include "system/system.h"
#include "dynet/expr.h"

namespace po = boost::program_options;

struct Parser {
  dynet::ParameterCollection& model;
  TransitionSystem& sys;
  std::string system_name;

  Parser(dynet::ParameterCollection & m,
         TransitionSystem& s,
         const std::string & sys_name) :
    model(m), sys(s), system_name(sys_name){}

  virtual Parser* copy_architecture(dynet::Model& new_model) = 0;
  virtual void activate_training() = 0;
  virtual void inactivate_training() = 0;
  virtual void new_graph(dynet::ComputationGraph& cg) = 0;
  virtual std::vector<dynet::Expression> get_params() = 0;

  void initialize(dynet::ComputationGraph& cg,
                  const InputUnits& input,
                  State& state);

  void initialize_state(const InputUnits& input,
                        State& state);

  virtual void initialize_parser(dynet::ComputationGraph& cg,
                                 const InputUnits& input) = 0;

  virtual void perform_action(const unsigned& action,
                              dynet::ComputationGraph& cg,
                              State& state) = 0;

  static std::pair<unsigned, float> get_best_action(const std::vector<float>& scores,
                                                    const std::vector<unsigned>& valid_actions);

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::Expression get_scores();
  
  virtual dynet::Expression get_confirm_values(unsigned wid) = 0;
  virtual dynet::Expression get_a_values() = 0;
};

#endif  //  end for RLPARSER_LEFT_TO_RIGHT_S2A_PARSER_H
