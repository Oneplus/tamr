#ifndef TRAIN_SUPERVISED_H
#define TRAIN_SUPERVISED_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "parser/parser.h"
#include "dynet/training.h"
#include "train.h"

namespace po = boost::program_options;

struct SupervisedTrainer : public Trainer {
  enum ORACLE_TYPE { kStatic, kDynamic, kPseudoDynamic };
  enum OBJECTIVE_TYPE { kCrossEntropy, kRank, kBipartieRank };
  ORACLE_TYPE oracle_type;
  OBJECTIVE_TYPE objective_type;
  Parser* parser;
  Parser* pseudo_dynamic_oracle;
  dynet::Model* pseudo_dynamic_oracle_model;
  float do_pretrain_iter;
  float do_explore_prob;
  std::string system;


  static po::options_description get_options();

  SupervisedTrainer(const po::variables_map& conf, Parser* parser);

  /* Code for supervised pretraining. */
  void train(const po::variables_map& conf,
             Corpus& corpus,
             const std::string& name,
             const std::string& output);

  float train_on_one_full_tree(const InputUnits& input_units,
                               const ActionUnits& action_units,
                               dynet::Trainer* trainer,
                               unsigned iter);
};

#endif  //  end for TRAIN_SUPERVISED_H