#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "corpus.h"
#include "dynet/model.h"
#include "dynet/training.h"

namespace po = boost::program_options;

void random_replace_singletons(const unsigned& unk_strategy,
                               const float& unk_prob,
                               const std::set<unsigned>& singletons,
                               const unsigned& kUNK,
                               InputUnits& units);

void restore_singletons(const unsigned& unk_strategy,
                        InputUnits& units);

void get_orders(Corpus& corpus,
                std::vector<unsigned>& order);

po::options_description get_optimizer_options();

dynet::Trainer* get_trainer(const po::variables_map& conf,
                            dynet::ParameterCollection& model);

void update_trainer(const po::variables_map& conf,
                    const float & eta0,
                    const float & iter,
                    dynet::Trainer* trainer);

std::string get_model_name(const po::variables_map& conf,
                           const std::string& prefix);

#endif  //  end for TRAIN_H