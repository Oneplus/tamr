#ifndef TESTING_H
#define TESTING_H

#include "parser/parser.h"

struct Tester {
  /*enum TEST_TARGET { kTrain, kDevelopment };
  TEST_TARGET test_target;
  bool enable_decision_acc_test;
  bool enable_pred_detail_test;

  Parser* parser;
  unsigned n_tests;*/
  
  static po::options_description get_options();

  /*Tester(const po::variables_map& conf, Parser* parser_);

  void test(const po::variables_map& conf,
            Corpus& corpus,
            const std::string& model_name);*/
};

#endif  //  end for TESTING_H