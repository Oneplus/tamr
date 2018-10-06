#include "testing.h"
#include "logging.h"
#include <boost/algorithm/string.hpp>


po::options_description Tester::get_options() {
  po::options_description cmd("Testing model options");
  cmd.add_options()
    ("test_model_path", po::value<std::string>(), "The path to the model")
    ("test_target", po::value<std::string>()->default_value("train"), "The evaluation target.")
    ("test_mode", po::value<std::string>()->default_value("decision_acc"), "The mode of testing [decision_acc, pred_detail].")
    ("test_num_tests", po::value<unsigned>()->default_value(1), "The number of tests run on each instance.")
    ;
  return cmd;
}