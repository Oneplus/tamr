#include <iostream>
#include <fstream>
#include <set>
#include "dynet/init.h"
#include "corpus.h"
#include "logging.h"
#include "sys_utils.h"
#include "trainer_utils.h"
#include "parser/parser_builder.h"
#include "system/swap.h"
#include "system/eager.h"
#include "train/algorithm.h"
#include "evaluate/evaluate.h"
#include "decode/testing.h"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description general("Transition-based AMR parser.");
  general.add_options()
    ("train,t", "Use to specify to perform training.")
    ("architecture", po::value<std::string>()->default_value("eager"), "The architecture [swap, eager].")
    ("algorithm", po::value<std::string>()->default_value("supervised"),
     "The choice of reinforcement learning algorithm [supervised]")
    ("training_data,T", po::value<std::string>(), "The path to the training data.")
    ("devel_data,d", po::value<std::string>(), "The path to the development data.")
    ("test_data,e", po::value<std::string>(), "The path to the test data.")
    ("pretrained,w", po::value<std::string>(), "The path to the word embedding.")
    ("devel_gold", po::value<std::string>(), "The path to the development data.")
    ("test_gold", po::value<std::string>(), "The path to the test data.")
    ("model,m", po::value<std::string>(), "The path to the model.")
    ("system", po::value<std::string>()->default_value("eager"), "The transition system [swap, eager].")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "The unknown word strategy.")
    ("unk_prob,u", po::value<float>()->default_value(0.2f), "The probability for replacing the training word.")
    ("layers", po::value<unsigned>()->default_value(2), "The number of layers in LSTM.")
    ("word_dim", po::value<unsigned>()->default_value(100), "Word dim")
    ("pos_dim", po::value<unsigned>()->default_value(20), "POS dim, set it as 0 to disable POS.")
    ("pretrained_dim", po::value<unsigned>()->default_value(100), "Pretrained input dimension.")
    ("char_dim", po::value<unsigned>()->default_value(50), "Character input dimension.")
    ("newnode_dim", po::value<unsigned>()->default_value(100), "Newnode embedding dimension.")
    ("action_dim", po::value<unsigned>()->default_value(20), "The dimension for action.")
    ("relation_dim", po::value<unsigned>()->default_value(32), "The dimension for relation.")
    ("entity_dim", po::value<unsigned>()->default_value(32), "The dimension for entity.")
    ("label_dim", po::value<unsigned>()->default_value(20), "The dimension for label.")
    ("lstm_input_dim", po::value<unsigned>()->default_value(100), "The dimension for lstm input.")
    ("hidden_dim", po::value<unsigned>()->default_value(100), "The dimension for hidden unit.")
    ("dropout", po::value<float>()->default_value(0.f), "The dropout rate.")
    ("reward_type", po::value<std::string>()->default_value("local"),
     "The type of reward [local, local0p10, local00n1, global, global_norm, global_maxout].")
    ("batch_size", po::value<unsigned>()->default_value(1), "The size of batch.")
    ("gamma", po::value<float>()->default_value(1.f), "The gamma, reward discount factor.")
    ("max_iter", po::value<unsigned>()->default_value(10), "The maximum number of iteration.")
    ("report_stops", po::value<unsigned>()->default_value(100), "The reporting stops")
    ("report_reward", "Use to specify to report reward and q-value in evaluation.")
    ("evaluate_oracle", "Use to specify use oracle.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "The evaluation stops")
    ("evaluate_skips", po::value<unsigned>()->default_value(0), "skip evaluation on the first n round.")
    ("external_eval", po::value<std::string>()->default_value("python -u ../scripts/eval.py"), "config the path for evaluation script")
    ("lambda", po::value<float>()->default_value(0.f), "The L2 regularizer, should not set in --dynet-l2.")
    ("output", po::value<std::string>(), "The path to the output file.")
    ("beam_size", po::value<unsigned>(), "The beam size.")
    ("random_seed", po::value<unsigned>()->default_value(7743), "The value of random seed.")
    ("verbose,v", "Details logging.")
    ("help,h", "show help information")
    ;

  po::options_description optimizer = get_optimizer_options();
  po::options_description supervise = SupervisedTrainer::get_options();
  po::options_description test = Tester::get_options();

  po::options_description cmd("Allowed options");
  cmd.add(general)
    .add(optimizer)
    .add(supervise)
    .add(test)
    ;

  po::store(po::parse_command_line(argc, argv, cmd), conf);
  if (conf.count("help")) {
    std::cerr << cmd << std::endl;
    exit(1);
  }
  init_boost_log(conf.count("verbose") > 0);
  if (!conf.count("training_data")) {
    std::cerr << "Please specify --training_data (-T), even in test" << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, false);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, conf);
  
  dynet::rndeng = new std::mt19937(conf["random_seed"].as<unsigned>());

  std::string model_name;
  if (conf.count("train")) {
    if (conf.count("model")) {
      model_name = conf["model"].as<std::string>();
      _INFO << "Main:: write parameters to: " << model_name;
    } else {
      std::string prefix("parser_l2r");
      prefix = prefix + "." + conf["algorithm"].as<std::string>();
      model_name = get_model_name(conf, prefix);
      _INFO << "Main:: write parameters to: " << model_name;
    }
  } else {
    model_name = conf["model"].as<std::string>();
    _INFO << "Main:: evaluating model from: " << model_name;
  }

  Corpus corpus;
  corpus.load_training_data(conf["training_data"].as<std::string>());
  corpus.stat();

  corpus.get_vocabulary_and_singletons();

  std::unordered_map<unsigned, std::vector<float>> pretrained;
  if (conf.count("pretrained")) {
    load_pretrained_word_embedding(conf["pretrained"].as<std::string>(),
                                   conf["pretrained_dim"].as<unsigned>(),
                                   pretrained, corpus);
  }
  _INFO << "Main:: after loading pretrained embedding, size(vocabulary)=" << corpus.word_map.size();

  dynet::ParameterCollection model;
  TransitionSystem* sys = nullptr;
  
  std::string system_name = conf["system"].as<std::string>();
  if (system_name == "swap") {
    sys = new Swap(corpus.action_map, corpus.node_map, corpus.rel_map, corpus.entity_map);
  } else if (system_name == "eager") {
    sys = new Eager(corpus.action_map, corpus.node_map, corpus.rel_map, corpus.entity_map);
  } else {
    _ERROR << "Main:: Unknown transition system: " << system_name;
    exit(1);
  }
  _INFO << "Main:: transition system: " << system_name;

  Parser* parser = ParserBuilder().build(conf, model, (*sys), corpus, pretrained);

  _INFO << "Main:: char_map unk id: " << corpus.char_map.get(corpus.UNK);

  corpus.load_devel_data(conf["devel_data"].as<std::string>());
  _INFO << "Main:: after loading development data, size(vocabulary)=" << corpus.word_map.size();

  if (conf.count("test_data")) {
    corpus.load_test_data(conf["test_data"].as<std::string>());
    _INFO << "Main:: after loading test data, size(vocabulary)=" << corpus.word_map.size();
  }

  std::string output;
  if (conf.count("output")) {
    output = conf["output"].as<std::string>();
  } else {
    int pid = portable_getpid();
#ifdef _MSC_VER
    output = "parser_l2r.evaluator." + boost::lexical_cast<std::string>(pid);
#else
    output = "/tmp/parser_l2r.evaluator." + boost::lexical_cast<std::string>(pid);
#endif
  }
  _INFO << "Main:: write tmp file to: " << output;

  if (conf.count("train")) {
    const std::string algorithm = conf["algorithm"].as<std::string>();
    _INFO << "Main:: algorithm: " << algorithm;
    if (algorithm == "supervised" || algorithm == "sup") {
      SupervisedTrainer trainer(conf, parser);
      trainer.train(conf, corpus, model_name, output);
    }/* else if (algorithm == "testing") {
      Tester tester(conf, parser);
      tester.test(conf, corpus, model_name);
    } else {
      _ERROR << "Main:: Unknown RL algorithm.";
    }*/
  }

  dynet::load_dynet_model(model_name, (&model));
  float dev_f, test_f;
  if (conf.count("evaluate_oracle")) {
    dev_f = evaluate_oracle(conf, corpus, (*parser), output, true);
    test_f = evaluate_oracle(conf, corpus, (*parser), output, false);
  } else {
    dev_f = evaluate(conf, corpus, (*parser), output, true);
    test_f = evaluate(conf, corpus, (*parser), output, false);
  }
  _INFO << "Final score: dev: " << dev_f << ", test: " << test_f;
  
  return 0;
}
