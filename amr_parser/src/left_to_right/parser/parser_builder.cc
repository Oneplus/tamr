#include "parser_swap.h"
#include "parser_eager.h"
#include "parser_builder.h"
#include "logging.h"

po::options_description ParserBuilder::get_options() {
  po::options_description cmd("Parser settings.");
  return cmd;
}

Parser * ParserBuilder::build(const po::variables_map& conf,
                              dynet::ParameterCollection & model,
                              TransitionSystem& sys,
                              const Corpus& corpus,
                              const std::unordered_map<unsigned, std::vector<float>>& pretrained) {
  std::string system_name = conf["system"].as<std::string>();
  Parser* parser = nullptr;
  std::string arch_name = conf["architecture"].as<std::string>();
  if (arch_name == "swap") {
    parser = new ParserSwap(model,
                            corpus.vocab.size() + 10,
                            conf["word_dim"].as<unsigned>(),
                            corpus.pos_map.size() + 10,
                            conf["pos_dim"].as<unsigned>(),
                            corpus.word_map.size() + 1,
                            conf["pretrained_dim"].as<unsigned>(),
                            corpus.char_map.size() + 1,
                            conf["char_dim"].as<unsigned>(),
                            sys.num_actions(),
                            conf["action_dim"].as<unsigned>(),
                            sys.node_map.size(),
                            conf["lstm_input_dim"].as<unsigned>(),
                            sys.rel_map.size(),
                            conf["relation_dim"].as<unsigned>(),
                            sys.entity_map.size(),
                            conf["entity_dim"].as<unsigned>(),
                            conf["layers"].as<unsigned>(),
                            conf["lstm_input_dim"].as<unsigned>(),
                            conf["hidden_dim"].as<unsigned>(),
                            system_name,
                            sys,
                            pretrained,
                            corpus.confirm_map,
                            corpus.char_map);
  } else if (arch_name == "eager") {
    parser = new ParserEager(model,
                             corpus.vocab.size() + 10,
                             conf["word_dim"].as<unsigned>(),
                             corpus.pos_map.size() + 10,
                             conf["pos_dim"].as<unsigned>(),
                             corpus.word_map.size() + 1,
                             conf["pretrained_dim"].as<unsigned>(),
                             corpus.char_map.size() + 1,
                             conf["char_dim"].as<unsigned>(),
                             sys.num_actions(),
                             conf["action_dim"].as<unsigned>(),
                             sys.node_map.size(),
                             conf["lstm_input_dim"].as<unsigned>(),
                             sys.rel_map.size(),
                             conf["relation_dim"].as<unsigned>(),
                             sys.entity_map.size(),
                             conf["entity_dim"].as<unsigned>(),
                             conf["layers"].as<unsigned>(),
                             conf["lstm_input_dim"].as<unsigned>(),
                             conf["hidden_dim"].as<unsigned>(),
                             system_name,
                             sys,
                             pretrained,
                             corpus.confirm_map,
                             corpus.char_map);
  } else {
    _ERROR << "Main:: Unknown architecture name: " << arch_name;
  }
  _INFO << "Main:: architecture: " << arch_name;
  return parser;
}
