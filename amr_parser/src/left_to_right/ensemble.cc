#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include "dynet/init.h"
#include "corpus.h"
#include "logging.h"
#include "parser/parser_builder.h"
#include "system/swap.h"
#include "system/eager.h"
#include "evaluate/evaluate.h"
#include "sys_utils.h"
#include "trainer_utils.h"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description general("Transition-based dependency parser with ensemble.");
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
    ("models,m", po::value<std::string>(), "The path to the model.")
    ("system", po::value<std::string>()->default_value("eager"), "The transition system [swap, eager].")
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
    ("external_eval", po::value<std::string>()->default_value("python -u ../scripts/eval.py"), "config the path for evaluation script")
    ("output", po::value<std::string>(), "The path to the output file.")
    ("verbose,v", "Details logging.")
    ("help,h", "show help information")
    ;

  po::options_description cmd("Allowed options");
  cmd.add(general);

  po::store(po::parse_command_line(argc, argv, cmd), conf);
  if (conf.count("help")) {
    std::cerr << cmd << std::endl;
    exit(1);
  }
  init_boost_log(conf.count("verbose") > 0);
}

float ensemble(const po::variables_map & conf,
               Corpus & corpus,
               const std::vector<Parser *> & parsers,
               const std::string & output,
               bool devel) {
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);

  std::ofstream ofs(output);
  for (auto parser : parsers) {
    parser->inactivate_training();
  }

  unsigned n = (devel ? corpus.n_devel : corpus.n_test);
  std::unordered_map<unsigned, InputUnits> & inputs = (devel ?
                                                       corpus.devel_inputs :
                                                       corpus.test_inputs);

  unsigned n_engines = parsers.size();

  for (unsigned sid = 0; sid < n; ++sid) {
    InputUnits & input_units = inputs[sid];

    ofs << "# ::tok";
    for (unsigned i = 0; i < input_units.size() - 1; ++i) {
      ofs << " " << input_units[i].w_str;
    }
    ofs << std::endl;

    for (InputUnit & u : input_units) {
      if (!corpus.vocab.count(u.wid)) { u.wid = kUNK; }
    }
    dynet::ComputationGraph cg;
    ActionUnits output;

    unsigned len = input_units.size();
    std::vector<State> states(n_engines, State(len));

    for (unsigned i = 0; i < n_engines; ++i) {
      parsers[i]->new_graph(cg);
      parsers[i]->initialize(cg, input_units, states[i]);
    }

    unsigned n_actions = 0;
    while (!states[0].terminated() && n_actions++ < 500) {
      std::vector<unsigned> valid_actions;
      (parsers[0]->sys).get_valid_actions(states[0], valid_actions);

      std::vector<float> scores = dynet::as_vector(cg.get_value(parsers[0]->get_scores()));

      for (unsigned i = 1; i < n_engines; ++i) {
        std::vector<float> another_scores =
          dynet::as_vector(cg.get_value(parsers[i]->get_scores()));

        for (unsigned j = 0; j < scores.size(); ++j) {
          scores[j] += another_scores[j];
        }
      }

      auto payload = Parser::get_best_action(scores, valid_actions);
      unsigned best_a = payload.first;
      unsigned best_c = 0;
      //if CONFIRM
      if (best_a == 0) {
        unsigned wid = 0;
        if (conf["system"].as<std::string>() == "swap") {
          wid = states[0].stack.back().first;
        } else if (conf["system"].as<std::string>() == "eager") {
          wid = states[0].buffer.back().first;
        } else {
          BOOST_ASSERT_MSG(false, "Illegal System");
        }

        // ensemble the confirm score.
        std::vector<float> confirm_scores =
          dynet::as_vector(cg.get_value(parsers[0]->get_confirm_values(wid)));
        for (unsigned i = 1; i < n_engines; ++i) {
          std::vector<float> another_confirm_scores =
            dynet::as_vector(cg.get_value(parsers[i]->get_confirm_values(wid)));
          for (unsigned  j = 0; j < confirm_scores.size(); ++j) {
            confirm_scores[j] += another_confirm_scores[j];
          }
        }

        float best_score = -1e9f;
        for (unsigned j = 0; j < confirm_scores.size(); ++j) {
          if (confirm_scores[j] > best_score) {
            best_score = confirm_scores[j];
            best_c = j;
          }
        }
        //std::cerr << "# ::action\t" << "CONFIRM\t" <<
        //  corpus.word_map.get(wid) << "\t";
        ofs << "# ::action\t"
            << "CONFIRM\t"
            << (corpus.word_map.contains(wid) ? corpus.word_map.get(wid) : std::string("_UNK_"))
            << "\t";
        if (corpus.confirm_map.find(wid) == corpus.confirm_map.end()) {
          //std::cerr << corpus.word_map.get(wid) << std::endl;
          ofs << (corpus.word_map.contains(wid) ? corpus.word_map.get(wid) : std::string("_UNK_")) << std::endl;
        } else {
          //std::cerr << corpus.confirm_map[wid].get(best_c) << std::endl;
          ofs << corpus.confirm_map[wid].get(best_c) << std::endl;
        }


      } else {
        //std::cerr << "# ::action\t" << parser.sys.action_map.get(best_a) << std::endl;
        ofs << "# ::action\t" << (parsers[0]->sys).action_map.get(best_a) << std::endl;
      }

      for (unsigned j = 0; j < n_engines; ++j) {
        parsers[j]->perform_action(best_a, cg, states[j]);
      }
    }

    ofs << std::endl;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  float f_score = execute_and_get_result(conf["external_eval"].as<std::string>() +
                                         " " +
                                         (devel ?
                                          conf["devel_gold"].as<std::string>() : conf["test_gold"].as<std::string>()) +
                                         " " +
                                         output);
  _INFO << "Evaluate:: Smatch " << f_score << " [" << corpus.n_devel <<
        " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, false);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, conf);

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


  TransitionSystem* sys = nullptr;
  std::string system_name = conf["system"].as<std::string>();
  if (system_name == "swap") {
    sys = new Swap(corpus.action_map, corpus.node_map, corpus.rel_map, corpus.entity_map);
  } else if (system_name == "eager") {
    sys = new Eager(corpus.action_map, corpus.node_map, corpus.rel_map, corpus.entity_map);
  } else {
    _ERROR << "Main:: unknown transition system: " << system_name;
    exit(1);
  }

  std::vector<std::string> model_paths;
  std::string model_path = conf["models"].as<std::string>();
  boost::split(model_paths, model_path, boost::is_any_of(","), boost::token_compress_on);

  unsigned n_engines = model_paths.size();
  assert(n_engines > 0);

  std::vector<dynet::ParameterCollection*> models(n_engines);
  std::vector<Parser*> parsers(n_engines, nullptr);

  for (unsigned i = 0; i < n_engines; ++i) {
    models[i] = new dynet::ParameterCollection;
    parsers[i] = ParserBuilder().build(conf, (*models[i]), (*sys), corpus, pretrained);

    dynet::load_dynet_model(model_paths[i], models[i]);
  }

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

  std::string devel_output = output + ".devel";
  float f_score = ensemble(conf, corpus, parsers, devel_output, true);
  _INFO << "Main:: devel f-score: " << f_score;

  std::string test_output = output + ".test";
  f_score = ensemble(conf, corpus, parsers, test_output, false);
  _INFO << "Main:: test f-score: " << f_score;
  return 0;
}
