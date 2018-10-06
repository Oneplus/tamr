#include "evaluate.h"
#include "logging.h"
#include "sys_utils.h"
#include <fstream>
#include <chrono>

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               Parser & parser,
               const std::string & output,
               bool devel) {
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);

  std::ofstream ofs(output);
  parser.inactivate_training();

  unsigned n = (devel ? corpus.n_devel : corpus.n_test);
  std::unordered_map<unsigned, InputUnits> & inputs = (devel ? corpus.devel_inputs : corpus.test_inputs);

  for (unsigned sid = 0; sid < n; ++sid) {

    ofs << "# ::tok";
    for (unsigned i = 0; i < inputs[sid].size() - 1; ++i) { //except for _ROOT_
      ofs << " " << inputs[sid][i].w_str;
    }
    ofs << std::endl;

    InputUnits& input_units = inputs[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.vocab.count(u.wid)) { u.wid = kUNK; }
    }
    dynet::ComputationGraph cg;
    ActionUnits output;

    unsigned len = input_units.size();
    State state(len);

    parser.new_graph(cg);

    parser.initialize(cg, input_units, state);
    unsigned n_actions = 0;
    while (!state.terminated() && n_actions++ < 500) {
      // collect all valid actions.
      std::vector<unsigned> valid_actions;
      parser.sys.get_valid_actions(state, valid_actions);
      //std::cerr << valid_actions.size() << std::endl;

      std::vector<float> scores = dynet::as_vector(cg.get_value(parser.get_scores()));

      auto payload = Parser::get_best_action(scores, valid_actions);
      unsigned best_a = payload.first;
      unsigned best_c = 0;
      //if CONFIRM
      if (best_a == 0) {
        unsigned wid = 0;
        if (conf["system"].as<std::string>() == "swap") {
          wid = state.stack.back().first;
        } else if (conf["system"].as<std::string>() == "eager") {
          wid = state.buffer.back().first;
        } else {
          BOOST_ASSERT_MSG(false, "Illegal System");
        }

        std::vector<float> confirm_scores = dynet::as_vector(cg.get_value(parser.get_confirm_values(wid)));
        float best_score = -1e9f;
        for (unsigned i = 0; i < confirm_scores.size(); i++) {
          if (confirm_scores[i] > best_score) {
            best_score = confirm_scores[i];
            best_c = i;
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
        ofs << "# ::action\t" << parser.sys.action_map.get(best_a) << std::endl;
      }
      parser.perform_action(best_a, cg, state);
    }

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }

    ofs << std::endl;

    //ofs && confirm
  }
  ofs.close();
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

float evaluate_oracle(const po::variables_map & conf,
                      Corpus & corpus,
                      Parser & parser,
                      const std::string & output,
                      bool devel) {
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);

  std::ofstream ofs(output);
  parser.inactivate_training();

  unsigned n = (devel ? corpus.n_devel : corpus.n_test);
  std::unordered_map<unsigned, InputUnits> & inputs = (devel ? corpus.devel_inputs : corpus.test_inputs);
  std::unordered_map<unsigned, ActionUnits> & actions = (devel ? corpus.devel_actions : corpus.test_actions);

  for (unsigned sid = 0; sid < n; ++sid) {

    ofs << "# ::tok";
    for (unsigned i = 0; i < inputs[sid].size() - 1; ++i) { //except for _ROOT_
      ofs << " " << inputs[sid][i].w_str;
    }
    ofs << std::endl;

    InputUnits& input_units = inputs[sid];
    ActionUnits & parse_units = actions[sid];

    for (InputUnit& u : input_units) {
      if (!corpus.vocab.count(u.wid)) { u.wid = kUNK; }
    }
    dynet::ComputationGraph cg;
    ActionUnits output;

    unsigned len = input_units.size();
    State state(len);

    parser.new_graph(cg);

    parser.initialize(cg, input_units, state);
    unsigned n_actions = 0;
    while (!state.terminated() && n_actions++ < 500) {
      // collect all valid actions.
      std::vector<unsigned> valid_actions;
      parser.sys.get_valid_actions(state, valid_actions);
      //std::cerr << valid_actions.size() << std::endl;

      std::vector<float> scores = dynet::as_vector(cg.get_value(parser.get_scores()));

      auto payload = Parser::get_best_action(scores, valid_actions);
      unsigned best_a = payload.first;
      unsigned best_c = 0;
      //if CONFIRM
      if (best_a == 0) {
        unsigned wid = 0;
        if (conf["system"].as<std::string>() == "swap") {
          wid = state.stack.back().first;
        } else if (conf["system"].as<std::string>() == "eager") {
          wid = state.buffer.back().first;
        } else {
          BOOST_ASSERT_MSG(false, "Illegal System");
        }

        std::vector<float> confirm_scores = dynet::as_vector(cg.get_value(parser.get_confirm_values(wid)));
        float best_score = -1e9f;
        for (unsigned i = 0; i < confirm_scores.size(); i++) {
          if (confirm_scores[i] > best_score) {
            best_score = confirm_scores[i];
            best_c = i;
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
        ofs << "# ::action\t" << parser.sys.action_map.get(best_a) << std::endl;
      }
      best_a = parse_units[n_actions].aid;
      parser.perform_action(best_a, cg, state);
    }

    for (InputUnit& u : input_units) { u.wid = u.aux_wid; }

    ofs << std::endl;

    //ofs && confirm
  }
  ofs.close();
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
