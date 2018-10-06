#include "trainer_utils.h"
#include "train_supervised.h"
#include "logging.h"
#include "evaluate/evaluate.h"

po::options_description SupervisedTrainer::get_options() {
  po::options_description cmd("Supervised options");
  cmd.add_options()
    ("supervised_oracle", po::value<std::string>()->default_value("static"), "The type of oracle in supervised learning [static|dynamic|pseduo_dynamic].")
    ("supervised_objective", po::value<std::string>()->default_value("crossentropy"), "The learning objective [crossentropy|rank|bipartie_rank]")
    ("supervised_do_pretrain_iter", po::value<unsigned>()->default_value(1), "The number of pretrain iteration on dynamic oracle.")
    ("supervised_do_explore_prob", po::value<float>()->default_value(0.9), "The probability of exploration.")
    ("supervised_pseudo_oracle_model", po::value<std::string>(), "The path to the pseudo dynamic oracle model, must in pseduo_dynamic mode.")
    ;
  return cmd;
}

SupervisedTrainer::SupervisedTrainer(const po::variables_map& conf, Parser * p) :
  Trainer(conf), 
  parser(p),
  pseudo_dynamic_oracle(nullptr),
  pseudo_dynamic_oracle_model(nullptr) {
  if (conf["supervised_oracle"].as<std::string>() == "static") {
    oracle_type = kStatic;
  } else {
    _ERROR << "Unknown oracle :" << conf["supervised_oracle"].as<std::string>();
  }

  if (conf["supervised_objective"].as<std::string>() == "crossentropy") {
    objective_type = kCrossEntropy;
  } else if (conf["supervised_objective"].as<std::string>() == "rank") {
    objective_type = kRank;
  } else {
    objective_type = kBipartieRank;
  }
  lambda_ = conf["lambda"].as<float>();
  _INFO << "SUP:: learning objective " << conf["supervised_objective"].as<std::string>();
  
  system = conf["system"].as<std::string>();
}

void SupervisedTrainer::train(const po::variables_map& conf,
                              Corpus& corpus,
                              const std::string& name,
                              const std::string& output) {
  dynet::ParameterCollection& model = parser->model;
  _INFO << "SUP:: start lstm-parser supervised training.";

  dynet::Trainer* trainer = get_trainer(conf, model);
  // unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  unsigned max_iter = conf["max_iter"].as<unsigned>();

  float llh = 0.f;
  float llh_in_batch = 0.f;
  float best_f = 0.f;

  std::vector<unsigned> order;
  get_orders(corpus, order);
  float n_train = order.size();

  unsigned logc = 0;
  // unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  // float unk_prob = conf["unk_prob"].as<float>();
  unsigned report_stops = conf["report_stops"].as<unsigned>();
  unsigned evaluate_stops = conf["evaluate_stops"].as<unsigned>();
  unsigned evaluate_skips = conf["evaluate_skips"].as<unsigned>();
  float eta0 = trainer->learning_rate;

  _INFO << "SUP:: will stop after " << max_iter << " iterations.";
  for (unsigned iter = 0; iter < max_iter; ++iter) {
    llh = 0;
    _INFO << "SUP:: start training iteration #" << iter << ", shuffled.";
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      _TRACE << "sid=" << sid;
      InputUnits& input_units = corpus.training_inputs[sid];
      const ActionUnits& parse_units = corpus.training_actions[sid];
      //random_replace_singletons(unk_strategy, unk_prob, corpus.singleton, kUNK, input_units);
      
      float lp;
      
      lp = train_on_one_full_tree(input_units, parse_units, trainer, iter);
      
      llh += lp;
      llh_in_batch += lp;
      //restore_singletons(unk_strategy, input_units);

      ++logc;
      if (logc % report_stops == 0) {
        float epoch = (float(logc) / n_train);
        _INFO << "SUP:: iter #" << iter << " (epoch " << epoch << ") loss " << llh_in_batch;
        llh_in_batch = 0.f;
      }
      if (iter >= evaluate_skips && logc % evaluate_stops == 0) {
        eval(conf, output, name, best_f, corpus, *parser);
      }
    }

    _INFO << "SUP:: end of iter #" << iter << " loss " << llh;
    eval(conf, output, name, best_f, corpus, *parser);

    update_trainer(conf, eta0, float(iter), trainer);
    trainer->status();
  }

  delete trainer;
}

float SupervisedTrainer::train_on_one_full_tree(const InputUnits& input_units,
                                                const ActionUnits& action_units,
                                                dynet::Trainer* trainer,
                                                unsigned iter) {
  dynet::ComputationGraph cg;
  parser->activate_training();
  parser->new_graph(cg);
  
  std::vector<dynet::Expression> loss;

  unsigned len = input_units.size();
  //for (int i = 0; i < len; i++) {
  //  std::cerr << input_units[i].w_str << " ";
  //}
  //std::cerr << std::endl;
  State state(len);
  parser->initialize(cg, input_units, state);

  unsigned illegal_action = parser->sys.num_actions();
  unsigned n_actions = 0;
  while (!state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    parser->sys.get_valid_actions(state, valid_actions);

    dynet::Expression score_exprs = parser->get_scores();
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));
    unsigned action = 0;

    unsigned best_gold_action = illegal_action;
    unsigned worst_gold_action = illegal_action;
    unsigned best_non_gold_action = illegal_action;

    best_gold_action = action_units[n_actions].aid;
    //std::cerr << action_units[n_actions].a_str << std::endl;
    action = action_units[n_actions].aid;

    if (objective_type == kRank || objective_type == kBipartieRank) {
      float best_non_gold_action_score = -1e10;
      for (unsigned i = 0; i < valid_actions.size(); ++i) {
        unsigned act = valid_actions[i];
        if (act != best_gold_action && (scores[act] > best_non_gold_action_score)) {
          best_non_gold_action = act;
          best_non_gold_action_score = scores[act];
        }
      }
    }

    if (objective_type == kCrossEntropy) {
      loss.push_back(dynet::pickneglogsoftmax(score_exprs, best_gold_action));
    } else if (objective_type == kRank) {
      if (best_gold_action != illegal_action && best_non_gold_action != illegal_action) {
        loss.push_back(dynet::pairwise_rank_loss(
          dynet::pick(score_exprs, best_gold_action),
          dynet::pick(score_exprs, best_non_gold_action)
        ));
      }
    } else {
      if (worst_gold_action != illegal_action && best_non_gold_action != illegal_action) {
        loss.push_back(dynet::pairwise_rank_loss(
          dynet::pick(score_exprs, worst_gold_action),
          dynet::pick(score_exprs, best_non_gold_action)
        ));
      }
    }

    //CONFIRM
    if (action == 0 && best_gold_action == 0) {
      dynet::Expression confirm_scores_expr;
      if (system == "eager") {
        confirm_scores_expr = parser->get_confirm_values(state.buffer.back().first);
      } else if (system == "swap") {
        confirm_scores_expr = parser->get_confirm_values(state.stack.back().first);
      } else {
        BOOST_ASSERT_MSG(false, "Illegal System");
      }
      //std::cerr << confirm_scores_expr.dim()[0] << " " << confirm_scores_expr.dim()[1] << std::endl;
      //std::cerr << "~" << action_units[n_actions].idx << " " << state.stack.back().first << " " << state.stack.back().second << std::endl;
      //std::cerr << action_units[n_actions].idx << std::endl;
      loss.push_back(dynet::pickneglogsoftmax(confirm_scores_expr, action_units[n_actions].idx));
      //std::cerr << action_units[n_actions].idx << std::endl;
    }

    parser->perform_action(action, cg, state);
    n_actions++;
  }
  float ret = 0.f;
  if (loss.size() > 0) {
    std::vector<dynet::Expression> all_params = parser->get_params();
    std::vector<dynet::Expression> reg;
    for (auto e : all_params) { reg.push_back(dynet::squared_norm(e)); }
    dynet::Expression l = dynet::sum(loss) + 0.5 * loss.size() * lambda_ * dynet::sum(reg);
    ret = dynet::as_scalar(cg.incremental_forward(l));
    cg.backward(l);
    trainer->update();
  }
  return ret;
}