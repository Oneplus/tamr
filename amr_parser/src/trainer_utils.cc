#include "trainer_utils.h"
#include "sys_utils.h"
#include "logging.h"
#include <fstream>
#include <sstream>

void random_replace_singletons(const unsigned & unk_strategy,
                               const float & unk_prob,
                               const std::set<unsigned>& singletons,
                               const unsigned& kUNK,
                               InputUnits & input_units) {
  if (unk_strategy != 1) { return; }
  for (auto& u : input_units) {
    if (singletons.count(u.wid) && dynet::rand01() < unk_prob) { u.wid = kUNK; }
  }
}

void restore_singletons(const unsigned & unk_strategy,
                        InputUnits & input_units) {
  if (unk_strategy != 1) { return; }  
  for (auto& u : input_units) { u.wid = u.aux_wid; }
}

void get_orders(Corpus& corpus,
                std::vector<unsigned>& order) {
  order.clear();
  for (unsigned i = 0; i < corpus.training_inputs.size(); ++i) {
    order.push_back(i);
  }
}

std::string get_model_name(const po::variables_map& conf,
                           const std::string& prefix) {
  std::ostringstream os;
  os << prefix << "." << portable_getpid();
  return os.str();
}

po::options_description get_optimizer_options() {
  po::options_description cmd("Optimizer options");
  cmd.add_options()
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The choice of optimizer [simple_sgd, momentum_sgd, adagrad, adadelta, adam].")
    ("optimizer_eta", po::value<float>(), "The initial value of learning rate (eta).")
    ("optimizer_final_eta", po::value<float>()->default_value(0.f), "The final value of eta.")
    ("optimizer_enable_eta_decay", po::value<bool>()->required(), "Specify to update eta at the end of each epoch.")
    ("optimizer_eta_decay", po::value<float>(), "The decay rate of eta.")
    ("optimizer_enable_clipping", po::value<bool>()->required(), "Enable clipping.")
    ("optimizer_adam_beta1", po::value<float>()->default_value(0.9f), "The beta1 hyper-parameter of adam")
    ("optimizer_adam_beta2", po::value<float>()->default_value(0.999f), "The beta2 hyper-parameter of adam.")
    ("optimizer_rmsprop_rho", po::value<float>()->default_value(0.99f), "The rho hyper-parameter of rmsprop.")
    ;

  return cmd;
}

dynet::Trainer* get_trainer(const po::variables_map& conf, dynet::ParameterCollection& model) {
  dynet::Trainer* trainer = nullptr;
  if (!conf.count("optimizer") || conf["optimizer"].as<std::string>() == "simple_sgd") {
    float eta0 = (conf.count("optimizer_eta") ? conf["optimizer_eta"].as<float>() : 0.1f);
    trainer = new dynet::SimpleSGDTrainer(model, eta0);
    // trainer->eta_decay = 0.08f;
  } else if (conf["optimizer"].as<std::string>() == "momentum_sgd") {
    trainer = new dynet::MomentumSGDTrainer(model);
    // trainer->eta_decay = 0.08f;
  } else if (conf["optimizer"].as<std::string>() == "adagrad") {
    trainer = new dynet::AdagradTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "adadelta") {
    trainer = new dynet::AdadeltaTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "rmsprop") {
    float eta0 = (conf.count("optimizer_eta") ? conf["optimizer_eta"].as<float>() : 0.001f);
    float rho = (conf.count("optimizer_rmsprop_rho") ? conf["optimizer_rmsprop_rho"].as<float>() : 0.99f);
    trainer = new dynet::RMSPropTrainer(model, eta0, 1e-8, rho);
  } else if (conf["optimizer"].as<std::string>() == "adam") {
    // default setting is same with Kingma and Ba (2015). 
    float eta0 = (conf.count("optimizer_eta") ? conf["optimizer_eta"].as<float>() : 0.001f);
    float beta1 = conf["optimizer_adam_beta1"].as<float>();
    float beta2 = conf["optimizer_adam_beta2"].as<float>();
    trainer = new dynet::AdamTrainer(model, eta0, beta1, beta2);
  } else {
    _ERROR << "Trainier:: unknown optimizer: " << conf["optimizer"].as<std::string>();
    exit(1);
  }
  _INFO << "Trainer:: using " << conf["optimizer"].as<std::string>() << " optimizer";
  _INFO << "Trainer:: eta = " << trainer->learning_rate;

  if (conf["optimizer_enable_clipping"].as<bool>()) {
    trainer->clipping_enabled = true;
    _INFO << "Trainer:: gradient clipping = enabled";
  } else {
    trainer->clipping_enabled = false;
    _INFO << "Trainer:: gradient clipping = false";
  }

  if (conf["optimizer_enable_eta_decay"].as<bool>()) {
    _INFO << "Trainer:: eta decay = enabled";
    if (conf.count("optimizer_eta_decay")) {
      // trainer->eta_decay = conf["optimizer_eta_decay"].as<float>();
      _INFO << "Trainer:: eta decay rate = " << conf["optimizer_eta_decay"].as<float>();
    } else {
      _INFO << "Trainer:: eta decay rate not set, use default = " << 0.08f;
    }
  } else {
    _INFO << "Trainer:: eta decay = disabled";
  }
  return trainer;
}

void update_trainer(const po::variables_map& conf, const float & eta0, const float & iter, dynet::Trainer* trainer) {
  if (conf.count("optimizer_enable_eta_decay")) {
    float final_eta = conf["optimizer_final_eta"].as<float>();
    float eta_decay = (conf.count("optimizer_eta_decay") ? conf["optimizer_eta_decay"].as<float>() : 0.08f);
    if (trainer->learning_rate > final_eta) {
      // trainer->update_epoch();
      // trainer->status();
      trainer->learning_rate = eta0 / (1.f + eta_decay * iter);
      _INFO << "Trainer:: trainer updated.";
    } else {
      trainer->learning_rate = final_eta;
      _INFO << "Trainer:: eta reach the final value " << final_eta;
    }
  }
}


