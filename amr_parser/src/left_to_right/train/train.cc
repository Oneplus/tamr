#include "train.h"
#include "logging.h"
#include "evaluate/evaluate.h"

Trainer::Trainer(const po::variables_map & conf) {
  gamma = conf["gamma"].as<float>();
  _INFO << "RL:: gamma = " << gamma;

  lambda_ = conf["lambda"].as<float>();
  _INFO << "RL:: lambda = " << lambda_;
}

void Trainer::eval(const po::variables_map& conf,
                   const std::string & output,
                   const std::string & model_name,
                   float & current_best,
                   Corpus & corpus,
                   Parser & parser,
                   bool update_and_save) {
  float f = evaluate(conf, corpus, parser, output, true);
  if (update_and_save && f > current_best) {
    current_best = f;
    dynet::save_dynet_model(model_name, (&(parser.model)));
    f = evaluate(conf, corpus, parser, output, false);
    _INFO << "Trainer:: new best record achieved " << current_best << ", test: " << f;
  }
}

dynet::Expression Trainer::l2(Parser & parser, unsigned n) {
  std::vector<dynet::Expression> reg;
  for (auto e : parser.get_params()) { reg.push_back(dynet::squared_norm(e)); }
  return (0.5 * n) * lambda_ * dynet::sum(reg);
}
