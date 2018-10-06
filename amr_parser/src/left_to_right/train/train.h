#ifndef TRAIN_H
#define TRAIN_H

#include <boost/program_options.hpp>
#include "parser/parser.h"
#include "corpus.h"
namespace po = boost::program_options;

struct Trainer {
  float gamma;
  float lambda_;
 
  Trainer(const po::variables_map& conf);

  void eval(const po::variables_map& conf,
            const std::string & output,
            const std::string & model_name,
            float & current_best,
            Corpus & corpus,
            Parser & parser,
            bool update_and_save = true);

  void eval(const po::variables_map& conf,
            const std::string & output,
            const std::string & model_name,
            float & current_best,
            Corpus & corpus,
            Parser & parser,
            Parser & parser2,
            bool update_and_save = true);

  dynet::Expression l2(Parser & parser, unsigned n);
};

#endif  //  end for TRAIN_H