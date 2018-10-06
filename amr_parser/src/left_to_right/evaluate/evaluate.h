#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <set>
#include "corpus.h"
#include "parser/parser.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

float evaluate(const po::variables_map & conf,
               Corpus & corpus,
               Parser & parser,
               const std::string& output,
               bool devel);

float evaluate_oracle(const po::variables_map & conf,
                      Corpus & corpus,
                      Parser & parser,
                      const std::string& output,
                      bool devel);


#endif  //  end for EVALUATE_H