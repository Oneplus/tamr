#ifndef RLPARSER_CORPUS_H
#define RLPARSER_CORPUS_H

#include <unordered_map>
#include <vector>
#include <set>
#include "ds.h"
#include <boost/serialization/vector.hpp>

struct InputUnit {
  unsigned wid;
  unsigned aux_wid;
  unsigned pid;
  std::vector<unsigned> c_id;

  std::string w_str;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) {
    ar & wid;
    ar & pid;
    ar & c_id;
    ar & aux_wid;
  }
};

struct ActionUnit {
  std::string a_str;
  std::string action_name;
  unsigned aid;
  unsigned idx; //for confirm, newnode, la and ra op

  ActionUnit(std::string a_str, std::string action_name): a_str(a_str), action_name(action_name){}
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) {
    ar & aid;
    ar & idx;
  }
};

typedef std::vector<InputUnit> InputUnits;
typedef std::vector<ActionUnit> ActionUnits;

struct Corpus {
  const static char* UNK;
  const static char* SPAN;
  const static char* BAD0;
  const static char* ROOT;

  unsigned n_train;
  unsigned n_devel;
  unsigned n_test;

  Alphabet word_map;
  Alphabet pos_map;
  Alphabet action_map;
  Alphabet char_map;
  Alphabet node_map;
  Alphabet rel_map;
  Alphabet entity_map;

  std::unordered_map<unsigned, Alphabet> confirm_map;

  std::unordered_map<unsigned, InputUnits> training_inputs;
  std::unordered_map<unsigned, ActionUnits> training_actions;
  std::unordered_map<unsigned, InputUnits> devel_inputs;
  std::unordered_map<unsigned, ActionUnits> devel_actions;
  std::unordered_map<unsigned, InputUnits> test_inputs;
  std::unordered_map<unsigned, ActionUnits> test_actions;

  std::set<unsigned> vocab;
  std::set<unsigned> singleton;
  
  Corpus();

  void load_training_data(const std::string& filename);

  void load_devel_data(const std::string& filename);

  void load_test_data(const std::string& filename);

  void parse_data(const std::string& data,
                  InputUnits& input_units, 
                  ActionUnits& action_units,
                  bool train);
 
  void get_vocabulary_and_singletons();

  unsigned get_or_add_word(const std::string& word);
  void stat();
};

void load_pretrained_word_embedding(const std::string& embedding_file,
                                    unsigned pretrained_dim,
                                    std::unordered_map<unsigned, std::vector<float> >& pretrained,
                                    Corpus& corpus);

#endif  //  end for RLPARSER_CORPUS_H
