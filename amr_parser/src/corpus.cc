#include "corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "logging.h"
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

const char* Corpus::UNK  = "_UNK_";
const char* Corpus::SPAN = "_SPAN_";
const char* Corpus::BAD0 = "_BAD0_";
const char* Corpus::ROOT = "_ROOT_";

Corpus::Corpus() : n_train(0), n_devel(0) {

}

void Corpus::load_training_data(const std::string& filename) {
  _INFO << "Corpus:: reading training data from: " << filename;

  word_map.insert(Corpus::ROOT);
  word_map.insert(Corpus::UNK);
  // word_map.insert(Corpus::SPAN);
  pos_map.insert(Corpus::ROOT);
  pos_map.insert(Corpus::UNK);
  char_map.insert(Corpus::UNK);
  action_map.insert("CONFIRM");
  action_map.insert(Corpus::UNK);

  confirm_map[word_map.get(Corpus::UNK)] = Alphabet();
  confirm_map[word_map.get(Corpus::UNK)].insert(Corpus::UNK);

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the training file.");

  n_train = 0;
  std::string data;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.empty()) {
      parse_data(data, training_inputs[n_train], training_actions[n_train], true);
      data = "";
      ++n_train;
    } else {
      data += (line + "\n");
    }
  }
  if (!data.empty()) {
    parse_data(data, training_inputs[n_train], training_actions[n_train], true);
    ++n_train;
  }
  _INFO << "Corpus:: loaded " << n_train << " training sentences.";
}

void Corpus::load_devel_data(const std::string& filename) {
  _INFO << "Corpus:: reading development data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
    "Corpus:: ROOT and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the devel file.");

  n_devel = 0;
  std::string data;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.empty()) {
      parse_data(data, devel_inputs[n_devel], devel_actions[n_devel], false);
      data = "";
      ++n_devel;
    } else {
      data += (line + "\n");
    }
  }
  if (!data.empty()) {
    parse_data(data, devel_inputs[n_devel], devel_actions[n_devel], false);
    ++n_devel;
  }
  _INFO << "Corpus:: loaded " << n_devel << " development sentences.";
}

void Corpus::load_test_data(const std::string & filename) {
  _INFO << "Corpus:: reading test data from: " << filename;
  BOOST_ASSERT_MSG(word_map.size() > 1,
                   "Corpus:: ROOT and UNK should be inserted before loading devel data.");

  std::ifstream in(filename);
  BOOST_ASSERT_MSG(in, "Corpus:: failed to open the test file.");

  n_test = 0;
  std::string data;
  std::string line;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) {
      parse_data(data, test_inputs[n_test], test_actions[n_test], false);
      data = "";
      ++n_test;
    } else {
      data += (line + "\n");
    }
  }
  if (!data.empty()) {
    parse_data(data, test_inputs[n_test], test_actions[n_test], false);
    ++n_test;
  }
  _INFO << "Corpus:: loaded " << n_test << " development sentences.";
}

void Corpus::parse_data(const std::string& data,
                        InputUnits& input_units,
                        ActionUnits& action_units,
                        bool train) {
  std::stringstream S(data);
  std::string line;

  input_units.clear();
  action_units.clear();

  while (std::getline(S, line)) {
    std::vector<std::string> tokens;
    boost::algorithm::trim(line);
    boost::algorithm::split(tokens, line, boost::is_any_of(" \t"), boost::token_compress_on);

    if (tokens[1] == "::tok") {
      for (int i = 2; i < tokens.size(); i++) {
        input_units.push_back(InputUnit());
        if (train) {
          unsigned wid = word_map.insert(tokens[i]);
          input_units[i - 2].wid = wid;
          input_units[i - 2].aux_wid = wid;
          input_units[i - 2].w_str = tokens[i];
          for (int j = 0; j < tokens[i].size(); ++j) {
            unsigned c_id = char_map.insert(std::string(1, tokens[i][j]));
            input_units[i - 2].c_id.push_back(c_id);
          }
        } else {
          unsigned wid = (word_map.contains(tokens[i])) ? word_map.get(tokens[i]) : word_map.get(UNK);
          input_units[i - 2].wid = wid;
          input_units[i - 2].aux_wid = wid;
          input_units[i - 2].w_str = tokens[i];
          for (int j = 0; j < tokens[i].size(); ++j) {
            unsigned c_id = (char_map.contains(std::string(1, tokens[i][j]))) ? char_map.get(std::string(1, tokens[i][j])) : char_map.get(UNK);
            input_units[i - 2].c_id.push_back(c_id);
          }
        }
      } 
    } else if (tokens[1] == "::pos") {
      for (int i = 2; i < tokens.size(); i++) {
        if (train) {
          unsigned pid = pos_map.insert(tokens[i]);
          input_units[i - 2].pid = pid;
        } else {
          unsigned pid = (pos_map.contains(tokens[i])) ? pos_map.get(tokens[i]) : pos_map.get(UNK);
          input_units[i - 2].pid = pid;
        }
      }
    } else if (tokens[1] == "::action") {
      std::string action = tokens[2];
      for (int i = 3; i < tokens.size(); i++) {
        action += "\t" + tokens[i];
      }
      ActionUnit action_unit = ActionUnit(action, tokens[2]);
      if (tokens[2] == "CONFIRM") {
        action_unit.action_name = "CONFIRM";
      } else {
        action_unit.action_name = action;
      }

      if (train) {
        std::vector<std::string> terms;
        boost::algorithm::split(terms, action, boost::is_any_of(" \t"), boost::token_compress_on);
        if (terms[0] == "CONFIRM") {
          unsigned wid = (word_map.contains(terms[1])) ? word_map.get(terms[1]) : word_map.get(UNK);
          if (wid == word_map.get(UNK)) {
            action_unit.idx = 0;
          } else {
            if (confirm_map.find(wid) == confirm_map.end()) {
              confirm_map[wid] = Alphabet();
              confirm_map[wid].insert(word_map.get(wid));
            }
            action_unit.idx = confirm_map[wid].insert(terms[2]);
          }
        } else if (terms[0] == "NEWNODE") {
          unsigned nid = node_map.insert(terms[1]);
          action_unit.idx = nid;
        } else if (terms[0] == "LEFT" || terms[0] == "RIGHT") {
          unsigned rid = rel_map.insert(terms[1]);
          action_unit.idx = rid;
        } else if (terms[0] == "ENTITY") {
          unsigned eid = entity_map.insert(terms[1]);
          action_unit.idx = eid;
        }
        unsigned aid = action_map.insert(action_unit.action_name);
        action_unit.aid = aid;
      } else {
        unsigned aid = (action_map.contains(action_unit.action_name)) ? action_map.get(action_unit.action_name) : action_map.get(UNK);
        action_unit.aid = aid;
      }
      action_units.push_back(action_unit);
    }
  }
  InputUnit input_unit;
  input_unit.wid = word_map.get(ROOT);
  input_unit.pid = pos_map.get(ROOT);
  input_unit.aux_wid = word_map.get(ROOT);
  input_unit.w_str = ROOT;
  input_units.push_back(input_unit);
}

unsigned Corpus::get_or_add_word(const std::string& word) {
  return word_map.insert(word);
}

void Corpus::stat() {
  _INFO << "Corpus:: # of words = " << word_map.size();
  _INFO << "Corpus:: # of pos = " << pos_map.size();
}

void Corpus::get_vocabulary_and_singletons() {
  std::map<unsigned, unsigned> counter;
  for (auto& payload : training_inputs) {
    for (auto& item : payload.second) {
      vocab.insert(item.wid);
      ++counter[item.wid];
    }
  }
  for (auto& payload : counter) {
    if (payload.second == 1) { singleton.insert(payload.first); }
  }
}

void load_pretrained_word_embedding(const std::string& embedding_file,
                                    unsigned pretrained_dim,
                                    std::unordered_map<unsigned, std::vector<float> >& pretrained,
                                    Corpus& corpus) {
  pretrained[corpus.get_or_add_word(Corpus::BAD0)] = std::vector<float>(pretrained_dim, 0.f);
  pretrained[corpus.get_or_add_word(Corpus::UNK)] = std::vector<float>(pretrained_dim, 0.f);
  _INFO << "Main:: Loading from " << embedding_file << " with " << pretrained_dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  // get the header in word2vec styled embedding.
  std::getline(ifs, line);
  std::vector<float> v(pretrained_dim, 0.);
  std::string word;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> word;
    // actually, there should be a checking about the embedding dimension.
    for (unsigned i = 0; i < pretrained_dim; ++i) { iss >> v[i]; }
    unsigned id = corpus.get_or_add_word(word);
    pretrained[id] = v;
  }
}
