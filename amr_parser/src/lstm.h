#ifndef LSTM_CONST_NEW_GRAPH_H
#define LSTM_CONST_NEW_GRAPH_H

#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet_layer/layer.h"
#include "ds.h"

struct LSTMBuilder : public dynet::CoupledLSTMBuilder {
  bool trainable;
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       dynet::ParameterCollection& model,
                       bool trainable=true);
  void active_training() { trainable = true; }
  void inactive_training() { trainable = false; }
  void new_graph(dynet::ComputationGraph& cg);
};

struct BiLSTMBuilder {
  bool trainable; 
  LSTMBuilder fw_lstm;
  LSTMBuilder bw_lstm;
  dynet::Parameter p_fw_guard;
  dynet::Parameter p_bw_guard;

  dynet::Expression fw_guard;
  dynet::Expression bw_guard;
  BiLSTMBuilder(unsigned layers,
                unsigned input_dim,
                unsigned hidden_dim,
                dynet::ParameterCollection& model,
    bool trainable = true);

  void active_training() { fw_lstm.active_training(); bw_lstm.active_training(); }
  void inactive_training() { fw_lstm.inactive_training(); bw_lstm.inactive_training(); }
  void new_graph(dynet::ComputationGraph &cg);
  dynet::Expression get_h(SymbolEmbedding &char_emb, const std::vector<unsigned> & c_id);
  
};



#endif  //  end for LSTM_CONST_NEW_GRAPH