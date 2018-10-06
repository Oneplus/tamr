#include "lstm.h"

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

LSTMBuilder::LSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         dynet::ParameterCollection& model,
                         bool trainable) :
  dynet::CoupledLSTMBuilder(layers, input_dim, hidden_dim, model),
  trainable(trainable) {
}

void LSTMBuilder::new_graph(dynet::ComputationGraph& cg) {
  if (trainable) {
    dynet::CoupledLSTMBuilder::new_graph(cg);
  } else {
    // cannot call sm.transition directly. this will waste some nodes
    // in computation graph.
    dynet::CoupledLSTMBuilder::new_graph(cg);
    param_vars.clear();
    for (unsigned i = 0; i < layers; ++i) {
      auto& p = params[i];

      //i
      dynet::Expression i_x2i = dynet::const_parameter(cg, p[X2I]);
      dynet::Expression i_h2i = dynet::const_parameter(cg, p[H2I]);
      dynet::Expression i_c2i = dynet::const_parameter(cg, p[C2I]);
      dynet::Expression i_bi = dynet::const_parameter(cg, p[BI]);
      //o
      dynet::Expression i_x2o = dynet::const_parameter(cg, p[X2O]);
      dynet::Expression i_h2o = dynet::const_parameter(cg, p[H2O]);
      dynet::Expression i_c2o = dynet::const_parameter(cg, p[C2O]);
      dynet::Expression i_bo = dynet::const_parameter(cg, p[BO]);
      //c
      dynet::Expression i_x2c = dynet::const_parameter(cg, p[X2C]);
      dynet::Expression i_h2c = dynet::const_parameter(cg, p[H2C]);
      dynet::Expression i_bc = dynet::const_parameter(cg, p[BC]);

      std::vector<dynet::Expression> vars = {
        i_x2i, i_h2i, i_c2i, i_bi,
        i_x2o, i_h2o, i_c2o, i_bo,
        i_x2c, i_h2c, i_bc
      };
      param_vars.push_back(vars);
    } //  layers
  }
}


BiLSTMBuilder::BiLSTMBuilder(unsigned layers,
                             unsigned input_dim,
                             unsigned hidden_dim,
                             dynet::ParameterCollection& model,
                             bool trainable):
  trainable(trainable), 
  fw_lstm(layers, input_dim, hidden_dim, model, trainable),
  bw_lstm(layers, input_dim, hidden_dim, model, trainable),
  p_fw_guard(model.add_parameters({ input_dim, 1 })),
  p_bw_guard(model.add_parameters({ input_dim, 1 })) {
}

void BiLSTMBuilder::new_graph(dynet::ComputationGraph &cg) {
  fw_lstm.new_graph(cg);
  bw_lstm.new_graph(cg);
  if (trainable) {
    fw_guard = dynet::parameter(cg, p_fw_guard);
    bw_guard = dynet::parameter(cg, p_bw_guard);
  }
  else {
    fw_guard = dynet::const_parameter(cg, p_fw_guard);
    bw_guard = dynet::const_parameter(cg, p_bw_guard);
  }
}

dynet::Expression BiLSTMBuilder::get_h(SymbolEmbedding &char_emb, const std::vector<unsigned> & c_id) {
  fw_lstm.start_new_sequence();
  bw_lstm.start_new_sequence();
  fw_lstm.add_input(fw_guard);
  bw_lstm.add_input(bw_guard);

  std::vector<dynet::Expression> inputs(c_id.size());
  for (int i = 0; i < c_id.size(); i++) {
    inputs[i] = char_emb.embed(c_id[i]);
  }
  for (int i = 0; i < inputs.size(); i++) {
    fw_lstm.add_input(inputs[i]);
    bw_lstm.add_input(inputs[inputs.size() - i - 1]);
  }
  return dynet::concatenate({ fw_lstm.get_h(inputs.size()).back(), bw_lstm.get_h(inputs.size()).back() });
}

