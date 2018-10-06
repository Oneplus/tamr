#include "math_utils.h"
#include <boost/assert.hpp>

void MeanStdevStreamer::clear() { n = 0; }

void MeanStdevStreamer::push(double x) {
  n++;
  if (n == 1) {
    old_m = new_m = x;
    old_s = 0.;
  } else {
    new_m = old_m + (x - old_m) / n;
    new_s = old_s + (x - old_m) * (x - new_m);

    old_m = new_m;
    old_s = new_s;
  }
}

int MeanStdevStreamer::num_data_values()  const { return n; }
double MeanStdevStreamer::mean()          const { return ((n > 0) ? new_m : 0.0); }
double MeanStdevStreamer::variance()      const { return ((n > 1) ? new_s / (n - 1) : 0.0); }
double MeanStdevStreamer::stdev()         const { return sqrt(variance()); }

void mean_and_stddev(const std::deque<float>& data,
                     float& mean, float& stddev) {
  float n = 0.;
  float sum1 = 0., sum2 = 0.;
  for (auto x : data) { sum1 += x; n += 1.; }
  mean = sum1 / n;
  for (auto x : data) { sum2 += (x - mean) * (x - mean); }
  stddev = sqrt(sum2 / (n - 1));
}

void softmax_copy(const std::vector<float>& input, std::vector<float>& output) {
  BOOST_ASSERT_MSG(input.size() > 0, "input should have one or more element.");
  float m = input[0];
  output.resize(input.size());
  for (unsigned i = 1; i < input.size(); ++i) { m = (input[i] > m ? input[i] : m); }
  float s = 0.;
  for (unsigned i = 0; i < input.size(); ++i) {
    output[i] = exp(input[i] - m);
    s += output[i];
  }
  for (unsigned i = 0; i < output.size(); ++i) { output[i] /= s; }
}

void softmax_inplace(std::vector<float>& x) {
  BOOST_ASSERT_MSG(x.size() > 0, "input should have one or more element.");
  float m = x[0];
  for (const float& _x : x) { m = (_x > m ? _x : m); }
  float s = 0.;
  for (unsigned i = 0; i < x.size(); ++i) {
    x[i] = exp(x[i] - m);
    s += x[i];
  }
  for (unsigned i = 0; i < x.size(); ++i) { x[i] /= s; }
}

void softmax_inplace_on_valid_indicies(std::vector<float>& x,
                                       const std::vector<unsigned>& valid_indices) {
  BOOST_ASSERT_MSG(x.size() > 0, "input should have one or more element.");
  BOOST_ASSERT_MSG(valid_indices.size() > 0, "input should have one or more indicces.");
  float m = x[valid_indices[0]];
  for (unsigned id : valid_indices) { m = (x[id] > m ? x[id] : m); }
  float s = 0.;
  for (unsigned id : valid_indices) {
    x[id] = exp(x[id] - m);
    s += x[id];
  }
  for (unsigned id : valid_indices) { x[id] /= s; }
}

void unnormalized_softmax_inplace(std::vector<float>& x) {
  BOOST_ASSERT_MSG(x.size() > 0, "input should have one or more element.");
  float m = x[0];
  for (const float& _x : x) { m = (_x > m ? _x : m); }
  for (unsigned i = 0; i < x.size(); ++i) { x[i] = exp(x[i] - m); }
}

std::vector<unsigned> fisher_yates_shuffle(unsigned size,
                                           unsigned max_size,
                                           std::mt19937& gen) {
  assert(size < max_size);
  std::vector<unsigned> b(size);

  for (unsigned i = 0; i < max_size; ++i) {
    std::uniform_int_distribution<> dis(0, i);
    unsigned j = dis(gen);
    if (j < b.size()) {
      if (i < j) {
        b[i] = b[j];
      }
      b[j] = i;
    }
  }
  return b;
}

unsigned distribution_sample(const std::vector<float>& prob,
                             std::mt19937& gen) {
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  // std::discrete_distribution produces random integers on the interval [0, n)
  // std::discrete_distribution<> d({40, 10, 10, 40});
  std::discrete_distribution<unsigned> distrib(prob.begin(), prob.end());
  return distrib(gen);
}

void reservoir_sample_n(const std::vector<unsigned>& S, unsigned N,
                        std::vector<unsigned>& R, unsigned K,
                        std::mt19937& gen) {
  for (unsigned i = 0; i < K; ++i) { R[i] = S[i]; }
  for (unsigned i = K; i < N; ++i) {
    std::uniform_int_distribution<> dis(0, i - 1);
    unsigned j = dis(gen);
    if (j < K) { R[j] = S[i]; }
  }
}

void fast_reservoir_sample_n(const std::vector<unsigned>& S, unsigned N,
                             std::vector<unsigned>& R, unsigned K,
                             std::mt19937& gen) {
  for (unsigned i = 0; i < K; ++i) { R[i] = S[i]; }
  unsigned t = 4 * K;
  unsigned j = 1 + K;
  while (j < N && j <= t) {
    std::uniform_int_distribution<> dis(0, j - 1);
    unsigned k = dis(gen);
    if (k < K) { R[k] = S[j]; }
    j++;
  }
  while (j < N) {
    float p = static_cast<float>(K) / j;
    std::uniform_real_distribution<> dis(0, 1);
    float u = dis(gen);
    unsigned g = static_cast<unsigned>(floor(log(u) / log(1 - p)));
    j = j + g;
    if (j < N) {
      std::uniform_int_distribution<> dis(0, K - 1);
      unsigned k = dis(gen);
      if (k < K) { R[k] = S[j]; }
    }
    j++;
  }
}
