#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <deque>
#include <vector>
#include <random>

struct MeanStdevStreamer {
  int n;
  double old_m, new_m, old_s, new_s;

  void clear();
  void push(double x);
  int num_data_values() const;
  double mean()         const;
  double variance()     const; 
  double stdev()        const;
};

void mean_and_stddev(const std::deque<float>& data,
                     float& mean,
                     float& stddev);

void softmax_copy(const std::vector<float>& input,
                  std::vector<float>& out);

void softmax_inplace(std::vector<float>& x);

void softmax_inplace_on_valid_indicies(std::vector<float>& x,
                                       const std::vector<unsigned>& valid_indices);

void unnormalized_softmax_inplace(std::vector<float>& x);

// Shuffle
std::vector<unsigned> fisher_yates_shuffle(unsigned size,
                                           unsigned max_size,
                                           std::mt19937& gen);

// Sample one
unsigned distribution_sample(const std::vector<float>& prob, std::mt19937& gen);

// Sample n
void reservoir_sample_n(const std::vector<unsigned>& S, unsigned N,
                        std::vector<unsigned>& R, unsigned K,
                        std::mt19937& gen);

void fast_reservoir_sample_n(const std::vector<unsigned>& S, unsigned N,
                             std::vector<unsigned>& R, unsigned K,
                             std::mt19937& gen);

#endif  //  end for MATH_UTILS_H
