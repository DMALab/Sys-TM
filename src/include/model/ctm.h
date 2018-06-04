#pragma once

#include <model/doc_first_base.h>
#include "util/numeric/eigenmvn.h"

namespace systm
{

class CTM : public DocFirstBase {
 public:
  CTM(int n_topics, float mu, float sigma, float beta,
      int large_word_threshold = 300, int mh_step = 2, int sgld_iter_num = 10);

 protected:
  float mu_, sigma_;
  Eigen::MatrixXd eta, exp_eta;
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma, inv_sigma;
  double logdet_sigma;

  int sgld_iter_num;

  virtual float DocProposal(int doc, int topic, bool is_large_word) {
    return exp_eta(doc, topic) / (topic_dist[topic] + cur_corpus->n_words * beta);
  }

  virtual void InitializeOthers(bool is_train);
  virtual void SampleVariables(bool is_train);
  virtual ResultBase* GenResult(bool is_train);

  virtual double Loglikelihood();
};

}
