#pragma once

#include <model/doc_first_base.h>
#include "util/numeric/eigenmvn.h"

namespace systm
{

class STM : public DocFirstBase {
 public:
  STM(int n_topics, float alpha, float beta, float sigma, int large_word_threshold = 300, int mh_step = 2);

 protected:
  float alpha, sigma;
  std::vector<float> eta, y;
  std::vector<float> eta_z, eta_z_origin;

  virtual void InitializeOthers(bool is_train);
  virtual void PrepareFTreeForDoc(int doc);
  virtual bool AcceptFTreeSample(int doc, int token, int old_topic, int topic);
  virtual void ClearTokenTopic(int doc, int token, int topic);
  virtual void SetTokenTopic(int doc, int token, int topic);
  virtual void EstimateParameters(bool is_train);
  virtual ResultBase* GenResult(bool is_train);
  virtual double Loglikelihood();

  virtual float DocProposal(int doc, int topic, bool is_large_word) {
    if (is_large_word) {
      float eta_nk = eta[topic] / DocSize(doc);
      float y_etaz = y[doc] - eta_z[doc];
      return exp(eta_nk * (2 * y_etaz - eta_nk) / (2 * sigma)) *
          (doc_topic_dist[doc][topic] + alpha) / (topic_dist[topic] + cur_corpus->n_words * beta);
    }
    else {
      float eta_nk = eta[topic] / DocSize(doc);
      float y_etaz = y[doc] - eta_z_origin[doc];
      return exp(eta_nk * (2 * y_etaz - eta_nk) / (2 * sigma)) *
          (doc_topic_dist[doc][topic] + alpha) / (topic_dist[topic] + cur_corpus->n_words * beta);
    }
  }
};

}
