#pragma once

#include <model/doc_first_base.h>

namespace systm
{

class TOT : public DocFirstBase {
 public:
  TOT(int n_topics, float alpha, float beta, int large_word_threshold = 300, int mh_step = 2);

 protected:
  float alpha;
  std::vector<double> phi_a, phi_b;
  std::vector<double> lgamma_phi;

  std::vector<double> doc_time, log_doc_time, log_1_doc_time;

  virtual float DocProposal(int doc, int topic, bool is_large_word) {
    return exp(CalLogBetaTime(doc, topic)) * (doc_topic_dist[doc][topic] + alpha) /
        (topic_dist[topic] + cur_corpus->n_words * beta);
  }

  virtual void InitializeOthers(bool is_train);

  virtual void EstimateParameters(bool is_train);

  virtual ResultBase* GenResult(bool is_train);


  virtual double Loglikelihood();
 private:
  double GetDocTime(int d) {
    return cur_corpus->doc_infos[d][0].double_value;
  }
  double CalLogBetaTime(int doc, int topic) {
    double P = (phi_a[topic] - 1) * log_doc_time[doc] +
        (phi_b[topic] - 1) * log_1_doc_time[doc];
    return P - lgamma_phi[topic];
  }
};

}
