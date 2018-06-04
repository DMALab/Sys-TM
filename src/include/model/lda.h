#pragma once

#include <model/word_first_base.h>

namespace systm
{

class LDA : public WordFirstBase {
 public:
  LDA(int n_topics, float alpha, float beta, int long_doc_threshold = 300, int mh_step = 2);

 protected:
  float beta;

  virtual float WordProposal(int word, int topic, bool is_long_doc) {
    return (word_topic_dist[word][topic] + beta) / (topic_dist[topic] + cur_corpus->n_words * beta);
  }

  virtual ResultBase* GenResult(bool is_train);

  virtual double Loglikelihood();
};

}
