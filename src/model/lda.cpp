#include <model/lda.h>

namespace systm {

LDA::LDA(int n_topics, float alpha, float beta, int long_doc_threshold, int mh_step)
: WordFirstBase(n_topics, alpha, long_doc_threshold, mh_step), beta(beta) {}

ResultBase* LDA::GenResult(bool is_train) {
  return nullptr;
}

double LDA::Loglikelihood() {
  double llh = 0;
  std::vector<int> doc_dist;

  llh += cur_corpus->n_docs * (lgamma(n_topics * alpha) - n_topics * lgamma(alpha));
#pragma omp parallel for schedule(dynamic) private(doc_dist) reduction(+:llh)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    llh -= lgamma(DocSize(doc) + n_topics * alpha);
    if (is_long_doc[doc]) {
      doc_dist.clear();
      doc_dist.resize(n_topics);
      for (int i = cur_corpus->doc_offset[doc]; i < cur_corpus->doc_offset[doc + 1]; ++i) {
        doc_dist[topics[cur_corpus->doc_to_word[i]]]++;
      }
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(doc_dist[topic] + alpha);
      }
    }
    else {
      for (auto &item : doc_topic_dist[doc].GetItem()) {
        llh += lgamma(item.count + alpha);
      }
      llh += lgamma(alpha) * (n_topics - doc_topic_dist[doc].GetItem().size());
    }
  }

  llh += n_topics * (lgamma(beta * cur_corpus->n_words) - cur_corpus->n_words * lgamma(beta));
#pragma omp parallel for schedule(dynamic) reduction(+:llh)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(word_topic_dist[word][topic] + beta);
    }
  }
  for (int topic = 0; topic < n_topics; ++topic) {
    llh -= lgamma(topic_dist[topic].load() + cur_corpus->n_words * beta);
  }
  return llh;
}

}
