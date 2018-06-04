#include <model/tot.h>

namespace systm
{

TOT::TOT(int n_topics, float alpha, float beta, int large_word_threshold, int mh_step)
  : DocFirstBase(n_topics, beta, large_word_threshold, mh_step), alpha(alpha) {}

void TOT::InitializeOthers(bool is_train) {
  phi_a.clear();
  phi_a.resize(n_topics);
  phi_b.clear();
  phi_b.resize(n_topics);
  lgamma_phi.clear();
  lgamma_phi.resize(n_topics);
  doc_time.resize(cur_corpus->n_docs);
  log_doc_time.resize(cur_corpus->n_docs);
  log_1_doc_time.resize(cur_corpus->n_docs);

  double min_time = 1e18, max_time = -1e18;
  for (int d = 0; d < cur_corpus->n_docs; ++d) {
    min_time = std::min(min_time, GetDocTime(d));
    max_time = std::max(max_time, GetDocTime(d));
  }
  for (int d = 0; d < cur_corpus->n_docs; ++d) {
    doc_time[d] = 0.1 + ((GetDocTime(d) - min_time)
        / (max_time - min_time + 1e-8)) * 0.8;
    log_doc_time[d] = log(doc_time[d] + 1e-16);
    log_1_doc_time[d] = log(1 - doc_time[d] + 1e-16);
  }
}

void TOT::EstimateParameters(bool is_train) {
  static std::vector<double> mean, var;
  mean.clear();
  var.clear();
  mean.resize(n_topics);
  var.resize(n_topics);
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    for (int t = cur_corpus->doc_offset[doc]; t < cur_corpus->doc_offset[doc + 1]; ++t) {
      mean[topics[t]] += doc_time[doc];
    }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (topic_dist[i] > 0) {
      mean[i] /= topic_dist[i];
    }
  }
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    for (int t = cur_corpus->doc_offset[doc]; t < cur_corpus->doc_offset[doc + 1]; ++t) {
      double diff = mean[topics[t]] - doc_time[doc];
      var[topics[t]] += diff * diff;
    }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (topic_dist[i] > 1) {
      var[i] /= (topic_dist[i] - 1);
    }
  }
  for (int i = 0; i < n_topics; ++i) {
    if (var[i] > mean[i] * (1 - mean[i]) - 1e-5) {
      phi_a[i] = (mean[i] + 1e-5) * 0.1;
      phi_b[i] = (1 - mean[i] + 1e-5) * 0.1;
      lgamma_phi[i] = lgamma(phi_a[i]) + lgamma(phi_b[i]) - lgamma(phi_a[i] + phi_b[i]);
      continue;
    }
    phi_a[i] = mean[i] * (mean[i] * (1 - mean[i]) / (var[i] + 1e-5) - 1);
    phi_b[i] = (1 - mean[i]) * (mean[i] * (1 - mean[i]) / (var[i] + 1e-5) - 1);
    lgamma_phi[i] = lgamma(phi_a[i]) + lgamma(phi_b[i]) - lgamma(phi_a[i] + phi_b[i]);
  }
}

double TOT::Loglikelihood() {
  double llh = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:llh)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    std::vector<int> &dist = doc_topic_dist[doc];
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(dist[topic] + alpha);
      llh += dist[topic] * CalLogBetaTime(doc, topic);
    }
  }
  std::vector<int> dist;
#pragma omp parallel for schedule(dynamic) private(dist) reduction(+:llh)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    if (!is_large_word[word]) {
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(word_topic_dist[word].Count(topic) + beta);
      }
    }
    else {
      dist.clear();
      dist.resize(n_topics);
      for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
        dist[topics[cur_corpus->word_to_doc[i]]]++;
      }
      for (int topic = 0; topic < n_topics; ++topic) {
        llh += lgamma(dist[topic] + beta);
      }
    }
  }
  for (int topic = 0; topic < n_topics; ++topic) {
    llh -= lgamma(topic_dist[topic].load() + cur_corpus->n_words * beta);
  }
  return llh;
}

ResultBase* TOT::GenResult(bool is_train) {
  return nullptr;
}

}
