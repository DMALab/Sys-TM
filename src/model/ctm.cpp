#include <model/ctm.h>
#include "util/numeric/logdet.h"

namespace systm
{

CTM::CTM(int n_topics, float mu, float sigma, float beta, int large_word_threshold,
         int mh_step, int sgld_iter_num)
    : DocFirstBase(n_topics, beta, large_word_threshold, mh_step), mu_(mu), sigma_(sigma),
      sgld_iter_num(sgld_iter_num) {}

void CTM::InitializeOthers(bool is_train) {
  eta.resize(cur_corpus->n_docs, n_topics);
  exp_eta.resize(cur_corpus->n_docs, n_topics);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < cur_corpus->n_docs; ++i) {
    int thread = omp_get_thread_num();
    for (int j = 0; j < n_topics; ++j) {
      eta(i, j) = rand[thread].RandNorm(mu_, sigma_);
      exp_eta(i, j) = exp(eta(i, j));
    }
    if (i % 100000 == 0) std::cout << i << std::endl;
  }
  mu.resize(n_topics);
  mu.fill(mu_);
  sigma.resize(n_topics, n_topics);
  for (int i = 0; i < n_topics; ++i) {
    sigma(i, i) = sigma_;
  }
  inv_sigma = sigma.inverse();
  logdet_sigma = logdet(sigma);
}

void CTM::SampleVariables(bool is_train) {
  Eigen::VectorXd delta(n_topics);
  for (int iter = 0; iter < sgld_iter_num; ++iter) {
    double eps = 0.01 * pow(30.0 + iter, -0.75);
#pragma omp parallel for schedule(dynamic) firstprivate(delta)
    for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
      int thread = omp_get_thread_num();
      delta = -(eta.row(doc) - mu.transpose()) / sigma_;// * inv_sigma;
      int Nd = DocSize(doc);
      for (int i = cur_corpus->doc_offset[doc]; i < cur_corpus->doc_offset[doc + 1]; ++i) {
        delta(topics[i]) += 1;
      }
      double sum_exp_eta = exp_eta.row(doc).sum();
      for (int k = 0; k < n_topics; ++k) {
        delta(k) -= Nd * exp_eta(doc, k) / sum_exp_eta;
      }
      delta *= 0.5 * eps;
      for (int k = 0; k < n_topics; ++k) {
        delta(k) += rand[thread].RandNorm(0.0, eps);
      }
      eta.row(doc) += delta;
      for (int i = 0; i < n_topics; ++i) {
        exp_eta(doc, i) = exp(eta(doc, i));
      }
    }
  }
  /*
  mu.fill(0);
  for (int d = 0; d < cur_corpus->n_docs; ++d) {
    mu += eta.row(d);
  }
  mu /= cur_corpus->n_docs;
  sigma.fill(0);
  for (int d = 0; d < cur_corpus->n_docs; ++d) {
    sigma += (eta.row(d).transpose() - mu) * (eta.row(d) - mu.transpose());
  }
  sigma /= cur_corpus->n_docs;
  inv_sigma = sigma.inverse();
  logdet_sigma = logdet(sigma);
   */
}

double CTM::Loglikelihood() {
  double llh = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:llh)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    std::vector<int> &dist = doc_topic_dist[doc];
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += dist[topic] * eta(doc, topic);
    }
    double sum_exp_eta = exp_eta.row(doc).sum();
    llh -= DocSize(doc) * log(sum_exp_eta);
    llh -= 0.5 / (sigma_ * (eta.row(doc) - mu.transpose()) * (eta.row(doc).transpose() - mu));
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

ResultBase* CTM::GenResult(bool is_train) {
  return nullptr;
}

}
