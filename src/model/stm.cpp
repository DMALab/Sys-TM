#include <model/stm.h>
#include "Eigen/Sparse"

namespace systm
{

STM::STM(int n_topics, float alpha, float beta, float sigma,
         int large_word_threshold, int mh_step)
    : DocFirstBase(n_topics, beta, large_word_threshold, mh_step), alpha(alpha), sigma(sigma) {}

void STM::InitializeOthers(bool is_train) {
  y.resize(cur_corpus->n_docs);
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    y[doc] = cur_corpus->doc_infos[doc][0].double_value;
  }
  eta.resize(n_topics + 1);
  eta_z.resize(cur_corpus->n_docs);
  eta_z_origin.resize(cur_corpus->n_docs);
}

void STM::PrepareFTreeForDoc(int doc) {
  eta_z_origin[doc] = eta_z[doc];
}

void STM::ClearTokenTopic(int doc, int token, int topic) {
  eta_z[doc] -= eta[topic] / DocSize(doc);
  DefaultClearTokenTopic(doc, cur_corpus->doc_content[token], topic);
}

void STM::SetTokenTopic(int doc, int token, int topic) {
  eta_z[doc] += eta[topic] / DocSize(doc);
  DefaultSetTokenTopic(doc, cur_corpus->doc_content[token], topic);
}

bool STM::AcceptFTreeSample(int doc, int token, int old_topic, int topic) {
  double accept = rand[omp_get_thread_num()].RandDouble();
  return accept < exp((eta[topic] - eta[old_topic]) / DocSize(doc) * (eta_z_origin[doc] - eta_z[doc]) / sigma);
}

void STM::EstimateParameters(bool is_train) {
  Eigen::SparseMatrix<double> Z(n_topics + 1, cur_corpus->n_docs);
  Eigen::VectorXd x, y(cur_corpus->n_docs), t;

  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    int N = DocSize(doc);

    y(doc) = cur_corpus->doc_infos[doc][0].double_value;

    std::vector<int> &doc_dist = doc_topic_dist[doc];

    Z.insert(n_topics, doc) = 1;
    for (int i = 0; i < n_topics; ++i) {
      if (doc_dist[i] > 0) {
        Z.insert(i, doc) = doc_dist[i] / double(N);
      }
    }
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(Z * Z.transpose());
  x = solver.solve(Z * y);

  for (int i = 0; i <= n_topics; ++i) {
    eta[i] = x(i);
  }
#pragma omp parallel for schedule(dynamic)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    this->eta_z[doc] = 1.2;
    eta_z[doc] = eta[n_topics];
    for (int k = 0; k < n_topics; ++k) {
      eta_z[doc] += (eta[k] * doc_topic_dist[doc][k]) / DocSize(doc);
    }
  }
}

double STM::Loglikelihood() {
  double llh = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:llh)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    std::vector<int>& dist = doc_topic_dist[doc];
    double y_etaz = y[doc] - eta_z[doc];
    for (int topic = 0; topic < n_topics; ++topic) {
      llh += lgamma(dist[topic] + alpha);
    }
    llh -= y_etaz * y_etaz / (2 * sigma);
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

ResultBase* STM::GenResult(bool is_train) {
  return nullptr;
}

}
