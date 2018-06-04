#pragma once

#include "corpus/corpus.h"
#include "result/result_base.h"
#include "util/atomic_integer.h"
#include "util/sparse_counter.h"
#include "util/random.h"

namespace systm {

class DocFirstBase {
 public:
  ResultBase* Train(const Corpus& corpus, int max_iter = 100, int thread_num = -1, bool verbose = true);
  ResultBase* Inference(const Corpus& corpus, int max_iter = 100, int thread_num = -1, bool verbose = false);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  DocFirstBase(int n_topics, float alpha, int large_word_threshold = 10000, int mh_step = 2);

 protected:
  int n_topics;
  float beta;
  int large_word_threshold;
  int mh_step;

  std::vector<bool> is_large_word;
  const Corpus* cur_corpus;
  int DocSize(int doc) { return cur_corpus->doc_offset[doc + 1] - cur_corpus->doc_offset[doc]; }
  int WordSize(int word) { return cur_corpus->word_offset[word + 1] - cur_corpus->word_offset[word]; }

  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<std::vector<int>> doc_topic_dist;
  std::vector<SparseCounter> word_topic_dist;
  std::vector<AtomicInt> topic_dist;

  // word-by-word
  std::vector<int> topics;
  std::vector<int> mh_proposal;

  Random rand[128];

  void Initialize(const Corpus& corpus, bool is_train);
  void FTreeIteration();
  void VisitByDoc();
  void VisitByWord();

  // Customized Processes
  virtual void InitializeOthers(bool is_train);
  virtual void SampleVariables(bool is_train);
  virtual void EstimateParameters(bool is_train);
  virtual ResultBase* GenResult(bool is_train);
  virtual void FinalizeAndClear(bool is_train);
  virtual double Loglikelihood();

  // User Defined Computations
  virtual void PrepareFTreeForDoc(int doc);
  virtual bool AcceptFTreeSample(int doc, int token, int old_topic, int topic);
  virtual float DocProposal(int doc, int topic, bool is_large_word);
  virtual void ClearTokenTopic(int doc, int token, int topic);
  virtual void SetTokenTopic(int doc, int token, int topic);

  // Default Behaviors
  void DefaultClearTokenTopic(int doc, int word, int topic) {
    if (is_large_word[word]) {
      word_topic_dist[word].Dec(topic);
    }
    doc_topic_dist[doc][topic]--;
    topic_dist[topic]--;
  }

  void DefaultSetTokenTopic(int doc, int word, int topic) {
    if (is_large_word[word]) {
      word_topic_dist[word].Inc(topic);
    }
    doc_topic_dist[doc][topic]++;
    topic_dist[topic]++;
  }

  void DefaultFinalizeAndClear(bool is_train) {
    if (!is_train) {
      for (int word = 0; word < cur_corpus->n_words; ++word) {
        for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
          int topic = topics[i];
          word_topic_dist[word].Dec(topic);
          topic_dist[topic]--;
        }
      }
    }
    doc_topic_dist.clear();
    is_large_word.clear();
  }
};

}
