#pragma once

#include "corpus/corpus.h"
#include "result/result_base.h"
#include "util/atomic_integer.h"
#include "util/sparse_counter.h"
#include "util/random.h"

namespace systm {

class WordFirstBase {
 public:
  ResultBase* Train(const Corpus& corpus, int max_iter = 100, int thread_num = -1, bool verbose = true);
  ResultBase* Inference(const Corpus& corpus, int max_iter = 100, int thread_num = -1, bool verbose = false);
  void LoadModel(const std::string &path);
  void SaveModel(const std::string &path);

  WordFirstBase(int n_topics, float alpha, int long_doc_threshold = 300, int mh_step = 2);

 protected:
  int n_topics;
  float alpha;
  int long_doc_threshold;
  int mh_step;

  std::vector<bool> is_long_doc;
  const Corpus* cur_corpus;
  int DocSize(int doc) { return cur_corpus->doc_offset[doc + 1] - cur_corpus->doc_offset[doc]; }
  int WordSize(int word) { return cur_corpus->word_offset[word + 1] - cur_corpus->word_offset[word]; }

  std::unordered_map<std::string, int> word_to_int;
  std::vector<std::string> word_list;

  std::vector<SparseCounter> doc_topic_dist;
  std::vector<std::vector<int>> word_topic_dist;
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
  virtual void PrepareFTreeForWord(int word);
  virtual bool AcceptFTreeSample(int word, int token, int old_topic, int topic);
  virtual float WordProposal(int word, int topic, bool is_long_doc);
  virtual void ClearTokenTopic(int word, int token, int topic);
  virtual void SetTokenTopic(int word, int token, int topic);

  // Default Behaviors
  void DefaultClearTokenTopic(int doc, int word, int topic) {
    if (!is_long_doc[doc]) {
      doc_topic_dist[doc].Dec(topic);
    }
    word_topic_dist[word][topic]--;
    topic_dist[topic]--;
  }

  void DefaultSetTokenTopic(int doc, int word, int topic) {
    if (!is_long_doc[doc]) {
      doc_topic_dist[doc].Inc(topic);
    }
    word_topic_dist[word][topic]++;
    topic_dist[topic]++;
  }

  void DefaultFinalizeAndClear(bool is_train) {
    if (!is_train) {
      for (int word = 0; word < cur_corpus->n_words; ++word) {
        for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
          int topic = topics[i];
          word_topic_dist[word][topic]--;
          topic_dist[topic]--;
        }
      }
    }
    doc_topic_dist.clear();
    is_long_doc.clear();
  }
};

}
