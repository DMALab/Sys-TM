#include <model/word_first_base.h>
#include <util/alias_table.h>
#include <util/ftree.h>
#include <omp.h>
#include <sys/time.h>

namespace systm {

WordFirstBase::WordFirstBase(int n_topics, float alpha, int long_doc_threshold, int mh_step)
    : n_topics(n_topics), alpha(alpha), long_doc_threshold(long_doc_threshold), mh_step(mh_step) {
  // Nothing to do
}

ResultBase* WordFirstBase::Train(const Corpus& corpus, int max_iter, int thread_num, bool verbose) {
  if (thread_num > 0 && thread_num <= 128) {
    omp_set_num_threads(thread_num);
  }
  Initialize(corpus, true);
  timeval time_start{}, time_end{};
  unsigned sample_time, total_time = 0;
  for (int i = 0; i < max_iter; ++i) {
    gettimeofday(&time_start, nullptr);
    VisitByDoc();
    VisitByWord();
    FTreeIteration();
    SampleVariables(true);
    EstimateParameters(true);
    gettimeofday(&time_end, nullptr);
    sample_time = (unsigned int) ((time_end.tv_sec - time_start.tv_sec) * 1000000 + time_end.tv_usec - time_start.tv_usec);
    total_time += sample_time;
    if (verbose) {
      std::cout << "iter " << i << " " << sample_time / 1e6 << "(s) " << total_time / 1e6
                << "(s)   " << "llh: " << Loglikelihood() << std::endl;
    }
  }
  ResultBase* result = GenResult(true);
  FinalizeAndClear(true);
  return result;
}

ResultBase* WordFirstBase::Inference(const Corpus& corpus, int max_iter, int thread_num, bool verbose) {
  if (thread_num > 0 && thread_num <= 128) {
    omp_set_num_threads(thread_num);
  }
  Initialize(corpus, false);
  timeval time_start{}, time_end{};
  unsigned sample_time, total_time = 0;
  for (int i = 0; i < max_iter; ++i) {
    gettimeofday(&time_start, nullptr);
    VisitByDoc();
    VisitByWord();
    FTreeIteration();
    SampleVariables(true);
    EstimateParameters(true);
    gettimeofday(&time_end, nullptr);
    sample_time = (unsigned int) ((time_end.tv_sec - time_start.tv_sec) * 1000000 + time_end.tv_usec - time_start.tv_usec);
    total_time += sample_time;
    if (verbose) {
      std::cout << "iter " << i << " " << sample_time / 1e6 << "(s) " << total_time / 1e6
                << "(s)   " << "llh: " << Loglikelihood() << std::endl;
    }
  }
  ResultBase* result = GenResult(false);
  FinalizeAndClear(false);
  return result;
}

void WordFirstBase::Initialize(const Corpus& corpus, bool is_train) {
  cur_corpus = &corpus;

  topics.clear();
  topics.resize(cur_corpus->n_tokens);
  mh_proposal.clear();
  mh_proposal.resize(cur_corpus->n_tokens * mh_step);
  topic_dist.clear();
  topic_dist.resize(n_topics);

  doc_topic_dist.clear();
  is_long_doc.resize(cur_corpus->n_docs);
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    is_long_doc[doc] = (DocSize(doc) > long_doc_threshold);
    if (!is_long_doc[doc]) {
      doc_topic_dist.emplace_back(n_topics);
    }
    else {
      doc_topic_dist.emplace_back(0);
    }
  }

  for (int word = 0; word < cur_corpus->n_words; ++word) {
    std::string word_s = corpus.word_list[word];
    if (word_to_int.count(word_s) == 0) {
      word_to_int[word_s] = word_list.size();
      word_list.push_back(cur_corpus->word_list[word]);
      word_topic_dist.emplace_back(n_topics);
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    std::string word_s = corpus.word_list[word];
    int cur_word_id = word_to_int[word_s];
    std::vector<int> &word_dist = word_topic_dist[cur_word_id];

    for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
      int doc = cur_corpus->word_content[i];
      int topic = topics[i] = random.RandInt(n_topics);
      if (is_long_doc[doc]) {
        for (int j = 0; j < mh_step; ++j) {
          mh_proposal[i * mh_step + j] = random.RandInt(n_topics);
        }
      }
      topic_dist[topic]++;
      word_topic_dist[cur_word_id][topic]++;
      if (!is_long_doc[doc]) {
        doc_topic_dist[doc].Inc(topic);
      }
    }
  }

  for (int word = 0; word < cur_corpus->n_words; ++word) {
    std::string word_s = corpus.word_list[word];
    int id = word_to_int[word_s];
    if (id != word) {
      std::swap(word_list[id], word_list[word]);
      word_topic_dist[id].swap(word_topic_dist[word]);
      word_to_int[word_list[id]] = id;
      word_to_int[word_list[word]] = word;
    }
  }

  InitializeOthers(is_train);
  EstimateParameters(is_train);
}

void WordFirstBase::FTreeIteration() {
  FTree tree(n_topics);
  std::vector<float> psum(n_topics);

#pragma omp parallel for schedule(dynamic) firstprivate(psum), firstprivate(tree)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    std::vector<int> &word_dist = word_topic_dist[word];
    PrepareFTreeForWord(word);

    for (int i = 0; i < n_topics; ++i) {
      tree.Set(i, WordProposal(word, i, false));
    }
    tree.Build();
    for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
      int doc = cur_corpus->word_content[i];
      if (is_long_doc[doc]) continue;
      int topic = topics[i];
      SparseCounter &doc_dist = doc_topic_dist[doc];
      doc_dist.Lock();

      ClearTokenTopic(word, i, topic);
      tree.Set(topic, WordProposal(word, topic, false));

      float prob_left = tree.Sum() * alpha;
      float prob_all = prob_left;
      const std::vector<CountItem> &items = doc_dist.GetItem();
      for (int t = 0, s = (int)items.size(); t < s; ++t) {
        float p = items[t].count * tree.Get(items[t].item);
        prob_all += p;
        psum[t] = p;
        if (t > 0) psum[t] += psum[t - 1];
      }

      float prob = random.RandDouble(prob_all);
      int new_topic;
      if (prob < prob_left) {
        new_topic = tree.Sample(prob / alpha);
      }
      else {
        prob -= prob_left;
        int p = (lower_bound(psum.begin(), psum.begin() + items.size(), prob) - psum.begin());
        new_topic = items[p].item;
      }
      if (!AcceptFTreeSample(word, i, topic, new_topic)) {
        new_topic = topic;
      }
      SetTokenTopic(word, i, new_topic);
      tree.Set(new_topic, WordProposal(word, new_topic, false));
      topics[i] = new_topic;
      doc_dist.Unlock();
    }
  }
}

void WordFirstBase::VisitByDoc() {
  std::vector<int> doc_dist;
  std::vector<int> local_topics;
  std::vector<int> local_mh_proposal;

#pragma omp parallel for schedule(dynamic) private(doc_dist) private(local_topics) private(local_mh_proposal)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    if (!is_long_doc[doc]) continue;
    int N = cur_corpus->doc_offset[doc + 1] - cur_corpus->doc_offset[doc];
    int offset = cur_corpus->doc_offset[doc];

    doc_dist.clear();
    doc_dist.resize(n_topics);
    local_topics.resize(N);
    local_mh_proposal.resize(N * mh_step);

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      int pos = cur_corpus->doc_to_word[t];
      int topic = topics[pos];
      local_topics[i] = topic;
      doc_dist[topic]++;
      for (int j = 0; j < mh_step; ++j) {
        local_mh_proposal[i * mh_step + j] = mh_proposal[pos * mh_step + j];
      }
    }

    for (int i = 0; i < N; ++i) {
      int topic = local_topics[i];

      for (int m = 0; m < mh_step; ++m) {
        int new_topic = local_mh_proposal[i * mh_step + m];
        double Cdj = doc_dist[new_topic] + alpha;
        double Cdi = doc_dist[topic] + alpha;
        double prob = Cdj / Cdi;
        if (random.RandDouble() < prob) {
          topic = new_topic;
        }
      }
      local_topics[i] = topic;
    }

    double prob = (n_topics * alpha) / (n_topics * alpha + N);
    for (int i = 0; i < N; ++i) {
      for (int m = 0; m < mh_step; ++m) {
        if (random.RandDouble() < prob) {
          local_mh_proposal[i * mh_step + m] = random.RandInt(n_topics);
        }
        else {
          local_mh_proposal[i * mh_step + m] = local_topics[random.RandInt(N)];
        }
      }
    }
    for (int i = 0, t = offset; i < N; ++i, ++t) {
      int pos = cur_corpus->doc_to_word[t];
      int word = cur_corpus->doc_content[t];
      int old_topic = topics[pos];
      int topic = local_topics[i];
      ClearTokenTopic(word, pos, old_topic);
      SetTokenTopic(word, pos, topic);
      for (int j = 0; j < mh_step; ++j) {
        mh_proposal[pos * mh_step + j] = local_mh_proposal[i * mh_step + j];
      }
    }
  }
}

void WordFirstBase::VisitByWord() {
  std::vector<int> local_topics;
  std::vector<float> prob;
  AliasTable alias;
#pragma omp parallel for schedule(dynamic) private(local_topics) private(prob) private(alias)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    int N = cur_corpus->word_offset[word + 1] - cur_corpus->word_offset[word];
    int offset = cur_corpus->word_offset[word];
    std::vector<int> &word_dist = word_topic_dist[word];
    local_topics.resize(N);

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_long_doc[cur_corpus->word_content[t]]) continue;
      int topic = topics[t];

      for (int m = 0; m < mh_step; ++m) {
        int new_topic = mh_proposal[t * 2 + m];
        float accept_rate = WordProposal(word, new_topic, true) / WordProposal(word, topic, true);
        if (random.RandDouble() < accept_rate) {
          topic = new_topic;
        }
      }

      local_topics[i] = topic;
    }

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_long_doc[cur_corpus->word_content[t]]) continue;
      int old_topic = topics[t];
      int topic = local_topics[i];
      ClearTokenTopic(word, t, old_topic);
      SetTokenTopic(word, t, topic);
    }

    prob.resize(n_topics);
    for (int i = 0; i < n_topics; ++i) {
      prob[i] = WordProposal(word, i, true);
    }
    alias.Init(prob);
    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_long_doc[cur_corpus->word_content[t]]) continue;
      for (int m = 0; m < mh_step; ++m) {
        mh_proposal[t * mh_step + m] = alias.Sample(random);
      }
    }
  }
}

// Customized Processes

void WordFirstBase::PrepareFTreeForWord(int word) {
  // Default : Empty
}

bool WordFirstBase::AcceptFTreeSample(int word, int token, int old_topic, int topic) {
  return true;
}

void WordFirstBase::InitializeOthers(bool is_train) {
  // Default: Emtpty
}

void WordFirstBase::SampleVariables(bool is_train) {
  // Default: Emtpty
}

void WordFirstBase::EstimateParameters(bool is_train) {
  // Default: Empty
}

void WordFirstBase::FinalizeAndClear(bool is_train) {
  DefaultFinalizeAndClear(is_train);
}

double WordFirstBase::Loglikelihood() {
  // Default:
  return 0.0;
}

// User Defined Computations
void WordFirstBase::ClearTokenTopic(int word, int token, int topic) {
  int doc = cur_corpus->word_content[token];
  DefaultClearTokenTopic(doc, word, topic);
}

void WordFirstBase::SetTokenTopic(int word, int token, int topic) {
  int doc = cur_corpus->word_content[token];
  DefaultSetTokenTopic(doc, word, topic);
}

void WordFirstBase::LoadModel(const std::string &path) {

}

void WordFirstBase::SaveModel(const std::string &path) {

}

ResultBase* systm::WordFirstBase::GenResult(bool is_train) {
  return nullptr;
}

float systm::WordFirstBase::WordProposal(int word, int topic, bool is_long_doc) {
  return 0.0;
}

}
