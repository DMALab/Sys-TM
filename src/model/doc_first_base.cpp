#include <model/doc_first_base.h>
#include <util/alias_table.h>
#include <util/ftree.h>
#include <omp.h>
#include <sys/time.h>

namespace systm {

ResultBase* DocFirstBase::Train(const Corpus& corpus, int max_iter, int thread_num, bool verbose) {
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

ResultBase* DocFirstBase::Inference(const Corpus& corpus, int max_iter, int thread_num, bool verbose) {
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
    SampleVariables(false);
    EstimateParameters(false);
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

DocFirstBase::DocFirstBase(int n_topics, float beta, int large_word_threshold, int mh_step)
    : n_topics(n_topics), beta(beta), large_word_threshold(large_word_threshold), mh_step(mh_step) {
  // Nothing to do
}

void DocFirstBase::Initialize(const Corpus& corpus, bool is_train) {
  cur_corpus = &corpus;

  topics.clear();
  topics.resize(cur_corpus->n_tokens);
  mh_proposal.clear();
  mh_proposal.resize(cur_corpus->n_tokens * mh_step);
  topic_dist.clear();
  topic_dist.resize(n_topics);

  doc_topic_dist.clear();
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    doc_topic_dist.emplace_back(n_topics);
  }

  is_large_word.resize(cur_corpus->n_words);
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    std::string word_s = corpus.word_list[word];
    is_large_word[word] = (WordSize(word) > large_word_threshold);
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
    SparseCounter &word_dist = word_topic_dist[cur_word_id];

    for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
      int pos = cur_corpus->word_to_doc[i];
      int topic = topics[pos] = random.RandInt(n_topics);
      if (is_large_word[word]) {
        for (int j = 0; j < mh_step; ++j) {
          mh_proposal[pos * mh_step + j] = random.RandInt(n_topics);
        }
      }
      topic_dist[topic]++;
      word_topic_dist[cur_word_id].Inc(topic);
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    for (int i = cur_corpus->doc_offset[doc]; i < cur_corpus->doc_offset[doc + 1]; ++i) {
      doc_topic_dist[doc][topics[i]]++;
    }
  }

  for (int word = 0; word < cur_corpus->n_words; ++word) {
    std::string word_s = corpus.word_list[word];
    int id = word_to_int[word_s];
    if (id != word) {
      std::swap(word_list[id], word_list[word]);
      word_topic_dist[id].Swap(word_topic_dist[word]);
      word_to_int[word_list[id]] = id;
      word_to_int[word_list[word]] = word;
    }
  }
  InitializeOthers(is_train);
  EstimateParameters(is_train);
}

void DocFirstBase::FTreeIteration() {
  FTree tree(n_topics);
  std::vector<float> psum(n_topics);

#pragma omp parallel for schedule(dynamic) firstprivate(psum), firstprivate(tree)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    std::vector<int> &doc_dist = doc_topic_dist[doc];
    PrepareFTreeForDoc(doc);

    for (int i = 0; i < n_topics; ++i) {
      tree.Set(i, DocProposal(doc, i, false));
    }
    tree.Build();
    for (int i = cur_corpus->doc_offset[doc]; i < cur_corpus->doc_offset[doc + 1]; ++i) {
      int word = cur_corpus->doc_content[i];
      if (is_large_word[word]) continue;
      int topic = topics[i];
      SparseCounter &word_dist = word_topic_dist[word];
      word_dist.Lock();

      ClearTokenTopic(doc, i, topic);
      tree.Set(topic, DocProposal(doc, topic, false));

      float prob_left = tree.Sum() * beta;
      float prob_all = prob_left;
      const std::vector<CountItem> &items = word_dist.GetItem();
      for (int t = 0, s = (int)items.size(); t < s; ++t) {
        float p = items[t].count * tree.Get(items[t].item);
        prob_all += p;
        psum[t] = p;
        if (t > 0) psum[t] += psum[t - 1];
      }

      float prob = random.RandDouble(prob_all);
      int new_topic;
      if (prob < prob_left) {
        new_topic = tree.Sample(prob / beta);
      }
      else {
        prob -= prob_left;
        int p = (lower_bound(psum.begin(), psum.begin() + items.size(), prob) - psum.begin());
        new_topic = items[p].item;
      }
      if (!AcceptFTreeSample(doc, i, topic, new_topic)) {
        new_topic = topic;
      }
      SetTokenTopic(doc, i, new_topic);
      tree.Set(new_topic, DocProposal(doc, new_topic, false));
      topics[i] = new_topic;
      word_dist.Unlock();
    }
  }
}

void DocFirstBase::VisitByWord() {
  std::vector<int> word_dist;
  std::vector<int> local_topics;
  std::vector<int> local_mh_proposal;

#pragma omp parallel for schedule(dynamic) private(word_dist) private(local_topics) private(local_mh_proposal)
  for (int word = 0; word < cur_corpus->n_words; ++word) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    if (!is_large_word[word]) continue;
    int N = cur_corpus->word_offset[word + 1] - cur_corpus->word_offset[word];
    int offset = cur_corpus->word_offset[word];

    word_dist.clear();
    word_dist.resize(n_topics);

    local_topics.resize(N);
    local_mh_proposal.resize(N * mh_step);

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      int pos = cur_corpus->word_to_doc[t];
      int topic = topics[pos];
      local_topics[i] = topic;
      word_dist[topic]++;
      for (int j = 0; j < mh_step; ++j) {
        local_mh_proposal[i * mh_step + j] = mh_proposal[pos * mh_step + j];
      }
    }

    for (int i = 0; i < N; ++i) {
      int topic = local_topics[i];

      for (int m = 0; m < mh_step; ++m) {
        int new_topic = local_mh_proposal[i * mh_step + m];
        double Cwj = word_dist[new_topic] + beta;
        double Cwi = word_dist[topic] + beta;
        double prob = Cwj / Cwi;
        if (random.RandDouble() < prob) {
          topic = new_topic;
        }
      }
      local_topics[i] = topic;
    }

    double prob = (n_topics * beta) / (n_topics * beta + N);
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
      int pos = cur_corpus->word_to_doc[t];
      int doc = cur_corpus->word_content[t];
      int old_topic = topics[pos];
      int topic = local_topics[i];
      ClearTokenTopic(doc, pos, old_topic);
      SetTokenTopic(doc, pos, topic);
      topics[pos] = topic;
      for (int j = 0; j < mh_step; ++j) {
        mh_proposal[pos * mh_step + j] = local_mh_proposal[i * mh_step + j];
      }
    }
  }
}

void DocFirstBase::VisitByDoc() {
  std::vector<int> local_topics;
  std::vector<float> prob;
  AliasTable alias;
#pragma omp parallel for schedule(dynamic) private(local_topics) private(prob) private(alias)
  for (int doc = 0; doc < cur_corpus->n_docs; ++doc) {
    int thread = omp_get_thread_num();
    Random& random = rand[thread];

    int N = cur_corpus->doc_offset[doc + 1] - cur_corpus->doc_offset[doc];
    int offset =cur_corpus-> doc_offset[doc];
    std::vector<int> &doc_dist = doc_topic_dist[doc];
    local_topics.resize(N);

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_large_word[cur_corpus->doc_content[t]]) continue;
      int topic = topics[t];

      for (int m = 0; m < mh_step; ++m) {
        int new_topic = mh_proposal[t * 2 + m];
        float accept_rate = DocProposal(doc, new_topic, true) / DocProposal(doc, topic, true);
        if (random.RandDouble() < accept_rate) {
          topic = new_topic;
        }
      }

      local_topics[i] = topic;
    }

    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_large_word[cur_corpus->doc_content[t]]) continue;
      int old_topic = topics[t];
      int topic = local_topics[i];
      ClearTokenTopic(doc, t, old_topic);
      SetTokenTopic(doc, t, topic);
    }

    prob.resize(n_topics);
    for (int i = 0; i < n_topics; ++i) {
      prob[i] = DocProposal(doc, i, true);
    }
    alias.Init(prob);
    for (int i = 0, t = offset; i < N; ++i, ++t) {
      if (!is_large_word[cur_corpus->doc_content[t]]) continue;
      for (int m = 0; m < mh_step; ++m) {
        mh_proposal[t * mh_step + m] = alias.Sample(random);
      }
    }
  }
}

// Customized Processes

void DocFirstBase::PrepareFTreeForDoc(int doc) {
  // Default : Empty
}
bool DocFirstBase::AcceptFTreeSample(int doc, int token, int old_topic, int topic) {
  return true;
}

void DocFirstBase::InitializeOthers(bool is_train) {
  // Default: Emtpty
}

void DocFirstBase::SampleVariables(bool is_train) {
  // Default: Emtpty
}

void DocFirstBase::EstimateParameters(bool is_train) {
  // Default: Empty
}

void DocFirstBase::FinalizeAndClear(bool is_train) {
  DefaultFinalizeAndClear(is_train);
}

double DocFirstBase::Loglikelihood() {
  // Default:
  return 0.0;
}

// User Defined Computations
void DocFirstBase::ClearTokenTopic(int doc, int token, int topic) {
  int word = cur_corpus->doc_content[token];
  DefaultClearTokenTopic(doc, word, topic);
}

void DocFirstBase::SetTokenTopic(int doc, int token, int topic) {
  int word = cur_corpus->doc_content[token];
  DefaultSetTokenTopic(doc, word, topic);
}

void DocFirstBase::LoadModel(const std::string &path) {

}

void DocFirstBase::SaveModel(const std::string &path) {

}

ResultBase* systm::DocFirstBase::GenResult(bool is_train) {
  return nullptr;
}

float systm::DocFirstBase::DocProposal(int doc, int topic, bool is_large_word) {
  return 0.0;
}

}
