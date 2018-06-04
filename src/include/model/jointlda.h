#pragma once

#include <model/word_first_base.h>
#include <set>
#include <omp.h>

namespace systm
{

class JointLDA : public WordFirstBase {
 public:
  JointLDA(int n_topics, float alpha, float beta, std::string dictionary_path,
           int long_doc_threshold = 300, int mh_step = 2)
      : WordFirstBase(n_topics, alpha, long_doc_threshold, mh_step), beta(beta) {
    
	std::ifstream fin(dictionary_path);
    char read_buffer[1024];
    while (fin.getline(read_buffer, 1024, '\n')) {
      std::stringstream line(read_buffer);

      std::vector<std::string> dic_item;
      std::string word;
      while (line >> word) {
        word_in_dictionary.insert(word);
        dic_item.push_back(word);
      }
      raw_dictionary.emplace_back(std::move(dic_item));
    }
    fin.close();
  }

 protected:
  float beta;
  std::set<std::string> word_in_dictionary;
  std::vector<std::vector<std::string>> raw_dictionary;
  std::vector<std::vector<int>> concept_to_word;
  std::vector<std::vector<int>>& concept_topic_dist = word_topic_dist;
  std::vector<std::vector<int>> word_to_concept;
  std::vector<int> concepts;
  int n_concepts;

  virtual void InitializeOthers(bool is_train) {
    word_to_concept.clear();
    word_to_concept.resize(cur_corpus->n_words);
    concept_to_word.clear();
    concept_to_word.resize(raw_dictionary.size());
    concepts.resize(cur_corpus->n_tokens);
    concept_topic_dist.clear();

    for (int c = 0; c < raw_dictionary.size(); ++c) {
      for (auto& word : raw_dictionary[c]) {
        if (word_to_int.count(word) > 0) {
          int word_id = word_to_int[word];
          if (word_id < cur_corpus->n_words) {
            word_to_concept[word_id].push_back(c);
            concept_to_word[c].push_back(word_id);
          }
        }
      }
      concept_topic_dist.emplace_back(n_topics);
    }

    n_concepts = raw_dictionary.size();
    for (int word = 0; word < cur_corpus->n_words; ++word) {
      if (word_to_concept[word].empty()) {
        word_to_concept[word].push_back(n_concepts);
        concept_to_word.emplace_back();
        concept_topic_dist.emplace_back(n_topics);
        concept_to_word[n_concepts].push_back(word);
        n_concepts++;
      }
      std::vector<int>& concepts = word_to_concept[word];
      int concept_num = concepts.size();
      for (int i = cur_corpus->word_offset[word]; i < cur_corpus->word_offset[word + 1]; ++i) {
        concepts[i] = concepts[rand[0].RandInt(concept_num)];
        concept_topic_dist[concepts[i]][topics[i]]++;
      }
    }
  }

  virtual float WordProposal(int word, int topic) {
    int concept_sum = 0;
    int concept_num = word_to_concept[word].size();
    for (int c : word_to_concept[word]) {
      concept_sum += concept_topic_dist[c][topic];
    }
    return (concept_sum + concept_num * beta) / (topic_dist[topic] + n_concepts * beta);
  }

  virtual void ClearTokenTopic(int word, int token, int topic) {
    int doc = cur_corpus->word_content[token];
    doc_topic_dist[doc].Dec(topic);
    topic_dist[topic]--;
    concept_topic_dist[concepts[token]][topic]--;
  }

  virtual void SetTokenTopic(int word, int token, int topic) {
    int doc = cur_corpus->word_content[token];
    doc_topic_dist[doc].Inc(topic);
    topic_dist[topic]++;
    std::vector<int>& concept_list = word_to_concept[word];
    int concept_num = concept_list.size();
    std::vector<float> psum(concept_num);
    for (int i = 0; i < concept_num; ++i) {
      psum[i] = concept_topic_dist[concept_list[i]][topic] + beta;
      if (i > 0) psum[i] += psum[i - 1];
    }
    float prob = rand[omp_get_thread_num()].RandDouble(psum[concept_num - 1]);
    int p = (lower_bound(psum.begin(), psum.begin() + concept_num, prob) - psum.begin());
    concepts[token] = concept_list[p];
    concept_topic_dist[concepts[token]][topic]++;
  }
};

}
