#include <result/result_base.h>
#include <model/lda.h>

namespace systm {

class LDAResult : public ResultBase {
 public:
  LDAResult(LDA* lda_model);
  std::vector<int> GetTopicDist(int doc);
  std::vector<int> GetTopTopics(int doc, int top_n);
  std::vector<std::string> GetTopWord(int topic, int top_n);


 private:
  std::vector<std::string> word_list;
  std::vector<std::vector<std::pair<int, int>>> doc_topics;
  std::vector<std::vector<std::pair<int, int>>> topic_words;
  std::vector<int> doc_size, topic_size;
  float alpha, beta;
  int n_topics;
};

}