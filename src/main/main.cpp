#include <iostream>
#include <fstream>
#include "corpus/corpus.h"
#include "model/lda.h"
#include "model/stm.h"
#include "model/ctm.h"
#include "model/tot.h"


namespace systm {

Corpus &get_corpus(std::string path) {
  srand(19931214);
  Corpus* corpus = new Corpus();
  corpus->ReadFromFile(path);
  for (int d = 0; d < corpus->n_docs; ++d) {
    double time = (rand() % 65536) / 65536.0;
	if (time < 0.5) time = 0; else time = 1;
    corpus->doc_infos[d].emplace_back(time);
  }
  return *corpus;
}

void test_lda(Corpus &corpus, int n_topics, int max_iter, int threashold) {
  LDA lda(n_topics, 50.0 / n_topics, 0.01, threashold);
  std::cout << "LDA:  N_TOPICS=" << n_topics << "  LONG_DOC>>" << threashold << std::endl;
  lda.Train(corpus, max_iter);
}

void test_tot(Corpus &corpus, int n_topics, int max_iter, int threashold) {
  TOT tot(n_topics, 50.0 / n_topics, 0.01, threashold);
  std::cout << "TOT: N_TOPICS=" << n_topics << " LARGE_WORD>" << threashold << std::endl;
  tot.Train(corpus, max_iter);
}

void test_slda(Corpus &corpus, int n_topics, int max_iter, int threashold) {
  STM slda(n_topics, 50.0 / n_topics, 0.01, 1.0, threashold);
  std::cout << "SLDA: N_TOPICS=" << n_topics << " LARGE_WORD>" << threashold << std::endl;
  slda.Train(corpus, max_iter);
}

void test_ctm(Corpus &corpus, int n_topics, int max_iter,int threashold) {
  CTM ctm(n_topics, 0.0, 0.1, 0.01, threashold);
  std::cout << "CTM: N_TOPICS=" << n_topics << " LARGE_WORD>" << threashold << std::endl;
  ctm.Train(corpus, max_iter);
}

}

int main() {
  systm::Corpus &nips = systm::get_corpus("/home/parallels/Documents/liblda/nips.train");
  systm::test_lda(nips, 100, 5, 200);
  systm::test_tot(nips, 100, 5, 10000);
  systm::test_slda(nips, 100, 5, 10000);
  systm::test_ctm(nips, 100, 5, 10000);

  return 0;
}
