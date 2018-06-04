//
// Created by parallels on 4/16/18.
//

#include "model/lda.h"
#include "model/ctm.h"
#include "model/stm.h"
#include "model/tot.h"
#include "corpus/corpus.h"

extern "C" {

systm::Corpus* Corpus_read_from_file(char* path)
{
    systm::Corpus* corpus = new systm::Corpus();
    corpus->ReadFromFile(path);
    for (int d = 0; d < corpus->n_docs; ++d) {
        double time = (rand() % 65536) / 65536;
        corpus->doc_infos[d].emplace_back(time);
    }
    return corpus;
}

systm::LDA* LDA_new(int n_topics, double alpha, double beta) {
    return new systm::LDA(n_topics, alpha, beta);
}

systm::CTM* CTM_new(int n_topics, double mu, double sigma, double beta) {
    return new systm::CTM(n_topics, mu, sigma, beta);
}

systm::STM* slda_new(int n_topics, double alpha, double beta, double sigma2)
{
    return new systm::STM(n_topics, alpha, beta, sigma2);
}

systm::TOT* tot_new(int n_topics, double alpha, double beta) {
    return new systm::TOT(n_topics, alpha, beta);
}

void LDA_train(systm::LDA *lda, systm::Corpus *corpus, int max_iter) {
    lda->Train(*corpus, max_iter);
}

void CTM_train(systm::CTM *ctm, systm::Corpus *corpus, int max_iter) {
    ctm->Train(*corpus, max_iter);
}

void STM_train(systm::STM *slda, systm::Corpus *corpus, int max_iter) {
    slda->Train(*corpus, max_iter);
}

void TOT_train(systm::TOT *tot, systm::Corpus *corpus, int max_iter) {
    tot->Train(*corpus, max_iter);
}

}
