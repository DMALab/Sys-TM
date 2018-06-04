#include <util/atomic_integer.h>
#include "corpus/corpus.h"
#include "util/random.h"

namespace systm {

void Corpus::ReadFromFile(const std::string &doc_path) {
  std::ifstream fin(doc_path);
  std::string doc;

  int read_buffer_size = 1U << 20;
  char read_buffer[read_buffer_size];
  std::vector<int> words(256);

  // read documents
  while (fin >> doc) {
    // read words
    fin.getline(read_buffer, read_buffer_size, '\n');
    std::stringstream doc_s(read_buffer);

    words.clear();

    std::string word;
    while (doc_s >> word) {
      int word_id = GetWordId(word);
      words.push_back(word_id);
    }
    AddDoc(doc, words);
  }
  fin.close();

  Indexing();
}

void Corpus::AddDoc(const std::string &doc, const std::vector<int> &words,
                    const std::vector<Value> &doc_info) {
  if (doc_to_int.count(doc) == 0) {
    GetDocId(doc);
    doc_content_raw.push_back(words);
    doc_infos.push_back(doc_info);
    n_tokens += words.size();
  }
}

void Corpus::Indexing() {
  doc_offset.clear();
  doc_offset.resize(n_docs + 1);
  word_offset.clear();
  word_offset.resize(n_words + 1);
  doc_content.resize(n_tokens);
  word_content.resize(n_tokens);
  word_to_doc.resize(n_tokens);
  doc_to_word.resize(n_tokens);

  for (int doc = 0; doc < n_docs; ++doc) {
    doc_offset[doc + 1] = doc_offset[doc] + doc_content_raw[doc].size();
  }

  std::vector<AtomicInt> word_count;
  word_count.resize(n_words);
#pragma omp parallel for schedule(dynamic)
  for (int doc = 0; doc < n_docs; ++doc) {
    for (auto w : doc_content_raw[doc]) {
      word_count[w]++;
    }
  }
  for (int word = 0; word < n_words; ++word) {
    word_offset[word + 1] = word_offset[word] + word_count[word];
  }

  word_count.clear();
  word_count.resize(n_words);
#pragma omp parallel for schedule(dynamic)
  for (int doc = 0; doc < n_docs; ++doc) {
    for (int i = doc_offset[doc]; i < doc_offset[doc + 1]; ++i) {
      doc_content[i] = doc_content_raw[doc][i - doc_offset[doc]];
      int word = doc_content[i];
      int word_pos = word_offset[word] + word_count[word].fetch_add(1);
      word_content[word_pos] = doc;
      word_to_doc[word_pos] = i;
      doc_to_word[i] = word_pos;
    }
  }
}

int Corpus::GetDocId(const std::string &doc) {
  if (doc_to_int.count(doc) == 0) {
    doc_list.push_back(doc);
    doc_to_int[doc] = n_docs;
    return n_docs++;
  }
  else {
    return doc_to_int[doc];
  }
}

int Corpus::GetWordId(const std::string &word) {
  if (word_to_int.count(word) == 0) {
    word_list.push_back(word);
    word_to_int[word] = n_words;
    return n_words++;
  }
  else {
    return word_to_int[word];
  }
}

}  // namespace liblda