#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mutex>
#include <cmath>

namespace systm {

enum class ValueType {
  INTEGER,
  DOUBLE,
};

struct Value {
  ValueType type;
  union {
    int int_value;
    double double_value;
  };
  Value(int val) {
    type = ValueType::INTEGER;
    int_value = val;
  }
  Value(double val) {
    type = ValueType::DOUBLE;
    double_value = val;
  }
};

struct Corpus {
  void ReadFromFile(const std::string &doc_path);

  void AddDoc(const std::string &doc, const std::vector<int> &word,
              const std::vector<Value> &doc_info = {});

  void Indexing();

  std::string GetWordById(int id) const;
  std::string GetDocById(int id) const;

  // get the ID of a document
  int GetDocId(const std::string &doc);

  // get the ID of a word
  int GetWordId(const std::string &word);

  // Number of documents
  int n_docs = 0;
  // Number of vocabulary
  int n_words = 0;
  // Number of tokens
  int n_tokens = 0;

  // Document contents
  std::vector<std::vector<int>> doc_content_raw;
  // Document infos
  std::vector<std::vector<Value>> doc_infos;

  std::vector<unsigned> doc_offset, word_offset;
  std::vector<unsigned> doc_content, word_content;
  std::vector<unsigned> doc_to_word, word_to_doc;

  // Vocabulary map
  std::unordered_map<std::string, int> word_to_int;
  // Document map
  std::unordered_map<std::string, int> doc_to_int;
  // Vocabulary list
  std::vector<std::string> word_list;
  // Document list
  std::vector<std::string> doc_list;
};

}  // namespace liblda
