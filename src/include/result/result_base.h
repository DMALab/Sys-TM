#pragma once

namespace systm {

class LDAResult;
class TOTResult;
class STMResult;
class CTMResult;

class ResultBase {
 public:
  LDAResult* GetLDAResult() { return reinterpret_cast<LDAResult*>(this); }
  TOTResult* GetTOTResult() { return reinterpret_cast<TOTResult*>(this); }
  STMResult* GetSTMResult() { return reinterpret_cast<STMResult*>(this); }
  CTMResult* GetCTMResult() { return reinterpret_cast<CTMResult*>(this); }
};

}