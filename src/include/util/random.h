#pragma once

#include <cmath>
#include <cstdlib>
#include <random>

namespace systm {

class Random {
 public:
  int RandInt(int n) {
    return next() % n;
  }

  float RandDouble(float x = 1.0) {
    return x * float(next()) / ~0U;
  }

  float RandNorm(float mean = 0, float var = 1) {
    float r = randn(gen);
    return mean + r * var;
  }

  Random() : randn(0.0, 1.0) {
    x = next_prime(rand());
    y = next_prime(rand());
    z = next_prime(rand());
  }

 private:
  unsigned x, y, z;
  std::default_random_engine gen;
  std::normal_distribution<float> randn;

  unsigned next_prime(int n) {
    while (true) {
      bool prime = true;
      for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
          prime = false;
          break;
        }
      }
      if (prime) break;
      ++n;
    }
    return n;
  }

  unsigned next() {
    return x = y * x + z;
  }
};

}
