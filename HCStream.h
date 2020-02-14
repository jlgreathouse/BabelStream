
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"
#include "hc.hpp"

#define IMPLEMENTATION_STRING "HC"

template <class T>
class HCStream : public Stream<T>
{
  static constexpr unsigned int sizeof_best_size{sizeof(unsigned int) * 4};
  static constexpr unsigned int elts_per_lane{sizeof_best_size / sizeof(T)};
protected:
  // Size of arrays
  const unsigned int array_size;
  const unsigned int lane_cnt;

  const bool evt_timing;

  // Device side pointers to arrays
  hc::array<T,1> d_a;
  hc::array<T,1> d_b;
  hc::array<T,1> d_c;

public:
  HCStream(const unsigned int, const bool, const int);
  ~HCStream();

  virtual float read() override;
  virtual float write() override;
  virtual float copy() override;
  virtual float add() override;
  virtual float mul() override;
  virtual float triad() override;
  virtual T dot() override;
  T dot_impl();

  virtual void init_arrays(T initA, T initB, T initC) override;
  virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
