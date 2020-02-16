
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"
#include "hip/hip_runtime.h"
#ifndef __HIP_PLATFORM_NVCC__
#include "hip/hip_ext.h"
#endif

#define IMPLEMENTATION_STRING "HIP"

template <class T>
class HIPStream : public Stream<T>
{
#ifdef __HIP_PLATFORM_NVCC__
  static constexpr unsigned int elts_per_lane{1};
#else
  static constexpr unsigned int best_size{sizeof(unsigned int) * 1};
  static constexpr unsigned int elts_per_lane{
    (best_size < sizeof(T)) ? 1 : (best_size / sizeof(T))};
  static constexpr unsigned int chunks_per_block{4};
#endif
  protected:
    // Size of arrays
    const unsigned int array_size;
    const unsigned int block_cnt;
    unsigned int dot_block_cnt;
    const bool evt_timing;
    hipEvent_t start_ev;
    hipEvent_t stop_ev;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;

  public:
    HIPStream(const unsigned int, const bool, const int);
    ~HIPStream();

    virtual float read() override;
    virtual float write() override;
    virtual float copy() override;
    virtual float add() override;
    virtual float mul() override;
    virtual float triad() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
