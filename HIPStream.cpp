// Copyright (c) 2014-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

#include <cfloat>

#define TBSIZE 1024

void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <typename... Args, typename F = void (*)(Args...)>
static void hipLaunchKernelWithEvents(F kernel, const dim3& numBlocks,
                           const dim3& dimBlocks, hipStream_t stream,
                           hipEvent_t startEvent, hipEvent_t stopEvent,
                           Args... args)
{
#ifdef __HIP_PLATFORM_NVCC__
  hipEventRecord(startEvent);
  check_error();
  hipLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                   0, stream, args...);
  check_error();
  hipEventRecord(stopEvent);
  check_error();
#else
  hipExtLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                      0, stream, startEvent, stopEvent, 0, args...);
  check_error();
#endif
}

template <typename... Args, typename F = void (*)(Args...)>
static void hipLaunchKernelSynchronous(F kernel, const dim3& numBlocks,
                           const dim3& dimBlocks, hipStream_t stream,
                           bool coherent, Args... args)
{
#ifdef __HIP_PLATFORM_NVCC__
  hipLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                   0, stream, args...);
  check_error();
  hipDeviceSynchronize();
  check_error();
#else
  hipEvent_t temp_event;
  unsigned int flag = coherent ? hipEventReleaseToSystem : 0;
  check_error();
  hipEventCreateWithFlags(&temp_event, flag);
  check_error();
  hipExtLaunchKernelGGL(kernel, numBlocks, dimBlocks,
                      0, stream, 0, temp_event, 0, args...);
  check_error();
  hipEventSynchronize(temp_event);
  check_error();
  hipEventDestroy(temp_event);
  check_error();
#endif
}

template <class T>
HIPStream<T>::HIPStream(const unsigned int ARRAY_SIZE, const bool event_timing,
  const int device_index)
  : array_size{ARRAY_SIZE}, evt_timing(event_timing),
    block_cnt(array_size / (TBSIZE * elts_per_lane * chunks_per_block))
{
  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  std::cerr << "block count " << block_cnt << std::endl;

  // Set device
  int count;
  hipGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  hipSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using HIP device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // Allocate the host array for partial sums for dot kernels
  // TODO would like to use hipHostMallocNonCoherent here, but it appears to
  // be broken with hipExtLaunchKernelGGL(). The data never becomes coherent
  // with the system, even if we device sync or wait on a system scope event
  hipHostMalloc(&sums, sizeof(T) * block_cnt, hipHostMallocCoherent);

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  hipMalloc(&d_a, ARRAY_SIZE * sizeof(T));
  check_error();
  hipMalloc(&d_b, ARRAY_SIZE * sizeof(T));
  check_error();
  hipMalloc(&d_c, ARRAY_SIZE * sizeof(T));
  check_error();

  hipEventCreate(&start_ev);
  check_error();
  hipEventCreate(&stop_ev);
  check_error();
}


template <class T>
HIPStream<T>::~HIPStream()
{
  hipHostFree(sums);
  check_error();
  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
  hipEventDestroy(start_ev);
  check_error();
  hipEventDestroy(stop_ev);
  check_error();
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void HIPStream<T>::init_arrays(T initA, T initB, T initC)
{
  hipLaunchKernelGGL(init_kernel<T>, dim3(array_size/TBSIZE), dim3(TBSIZE), 0,
                     nullptr, d_a, d_b, d_c, initA, initB, initC);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  hipDeviceSynchronize();
  check_error();
  // Copy device memory to host
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void read_kernel(const T * __restrict a, T * __restrict c)
{
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  T tmp{0};
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      tmp += a[gidx + i * dx + j];
    }
  }

  // Prevent side-effect free loop from being optimised away.
  if (tmp == FLT_MIN)
  {
    c[gidx] = tmp;
  }
}

template <class T>
float HIPStream<T>::read()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(read_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_a, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(read_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_a, d_c);
  }
  return kernel_time;
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void write_kernel(T * __restrict c)
{
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      c[gidx + i * dx + j] = startC;
    }
  }
}

template <class T>
float HIPStream<T>::write()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(write_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(write_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_c);
  }
  return kernel_time;
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      c[gidx + i * dx + j] = a[gidx + i * dx + j];
    }
  }
}

template <class T>
float HIPStream<T>::copy()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(copy_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_a, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(copy_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_a, d_c);
  }
  return kernel_time;
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      b[gidx + i * dx + j] = scalar * c[gidx + i * dx + j];
    }
  }
}

template <class T>
float HIPStream<T>::mul()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(mul_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_b, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(mul_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_b, d_c);
  }
  return kernel_time;
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void add_kernel(const T * __restrict a, const T * __restrict b,
                T * __restrict c)
{
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  T temp_a[chunks_per_block * elts_per_lane];
  T temp_b[chunks_per_block * elts_per_lane];

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
        temp_a[i*elts_per_lane + j] = a[gidx + i * dx + j];
    }
  }
  __threadfence();
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
        temp_b[i*elts_per_lane + j] = b[gidx + i * dx + j];
    }
  }
  __threadfence();
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      c[gidx + i * dx + j] = temp_a[i*elts_per_lane + j] + temp_b[i*elts_per_lane + j];
    }
  }
}

template <class T>
float HIPStream<T>::add()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(add_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_a, d_b, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(add_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_a, d_b, d_c);
  }
  return kernel_time;
}

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void triad_kernel(T * __restrict a, const T * __restrict b,
                  const T * __restrict c)
{
  const T scalar = startScalar;
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  T temp_b[chunks_per_block * elts_per_lane];
  T temp_c[chunks_per_block * elts_per_lane];

  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
        temp_b[i*elts_per_lane + j] = b[gidx + i * dx + j];
    }
  }
  __threadfence();
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
        temp_c[i*elts_per_lane + j] = c[gidx + i * dx + j];
    }
  }
  __threadfence();
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      a[gidx + i * dx + j] = temp_b[i*elts_per_lane + j] + scalar * temp_c[i*elts_per_lane + j];
    }
  }
}

template <class T>
float HIPStream<T>::triad()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    hipLaunchKernelWithEvents(triad_kernel<elts_per_lane, chunks_per_block, T>,
                              dim3(block_cnt), dim3(TBSIZE), nullptr, start_ev,
                              stop_ev, d_a, d_b, d_c);
    hipEventSynchronize(stop_ev);
    check_error();
    hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    hipLaunchKernelSynchronous(triad_kernel<elts_per_lane, chunks_per_block, T>,
                               dim3(block_cnt), dim3(TBSIZE), nullptr, false,
                               d_a, d_b, d_c);
  }
  return kernel_time;
}

template<unsigned int n = TBSIZE>
struct Reducer {
  template<typename I>
  __device__
  static
  void reduce(I it) noexcept
  {
    if (n == 1) return;

#if defined(__HIP_PLATFORM_NVCC__)
    constexpr unsigned int warpSize = 32;
#endif
    constexpr bool is_same_warp{n <= warpSize * 2};
    if (static_cast<int>(threadIdx.x) < n / 2)
    {
      it[threadIdx.x] += it[threadIdx.x + n / 2];
    }
    is_same_warp ? __threadfence_block() : __syncthreads();

    Reducer<n / 2>::reduce(it);
  }
};

template<>
struct Reducer<1u> {
  template<typename I>
  __device__
  static
  void reduce(I) noexcept
  {}
};

template <unsigned int elts_per_lane, unsigned int chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void dot_kernel(const T * __restrict a, const T * __restrict b,
                T * __restrict sum)
{
  const auto dx = gridDim.x * blockDim.x * elts_per_lane;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  T tmp{0};
  for (auto i = 0u; i != chunks_per_block; ++i)
  {
    for (auto j = 0u; j != elts_per_lane; ++j)
    {
      tmp += a[gidx + i * dx + j] * b[gidx + i * dx + j];
    }
  }

  __shared__ T tb_sum[TBSIZE];
  tb_sum[threadIdx.x] = tmp;

  __syncthreads();

  Reducer<>::reduce(tb_sum);

  if (threadIdx.x)
  {
    return;
  }
  sum[blockIdx.x] = tb_sum[0];
}

template <class T>
T HIPStream<T>::dot()
{
  hipLaunchKernelSynchronous(dot_kernel<elts_per_lane, chunks_per_block, T>,
                             dim3(block_cnt), dim3(TBSIZE), nullptr, true,
                             d_a, d_b, sums);

  T sum{0};
  for (auto i = 0u; i != block_cnt; ++i)
  {
    sum += sums[i];
  }

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  hipGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  hipSetDevice(device);
  check_error();
  int driver;
  hipDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class HIPStream<float>;
template class HIPStream<double>;
