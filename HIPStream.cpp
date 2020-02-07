// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

#include <numeric>

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

template <class T>
HIPStream<T>::HIPStream(const unsigned int ARRAY_SIZE, const int device_index)
  : array_size{ARRAY_SIZE}, block_cnt(array_size / (TBSIZE * elts_per_lane))
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

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
  hipHostMalloc(&sums, sizeof(T) * block_cnt, hipHostMallocNonCoherent);

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
  hipMalloc(&d_sum, block_cnt * sizeof(T));
  check_error();
}


template <class T>
HIPStream<T>::~HIPStream()
{
  hipFree(sums);

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
  hipFree(d_sum);
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
  // Copy device memory to host
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
}

template <unsigned int elts_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;
  for (auto i = 0u; i != elts_per_lane; ++i) c[gidx + i] = a[gidx + i];
}

template <class T>
void HIPStream<T>::copy()
{
  hipLaunchKernelGGL(copy_kernel<elts_per_lane>, dim3(block_cnt), dim3(TBSIZE),
                     0, nullptr, d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <unsigned int elts_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;
  for (auto i = 0u; i != elts_per_lane; ++i) b[gidx + i] = scalar * c[gidx + i];
}

template <class T>
void HIPStream<T>::mul()
{
  hipLaunchKernelGGL(mul_kernel<elts_per_lane>, dim3(block_cnt), dim3(TBSIZE),
                     0, nullptr, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <unsigned int elts_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void add_kernel(const T * __restrict a, const T * __restrict b,
                           T * __restrict c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;
  for (auto i = 0u; i != elts_per_lane; ++i) {
    c[gidx + i] = a[gidx + i] + b[gidx + i];
  }
}

template <class T>
void HIPStream<T>::add()
{
  hipLaunchKernelGGL(add_kernel<elts_per_lane>, dim3(block_cnt), dim3(TBSIZE),
                     0, nullptr, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <unsigned int elts_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void triad_kernel(T * __restrict a, const T * __restrict b,
                             const T * __restrict c)
{
  const T scalar = startScalar;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;
  for (auto i = 0u; i != elts_per_lane; ++i) {
    a[gidx + i] = b[gidx + i] + scalar * c[gidx + i];
  }
}

template <class T>
void HIPStream<T>::triad()
{
  hipLaunchKernelGGL(triad_kernel<elts_per_lane>, dim3(block_cnt), dim3(TBSIZE),
                     0, nullptr, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <unsigned int elts_per_lane, class T>
__launch_bounds__(TBSIZE)
__global__ void dot_kernel(const T * __restrict a, const T * __restrict b,
                           T * __restrict sum)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x) * elts_per_lane;

  T tmp{0.0};
  for (auto i = 0u; i != elts_per_lane; ++i) tmp += a[gidx + i] * b[gidx + i];

  const auto local_i = threadIdx.x;

  __shared__ T tb_sum[TBSIZE];
  tb_sum[local_i] = tmp;

  #pragma unroll
  for (auto offset = TBSIZE / 2; offset > 0; offset /= 2) {
    if (warpSize < offset) __syncthreads();
    if (local_i >= offset) continue;

    tb_sum[local_i] += tb_sum[local_i + offset];
  }

  if (local_i) return;

  sum[blockIdx.x] = tb_sum[0];
}

template <class T>
T HIPStream<T>::dot()
{
  hipLaunchKernelGGL(dot_kernel<elts_per_lane>, dim3(block_cnt), dim3(TBSIZE),
                     0, nullptr, d_a, d_b, d_sum);
  check_error();

  hipMemcpy(sums, d_sum, block_cnt * sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  return std::accumulate(sums, sums + block_cnt, T{0});
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
