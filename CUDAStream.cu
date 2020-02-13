
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUDAStream.h"

#define TBSIZE 1024

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
CUDAStream<T>::CUDAStream(const unsigned int ARRAY_SIZE, const bool event_timing,
  const int device_index)
  : array_size(ARRAY_SIZE), evt_timing(event_timing),
    block_cnt(ARRAY_SIZE / TBSIZE),
    dot_block_cnt(256)
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
  cudaGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  cudaSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // Allocate the host array for partial sums for dot kernels
  cudaHostAlloc(&sums, sizeof(T) * dot_block_cnt, 0);

  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  cudaMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();

  cudaEventCreate(&start_ev);
  check_error();
  cudaEventCreate(&stop_ev);
  check_error();
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  cudaFreeHost(sums);
  check_error();
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaEventDestroy(start_ev);
  check_error();
  cudaEventDestroy(stop_ev);
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
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void read_kernel(const T * a, T * c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  T local_temp = 0.;
  local_temp += a[gidx];
  if (local_temp == 126789.)
      c[gidx] += local_temp;
}

template <class T>
float CUDAStream<T>::read()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  read_kernel<<<block_cnt, TBSIZE>>>(d_a, d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void write_kernel(T * __restrict c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  c[gidx] = 0.;
}

template <class T>
float CUDAStream<T>::write()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  write_kernel<<<block_cnt, TBSIZE>>>(d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  c[gidx] = a[gidx];
}

template <class T>
float CUDAStream<T>::copy()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  copy_kernel<<<block_cnt, TBSIZE>>>(d_a, d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  b[gidx] = scalar * c[gidx];
}

template <class T>
float CUDAStream<T>::mul()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  mul_kernel<<<block_cnt, TBSIZE>>>(d_b, d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void add_kernel(const T * __restrict a, const T * __restrict b,
                           T * __restrict c)
{
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  c[gidx] = a[gidx] + b[gidx];
}

template <class T>
float CUDAStream<T>::add()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  add_kernel<<<block_cnt, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void triad_kernel(T * __restrict a, const T * __restrict b,
                             const T * __restrict c)
{
  const T scalar = startScalar;
  const auto gidx = (blockDim.x * blockIdx.x + threadIdx.x);
  a[gidx] = b[gidx] + scalar * c[gidx];
}

template <class T>
float CUDAStream<T>::triad()
{
  float kernel_time = 0.;
  if (evt_timing)
  {
    cudaEventRecord(start_ev);
    check_error();
  }
  triad_kernel<<<block_cnt, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  if (evt_timing)
  {
    cudaEventRecord(stop_ev);
    check_error();
    cudaEventSynchronize(stop_ev);
    check_error();
    cudaEventElapsedTime(&kernel_time, start_ev, stop_ev);
    check_error();
  }
  else
  {
    cudaDeviceSynchronize();
    check_error();
  }
  return kernel_time;
}

template <typename T>
__launch_bounds__(TBSIZE)
__global__ void dot_kernel(const T * __restrict a, const T * __restrict b,
                           T * __restrict sum, const unsigned int total_blocks)
{
  __shared__ T tb_sum[TBSIZE];
  const auto local_i = threadIdx.x;
  T tmp{0.0};

  for (unsigned int vblock = blockIdx.x; vblock < total_blocks; vblock += gridDim.x)
  {
    const auto gidx = (blockDim.x * vblock + threadIdx.x);
    tmp += a[gidx] * b[gidx];
  }

  tb_sum[local_i] = tmp;

  #pragma unroll
  for (auto offset = TBSIZE / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (local_i >= offset) continue;

    tb_sum[local_i] += tb_sum[local_i + offset];
  }

  if (local_i) return;

  sum[blockIdx.x] = tb_sum[0];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<dot_block_cnt, TBSIZE>>>(d_a, d_b, sums, block_cnt);
  check_error();
  cudaDeviceSynchronize();
  check_error();

  T sum = 0.0;
  for (int i = 0; i < dot_block_cnt; i++)
    sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
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
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
