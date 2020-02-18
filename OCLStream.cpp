
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"

// Cache list of devices
bool cached = false;
std::vector<cl::Device> devices;
void getDeviceList(void);

std::string kernels{R"CLC(

  constant TYPE scalar = startScalar;

  kernel void init(
    global TYPE * restrict a,
    global TYPE * restrict b,
    global TYPE * restrict c,
    TYPE initA, TYPE initB, TYPE initC)
  {
    const size_t i = get_global_id(0);
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }

  kernel void read(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    TYPE local_temp = 0.;
    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        local_temp += a[gidx + i * dx + j];
      }
    }
    if (local_temp == 126789.)
      c[gidx] = local_temp;
  }

  kernel void write(
    global TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        c[gidx + i * dx + j] = STARTC;
      }
    }
  }

  kernel void copy(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        c[gidx + i * dx + j] = a[gidx + i * dx + j];
      }
    }
  }

  kernel void mul(
    global TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        b[gidx + i * dx + j] = scalar * c[gidx + i * dx + j];
      }
    }
  }

  kernel void add(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        c[gidx + i * dx + j] = a[gidx + i * dx + j] + b[gidx + i * dx + j];
      }
    }
  }

  kernel void triad(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;

    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        a[gidx + i * dx + j] = b[gidx + i * dx + j] + scalar * c[gidx + i * dx + j];
      }
    }
  }

  kernel void stream_dot(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict sum,
    local TYPE * restrict wg_sum,
    uint array_size)
  {
    const size_t dx = get_num_groups(0) * get_local_size(0) * ELTS_PER_LANE;
    const size_t gidx = get_global_id(0) * ELTS_PER_LANE;
    const size_t local_i = get_local_id(0);

    TYPE temp = 0.;
    for (int i = 0; i != CHUNKS_PER_WG; ++i) {
      for (int j = 0; j != ELTS_PER_LANE; ++j) {
        temp += a[gidx + i * dx + j] * b[gidx + i * dx + j];
      }
    }

    wg_sum[local_i] = temp;

    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_i >= offset) continue;

      wg_sum[local_i] += wg_sum[local_i + offset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i)
      return;

    sum[get_group_id(0)] = wg_sum[0];
  }

)CLC"};

static unsigned int getDeviceVendor(const int device)
{
  if (!cached)
    getDeviceList();

  unsigned int vendor_id;
  cl_device_info info = CL_DEVICE_VENDOR_ID;

  if (device < devices.size())
    devices[device].getInfo(info, &vendor_id);
  else
    throw std::runtime_error("Error asking for name for non-existant device");

  return vendor_id;
}

template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const bool event_timing,
  const int device_index)
  : array_size(ARRAY_SIZE), evt_timing(event_timing)
{
  if (!cached)
    getDeviceList();

  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];

  unsigned int size_of_good_load = sizeof(unsigned int);
  if (getDeviceVendor(device_index) == 0x1002) // AMD
  {
      size_of_good_load *= 4;
      chunks_per_wg = 1;
  }
  else
  {
      chunks_per_wg = 8;
  }
  elts_per_lane = size_of_good_load / sizeof(T);
  if (elts_per_lane == 0)
    elts_per_lane++;

  // Determine sensible dot kernel NDRange configuration
  if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU)
    dot_wgsize     = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>() * 2;
  else
    dot_wgsize     = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  dot_num_groups = array_size/(elts_per_lane*chunks_per_wg*dot_wgsize);

  // Print out device information
  std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  context = cl::Context(device);
  if (evt_timing)
    queue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);
  else
    queue = cl::CommandQueue(context);

  // Create program
  cl::Program program(context, kernels);
  std::ostringstream args;
  args << "-DstartScalar=" << startScalar << " ";
  args << "-DELTS_PER_LANE=" << elts_per_lane << " ";
  args << "-DCHUNKS_PER_WG=" << chunks_per_wg << " ";
  args << "-DSTARTC=" << startC << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      throw std::runtime_error("Device does not support double precision, please use --float");
    try
    {
      program.build(args.str().c_str());
    }
    catch (cl::Error& err)
    {
      if (err.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << std::endl;
        throw err;
      }
    }
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float";
    program.build(args.str().c_str());
  }

  // Create kernels
  init_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, T, T, T>(program, "init");
  read_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "read");
  write_kernel = new cl::KernelFunctor<cl::Buffer>(program, "write");
  copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");
  dot_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int>(program, "stream_dot");

  // Check buffers fit on the device
  cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  if (maxbuffer < sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device cannot allocate a buffer big enough");
  if (totalmem < 3*sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create buffers
  d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_sum = cl::Buffer(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(T) * dot_num_groups);
}

template <class T>
OCLStream<T>::~OCLStream()
{
  delete init_kernel;
  delete read_kernel;
  delete write_kernel;
  delete copy_kernel;
  delete mul_kernel;
  delete add_kernel;
  delete triad_kernel;

  devices.clear();
}

template <class T>
float OCLStream<T>::read()
{
  float kernel_time = 0.;
  cl::Event evt = (*read_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_a, d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
float OCLStream<T>::write()
{
  float kernel_time = 0.;
  cl::Event evt = (*write_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
float OCLStream<T>::copy()
{
  float kernel_time = 0.;
  cl::Event evt = (*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_a, d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
float OCLStream<T>::mul()
{
  float kernel_time = 0.;
  cl::Event evt = (*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_b, d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
float OCLStream<T>::add()
{
  float kernel_time = 0.;
  cl::Event evt = (*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_a, d_b, d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
float OCLStream<T>::triad()
{
  float kernel_time = 0.;
  cl::Event evt = (*triad_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size/(elts_per_lane*chunks_per_wg))),
    d_a, d_b, d_c
  );
  evt.wait();
  if (evt_timing)
  {
    kernel_time = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                    evt.getProfilingInfo<CL_PROFILING_COMMAND_START>())
                    / 1000000.;
  }
  return kernel_time;
}

template <class T>
T OCLStream<T>::dot()
{
  cl::Event evt = (*dot_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(dot_num_groups * dot_wgsize), cl::NDRange(dot_wgsize)),
    d_a, d_b, d_sum, cl::Local(sizeof(T) * dot_wgsize), array_size
  );
  T *local_sums = static_cast<T*>(queue.enqueueMapBuffer(d_sum, CL_BLOCKING,
                                  0, 0, dot_num_groups*sizeof(T)));

  T sum = 0.0;
  for (int i = 0; i < dot_num_groups; i++)
    sum += local_sums[i];

  return sum;
}

template <class T>
void OCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  cl::Event evt = (*init_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c, initA, initB, initC
  );
  evt.wait();
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::copy(queue, d_a, a.begin(), a.end());
  cl::copy(queue, d_b, b.begin(), b.end());
  cl::copy(queue, d_c, c.begin(), c.end());
}

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }


}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;
  cl_device_info info = CL_DEVICE_NAME;

  if (device < devices.size())
  {
    devices[device].getInfo(info, &name);
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;

}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


template class OCLStream<float>;
template class OCLStream<double>;
