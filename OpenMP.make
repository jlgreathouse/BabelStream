
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  CLANG CRAY GNU INTEL XL PGI NEC

endef
$(info $(compiler_help))
COMPILER=GNU
endif

ifndef TARGET
define target_help
Set TARGET to change device (defaulting to CPU).
Available targets are:
  CPU NVIDIA AMD

endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ("$(TARGET)","AMD")
# to override AOMP_GPU: $ export AOMP_GPU=gfx906
INSTALLED_GPU  = $(shell /opt/rocm/bin/rocm_agent_enumerator | grep gfx9 | grep -v gfx000 | head -n 1)
AOMP_GPU ?= $(INSTALLED_GPU)
endif

COMPILER_GNU = g++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
COMPILER_CLANG = clang++
COMPILER_XL = xlc++
COMPILER_PGI = pgc++
COMPILER_NEC = /opt/nec/ve/bin/nc++
CXX = $(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=c++11 -mcpu=native
FLAGS_INTEL = -O3 -std=c++11 -xHOST -qopt-streaming-stores=always
FLAGS_CRAY = -O3 -hstd=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_XL = -O5 -qarch=auto -qtune=auto -std=c++11
FLAGS_PGI = -O3 -std=c++11
FLAGS_NEC = -O4 -finline -std=c++11
CXXFLAGS = $(FLAGS_$(COMPILER))

# OpenMP flags for CPUs
OMP_GNU_CPU   = -fopenmp
OMP_INTEL_CPU = -qopenmp
OMP_CRAY_CPU  = -homp
OMP_CLANG_CPU = -fopenmp=libomp
OMP_XL_CPU = -qsmp=omp -qthreaded
OMP_PGI_CPU = -mp
OMP_NEC_CPU = -fopenmp

# OpenMP flags for NVIDIA
OMP_CRAY_NVIDIA  = -DOMP_TARGET_GPU
OMP_CLANG_NVIDIA = -DOMP_TARGET_GPU -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda

OMP_CLANG_AMD = -DOMP_TARGET_GPU -O3 -target x86_64-pc-linux-gnu -fopenmp \
                -fopenmp-targets=amdgcn-amd-amdhsa \
                -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU)

ifndef OMP_$(COMPILER)_$(TARGET)
$(error Targeting $(TARGET) with $(COMPILER) not supported)
endif

OMP = $(OMP_$(COMPILER)_$(TARGET))

omp-stream: main.cpp OMPStream.cpp
	$(CXX) $(CXXFLAGS) -DOMP $^ $(OMP) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f omp-stream
