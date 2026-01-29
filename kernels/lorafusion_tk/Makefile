# Makefile for LoRAFusion ThunderKittens kernel

# ThunderKittens path
TK_PATH ?= /local/scratch/muchane_686487/ThunderKittens

# Python configuration
PYTHON := python3
PY_EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Output
OUT := lorafusion_tk$(PY_EXT)

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

# Compiler flags
NVCCFLAGS := -DTORCH_COMPILE \
             -DKITTENS_HOPPER \
             -arch=sm_90a \
             -std=c++20 \
             -O3 \
             --use_fast_math \
             --expt-relaxed-constexpr \
             -Xcompiler=-fPIC \
             --shared

# Include paths
PY_INCLUDES := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as ext; print(' '.join(['-I'+p for p in ext.include_paths()]))")
TK_INCLUDES := -I$(TK_PATH)/include
INCLUDES := $(TK_INCLUDES) $(PY_INCLUDES)

# Library paths
PY_LDFLAGS := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as ext; print(' '.join(['-L'+p for p in ext.library_paths()]))")
LDFLAGS := $(PY_LDFLAGS) -lc10 -ltorch -ltorch_cpu -ltorch_cuda

.PHONY: all clean test

all: $(OUT)

$(OUT): lorafusion_fused.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

test: $(OUT)
	$(PYTHON) test_lorafusion_tk.py

clean:
	rm -f $(OUT) *.o
