# Makefile for LoRAFusion ThunderKittens kernel
# Uses the project's uv environment for Python/PyTorch paths

# ThunderKittens path
TK_PATH := /local/scratch/muchane_686487/ThunderKittens

# Project root (for uv)
PROJECT_ROOT := /local/scratch/muchane_686487/weightshared_torchtitan

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

# Python/PyTorch paths (hardcoded from uv run python output)
PYTHON_VERSION := 3.13
PYTHON_INCLUDES := -I/home/muchane/.local/share/uv/python/cpython-3.13.1-linux-x86_64-gnu/include/python3.13
PYTHON_LIBDIR := -L/home/muchane/.local/share/uv/python/cpython-3.13.1-linux-x86_64-gnu/lib
PYBIND_INCLUDES := -I$(PROJECT_ROOT)/.venv/lib/python3.13/site-packages/pybind11/include
PYTORCH_INCLUDES := -I$(PROJECT_ROOT)/.venv/lib/python3.13/site-packages/torch/include \
                    -I$(PROJECT_ROOT)/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include
PYTORCH_LIBDIR := -L$(PROJECT_ROOT)/.venv/lib/python3.13/site-packages/torch/lib

# Output filename
OUT := lorafusion_tk.cpython-313-x86_64-linux-gnu.so

# Compiler flags (based on TK common.mk)
NVCCFLAGS := -std=c++20 -O3 --use_fast_math
NVCCFLAGS += -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
NVCCFLAGS += -DNDEBUG -lineinfo

# Shared library flags
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -diag-suppress 3189
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
NVCCFLAGS += -DTORCH_EXTENSION_NAME=lorafusion_tk

# ThunderKittens includes
NVCCFLAGS += -I$(TK_PATH)/include -I$(TK_PATH)/prototype

# Python/PyTorch includes and libs
NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTHON_LIBDIR)
NVCCFLAGS += $(PYBIND_INCLUDES)
NVCCFLAGS += $(PYTORCH_INCLUDES) $(PYTORCH_LIBDIR)
NVCCFLAGS += -lpython$(PYTHON_VERSION)
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

# Enable TK TORCH_COMPILE mode
NVCCFLAGS += -DTORCH_COMPILE

# H100/H200 (Hopper) architecture
NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a

.PHONY: all clean test run

all: $(OUT)

$(OUT): lorafusion_fused.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

test: $(OUT)
	cd $(PROJECT_ROOT) && uv run python kernels/lorafusion_tk/test_lorafusion_tk.py --check-correctness

run: $(OUT)
	cd $(PROJECT_ROOT) && uv run python kernels/lorafusion_tk/test_lorafusion_tk.py

clean:
	rm -f $(OUT) *.ptx *.o
