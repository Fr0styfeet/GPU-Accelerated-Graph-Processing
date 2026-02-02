# Compiler
NVCC = nvcc

# Include directories
INCLUDES = -Iinclude

# Source files
SRC = src/main.cu src/bellmanGpu.cu src/bellmanCpu.cu src/deltaStep.cu

# Output executable
OUT = build/graphApp.exe

# Compile flags
CFLAGS = -O3 --use_fast_math -std=c++17 -arch=sm_86

# Default target
all: $(OUT)

# Build executable
$(OUT): $(SRC)
	@if not exist build mkdir build
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(OUT)

# Clean build
clean:
	if exist build rmdir /s /q build
