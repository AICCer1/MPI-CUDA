# Required variables  
ARCH = sm_60  
HOST_COMP = mpicc  

# Compiler and flags  
NVCC = nvcc  
NVCC_FLAGS = -O3 -std=c++11 -arch=$(ARCH)  
MPI_FLAGS = -I/opt/ibm/spectrum_mpi/include  
MPI_LIBS = -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm -lmpiprofilesupport  
EXTRA_FLAGS = -Xcompiler -fopenmp  

# Output executable name  
TARGET = mpigpu-1  

# Source file  
SRC = mpigpu-1.cu  

# Default target  
all: $(TARGET)  

# Compilation rule  
$(TARGET): $(SRC)  
	$(NVCC) $(NVCC_FLAGS) $(MPI_FLAGS) $(EXTRA_FLAGS) -o $@ $< $(MPI_LIBS)  

# Clean rule  
clean:  
	rm -f $(TARGET)