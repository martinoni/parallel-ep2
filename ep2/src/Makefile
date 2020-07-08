OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11
CC_OPT2=-std=gnu11


MCC=mpicc
NVCC=nvcc
CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_ompi $(OUTPUT)_cuda $(OUTPUT)_seq

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT2) $(CC_OMP) $(OUTPUT)_omp.c

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT2) $(CC_PTH) $(OUTPUT)_pth.c

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(NVCC) --link -o $(OUTPUT)_cuda $(OUTPUT)_cuda.cu

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(MCC) -o $(OUTPUT)_ompi $(OUTPUT)_ompi.c -lm

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT2) $(OUTPUT)_seq.c

.PHONY: clean
clean:
	rm $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_seq *$(IMAGE)
 