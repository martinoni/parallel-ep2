OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11

CC_OMPI=-openmpi
NVCC=nvcc


.PHONY: all
all: $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_seq

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(NVCC) -o $(OUTPUT)_cuda $(OUTPUT)_cuda.cu

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(CC) -o $(OUTPUT)_ompi $(CC_OPT) $(CC_OMPI) $(OUTPUT)_ompi.c

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

.PHONY: clean
clean:
	rm $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_seq *$(IMAGE)
