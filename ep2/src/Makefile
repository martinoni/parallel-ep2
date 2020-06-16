OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11

MCC=mpicc
NVCC=nvcc


.PHONY: all
all: $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_seq

# $(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
# 	$(NVCC) --link -o $(OUTPUT)_cuda $(OUTPUT)_cuda.cu

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(MCC) -o $(OUTPUT)_ompi $(OUTPUT)_ompi.c -lm

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

.PHONY: clean
clean:
	rm $(OUTPUT)_cuda $(OUTPUT)_ompi $(OUTPUT)_seq *$(IMAGE)
 