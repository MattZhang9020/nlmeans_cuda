NVCC = nvcc
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -diag-suppress 550

CXX = g++
CXXFLAGS = -std=c++11 -O3

LDFLAGS = -lm

HEADERS = ./include/stb_image.h ./include/stb_image_write.h

TARGETS = bin/nlmeans_seq bin/nlmeans_cuda_1D_uncoaleased bin/nlmeans_cuda_1D_coaleased bin/nlmeans_cuda_2D bin/nlmeans_cuda_shared bin/nlmeans_cuda_stream bin/nlmeans_cuda_shared_stream

.PHONY: all clean

all: bin $(TARGETS)

bin:
	mkdir -p bin

bin/nlmeans_seq: ./src/nlmeans_seq.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_1D_uncoaleased: ./src/nlmeans_cuda_1D_uncoaleased.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_1D_coaleased: ./src/nlmeans_cuda_1D_coaleased.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_2D: ./src/nlmeans_cuda_2D.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_shared: ./src/nlmeans_cuda_shared.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_stream: ./src/nlmeans_cuda_stream.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

bin/nlmeans_cuda_shared_stream: ./src/nlmeans_cuda_shared_stream.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o