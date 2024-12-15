#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define SIGMA 2
#define BZ 32
#define SEARCH_WINDOW 11
#define PATCH_SIZE 5
#define NUM_STREAMS 16
#define HALO (SEARCH_WINDOW / 2 + PATCH_SIZE / 2)
#define SHARED_SIZE (BZ + 2 * HALO)

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

using namespace std;

int width, height, channels;

void read_image(const char* filename, vector<float>& noisy) {
    unsigned char* image_data = stbi_load(filename, &width, &height, &channels, 3);

    channels = 3;

    if (!image_data) {
        cerr << "Error: Cannot open input file or unsupported format!" << endl;
        exit(EXIT_FAILURE);
    }

    noisy.reserve(width * height * channels);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                noisy[idx] = image_data[idx] / 255.0f;
            }
        }
    }

    stbi_image_free(image_data);
}

void save_image(const char* filename, const vector<float>& data) {
    vector<unsigned char> image_data(width * height * channels);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                image_data[idx] = static_cast<unsigned char>(data[idx] * 255.0f);
            }
        }
    }

    string file_ext = string(filename).substr(string(filename).find_last_of(".") + 1);
    if (file_ext == "jpg" || file_ext == "jpeg") {
        stbi_write_jpg(filename, width, height, channels, image_data.data(), 100);
    } else if (file_ext == "png") {
        stbi_write_png(filename, width, height, channels, image_data.data(), width * channels);
    } else {
        cerr << "Error: Unsupported output format! Use .jpg or .png" << endl;
        exit(EXIT_FAILURE);
    }
}

__device__ float calculate_patch_distance_shared(const float shared_mem[][SHARED_SIZE][3], int x1, int y1, int x2, int y2, int patch_radius, int channels, int halo, int width, int height) {
    float distance = 0.0f;

    for (int py = -patch_radius; py <= patch_radius; ++py) {
        for (int px = -patch_radius; px <= patch_radius; ++px) {
            int global_yy1 = blockIdx.y * blockDim.y + y1 + py;
            int global_xx1 = blockIdx.x * blockDim.x + x1 + px;

            int global_yy2 = blockIdx.y * blockDim.y + y2 + py;
            int global_xx2 = blockIdx.x * blockDim.x + x2 + px;

            if (global_yy1 >= 0 && global_yy1 < height && global_xx1 >= 0 && global_xx1 < width && global_yy2 >= 0 && global_yy2 < height && global_xx2 >= 0 && global_xx2 < width) {
                int yy1 = y1 + py + halo, xx1 = x1 + px + halo;
                int yy2 = y2 + py + halo, xx2 = x2 + px + halo;

                for (int c = 0; c < channels; ++c) {
                    float diff = shared_mem[yy1][xx1][c] - shared_mem[yy2][xx2][c];
                    distance += diff * diff;
                }
            }
        }
    }

    float result = sqrtf(distance) / ((patch_radius * 2 + 1) * (patch_radius * 2 + 1) * channels);

    return result;
}

__global__ void nlmeans_kernel(const float* d_noisy, float* d_denoised, int width, int height, int channels, int patch_radius, int search_radius, int start_row, int end_row) {
    __shared__ float shared_mem[SHARED_SIZE][SHARED_SIZE][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = start_row + blockIdx.y * blockDim.y + threadIdx.y;

    if (global_x < 0 || global_y < 0 || global_x >= width || global_y >= height) return;

    // Load shared memory including halo region
    int halo = search_radius + patch_radius;

    // Every thread loads itself
    shared_mem[ty + halo][tx + halo][0] = d_noisy[(global_y * width + global_x) * channels + 0];
    shared_mem[ty + halo][tx + halo][1] = d_noisy[(global_y * width + global_x) * channels + 1];
    shared_mem[ty + halo][tx + halo][2] = d_noisy[(global_y * width + global_x) * channels + 2];

    // Load halo region - top
    if (ty < halo) {
        int y = max(global_y - halo, 0);
        shared_mem[ty][tx + halo][0] = d_noisy[(y * width + global_x) * channels + 0];
        shared_mem[ty][tx + halo][1] = d_noisy[(y * width + global_x) * channels + 1];
        shared_mem[ty][tx + halo][2] = d_noisy[(y * width + global_x) * channels + 2];
    }

    if (ty >= blockDim.y - halo) {
        int y = min(global_y + halo, height - 1);
        shared_mem[ty + 2 * halo][tx + halo][0] = d_noisy[(y * width + global_x) * channels + 0];
        shared_mem[ty + 2 * halo][tx + halo][1] = d_noisy[(y * width + global_x) * channels + 1];
        shared_mem[ty + 2 * halo][tx + halo][2] = d_noisy[(y * width + global_x) * channels + 2];
    }

    // Load halo region - left
    if (tx < halo) {
        int x = max(global_x - halo, 0);
        shared_mem[ty + halo][tx][0] = d_noisy[(global_y * width + x) * channels + 0];
        shared_mem[ty + halo][tx][1] = d_noisy[(global_y * width + x) * channels + 1];
        shared_mem[ty + halo][tx][2] = d_noisy[(global_y * width + x) * channels + 2];
    }

    // Load halo region - right
    if (tx >= blockDim.x - halo) {
        int x = min(global_x + halo, width - 1);
        shared_mem[ty + halo][tx + 2 * halo][0] = d_noisy[(global_y * width + x) * channels + 0];
        shared_mem[ty + halo][tx + 2 * halo][1] = d_noisy[(global_y * width + x) * channels + 1];
        shared_mem[ty + halo][tx + 2 * halo][2] = d_noisy[(global_y * width + x) * channels + 2];
    }

    // Load corner regions
    if (tx < halo && ty < halo) {
        // Top-left
        int x = max(global_x - halo, 0);
        int y = max(global_y - halo, 0);
        shared_mem[ty][tx][0] = d_noisy[(y * width + x) * channels + 0];
        shared_mem[ty][tx][1] = d_noisy[(y * width + x) * channels + 1];
        shared_mem[ty][tx][2] = d_noisy[(y * width + x) * channels + 2];
    }

    if (tx >= blockDim.x - halo && ty < halo) {
        // Top-right
        int x = min(global_x + halo, width - 1);
        int y = max(global_y - halo, 0);
        shared_mem[ty][tx + 2 * halo][0] = d_noisy[(y * width + x) * channels + 0];
        shared_mem[ty][tx + 2 * halo][1] = d_noisy[(y * width + x) * channels + 1];
        shared_mem[ty][tx + 2 * halo][2] = d_noisy[(y * width + x) * channels + 2];
    }

    if (tx < halo && ty >= blockDim.y - halo) {
        // Bottom-left
        int x = max(global_x - halo, 0);
        int y = min(global_y + halo, height - 1);
        shared_mem[ty + 2 * halo][tx][0] = d_noisy[(y * width + x) * channels + 0];
        shared_mem[ty + 2 * halo][tx][1] = d_noisy[(y * width + x) * channels + 1];
        shared_mem[ty + 2 * halo][tx][2] = d_noisy[(y * width + x) * channels + 2];
    }

    if (tx >= blockDim.x - halo && ty >= blockDim.y - halo) {
        // Bottom-right
        int x = min(global_x + halo, width - 1);
        int y = min(global_y + halo, height - 1);
        shared_mem[ty + 2 * halo][tx + 2 * halo][0] = d_noisy[(y * width + x) * channels + 0];
        shared_mem[ty + 2 * halo][tx + 2 * halo][1] = d_noisy[(y * width + x) * channels + 1];
        shared_mem[ty + 2 * halo][tx + 2 * halo][2] = d_noisy[(y * width + x) * channels + 2];
    }

    __syncthreads();

    float weight_sum[3] = {0.0f, 0.0f, 0.0f};
    float pixel_value[3] = {0.0f, 0.0f, 0.0f};

    for (int wy = -search_radius; wy <= search_radius; ++wy) {
        for (int wx = -search_radius; wx <= search_radius; ++wx) {
            int ny = ty + wy, nx = tx + wx;

            int global_ny = start_row + blockIdx.y * blockDim.y + ny;
            int global_nx = blockIdx.x * blockDim.x + nx;

            if (global_ny >= 0 && global_ny < height && global_nx >= 0 && global_nx < width) {
                float distance = calculate_patch_distance_shared(shared_mem, tx, ty, nx, ny, patch_radius, channels, halo, width, height);
                float weight = expf(-distance / (2 * SIGMA * SIGMA));

                for (int c = 0; c < channels; ++c) {
                    weight_sum[c] += weight;
                    pixel_value[c] += weight * shared_mem[ny + halo][nx + halo][c];
                }
            }
        }
    }

    for (int c = 0; c < channels; ++c) {
        d_denoised[(global_y * width + global_x) * channels + c] = pixel_value[c] / weight_sum[c];
    }
}

void nlmeans_cuda(const vector<float>& noisy, vector<float>& denoised) {
    size_t total_size = width * height * channels * sizeof(float);

    float *h_noisy, *h_denoised;
    cudaMallocHost(&h_noisy, total_size);
    cudaMallocHost(&h_denoised, total_size);

    memcpy(h_noisy, noisy.data(), total_size);

    float *d_noisy, *d_denoised;
    cudaMalloc(&d_noisy, total_size);
    cudaMalloc(&d_denoised, total_size);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int rows_per_stream = (height + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int start_row = i * rows_per_stream;
        const int start = start_row * width * channels;
        const int num_rows = min(rows_per_stream, height - start_row);

        const int load_start_row = max(0, start_row - HALO);
        const int load_start = load_start_row * width * channels;
        const int load_num_rows = min(num_rows + 2 * HALO, height - load_start_row);

        cudaMemcpyAsync(d_noisy + load_start, h_noisy + load_start, load_num_rows * width * channels * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        dim3 block(BZ, BZ);
        dim3 grid((width + block.x - 1) / block.x, (num_rows + block.y - 1) / block.y);

        nlmeans_kernel<<<grid, block, 0, streams[i]>>>(
            d_noisy, d_denoised,
            width, height, channels,
            PATCH_SIZE / 2, SEARCH_WINDOW / 2,
            start_row, start_row + num_rows);

        cudaMemcpyAsync(h_denoised + start, d_denoised + start, num_rows * width * channels * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    memcpy(denoised.data(), h_denoised, total_size);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_noisy);
    cudaFreeHost(h_denoised);

    cudaFree(d_noisy);
    cudaFree(d_denoised);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << endl;
        return EXIT_FAILURE;
    }

    vector<float> noisy;

    read_image(argv[1], noisy);

    vector<float> denoised(width * height * channels);

    nlmeans_cuda(noisy, denoised);

    save_image(argv[2], denoised);

    return EXIT_SUCCESS;
}