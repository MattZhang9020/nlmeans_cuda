#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define SIGMA 2
#define PATCH_SIZE 3
#define SEARCH_WINDOW 7

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

using namespace std;

int width, height, channels;

void read_image(const char *filename, vector<float> &noisy) {
    unsigned char *image_data = stbi_load(filename, &width, &height, &channels, 3);

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

void save_image(const char *filename, const vector<float> &data) {
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

float calculate_patch_distance(const vector<float> &noisy, int x1, int y1, int x2, int y2, int patch_radius) {
    float distance = 0.0f;

    for (int py = -patch_radius; py <= patch_radius; ++py) {
        for (int px = -patch_radius; px <= patch_radius; ++px) {
            int yy1 = y1 + py, xx1 = x1 + px;
            int yy2 = y2 + py, xx2 = x2 + px;

            if (yy1 >= 0 && yy1 < height && xx1 >= 0 && xx1 < width && yy2 >= 0 && yy2 < height && xx2 >= 0 && xx2 < width) {
                for (int c = 0; c < channels; ++c) {
                    float diff = noisy[(yy1 * width + xx1) * channels + c] - noisy[(yy2 * width + xx2) * channels + c];

                    distance += diff * diff;
                }
            }
        }
    }

    float result = sqrt(distance) / (patch_radius * 2 + 1) / (patch_radius * 2 + 1) / channels;

    return result;
}

void nlmeans(const vector<float> &noisy, vector<float> &denoised) {
    int patch_radius = PATCH_SIZE / 2;
    int search_radius = SEARCH_WINDOW / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            vector<float> weight_sum(channels, 0.0f);
            vector<float> pixel_value(channels, 0.0f);

            for (int wy = -search_radius; wy <= search_radius; ++wy) {
                for (int wx = -search_radius; wx <= search_radius; ++wx) {
                    int ny = y + wy, nx = x + wx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        float distance = calculate_patch_distance(noisy, x, y, nx, ny, patch_radius);
                        float weight = exp(-distance / (2 * SIGMA * SIGMA));

                        for (int c = 0; c < channels; ++c) {
                            weight_sum[c] += weight;
                            pixel_value[c] += weight * noisy[(ny * width + nx) * channels + c];
                        }
                    }
                }
            }

            for (int c = 0; c < channels; ++c) {
                denoised[(y * width + x) * channels + c] = pixel_value[c] / weight_sum[c];
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << endl;
        return EXIT_FAILURE;
    }

    vector<float> noisy;

    read_image(argv[1], noisy);

    vector<float> denoised(width * height * channels);

    nlmeans(noisy, denoised);

    save_image(argv[2], denoised);

    return EXIT_SUCCESS;
}