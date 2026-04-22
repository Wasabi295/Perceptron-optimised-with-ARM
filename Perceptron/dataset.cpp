#include "dataset.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DatasetGenerator::DatasetGenerator() : gen(std::random_device{}()) {}
DatasetGenerator::DatasetGenerator(unsigned int seed) : gen(seed) {}

float DatasetGenerator::random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

Dataset DatasetGenerator::generate_linear_separable(size_t samples, size_t features, float noise) {
    Dataset dataset;
    dataset.name = "Linear Separable";
    dataset.description = "Linearly separable data with random hyperplane";

    dataset.inputs.resize(samples, std::vector<float>(features));
    dataset.targets.resize(samples);

    std::vector<float> true_weights(features);
    float true_bias = random_float(-1.0f, 1.0f);

    for (size_t i = 0; i < features; ++i) {
        true_weights[i] = random_float(-1.0f, 1.0f);
    }

    for (size_t i = 0; i < samples; ++i) {
        float dot_product = 0.0f;

        for (size_t j = 0; j < features; ++j) {
            dataset.inputs[i][j] = random_float(-1.0f, 1.0f);
            dot_product += dataset.inputs[i][j] * true_weights[j];
        }

        dot_product += random_float(-noise, noise);
        dataset.targets[i] = (dot_product + true_bias) >= 0 ? 1 : -1;
    }

    return dataset;
}

Dataset DatasetGenerator::generate_xor_problem(size_t samples_per_class) {
    Dataset dataset;
    dataset.name = "XOR Problem";
    dataset.description = "Classic XOR problem - non-linearly separable";

    size_t total_samples = samples_per_class * 4;
    dataset.inputs.resize(total_samples, std::vector<float>(2));
    dataset.targets.resize(total_samples);

    size_t sample_idx = 0;

    for (size_t i = 0; i < samples_per_class; ++i) {
        dataset.inputs[sample_idx][0] = random_float(-1.5f, -0.5f);
        dataset.inputs[sample_idx][1] = random_float(-1.5f, -0.5f);
        dataset.targets[sample_idx] = -1;
        sample_idx++;
    }

    for (size_t i = 0; i < samples_per_class; ++i) {
        dataset.inputs[sample_idx][0] = random_float(0.5f, 1.5f);
        dataset.inputs[sample_idx][1] = random_float(0.5f, 1.5f);
        dataset.targets[sample_idx] = -1;
        sample_idx++;
    }

    for (size_t i = 0; i < samples_per_class; ++i) {
        dataset.inputs[sample_idx][0] = random_float(-1.5f, -0.5f);
        dataset.inputs[sample_idx][1] = random_float(0.5f, 1.5f);
        dataset.targets[sample_idx] = 1;
        sample_idx++;
    }

    for (size_t i = 0; i < samples_per_class; ++i) {
        dataset.inputs[sample_idx][0] = random_float(0.5f, 1.5f);
        dataset.inputs[sample_idx][1] = random_float(-1.5f, -0.5f);
        dataset.targets[sample_idx] = 1;
        sample_idx++;
    }

    return dataset;
}

void DatasetGenerator::display_info(const Dataset& dataset) const {
    std::cout << "\n=== Dataset Information ===\n";
    std::cout << "Name: " << dataset.name << "\n";
    std::cout << "Description: " << dataset.description << "\n";
    std::cout << "Samples: " << dataset.inputs.size() << "\n";
    if (!dataset.inputs.empty()) {
        std::cout << "Features: " << dataset.inputs[0].size() << "\n";
    }

    int positive_count = 0, negative_count = 0;
    for (int t : dataset.targets) (t == 1) ? positive_count++ : negative_count++;

    std::cout << "Class distribution: +1(" << positive_count << "), -1(" << negative_count << ")\n";
    std::cout << "===========================\n\n";
}
