#include "perceptron.h"
#include <iostream>
#include <cmath>

Perceptron::Perceptron(size_t input_dim, float lr)
    : bias(0.0f), learning_rate(lr), input_size(input_dim), init_seed(0u), has_seed(false) {
    initialize_weights();
}

Perceptron::Perceptron(size_t input_dim, float lr, unsigned int seed)
    : bias(0.0f), learning_rate(lr), input_size(input_dim), init_seed(seed), has_seed(true) {
    initialize_weights_with_seed(seed);
}

void Perceptron::initialize_weights() {
    if (has_seed) {
        initialize_weights_with_seed(init_seed);
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = 1.0f / std::sqrt(static_cast<float>(input_size));
    std::uniform_real_distribution<float> dist(-scale, scale);

    weights.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) weights[i] = dist(gen);
    bias = dist(gen);
}

void Perceptron::initialize_weights_with_seed(unsigned int seed) {
    std::mt19937 gen(seed);
    float scale = 1.0f / std::sqrt(static_cast<float>(input_size));
    std::uniform_real_distribution<float> dist(-scale, scale);

    weights.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) weights[i] = dist(gen);
    bias = dist(gen);
}

void Perceptron::reset_weights() {
    initialize_weights();
}

float Perceptron::activation(float x) const {
    return x >= 0 ? 1.0f : -1.0f;
}

float Perceptron::dot_product(const std::vector<float>& input) const {
    float result = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        result += weights[i] * input[i];
    }
    return result;
}

int Perceptron::predict(const std::vector<float>& input) const {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size doesn't match perceptron input size");
    }
    float weighted_sum = dot_product(input) + bias;
    return activation(weighted_sum) >= 0 ? 1 : -1;
}

std::vector<int> Perceptron::predict_batch(const std::vector<std::vector<float>>& inputs) const {
    std::vector<int> preds;
    preds.reserve(inputs.size());
    for (const auto& in : inputs) preds.push_back(predict(in));
    return preds;
}

void Perceptron::weight_update(const std::vector<float>& input, int target, int prediction) {
    float error = static_cast<float>(target - prediction);
    float update = learning_rate * error;

    for (size_t i = 0; i < input_size; ++i) {
        weights[i] += update * input[i];
    }
    bias += update;
}

void Perceptron::train(const std::vector<std::vector<float>>& inputs,
    const std::vector<int>& targets,
    size_t epochs) {

    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            int pred = predict(inputs[i]);
            if (pred == targets[i]) correct++;
            else weight_update(inputs[i], targets[i], pred);
        }

        float acc = static_cast<float>(correct) / inputs.size();
        if (acc >= 0.999f) break;
    }
}

double Perceptron::train_and_measure(const std::vector<std::vector<float>>& inputs,
    const std::vector<int>& targets,
    size_t epochs) {

    auto t0 = std::chrono::high_resolution_clock::now();
    train(inputs, targets, epochs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d = t1 - t0;
    return d.count();
}

void Perceptron::print_weights() const {
    std::cout << "Weights: [";
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << weights[i];
        if (i + 1 < weights.size()) std::cout << ", ";
    }
    std::cout << "], Bias: " << bias << "\n";
}

float Perceptron::get_accuracy(const std::vector<std::vector<float>>& inputs,
    const std::vector<int>& targets) const {

    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }

    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (predict(inputs[i]) == targets[i]) correct++;
    }
    return static_cast<float>(correct) / inputs.size();
}
