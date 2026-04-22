#include "perceptron_simd.h"
#include <chrono>
#include <stdexcept>

PerceptronSIMD::PerceptronSIMD(size_t input_dim, float lr)
    : Perceptron(input_dim, lr) {
}

PerceptronSIMD::PerceptronSIMD(size_t input_dim, float lr, unsigned int seed)
    : Perceptron(input_dim, lr, seed) {
}

void PerceptronSIMD::reset_weights() {
    initialize_weights();
}

float PerceptronSIMD::dot_product(const std::vector<float>& input) const {
    return simd_dot_product_optimized(input);
}

float PerceptronSIMD::simd_dot_product_optimized(const std::vector<float>& input) const {
    const float* w = weights.data();
    const float* x = input.data();
    float result = 0.0f;

    size_t i = 0;
    size_t n = input_size;

#ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vw, vx));
    }
    __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0),
        _mm256_extractf128_ps(sum, 1));
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    result = _mm_cvtss_f32(s128);

#elif defined(__ARM_NEON)
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t vw = vld1q_f32(w + i);
        float32x4_t vx = vld1q_f32(x + i);
        sum = vaddq_f32(sum, vmulq_f32(vw, vx));
    }
    float32x2_t s2 = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
    result = vget_lane_f32(vpadd_f32(s2, s2), 0);
#endif

    for (; i < n; ++i) result += w[i] * x[i];
    return result;
}

void PerceptronSIMD::weight_update(const std::vector<float>& input, int target, int prediction) {
    simd_weight_update_optimized(input, target, prediction);
}

void PerceptronSIMD::simd_weight_update_optimized(const std::vector<float>& input, int target, int prediction) {
    float error = static_cast<float>(target - prediction);
    float update = learning_rate * error;

    float* w = weights.data();
    const float* x = input.data();
    size_t i = 0;

#ifdef __AVX2__
    __m256 u = _mm256_set1_ps(update);
    for (; i + 7 < input_size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        vw = _mm256_add_ps(vw, _mm256_mul_ps(u, vx));
        _mm256_storeu_ps(w + i, vw);
    }

#elif defined(__ARM_NEON)
    float32x4_t u = vdupq_n_f32(update);
    for (; i + 3 < input_size; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(w + i);
        vw = vaddq_f32(vw, vmulq_f32(u, vx));
        vst1q_f32(w + i, vw);
    }
#endif

    for (; i < input_size; ++i) w[i] += update * x[i];
    bias += update;
}

int PerceptronSIMD::predict(const std::vector<float>& input) const {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size doesn't match perceptron input size");
    }
    float s = dot_product(input) + bias;
    return activation(s) >= 0 ? 1 : -1;
}

std::vector<int> PerceptronSIMD::predict_batch(const std::vector<std::vector<float>>& inputs) const {
    std::vector<int> preds;
    preds.reserve(inputs.size());
    for (const auto& in : inputs) preds.push_back(predict(in));
    return preds;
}

void PerceptronSIMD::train(const std::vector<std::vector<float>>& inputs,
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

double PerceptronSIMD::train_and_measure(const std::vector<std::vector<float>>& inputs,
    const std::vector<int>& targets,
    size_t epochs) {

    auto t0 = std::chrono::high_resolution_clock::now();
    train(inputs, targets, epochs);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d = t1 - t0;
    return d.count();
}
