#ifndef PERCEPTRON_SIMD_H
#define PERCEPTRON_SIMD_H

#include "perceptron.h"
#include <string>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

class PerceptronSIMD : public Perceptron {
public:
    PerceptronSIMD(size_t input_dim, float lr = 0.01f);
    PerceptronSIMD(size_t input_dim, float lr, unsigned int seed);

    void train(const std::vector<std::vector<float>>& inputs,
        const std::vector<int>& targets,
        size_t epochs) override;

    int predict(const std::vector<float>& input) const override;
    std::vector<int> predict_batch(const std::vector<std::vector<float>>& inputs) const override;

    double train_and_measure(const std::vector<std::vector<float>>& inputs,
        const std::vector<int>& targets,
        size_t epochs) override;

    std::string get_type() const override { return "SIMD"; }

    void reset_weights() override;

private:
    float dot_product(const std::vector<float>& input) const override;
    void weight_update(const std::vector<float>& input, int target, int prediction) override;

    float simd_dot_product_optimized(const std::vector<float>& input) const;
    void simd_weight_update_optimized(const std::vector<float>& input, int target, int prediction);
};

#endif
