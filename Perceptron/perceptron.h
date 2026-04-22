#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <stdexcept>

class Perceptron {
protected:
    std::vector<float> weights;
    float bias;
    float learning_rate;
    size_t input_size;

    unsigned int init_seed;
    bool has_seed;

public:
    Perceptron(size_t input_dim, float lr = 0.01f);
    Perceptron(size_t input_dim, float lr, unsigned int seed);
    virtual ~Perceptron() = default;

    virtual void train(const std::vector<std::vector<float>>& inputs,
        const std::vector<int>& targets,
        size_t epochs);

    virtual int predict(const std::vector<float>& input) const;
    virtual std::vector<int> predict_batch(const std::vector<std::vector<float>>& inputs) const;

    virtual void print_weights() const;
    virtual float get_accuracy(const std::vector<std::vector<float>>& inputs,
        const std::vector<int>& targets) const;

    virtual double train_and_measure(const std::vector<std::vector<float>>& inputs,
        const std::vector<int>& targets,
        size_t epochs);

    virtual std::string get_type() const { return "Scalar"; }

    virtual void reset_weights();

protected:
    virtual float activation(float x) const;
    virtual float dot_product(const std::vector<float>& input) const;
    virtual void weight_update(const std::vector<float>& input, int target, int prediction);

    virtual void initialize_weights();
    virtual void initialize_weights_with_seed(unsigned int seed);
};

#endif
