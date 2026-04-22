#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <string>

struct Dataset {
    std::vector<std::vector<float>> inputs;
    std::vector<int> targets;
    std::string name;
    std::string description;
};

class DatasetGenerator {
private:
    std::mt19937 gen;

public:
    DatasetGenerator();
    DatasetGenerator(unsigned int seed);

    Dataset generate_linear_separable(size_t samples = 1000,
        size_t features = 10,
        float noise = 0.2f);

    Dataset generate_xor_problem(size_t samples_per_class = 250);

    void display_info(const Dataset& dataset) const;

private:
    float random_float(float min = -1.0f, float max = 1.0f);
};

#endif
