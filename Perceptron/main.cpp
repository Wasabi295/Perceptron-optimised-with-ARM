#include "perceptron.h"
#include "perceptron_simd.h"
#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

static bool cpu_supports_avx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4] = { 0,0,0,0 };

    __cpuid(info, 1);
    bool osxsave = (info[2] & (1 << 27)) != 0;
    bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    unsigned long long xcr0 = _xgetbv(0);
    bool xmm_ok = (xcr0 & 0x2) != 0;
    bool ymm_ok = (xcr0 & 0x4) != 0;
    if (!xmm_ok || !ymm_ok) return false;

    __cpuidex(info, 7, 0);
    bool avx2 = (info[1] & (1 << 5)) != 0; // EBX bit 5
    return avx2;
#else
    return false;
#endif
}

static std::string get_mode(int argc, char** argv) {
    std::string mode = "auto";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--mode" && i + 1 < argc) {
            mode = argv[i + 1];
            break;
        }
        const std::string pfx = "--mode=";
        if (a.rfind(pfx, 0) == 0) {
            mode = a.substr(pfx.size());
            break;
        }
    }

    std::transform(mode.begin(), mode.end(), mode.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });

    if (mode != "auto" && mode != "simd" && mode != "standard") mode = "auto";
    return mode;
}

static bool resolve_use_simd(const std::string& mode) {
    bool avx2_ok = cpu_supports_avx2();

    if (mode == "standard") return false;
    if (mode == "simd") return avx2_ok;
    return avx2_ok; // auto
}

static std::unique_ptr<Perceptron> make_model(size_t features, float lr, unsigned int seed, bool use_simd) {
    if (use_simd) return std::make_unique<PerceptronSIMD>(features, lr, seed);
    return std::make_unique<Perceptron>(features, lr, seed);
}

void performance_comparison() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "PERFORMANCE COMPARISON: SCALAR vs SIMD\n";
    std::cout << std::string(70, '=') << "\n";

    DatasetGenerator generator(42);

    std::vector<std::pair<size_t, size_t>> test_cases = {
        {500, 8},
        {2000, 16},
        {5000, 32},
        {10000, 64},
    };

    const size_t epochs = 50;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << std::setw(12) << "Samples" << std::setw(12) << "Features"
        << std::setw(15) << "Scalar (s)" << std::setw(15) << "SIMD (s)"
        << std::setw(15) << "Speedup" << std::setw(15) << "Acc S / Acc V" << "\n";
    std::cout << std::string(80, '=') << "\n";

    for (const auto& tc : test_cases) {
        size_t samples = tc.first;
        size_t features = tc.second;

        Dataset dataset = generator.generate_linear_separable(samples, features, 0.1f);

        const unsigned int init_seed = 1337u;

        Perceptron scalar(features, 0.01f, init_seed);
        double scalar_time = scalar.train_and_measure(dataset.inputs, dataset.targets, epochs);
        float  scalar_acc = scalar.get_accuracy(dataset.inputs, dataset.targets);

        PerceptronSIMD simd(features, 0.01f, init_seed);
        double simd_time = simd.train_and_measure(dataset.inputs, dataset.targets, epochs);
        float  simd_acc = simd.get_accuracy(dataset.inputs, dataset.targets);

        double speedup = scalar_time / simd_time;

        std::cout << std::setw(12) << samples << std::setw(12) << features
            << std::setw(15) << scalar_time << std::setw(15) << simd_time
            << std::setw(15) << speedup << "x"
            << std::setw(7) << (scalar_acc * 100) << "%"
            << std::setw(7) << (simd_acc * 100) << "%" << "\n";
    }

    std::cout << std::string(80, '=') << "\n";
}

void xor_problem_demo(bool use_simd) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "XOR Demonstratie (" << (use_simd ? "SIMD" : "STANDARD") << ")\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "Un singur perceptron NU poate rezolva XOR (non-linearly separable)\n";
    std::cout << "Acuratete asteptata ~50%\n";

    DatasetGenerator generator(42);
    Dataset xor_data = generator.generate_xor_problem(100);
    generator.display_info(xor_data);

    const int num_tests = 5;
    std::cout << "Running " << num_tests << " tests with different initializations:\n\n";

    for (int test = 1; test <= num_tests; ++test) {
        std::cout << "TEST " << test << ":\n";

        
        unsigned int seed = 1000u + (unsigned int)test;

        auto model = make_model(2, 0.1f, seed, use_simd);
        double t = model->train_and_measure(xor_data.inputs, xor_data.targets, 100);
        float acc = model->get_accuracy(xor_data.inputs, xor_data.targets);

        std::cout << (use_simd ? "SIMD:   " : "Scalar: ")
            << std::fixed << std::setprecision(1) << acc * 100
            << "% (Time: " << std::setprecision(3) << t << "s)\n";

        if (test < num_tests) std::cout << "---\n";
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << "Concluzie:\n";
    std::cout << "XOR necesita mai multe straturi (MLP), un singur perceptron nu poate invata functia XOR.\n";
    std::cout << std::string(70, '=') << "\n";
}

int main(int argc, char** argv) {
    std::cout << "PERCEPTRON WITH SIMD OPTIMIZATIONS\n";
    std::cout << "======================================\n";

    std::string mode = get_mode(argc, argv);
    bool avx2_ok = cpu_supports_avx2();
    bool use_simd = resolve_use_simd(mode);

#ifdef __AVX2__
    std::cout << "Compiled with AVX2 support: YES\n";
#else
    std::cout << "Compiled with AVX2 support: NO (set /arch:AVX2)\n";
#endif

    std::cout << "CPU AVX2 available: " << (avx2_ok ? "YES" : "NO") << "\n";

    if (mode == "simd" && !use_simd) {
        std::cout << "Mode requested: SIMD, but AVX2 is not available -> falling back to STANDARD\n";
    }

    std::cout << "Mode selected: " << (use_simd ? "SIMD" : "STANDARD") << " (mode=" << mode << ")\n";

    int choice;
    do {
        std::cout << "\n" << std::string(40, '=') << "\n";
        std::cout << "MAIN MENU\n";
        std::cout << std::string(40, '=') << "\n";
        std::cout << "1. Performance Comparison \n";
        std::cout << "2. XOR Problem Demonstration S\n";
        std::cout << "3. Exit\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << "Enter your choice (1-3): ";
        std::cin >> choice;

        switch (choice) {
        case 1:
            performance_comparison();
            break;
        case 2:
            xor_problem_demo(use_simd);
            break;
        case 3:
            std::cout << "\nCiao\n";
            break;
        default:
            std::cout << "Invalid choice. Please try again.\n";
        }
    } while (choice != 3);

    return 0;
}
