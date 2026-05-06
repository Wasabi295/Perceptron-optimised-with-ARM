# Perceptron — SIMD Optimisations (AVX2 & ARM NEON)

A C++ implementation of a single-layer **perceptron** with two interchangeable backends — a clean scalar baseline and a vectorised version that uses **AVX2** on x86_64 or **ARM NEON** on ARM. The program ships with a CLI that auto-detects the best backend at runtime, generates datasets on the fly, and runs head-to-head benchmarks (scalar vs. SIMD) across several problem sizes, plus a small demo that shows why a single perceptron cannot learn XOR.

## Highlights

- **Two backends, one interface** — `Perceptron` (scalar) and `PerceptronSIMD : public Perceptron` (vectorised). The SIMD subclass only overrides the inner loops (`dot_product`, `weight_update`); the rest of the training logic is shared.
- **AVX2 + ARM NEON** — `simd_dot_product_optimized` and `simd_weight_update_optimized` both have an `__AVX2__` branch using `_mm256_*` intrinsics and an `__ARM_NEON` branch using `vld1q_f32` / `vmulq_f32` / `vaddq_f32`, with a scalar tail loop for the leftover elements.
- **Runtime CPU detection** — `cpu_supports_avx2()` checks both `CPUID.1:ECX[28]` (AVX) and `CPUID.7:EBX[5]` (AVX2), plus the `XCR0` bits to confirm the OS preserves the YMM state. If AVX2 is missing or the binary was built without `/arch:AVX2`, the SIMD backend cleanly falls back to scalar.
- **CLI mode flag** — `--mode=auto|simd|standard` (also accepted as `--mode simd`, etc.). Default is `auto`.
- **Reproducible benchmarks** — both backends are seeded with the same value (`1337`) so the comparison is apples-to-apples.
- **XOR demo** — runs five tests with different seeds and prints accuracy + training time, illustrating that a single perceptron stays around 50 % on XOR (the obligatory "this is why we need MLPs" moment).

## Repository layout

```
Perceptron-optimised-with-ARM/
└── Perceptron/
    ├── Perceptron.sln
    ├── Perceptron.vcxproj          # Visual Studio project (target the desired arch)
    ├── main.cpp                    # CLI, CPU detection, menu, benchmarks
    ├── perceptron.h / .cpp         # Scalar baseline (training, prediction, accuracy)
    ├── perceptron_simd.h / .cpp    # AVX2 + NEON overrides
    ├── dataset.h / .cpp            # DatasetGenerator: linearly-separable + XOR
    └── data/
        └── dataset.csv             # Sample 2D dataset (x1, x2, label)
```

## How the SIMD path works

The dot-product on AVX2 accumulates eight floats per iteration into a `__m256`, then horizontally reduces it to a scalar via `_mm_add_ps` + `_mm_hadd_ps`:

```cpp
__m256 sum = _mm256_setzero_ps();
for (; i + 7 < n; i += 8) {
    __m256 vw = _mm256_loadu_ps(w + i);
    __m256 vx = _mm256_loadu_ps(x + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(vw, vx));
}
```

The NEON branch does the same with `float32x4_t` (4 floats per iteration) and a `vpadd_f32` reduction. Both paths fall through to a small scalar loop for any remaining elements, which keeps the code correct for input sizes that are not multiples of the SIMD width.

The weight update follows the same shape: broadcast the scalar `update = learning_rate * error` to a vector and perform a fused `w += update * x` over the whole weight buffer.

## Build

### Visual Studio (Windows, x86_64)

Open `Perceptron.sln`, set the configuration to `Release / x64`, and **Build → Build Solution**. To enable AVX2 you must add **/arch:AVX2** in the project's *C/C++ → Code Generation → Enable Enhanced Instruction Set* setting; otherwise the binary will still run, but always in scalar mode.

### Command line (g++ / clang)

```bash
# x86_64 with AVX2
g++ -O3 -std=c++17 -mavx2 main.cpp dataset.cpp perceptron.cpp perceptron_simd.cpp -o perceptron

# ARM (e.g. Apple Silicon, Raspberry Pi 64-bit)
g++ -O3 -std=c++17 -march=armv8-a+simd main.cpp dataset.cpp perceptron.cpp perceptron_simd.cpp -o perceptron
```

## Run

```bash
./perceptron                  # auto-detects CPU
./perceptron --mode=simd      # force SIMD (falls back to scalar if AVX2 unavailable)
./perceptron --mode=standard  # force scalar
```

A small interactive menu opens:

1. **Performance Comparison** — trains both backends on `{500×8, 2000×16, 5000×32, 10000×64}` synthetic linearly-separable problems for 50 epochs each and prints a table:

   ```
   Samples  Features  Scalar (s)  SIMD (s)  Speedup  Acc S / Acc V
   ```

2. **XOR Problem Demonstration** — runs 5 trainings with different seeds on a small XOR dataset and confirms that a single perceptron is stuck around ~50 % accuracy.
3. **Exit**.

## Why this is interesting

The perceptron's hot loop is essentially a dot product followed by a fused multiply-add — a textbook target for SIMD. With identical seeds and dataset, the scalar and vectorised versions converge to the same accuracy, which makes any change in wall-clock time a clean measurement of vectorisation overhead vs. throughput. The same source compiles unchanged on x86_64 (AVX2) and ARM (NEON) thanks to the `#ifdef __AVX2__ / #elif __ARM_NEON` switch, so the project doubles as a small case study in writing portable SIMD code.
