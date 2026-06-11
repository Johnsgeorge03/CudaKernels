#pragma once

// C = A dot B
// d_C is a single-float accumulator; the launcher zeroes it before the
// kernel runs, so callers must not pre-fill it.

void launchDotProd(const float* d_A,
                    const float* d_B,
                    float* d_C,
                    int N);