#pragma once

// C = A dot B 

void launchDotProd(const float* d_A,
                    const float* d_B,
                    float* d_C,
                    int N);