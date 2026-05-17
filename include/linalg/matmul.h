#pragma once

// C = A * B
// A is A_rows x A_cols
// B is A_cols x B_cols
// C is A_rows x B_cols

void launchMatMulNaive(const float* d_A, 
                      const float* d_B, 
                      float* d_C, 
                      int A_rows, 
                      int A_cols, 
                      int B_cols);

// C = A * B
void launchMatMulTiled(const float* d_A, 
                      const float* d_B, 
                      float* d_C, 
                      int A_rows, 
                      int A_cols, 
                      int B_cols);