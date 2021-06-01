#include "net.h"

//  maxpoolPE function
//  the variable H,W represent the output size of the matrix

void maxpoolPE(int H, int W, FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
               FIX_FM output[CHANNEL][MATRIX_H][MATRIX_W]) {

  int ch, h, w;
  for (ch = 0; ch < CHANNEL; ch++) {
    for (h = 1; h < H/2; h++) {
      for (w = 1; w < W/2; w++) {
        output[ch][h][w] =
            max(input[ch][h * 2 - 1][w * 2 - 1], input[ch][h * 2 - 1][w * 2],
                input[ch][h * 2][w * 2 - 1], input[ch][h * 2][w * 2]);
      }
    }
  }
}

FIX_FM max(FIX_FM in1, FIX_FM in2, FIX_FM in3, FIX_FM in4) {
  FIX_FM result = 0;
  if (in1 < in2) in1 = in2;
  if (in3 < in4) in3 = in4;
  if (in1 > in3) result = in1;
  else result = in3;

  return result;
}
