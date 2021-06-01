#include "net.h"

//  dwcompute funtion
//    finish the multiplication of correspongding elements of two vectors
//    the result is a vector


void dwcompute(FIX_FM weight[CHANNEL], FIX_FM input[CHANNEL],
               FIX_ACC output[CHANNEL]) {

  FIX_FM fm_buf[16];
  FIX_WT weight_buf[16];
  FIX_ACC *output_ptr = output;
  // the output_ptr marks where the output result start
  int cin, i;
  cin = CHANNEL / 16;
  for (i = 0; i < cin; i++) {
    load_weights(weight_buf, weight, i);
    load_fm(fm_buf, input, i);
    compute(weight_buf[0], fm_buf[0], weight_buf[1], fm_buf[1], weight_buf[2],
            fm_buf[2], weight_buf[3], fm_buf[3], weight_buf[4], fm_buf[4],
            weight_buf[5], fm_buf[5], weight_buf[6], fm_buf[6], weight_buf[7],
            fm_buf[7], weight_buf[8], fm_buf[8], weight_buf[9], fm_buf[9],
            weight_buf[10], fm_buf[10], weight_buf[11], fm_buf[11], weight_buf[12],
            fm_buf[12], weight_buf[13], fm_buf[13], weight_buf[14], fm_buf[14],
            weight_buf[15], fm_buf[15], 0,output_ptr);
    output_ptr += 16;
  }
}

