
#include "net.h"

// the funciton of compute
//  if the variable add is true 1
//     we calculate the multiplication first and then the addition
//     and return the result
//  if the variable add is false 0
//     we only caculate the multiplication
//     and store the result in the array named ouput

FIX_ACC compute(FIX_WT w0, FIX_FM b0, FIX_WT w1, FIX_FM b1, FIX_WT w2, FIX_FM b2,
               FIX_WT w3, FIX_FM b3, FIX_WT w4, FIX_FM b4, FIX_WT w5, FIX_FM b5,
               FIX_WT w6, FIX_FM b6, FIX_WT w7, FIX_FM b7, FIX_WT w8, FIX_FM b8,
               FIX_WT w9, FIX_FM b9, FIX_WT w10, FIX_FM b10, FIX_WT w11,
               FIX_FM b11, FIX_WT w12, FIX_FM b12, FIX_WT w13, FIX_FM b13,
               FIX_WT w14, FIX_FM b14, FIX_WT w15, FIX_FM b15, int add,
               FIX_FM output[16]) {
  FIX_ACC mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7;
  FIX_ACC mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
  FIX_ACC add0, add1, add2, add3, add4, add5, add6;
  FIX_ACC add7, add8, add9, add10, add11, add12, add13, add14;
  add14 = 0;

  if (add == 1) {
    mul0 = w0 * b0;
    mul1 = w1 * b1;
    mul2 = w2 * b2;
    mul3 = w3 * b3;
    mul4 = w4 * b4;
    mul5 = w5 * b5;
    mul6 = w6 * b6;
    mul7 = w7 * b7;
    mul8 = w8 * b8;
    mul9 = w9 * b9;
    mul10 = w10 * b10;
    mul11 = w11 * b11;
    mul12 = w12 * b12;
    mul13 = w13 * b13;
    mul14 = w14 * b14;
    mul15 = w15 * b15;

    add0 = mul0 + mul1;
    add1 = mul2 + mul3;
    add2 = mul4 + mul5;
    add3 = mul6 + mul7;
    add4 = mul8 + mul9;
    add5 = mul10 + mul11;
    add6 = mul12 + mul13;
    add7 = mul14 + mul15;

    add8 = add0 + add1;
    add9 = add2 + add3;
    add10 = add4 + add5;
    add11 = add6 + add7;

    add12 = add8 + add9;
    add13 = add10 + add11;

    add14 = add12 + add13;

  }

  else {
    output[0] = w0 * b0;
    output[1] = w1 * b1;
    output[2] = w2 * b2;
    output[3] = w3 * b3;
    output[4] = w4 * b4;
    output[5] = w5 * b5;
    output[6] = w6 * b6;
    output[7] = w7 * b7;
    output[8] = w8 * b8;
    output[9] = w9 * b9;
    output[10] = w10 * b10;
    output[11] = w11 * b11;
    output[12] = w12 * b12;
    output[13] = w13 * b13;
    output[14] = w14 * b14;
    output[15] = w15 * b15;
  }

  return add14;
}

//  load data for the coming compute

void load_weights(FIX_WT weight_buf[16], FIX_WT weight[CHANNEL], int cin) {
  for (int i = 0; i < 16; i++) {
    weight_buf[i] = weight[cin * 16 + i];
  }
}

void load_fm(FIX_FM fm_buf[16], FIX_FM fm[CHANNEL], int cin) {
  for (int i = 0; i < 16; i++) {
    fm_buf[i] = fm[cin * 16 + i];
  }
}

//  pwcompute function
//    finish the vector inner product of feather map and weight
//    the result is a number
//    we devide the CHANNEL size vector into several 16 size vector  
FIX_ACC pwcompute(FIX_FM fm[CHANNEL], FIX_WT weight[CHANNEL]) {


  FIX_WT weight_buf[16];
  FIX_FM fm_buf[16];
  FIX_ACC result = 0;
  int cin, i;
  cin = CHANNEL / 16;
  for (i = 0; i < cin; i++) {
    load_weights(weight_buf, weight, i);
    load_fm(fm_buf, fm, i);
    result += compute(
        weight_buf[0], fm_buf[0], weight_buf[1], fm_buf[1], weight_buf[2],
        fm_buf[2], weight_buf[3], fm_buf[3], weight_buf[4], fm_buf[4],
        weight_buf[5], fm_buf[5], weight_buf[6], fm_buf[6], weight_buf[7],
        fm_buf[7], weight_buf[8], fm_buf[8], weight_buf[9], fm_buf[9],
        weight_buf[10], fm_buf[10], weight_buf[11], fm_buf[11], weight_buf[12],
        fm_buf[12], weight_buf[13], fm_buf[13], weight_buf[14], fm_buf[14],
        weight_buf[15], fm_buf[15], 1,fm_buf);
  }
  //  here the fm_buf is not used in the compute function
  //  so we can configure it casually
  return result;
}
