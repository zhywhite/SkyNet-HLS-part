#include "net.h"

// load the data
void copy_input(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_FM input_blk[CHANNEL], int h, int w)
{
  int i;
  for (i = 0; i < CHANNEL; i++)
  {
    input_blk[i] = input[i][h][w];
  }
}

void copy_weight(FIX_WT weight[CHANNEL][CHANNEL], FIX_WT weight_buf[CHANNEL],
                 int num)
{
  for (int i = 0; i < CHANNEL; i++)
  {
    weight_buf[i] = weight[num][i];
  }
}

//  compute a pwconv_blk
void pwconv_blk(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_ACC output[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_WT weight[CHANNEL][CHANNEL])
{

  /*#pragma HLS INTERFACE m_axi depth=1024 port=input bundle=DATA
#pragma HLS INTERFACE m_axi depth=1024 port=output bundle=DATA
#pragma HLS INTERFACE m_axi depth=1024 port=weight bundle=DATA

#pragma HLS INTERFACE s_axilite port=input bundle=CTRL
#pragma HLS INTERFACE s_axilite port=output bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weight bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=return bundle=CTRL*/
  int i, h, w;
  int MATRIXH_CONV, MATRIXW_CONV;
  compute_matrix_size(MAX_N, MATRIXH_CONV, MATRIXW_CONV);

  //  MATRIXH_CONV = MATRIX_H;
  //  MATRIXW_CONV = MATRIX_W;
  FIX_FM input_blk[CHANNEL];
  FIX_WT weight_blk[CHANNEL];
  for (i = 0; i < CHANNEL; i++)
  {
    copy_weight(weight, weight_blk, i);
    for (h = 0; h < MATRIXH_CONV; h++)
    {
#pragma HLS pipeline
      for (w = 0; w < MATRIXW_CONV; w++)
      {
#pragma HLS unroll
        copy_input(input, input_blk, h, w);
        output[i][h][w] += pwcompute(input_blk, weight_blk);
      }
    }
  }
}
