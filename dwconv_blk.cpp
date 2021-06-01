#include "net.h"

//  we fetch the One-dimensional vector from Three-dimensional vector
//  then compute the vector mult vector
//  finally we wirte the One-dimensional vector result back to Three-dimensional
//  vector

void store_output(FIX_ACC output[CHANNEL][MATRIX_H][MATRIX_W],
                  FIX_ACC output_buf[CHANNEL], int h, int w)
{
  for (int i = 0; i < CHANNEL; i++)
  {
    output[i][h][w] += output_buf[i];
  }
}

void load_data(FIX_FM data[CHANNEL][MATRIX_H][MATRIX_W],
               FIX_FM data_buf[CHANNEL], int h, int w)
{
  for (int i = 0; i < CHANNEL; i++)
  {
    data_buf[i] = data[i][h][w];
  }
}

void load_weight(FIX_WT weight[CHANNEL][3][3], FIX_WT weight_buf[CHANNEL],
                 int h, int w)
{
  for (int i = 0; i < CHANNEL; i++)
  {
    weight_buf[i] = weight[i][h][w];
  }
}
//  MAX_N represent the maxpooling numbers
//  through MAX_N we can determine the size of the matrix we will compute
void dwconv_blk(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_ACC output[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_WT weight[CHANNEL][3][3] ，int MAX_N)
{
  /*#pragma HLS INTERFACE m_axi depth=32*5*5 port=weight bundle=DATA
#pragma HLS INTERFACE m_axi depth=32*5*5 port=input bundle=DATA
#pragma HLS INTERFACE m_axi depth=32*5*5 port=output bundle=DATA

#pragma HLS INTERFACE s_axilite register port=input bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=output bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite register port=weight bundle=CTRL*/
  int i, j, h, w;
  int MATRIXH_CONV, MATRIXW_CONV;
  // MATRIXH_CONV, MATRIXW_CONV represent the size of the matrix we will compute

  FIX_FM input_buf1[CHANNEL];
  FIX_FM input_buf2[CHANNEL];
  FIX_FM weight_buf[CHANNEL];
  FIX_ACC output_buf1[CHANNEL];
  FIX_ACC output_buf2[CHANNEL];
  // compute the actual matrix size
  compute_matrix_size(MAX_N, MATRIXH_CONV, MATRIXW_CONV);
  //  load_data(input, input_buf1, 1, 1);
  //  MATRIXH_CONV = 5;
  //  MATRIXW_CONV = 5;
  for (i = 0; i < 3; i++)
  {
    for (j = 0; j < 3; j++)
    {
      load_weight(weight, weight_buf, i, j);
      for (h = 1; h < MATRIXH_CONV - 1; h++)
      {
#pragma HLS pipeline
        load_data(input, input_buf1, h + i - 1, j);
        for (w = 1; w < MATRIXW_CONV - 1; w++)
        {
#pragma HLS unroll
          // use the pingpong bufers
          if (w % 2 == 1)
          {
            dwcompute(weight_buf, input_buf1, output_buf1);
            store_output(output, output_buf1, h, w);
            load_data(input, input_buf2, h + i - 1, w + j);
          }
          else
          {
            dwcompute(weight_buf, input_buf2, output_buf2);
            store_output(output, output_buf2, h, w);
            load_data(input, input_buf1, h + i - 1, w + j);
          }
        }
      }
    }
  }
}
