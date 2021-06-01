#include "net.h"

// the total feature map and weight
uint512 dw_weight_all[DW_WEIGHT_TOTAL][3][3];
uint512 fm_all[FM_TOTAL * MATRIX_H * MATRIX_W];
uint512 pw_weight_all[PW_WEIGHT_TOTAL][CHANNEL];

// bufs for the block of the feature map
FIX_FM FM_BUF1[CHANNEL][MATRIX_H][MATRIX_W];
FIX_FM FM_BUF2[CHANNEL][MATRIX_H][MATRIX_W];
FIX_FM FM_BUF3[CHANNEL][MATRIX_H][MATRIX_W];
FIX_FM FM_BUF4[CHANNEL][MATRIX_H][MATRIX_W];
FIX_FM FM_BUF_ACC[CHANNEL][MATRIX_H][MATRIX_W];

// bufs for the block of the weight
FIX_WT WT_DW_BUF1[CHANNEL][3][3];
FIX_WT WT_DW_BUF2[CHANNEL][3][3];
FIX_WT WT_PW_BUF[CHANNEL][CHANNEL];

// fmi_index where the input feature map start
// fmo_index where the output feature map start
// pool determine whether do the maxpool
// CI_N the block num of channel_in
// CO_N the block num of channel_out
// row the block num of row
// col the block num if column
// Layer function do the dwconv ,pwconv ,maxpool

void Layer(int pw_weight_index, int dw_weight_index, int fmi_index,
           int fmo_index, int pool, int CI_N, int CO_N, int row, int col,
           int MAX_N)
{

  int i, j, co, ci;
  for (i = 0; i < row; i++)
  {
    for (j = 0; j < col; j++)
    {
      for (co = 0; co < CO_N; co++)
      {
#pragma HLS pipeline
        load_weightpw_from_DDR(WT_PW_BUF, pw_weight_all[pw_weight_index]);
        load_weightdw_from_DDR(WT_DW_BUF1, dw_weight_all[dw_weight_index]);
        load_fm_from_DDR(FM_BUF1, fm_all, fmi_index, MAX_N);
        for (ci = 0; ci < CI_N; ci++)
        {
#pragma HLS unroll
          // use the pingpong bufers
          if (ci % 2 == 0)
          {
            dwconv_blk(FM_BUF1, FM_BUF2, WT3x3_BUF1, MAX_N);
            pwconv_blk(FM_BUF2, FM_BUF_ACC, WT1x1_BUF, MAX_N);
            fmi_index += 1;
            dw_weight_index += 1;
            load_fm_from_DDR(FM_BUF3, fm_all, fmi_index, MAX_N);
            load_weightdw_from_DDR(WT3x3_BUF2, dw_weight_all[dw_weight_index]);
          }
          else
          {
            dwconv_blk(FM_BUF3, FM_BUF2, WT3x3_BUF2, MAX_N);
            pwconv_blk(FM_BUF2, FM_BUF_ACC, WT1x1_BUF, MAX_N);
            fmi_index += 1;
            dw_weight_index += 1;
            load_fm_from_DDR(FM_BUF1, fm_all, fmi_index, MAX_N);
            load_weightdw_from_DDR(WT3x3_BUF1, dw_weight_all[dw_weight_index]);
          }
        }
        if (pool)
          maxpool_blk(MAX_N, FM_BUF_ACC, fmo_index);
        else
          store_fm_to_DDR(FM_BUF_ACC, fm_all, fmo_index, MAX_N);
        fmo_index += 1;
        pw_weight_index += 1;
      }
    }
  }
}
