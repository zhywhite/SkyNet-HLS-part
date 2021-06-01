#include "net.h"

// load FIX_WT bit weigthdw from 512 bit weightdw_all
// a total CHANNEL weightdw stored in one 512 bit weightdw_all
void load_weightdw_from_DDR(FIX_WT weightdw_buf[CHANNEL][3][3],
                            uint512 weightdw[3][3]) {
  int m, n, c;
  uint512 DATA = 0;
  for (m = 0; m < 3; m++) {
    for (n = 0; n < 3; n++) {
#pragma HLS pipeline
      DATA.range(511, 0) = weightdw[m][n].range(511, 0);
      for (c = 0; c < CHANNEL; c++) {
#pragma HLS unroll
        weightdw_buf[c][m][n].range(WEIGHT_SIZE-1, 0) =
            DATA.range(WEIGHT_SIZE + c * WEIGHT_SIZE-1, c * WEIGHT_SIZE);
      }
    }
  }
}

// load FIX_WT bit weigthpw from 512 bit weightpw_all
// a total CHANNEL weightpw stored in one 512 bit weightpw_all
void load_weightpw_from_DDR(FIX_WT weightpw_buf[CHANNEL][CHANNEL],
                            uint512 weightpw[CHANNEL]) {
  int ci, co;
  uint512 DATA = 0;
  for (ci = 0; ci < CHANNEL; ci++) {
#pragma HLS pipeline
    DATA.range(511, 0) = weightpw[ci].range(511, 0);
    for (co = 0; co < CHANNEL; co++) {
#pragma HLS unroll
      weightpw_buf[ci][co].range(WEIGHT_SIZE-1, 0) =
          DATA.range(WEIGHT_SIZE + WEIGHT_SIZE * co-1, WEIGHT_SIZE * co);
    }
  }
}

//  load the feature map
void load_fm_from_DDR(FIX_FM fm[CHANNEL][MATRIX_H][MATRIX_W], uint512 *fm_buf,
                      int buf_id, int MAX_N) {
  uint512 *fm_buf_ptr = fm_buf + buf_id * MATRIX_W * MATRIX_H;
  int h, w, c;
  int MATRIXH_CONV, MATRIXW_CONV;
  compute_matrix_size(MAX_N, MATRIXH_CONV, MATRIXW_CONV);
  uint512 DATA = 0;
  for (h = 0; h < MATRIXH_CONV; h++) {
    for (w = 0; w < MATRIXW_CONV; w++) {
#pragma HLS pipeline
      DATA = fm_buf_ptr[w];
      for (c = 0; c < CHANNEL; c++) {
#pragma HLS unroll
        fm[c][h][w].range(FM_SIZE-1, 0) =
            DATA.range(FM_SIZE + c * FM_SIZE-1, c * FM_SIZE);
      }
    }
    fm_buf_ptr += MATRIXW_CONV;
  }
}


//  store the feature map
void store_fm_to_DDR(FIX_FM fm[CHANNEL][MATRIX_H][MATRIX_W], uint512 *fm_buf,
                     int buf_id, int MAX_N) {
  uint512 *fm_buf_ptr = fm_buf + buf_id * MATRIX_H * MATRIX_W;
  int c, h, w;
  uint512 DATA = 0;
  int MATRIXH_CONV, MATRIXW_CONV;
  compute_matrix_size(MAX_N, MATRIXH_CONV, MATRIXW_CONV);
  for (h = 0; h < MATRIXH_CONV; h++) {
    for (w = 0; w < MATRIXW_CONV; w++) {
#pragma HLS pipeline
      DATA = fm_buf_ptr[w];
      for (c = 0; c < CHANNEL; c++) {
#pragma HLS unroll
        fm[c][h][w].range(FM_SIZE-1, 0) =
            DATA.range(FM_SIZE + c * FM_SIZE-1, c * FM_SIZE);
      }
    }
    fm_buf_ptr += MATRIXW_CONV;
  }
}
