#include "net.h"

// compute the maxpool and store the result to DDR
void maxpool_blk(int MAX_N, FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                 int fmo_index) {

  FIX_FM output[CHANNEL][MATRIX_H][MATRIX_W];
  int MATRIXH_CONV, MATRIXW_CONV;
  compute_matrix_size(MAX_N, MATRIXH_CONV, MATRIXW_CONV);
  maxpoolPE(MATRIXH_CONV,MATRIXW_CONV,input,output);
  store_fm_to_DDR(output, fm_all, fmo_index, MAX_N);
}

// compute the current matrix size 
void compute_matrix_size(int MAX_N, int MATRIXH_CONV, int MATRIXW_CONV) {
  if (MAX_N == 1) {
    MATRIXH_CONV = MATRIX_H_M1;
    MATRIXW_CONV = MATRIX_W_M1;
  }

  else if (MAX_N == 2) {
    MATRIXH_CONV = MATRIX_H_M2;
    MATRIXW_CONV = MATRIX_W_M2;
  }

  else if (MAX_N == 3) {
    MATRIXH_CONV = MATRIX_H_M3;
    MATRIXW_CONV = MATRIX_W_M3;
  } else if (MAX_N == 0) {
    MATRIXH_CONV = MATRIX_H;
    MATRIXW_CONV = MATRIX_W;
  }
}
