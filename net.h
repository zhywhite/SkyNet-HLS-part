#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <stdio.h>

// define the weight and feature map size
#define FM_SIZE 8
#define WEIGHT_SIZE 8
\
#define ACC_SIZE 32

// define the block size channel, height ,width
#define MATRIX_H 16
#define MATRIX_W 16
#define CHANNEL 32

// define data type
typedef ap_uint<FM_SIZE> FIX_FM;
typedef ap_uint<WEIGHT_SIZE> FIX_WT;
typedef ap_uint<ACC_SIZE> FIX_ACC;
typedef ap_uint<512> uint512;


// define the total number of weight and feature map
#define DW_WEIGHT_TOTAL 1000
#define PW_WEIGHT_TOTAL 1000
#define FM_TOTAL 1000

// define the block size after three maxpooling respectively
#define MATRIX_H_M1 22;
#define MATRIX_W_M1 42;
#define MATRIX_H_M2 12;
#define MATRIX_W_M2 22;
#define MATRIX_H_M3 7;
#define MATRIX_W_M3 12;

//total data
extern uint512 dw_weight_all[DW_WEIGHT_TOTAL][3][3];
extern uint512 fm_all[FM_TOTAL * MATRIX_H * MATRIX_W];
extern uint512 pw_weight_all[PW_WEIGHT_TOTAL][CHANNEL];

// functions of dwconv_blk
void dwconv_blk(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_FM output[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_WT weight[CHANNEL][3][3], int MAX_N);

void store_output(FIX_FM output[CHANNEL][MATRIX_H][MATRIX_W],
                  FIX_FM output_buf[CHANNEL], int h, int w);

void load_data(FIX_FM data[CHANNEL][MATRIX_H][MATRIX_W],
               FIX_FM data_buf[CHANNEL], int h, int w);
//    dwconv_PE
void dwcompute(FIX_FM weight[CHANNEL], FIX_FM input[CHANNEL],
               FIX_ACC output[CHANNEL]);

// functions of pwconv_blk

void pwconv_blk(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_ACC output[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_WT weight[CHANNEL][CHANNEL]);

void copy_weight(FIX_WT weight[CHANNEL][CHANNEL], FIX_WT weight_buf[CHANNEL],
                 int num);

void copy_input(FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                FIX_FM input_blk[CHANNEL], int h, int w);

//    pwconv_PE

FIX_ACC pwcompute(FIX_FM fm[CHANNEL], FIX_WT weight[CHANNEL]);

FIX_ACC compute(FIX_WT w0, FIX_FM b0, FIX_WT w1, FIX_FM b1, FIX_WT w2, FIX_FM b2,
               FIX_WT w3, FIX_FM b3, FIX_WT w4, FIX_FM b4, FIX_WT w5, FIX_FM b5,
               FIX_WT w6, FIX_FM b6, FIX_WT w7, FIX_FM b7, FIX_WT w8, FIX_FM b8,
               FIX_WT w9, FIX_FM b9, FIX_WT w10, FIX_FM b10, FIX_WT w11,
               FIX_FM b11, FIX_WT w12, FIX_FM b12, FIX_WT w13, FIX_FM b13,
               FIX_WT w14, FIX_FM b14, FIX_WT w15, FIX_FM b15, int add,
               FIX_ACC output[16]);

void load_weights(FIX_WT weight_buf[16], FIX_WT weight[CHANNEL], int cin);

void load_fm(FIX_WT fm_buf[16], FIX_WT fm[CHANNEL], int cin);

// functions of maxpool_blk
void maxpool_blk(int MAX_N, FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
                 int fmo_index);

void compute_matrix_size(int MAX_N, int MATRIXH_CONV, int MATRIXW_CONV);
//    maxpoolPE
void maxpoolPE(int H, int W, FIX_FM input[CHANNEL][MATRIX_H][MATRIX_W],
               FIX_FM output[CHANNEL][MATRIX_H][MATRIX_W]);

FIX_FM max(FIX_FM in1, FIX_FM in2, FIX_FM in3, FIX_FM in4);

// functions of trans_data

void load_weightdw_from_DDR(FIX_WT weightdw_buf[CHANNEL][3][3],
                            uint512 weightdw[3][3]);

void load_weightpw_from_DDR(FIX_WT weightpw_buf[CHANNEL][CHANNEL],
                            uint512 weightpw[CHANNEL]);

void load_fm_from_DDR(FIX_FM fm[CHANNEL][MATRIX_H][MATRIX_W], uint512 *fm_buf,
                      int buf_id, int MAX_N);

void store_fm_to_DDR(FIX_FM fm[CHANNEL][MATRIX_H][MATRIX_W], uint512 *fm_buf,
                     int buf_id, int MAX_N);
