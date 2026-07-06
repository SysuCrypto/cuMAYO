#ifndef MAYO_PARAMS_H
#define MAYO_PARAMS_H

#include <cstdint>

#ifndef MAYO_VERSION
#error "MAYO_VERSION =1,2,3,5"
#endif

#if   MAYO_VERSION == 1

#define MAYO_NAME       "MAYO_1"
#define MAYO_N          86
#define MAYO_M          78
#define MAYO_Menc       39
#define MAYO_O          8
#define MAYO_K          10
#define MAYO_SK_BYTES   24
#define MAYO_PK_BYTES   1420
#define MAYO_P_BYTES    144495
#define MAYO_O_BYTES    312
#define MAYO_P1_BYTES   120159
#define MAYO_P2_BYTES   24336
#define MAYO_DIGEST_BYTES 32
#define MAYO_Pair 55
#define MAYO_POLY_C0 0x8
#define MAYO_POLY_C1 0x1
#define MAYO_POLY_C2 0x1
#define MAYO_POLY_C3 0x0

#elif MAYO_VERSION == 2

#define MAYO_NAME       "MAYO_2"
#define MAYO_N          81
#define MAYO_M          64
#define MAYO_Menc       32
#define MAYO_O          17
#define MAYO_K          4
#define MAYO_SK_BYTES   24
#define MAYO_PK_BYTES   4912
#define MAYO_P_BYTES    101376
#define MAYO_O_BYTES    544
#define MAYO_P1_BYTES   66560
#define MAYO_P2_BYTES   34816
#define MAYO_DIGEST_BYTES 32
#define MAYO_Pair 10
#define MAYO_POLY_C0 0x8
#define MAYO_POLY_C1 0x0
#define MAYO_POLY_C2 0x2
#define MAYO_POLY_C3 0x8

#elif MAYO_VERSION == 3

#define MAYO_NAME       "MAYO_3"
#define MAYO_N          118
#define MAYO_M          108
#define MAYO_Menc       54
#define MAYO_O          10
#define MAYO_K          11
#define MAYO_SK_BYTES   32
#define MAYO_PK_BYTES   2986
#define MAYO_P_BYTES    376164
#define MAYO_O_BYTES    540
#define MAYO_P1_BYTES   317844
#define MAYO_P2_BYTES   58320
#define MAYO_DIGEST_BYTES 48
#define MAYO_Pair 66
#define MAYO_POLY_C0 0x8
#define MAYO_POLY_C1 0x0
#define MAYO_POLY_C2 0x1
#define MAYO_POLY_C3 0x7

#elif MAYO_VERSION == 5

#define MAYO_NAME       "MAYO_5"
#define MAYO_N          154
#define MAYO_M          142
#define MAYO_Menc       71
#define MAYO_O          12
#define MAYO_K          12
#define MAYO_SK_BYTES   40
#define MAYO_PK_BYTES   5554
#define MAYO_P_BYTES    841847
#define MAYO_O_BYTES    852
#define MAYO_P1_BYTES   720863
#define MAYO_P2_BYTES   120984
#define MAYO_DIGEST_BYTES 64
#define MAYO_Pair 78
#define MAYO_POLY_C0 0x4
#define MAYO_POLY_C1 0x0
#define MAYO_POLY_C2 0x8
#define MAYO_POLY_C3 0x1

#else
#error "MAYO_VERSION =1,2,3,5"
#endif

#endif // MAYO_PARAMS_H
