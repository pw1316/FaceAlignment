#ifndef __LBF_H__
#define __LBF_H__

#define _CRT_SECURE_NO_WARNINGS

#define SET_BIOID 1
#define SET_FACE_WAREHOUSE 2
#define DATA_SET SET_BIOID // select dataset

#if DATA_SET==SET_BIOID
#define TRAIN_DATA_SIZE 30 //1521
#define NUM_LANDMARKS 20
#elif DATA_SET==SET_FACE_WAREHOUSE
const int TRAIN_DATA_SIZE = 60;
const int NUM_LANDMARKS = 74;
#else
#error DATA_SET not specified
#endif

#define NUM_AUG 20
#define NUM_TREES (1200 / NUM_LANDMARKS)
#define TREE_DEPTH 7
#define NUM_STAGES 5
#define NUM_SAMPLE_PIXELS 400
#define LAMBDA 1.0f // regularization strength

#define NO_READ_RF
#define NO_READ_W
//#define DUMP_TRAINING_SHAPES
//#define TREE_BOOSTING

#endif
