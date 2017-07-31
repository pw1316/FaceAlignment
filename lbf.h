#ifndef __LBF_H__
#define __LBF_H__

#define SET_BIOID 1
#define SET_FACE_WAREHOUSE 2
#define DATA_SET SET_FACE_WAREHOUSE // select dataset

#if DATA_SET == SET_BIOID
#define TRAIN_DATA_SIZE 30//30 //1521
#define NUM_LANDMARKS 20
#elif DATA_SET == SET_FACE_WAREHOUSE
#define TRAIN_DATA_SIZE 20 //60
#define NUM_LANDMARKS 74
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
//#define TREE_BOOSTING

#define CASCADE_NAME ".\\3rdparty\\haarcascade_frontalface_alt2.xml"

#endif
