#ifndef __LBF_FOREST_H__
#define __LBF_FOREST_H__

#include "tree.h"

struct LbfRandomForest
{
    LbfRandomForest() :nLeaves(0) {}
    LbfTree trees[NUM_TREES];
    int nLeaves; // count leaves

    void build(const Image* trainImages, const Matrix2Df& inputRegShapes, const Matrix2Df& inputGtOffsets);
    float2 run(const Image& img, const float2& landmark, int* reachedLeaves);
    bool load(std::string fpath);
    bool dump(std::string fpath);
};

#endif // !__LBF_FOREST_H__
