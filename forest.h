#ifndef __LBF_FOREST_H__
#define __LBF_FOREST_H__

#include "tree.h"

struct LbfRandomForest
{
    LbfRandomForest() :nLeaves(0) {}
    LbfTree trees[NUM_TREES];
    int nLeaves; // count leaves

    void build(const Image* Train_Images, const Matrix2Df& landmarks, const Matrix2Df& gt_offsets);
    // output int*NUM_TREES: ids of reached leaves in all trees
    float2 run(const Image& img, const float2& landmark, int* output_leaves);

    bool load(std::string fpath)
    {
		FILE* fp;
        if (fopen_s(&fp, fpath.c_str(), "rb")) return false;
        fread(&nLeaves, sizeof(int), 1, fp);
        for (int i = 0; i < NUM_TREES; ++i)
            trees[i].load(fp);
        fclose(fp);
        return true;
    }
    bool dump(std::string fpath)
    {
		FILE* fp;
        if (fopen_s(&fp, fpath.c_str(), "wb")) return false;
        fwrite(&nLeaves, sizeof(int), 1, fp);
        for (int i = 0; i < NUM_TREES; ++i)
            trees[i].dump(fp);
        fclose(fp);
        return true;
    }
};

#endif // !__LBF_FOREST_H__
