#ifndef __LBF_TREE_H__
#define __LBF_TREE_H__

#include "node.h"
#include "image.h"
#include <vector>

struct LbfTree
{
    std::vector<LbfNode> nodes;
    int root;
    int nLeaves;

    void build(
        const int* sampleIdxs, int nSamples, // part of training samples
        const Matrix2Df& gt_offsets, // GTlandmarks-landmarks [0]: x, [1]: y
        const float2* pixel_rel_coords,
        const Matrix2Di& pixelValues // pixel values at above position(absolute) for all training samples; nsamples X npixels
    );
    float2 run(const Image& img, const float2& landmark, int*);

    void load(FILE* fp);
    void dump(FILE* fp);
private:
    int build(int& leaf_id, // count leaves
        const int* sampleIdxs, int Nsamples,
        const Matrix2Df& gt_offsets,
        const float2* pixel_rel_coords,
        const Matrix2Di& pixelValues,
        int cur_d, int D // current depth, max depth
    );
};

#endif
