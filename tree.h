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
        const int* sampleIdxs, int nSample, // part of training samples
        const Matrix2Df& gtOffsetsRow, // GTlandmarks-landmarks [0]: x, [1]: y
        const float2* randomPixels,
        const Matrix2Di& pixelValues // pixel values at above position(absolute) for all training samples; nsamples X npixels
    );
    float2 run(const Image& img, const float2& landmark, int* reachedLeaf);

    void load(FILE* fp);
    void dump(FILE* fp);
private:
    int build(int& leafId, // count leaves
        const int* sampleIdxs, int nSample,
        const Matrix2Df& gtOffsetsRow,
        const float2* randomPixels,
        const Matrix2Di& pixelValues,
        int curDepth, int maxDepth
    );
};

#endif
