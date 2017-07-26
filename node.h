#ifndef __LBF_NODE_H__
#define __LBF_NODE_H__

#include "struct.hpp"

struct LbfNode
{
    LbfNode(int l, int r) :left(l), right(r) {}
    LbfNode() :left(-1), right(-1) {}

    int left, right;
    union
    {
        struct
        {
            float2 offset; // average offset for all training samples in the leaf
            int id;
        }leaf;
        struct
        {
            float2 pt1, pt2;
            int threshold;
        }split;
    };
};

#endif
