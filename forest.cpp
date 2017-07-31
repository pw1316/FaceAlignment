#include "forest.h"
#include <omp.h>

static const float lbf_PI = 3.14159265358979323846f;

void LbfRandomForest::build(const Image* trainImages, const Matrix2Df& inputRegShapes, const Matrix2Df& inputGtOffsets)
{
    int nSample = inputRegShapes.cols;
#if DATA_SET == SET_FACE_WAREHOUSE
    float radiusBase = 120.f; //mean_shape(1);
#elif DATA_SET == SET_BIOID
    float radiusBase = 71.f;
#endif

#define NUM_RADIUS 10
    float Rs[NUM_RADIUS];
    float sqrt2 = sqrtf(2.f), sqrt2Inv = 1.f / sqrt2;
    Rs[0] = radiusBase * sqrt2Inv; // largest radius set to sqrt(2)*pupil_distance
    for (int i = 1; i < NUM_RADIUS; ++i)
        Rs[i] = Rs[i - 1] * sqrt2Inv;
    float minOobError = FLT_MAX;
    float minR = Rs[0]; // corresponding radius
    Matrix2Di sampleIdxs(NUM_TREES, nSample);
    LbfTree candidateTrees[NUM_TREES];

    // constant
#ifndef TREE_BOOSTING
    const Matrix2Df& regShapes = inputRegShapes;
    const Matrix2Df& gtOffsets = inputGtOffsets;
#endif
    for (int radiusId = 0; radiusId < NUM_RADIUS; ++radiusId)
    {
        float R = Rs[radiusId];
        if (0 == omp_get_thread_num())
            printf("****** radius = %f ******\n", R);
#ifdef TREE_BOOSTING
        // modifications on shape and offset
        Matrix2Df regShapes = inputRegShapes;
        Matrix2Df gtOffsets = inputGtOffsets;
#endif
        // sample pixels (relative)
        float2 randomPixels[NUM_SAMPLE_PIXELS];
        for (int pixelId = 0; pixelId < NUM_SAMPLE_PIXELS; ++pixelId)
        {
            float r = sqrtf(randf()) * R;
            float theta = randf()*(2.f * lbf_PI);
            randomPixels[pixelId].x = r * cosf(theta);
            randomPixels[pixelId].y = r * sinf(theta);
        }

        Matrix2Di pixelValues(nSample, NUM_SAMPLE_PIXELS);
        for (int i = 0; i < nSample; ++i)
        {
            const Image& img = trainImages[i / NUM_AUG];
            float cx = regShapes[0][i], cy = regShapes[1][i];
            // Normalized value -> real value
            cx = cx * img.faceWidth + img.faceLeftTop.x;
            cy = cy * img.faceHeight + img.faceLeftTop.y;

            for (int pixelId = 0; pixelId < NUM_SAMPLE_PIXELS; ++pixelId)
            {
                int x = clampi(lbf_roundf(randomPixels[pixelId].x + cx), 0, img.width - 1);
                int y = clampi(lbf_roundf(randomPixels[pixelId].y + cy), 0, img.height - 1);
                pixelValues[i][pixelId] = img[y][x]; // same feature even if boosting ?
            }
        }

        for (int treeId = 0; treeId < NUM_TREES; ++treeId)
        {
            if (0 == omp_get_thread_num())
                printf("\rbuilding tree: %04d/%04d   ", treeId + 1, NUM_TREES);

            int* idxs = sampleIdxs[treeId];
            // sample training data with replacement
            for (int i = 0; i < nSample; ++i)
                idxs[i] = randM(nSample);
            candidateTrees[treeId].build(idxs, nSample, gtOffsets, randomPixels, pixelValues);

            // set sampled idxs to -1
            for (int i = 0; i < nSample; ++i)
            {
                if (idxs[i] >= 0)
                {
                    int next = idxs[i], current;
                    while (next >= 0)
                    {
                        current = next;
                        next = idxs[next];
                        idxs[current] = -1;
                    }
                }
            }

#ifdef TREE_BOOSTING
            // update shape and gt_offsets
            for (int i = 0; i < nSample; ++i)
            {
                const Image& img = trainImages[i / NUM_AUG];
                float2 regShape(regShapes[0][i], regShapes[1][i]);
                int leaf;
                float2 offset = candidateTrees[nt].run(img, regShape, &leaf);
                regShapes[0][i] += offset.x;
                regShapes[1][i] += offset.y;
                gtOffsets[0][i] -= offset.x;
                gtOffsets[1][i] -= offset.y;
            }
#endif
        }
        if (0 == omp_get_thread_num())
            printf("\n");

        // compute out-of-bag error
        float oobError = 0.f;
        for (int i = 0; i < nSample; ++i)
        {
            const Image& img = trainImages[i / NUM_AUG];
            float2 regShape(regShapes[0][i], regShapes[1][i]);
            float2 offset(0.f);
            int testCount = 0;
            for (int treeId = 0; treeId < NUM_TREES; ++treeId)
            {
                if (sampleIdxs[treeId][i] == -1) // already included in the tree's training samples
                    continue;
                int leaf;// <not used>
                offset += candidateTrees[treeId].run(img, regShape, &leaf);
                ++testCount;
            }

            // [check] paper says summation, I use average
            // if not boosting, use average, else use summation
#ifndef TREE_BOOSTING
            if (testCount > 0)
            {
                offset.x /= (float)testCount;
                offset.y /= (float)testCount;
            }
#endif
            float2 gtOffset(gtOffsets[0][i], gtOffsets[1][i]); // ground-truth delta
            float2 error = gtOffset - offset;
            oobError += error * error;
        }

        // keep the tree with minimal error
        if (oobError < minOobError)
        {
            // copy trees
            nLeaves = 0;
            for (int treeId = 0; treeId < NUM_TREES; ++treeId)
            {
                this->trees[treeId] = candidateTrees[treeId]; // enough?
                nLeaves += this->trees[treeId].nLeaves;
            }
            minOobError = oobError;
            minR = R;
        }
    }

    if (0 == omp_get_thread_num())
        printf("select Random Forest trained on radius=%f with least oob error\n", minR);
}

float2 LbfRandomForest::run(const Image& img, const float2& landmark, int* reachedLeaves)
{
    float2 offset(0.f);
    int leafIdOffset = 0;
    for (int treeId = 0; treeId < NUM_TREES; ++treeId)
    {
        int leaf;
        offset += this->trees[treeId].run(img, landmark, &leaf);
        reachedLeaves[treeId] = leaf + leafIdOffset;
        leafIdOffset += this->trees[treeId].nLeaves;
    }
#ifdef TREE_BOOSTING
    return offset; // summation
#else
    return (1.f / (float)NUM_TREES) * offset; // average?
#endif
}

bool LbfRandomForest::load(std::string fpath)
{
    FILE* fp;
    if (fopen_s(&fp, fpath.c_str(), "rb")) return false;
    fread(&nLeaves, sizeof(int), 1, fp);
    for (int i = 0; i < NUM_TREES; ++i)
        trees[i].load(fp);
    fclose(fp);
    return true;
}

bool LbfRandomForest::dump(std::string fpath)
{
    FILE* fp;
    if (fopen_s(&fp, fpath.c_str(), "wb")) return false;
    fwrite(&nLeaves, sizeof(int), 1, fp);
    for (int i = 0; i < NUM_TREES; ++i)
        trees[i].dump(fp);
    fclose(fp);
    return true;
}
