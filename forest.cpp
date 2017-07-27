#include "forest.h"
#include <omp.h>

static const float lbf_PI = 3.14159265358979323846f;

void LbfRandomForest::build(const Image* trainImages, const Matrix2Df& inputRegShapes, const Matrix2Df& inputGtOffsets)
{
    int Nsamples = inputRegShapes.cols;
    int nFeatures = lbf_C_n_2(NUM_SAMPLE_PIXELS);
#if DATA_SET == SET_FACE_WAREHOUSE
    float pupil_d = 120.f; //mean_shape(1);
#elif DATA_SET == SET_BIOID
    float pupil_d = 71.f;
#endif

#define NUM_RADIUS 10
    float Rs[NUM_RADIUS];
    float sqrt2 = sqrtf(2.f), inv_sqrt2 = 1.f / sqrt2;
    Rs[0] = pupil_d * inv_sqrt2; // largest radius set to sqrt(2)*pupil_distance
    for (int i = 1; i < NUM_RADIUS; ++i)
        Rs[i] = Rs[i - 1] * inv_sqrt2;
    float min_oob_error = FLT_MAX;
    float min_R = Rs[0]; // corresponding radius
    Matrix2Di sampleIdxs(NUM_TREES, Nsamples);
    LbfTree candidate_trees[NUM_TREES];

    // constant
#ifndef TREE_BOOSTING
    const Matrix2Df& landmarks = inputRegShapes;
    const Matrix2Df& gt_offsets = inputGtOffsets;
#endif
    for (int ri = 0; ri < NUM_RADIUS; ++ri)
    {
        float R = Rs[ri];
        if (0 == omp_get_thread_num())
            printf("****** radius = %f ******\n", R);
#ifdef TREE_BOOSTING
        // modifications on shape and offset
        Matrix2Df landmarks = input_landmarks;
        Matrix2Df gt_offsets = input_gt_offsets;
#endif
        // sample pixels (relative)
        float2 pixel_rel_coords[NUM_SAMPLE_PIXELS];
        for (int pi = 0; pi < NUM_SAMPLE_PIXELS; ++pi)
        {
            float r = sqrtf(randf())*R;
            float theta = randf()*(2.f*lbf_PI);
            pixel_rel_coords[pi].x = r*cosf(theta);
            pixel_rel_coords[pi].y = r*sinf(theta);
        }

        Matrix2Di pixelValues(Nsamples, NUM_SAMPLE_PIXELS);
        for (int i = 0; i < Nsamples; ++i)
        {
            const Image& img = trainImages[i / NUM_AUG];
            float cx = landmarks[0][i], cy = landmarks[1][i];

            for (int pi = 0; pi < NUM_SAMPLE_PIXELS; ++pi)
            {
                int x = clampi(lbf_roundf(pixel_rel_coords[pi].x + cx), 0, img.width - 1);
                int y = clampi(lbf_roundf(pixel_rel_coords[pi].y + cy), 0, img.height - 1);
                pixelValues[i][pi] = img[y][x]; // same feature even if boosting ?
            }
        }

        for (int nt = 0; nt < NUM_TREES; ++nt)
        {
            if (0 == omp_get_thread_num())
                printf("\rbuilding tree: %d/%d   ", nt + 1, NUM_TREES);

            int* idxs = sampleIdxs[nt];
            // sample training data with replacement
            for (int i = 0; i < Nsamples; ++i)
                idxs[i] = randM(Nsamples);
            candidate_trees[nt].build(idxs, Nsamples, gt_offsets, pixel_rel_coords, pixelValues);

            // set sampled idxs to -1
            for (int i = 0; i < Nsamples; ++i)
            {
                if (idxs[i] >= 0)
                {
                    int next = idxs[i], save_next;
                    while (next >= 0)
                    {
                        save_next = next;
                        next = idxs[next];
                        idxs[save_next] = -1;
                    }
                }
            }

#ifdef TREE_BOOSTING
            // update shape and gt_offsets
            for (int i = 0; i < Nsamples; ++i)
            {
                const Image& img = Train_Images[i / NUM_AUG];
                float2 landmark(landmarks[0][i], landmarks[1][i]);

                int leaf;
                float2 offset = candidate_trees[nt].run(img, landmark, &leaf);

                landmarks[0][i] += offset.x;
                landmarks[1][i] += offset.y;
                gt_offsets[0][i] -= offset.x;
                gt_offsets[1][i] -= offset.y;
            }
#endif
        }
        if (0 == omp_get_thread_num())
            printf("\n");

        // compute out-of-bag error
        float oob_error = 0.f;
        for (int i = 0; i < Nsamples; ++i)
        {
            const Image& img = trainImages[i / NUM_AUG];
            float2 landmark(landmarks[0][i], landmarks[1][i]);

            float2 offset(0.f);
            int test_tr_cnt = 0;
            for (int nt = 0; nt < NUM_TREES; ++nt)
            {
                if (sampleIdxs[nt][i] == -1) // already included in the tree's training samples
                    continue;

                int leaf;// <not used>
                offset += candidate_trees[nt].run(img, landmark, &leaf);
                ++test_tr_cnt;
            }

            // [check] paper says summation, I use average
            // if not boosting, use average, else use summation
#ifndef TREE_BOOSTING
            if (test_tr_cnt > 0)
            {
                offset.x /= (float)test_tr_cnt;
                offset.y /= (float)test_tr_cnt;
            }
#endif
            float2 gt_delta(gt_offsets[0][i], gt_offsets[1][i]); // ground-truth delta
            float2 error2 = gt_delta - offset;
            oob_error += error2*error2; // sqrt?
        }

        // keep the tree with minimal error
        if (oob_error < min_oob_error)
        {
            // copy trees
            int nl = 0;
            for (int tri = 0; tri < NUM_TREES; ++tri)
            {
                this->trees[tri] = candidate_trees[tri]; // enough?
                nl += this->trees[tri].nLeaves;
            }
            this->nLeaves = nl;
            min_oob_error = oob_error;
            min_R = R;
        }
    }

    if (0 == omp_get_thread_num())
        printf("select Random Forest trained on radius=%f with least oob error\n", min_R);
}

float2 LbfRandomForest::run(const Image& img, const float2& landmark, int* resultLeaves)
{
    float2 offset(0.f);
    int leaf_id_offset = 0;
    for (int ti = 0; ti < NUM_TREES; ++ti)
    {
        int leaf;
        offset += this->trees[ti].run(img, landmark, &leaf);
        resultLeaves[ti] = leaf + leaf_id_offset;
        leaf_id_offset += this->trees[ti].nLeaves;
    }

#ifdef TREE_BOOSTING
    return offset; // summation
#else
    return (1.f / (float)NUM_TREES) * offset; // average?
#endif
}
