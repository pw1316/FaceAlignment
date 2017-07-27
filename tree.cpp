#include "tree.h"

using std::vector;

static int2 getcombination(int idx, int n)
{
    ++idx;
    float b = (float)(n + n - 1);
    int row = (int)ceilf(0.5f*(b - sqrtf(b*b - 8.f*(float)idx)));
    int i = row - 1;
    int j = idx - (n + n - row)*i / 2 + i;
    return int2(i, j);
}

static __forceinline double _Var(double sum, double sum2, double invN)
{
    return fabs(sum2 - invN*sum*sum); // invN
}

// per component
static __forceinline double _Var(const double2& sum, const double2& sum2, double invN)
{
    return _Var(sum.x, sum2.x, invN) + _Var(sum.y, sum2.y, invN);
}

void LbfTree::build(const int * sampleIdxs, int nSamples, const Matrix2Df & gt_offsets, const float2 * pixel_rel_coords, const Matrix2Di & pixelValues)
{
    nodes.clear();
    int leaf = 0;
    root = build(leaf, sampleIdxs, nSamples, gt_offsets, pixel_rel_coords, pixelValues, 0, TREE_DEPTH); // tree_depth 1 means only root or having one-level children?
    nLeaves = leaf;
}

float2 LbfTree::run(const Image& img, const float2& landmark, int* leaf)
{
    LbfNode* plbfnode = &(this->nodes[this->root]);
    while (1)
    {
        if (plbfnode->left == -1) // leaf
        {
            *leaf = plbfnode->leaf.id;
            return float2(plbfnode->leaf.offset.x, plbfnode->leaf.offset.y);
        }
        else
        {
            float2 pt1 = landmark + float2(plbfnode->split.pt1.x, plbfnode->split.pt1.y);
            float2 pt2 = landmark + float2(plbfnode->split.pt2.x, plbfnode->split.pt2.y);
            int col1 = img[clampi(lbf_roundf(pt1.y), 0, img.height - 1)][clampi(lbf_roundf(pt1.x), 0, img.width - 1)];
            int col2 = img[clampi(lbf_roundf(pt2.y), 0, img.height - 1)][clampi(lbf_roundf(pt2.x), 0, img.width - 1)];
            int fea = col1 - col2;
            if (fea <= plbfnode->split.threshold)
                plbfnode = &(this->nodes[plbfnode->left]);
            else
                plbfnode = &(this->nodes[plbfnode->right]);
        }
    }
}

void LbfTree::load(FILE * fp)
{
    fread(&root, sizeof(int), 1, fp);
    fread(&nLeaves, sizeof(int), 1, fp);
	size_t sz;
    fread(&sz, sizeof(size_t), 1, fp); nodes.resize(sz);
    fread(&nodes[0], sizeof(LbfNode)*sz, 1, fp);
}

void LbfTree::dump(FILE * fp)
{
    fwrite(&root, sizeof(int), 1, fp);
    fwrite(&nLeaves, sizeof(int), 1, fp);
    size_t sz = nodes.size();
    fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(&nodes[0], sizeof(LbfNode)*sz, 1, fp);
}

int LbfTree::build(int & leaf_id, const int * sampleIdxs, int Nsamples, const Matrix2Df & gt_offsets, const float2 * pixel_rel_coords, const Matrix2Di & pixelValues, int cur_d, int D)
{
    if (cur_d == D || Nsamples == 1) // leaf
    {
        float offset_x = 0.f, offset_y = 0.f;
        for (int i = 0; i < Nsamples; ++i)
        {
            offset_x += gt_offsets[0][sampleIdxs[i]];
            offset_y += gt_offsets[1][sampleIdxs[i]];
        }
        float inv = 1.f / (float)Nsamples;

        LbfNode node;
        node.leaf.offset.x = inv * offset_x; // average offset of all samples in the leaves
        node.leaf.offset.y = inv * offset_y;
        node.leaf.id = (leaf_id++);
        this->nodes.push_back(node);
        return (int)(this->nodes.size() - 1);
    }

    int M = lbf_C_n_2(NUM_SAMPLE_PIXELS);
    const int m = 500;
    double splitVar = 0.f; // variance reduction
    int split_on_fea = -1; // split on which feature
    float2 split_pt1, split_pt2; // feature by which two pixels (relative position)
    int split_threshold = 0;
    double invN = 1.f / (double)Nsamples;
    vector<int> mFeatures(Nsamples), // a selected features for all samples 
        save_mFeatures(Nsamples);
    vector<int> selectedFeaMap(M, 0); // select m features without replacement

    const int FeatureValuesRange = 255 + 256; // [-255,255]
    int countFeatureValue[FeatureValuesRange];
    double2 offset_sum_by_FeatureValue[FeatureValuesRange];
    double2 offset_sum2_by_FeatureValue[FeatureValuesRange];

    for (int j = 0; j < m; ++j)
    {
        memset(countFeatureValue, 0, sizeof(int)*FeatureValuesRange);
        memset(offset_sum_by_FeatureValue, 0, sizeof(double2)*FeatureValuesRange);
        memset(offset_sum2_by_FeatureValue, 0, sizeof(double2)*FeatureValuesRange);
        // !! sample without replacement
        int feaIdx = randM(M); // [check] select same feature as previous levels?
        while (selectedFeaMap[feaIdx] == -1)
            feaIdx = randM(M);
        selectedFeaMap[feaIdx] = -1;
        int2 feaPx2 = getcombination(feaIdx, NUM_SAMPLE_PIXELS);

        double2 tot_sum(0.), tot_sum2(0.);
        // get features
        for (int i = 0; i < Nsamples; ++i)
        {
            int smpl_idx = sampleIdxs[i];
            int* pxvals = pixelValues[smpl_idx];
            mFeatures[i] = pxvals[feaPx2.x] - pxvals[feaPx2.y];
            double2 oxoy(gt_offsets[0][smpl_idx], gt_offsets[1][smpl_idx]);
            double2 oxoy2 = oxoy*oxoy;
            tot_sum += oxoy;
            tot_sum2 += oxoy2;
            int fv = mFeatures[i] + 255;
            ++countFeatureValue[fv];
            offset_sum_by_FeatureValue[fv] += oxoy;
            offset_sum2_by_FeatureValue[fv] += oxoy2;
        }
        double tot_var = _Var(tot_sum, tot_sum2, invN);

        // count
        for (int fv = 1; fv < FeatureValuesRange; ++fv)
        {
            countFeatureValue[fv] += countFeatureValue[fv - 1];
            offset_sum_by_FeatureValue[fv] += offset_sum_by_FeatureValue[fv - 1];
            offset_sum2_by_FeatureValue[fv] += offset_sum2_by_FeatureValue[fv - 1];
        }

        double maxReduce = 0.f;
        int max_fv = -1;
        // find threshold
        for (int fv = 0; fv < FeatureValuesRange; ++fv)
            if (countFeatureValue[fv] > 0 && countFeatureValue[fv] < Nsamples // not the last feature value (at least one value reserved for the right part)
                && (!fv || countFeatureValue[fv] != countFeatureValue[fv - 1])) // has samples for this value
            {
                double2& left_sum = offset_sum_by_FeatureValue[fv],
                    &left_sum2 = offset_sum2_by_FeatureValue[fv];
                // split include this feature value
                double invL = 1.f / (double)countFeatureValue[fv];
                double invR = 1.f / (double)(Nsamples - countFeatureValue[fv]);
                double2 right_sum = tot_sum - left_sum;
                double2 right_sum2 = tot_sum2 - left_sum2;
                double var1 = _Var(left_sum, left_sum2, invL);
                double var2 = _Var(right_sum, right_sum2, invR);
                double vreduce = tot_var - var1 - var2;
                if (vreduce > maxReduce)
                {
                    maxReduce = vreduce;
                    max_fv = fv;
                }
            }
        if (maxReduce > splitVar)
        {
            splitVar = maxReduce;
            split_on_fea = j;
            split_threshold = max_fv - 255;
            split_pt1 = pixel_rel_coords[feaPx2.x]; // todo: store index?
            split_pt2 = pixel_rel_coords[feaPx2.y];
            save_mFeatures = mFeatures;
        }
    }

    if (split_on_fea < 0) // cannot split, set as leaf
    {
        float offset_x = 0.f, offset_y = 0.f;
        for (int i = 0; i < Nsamples; ++i)
        {
            offset_x += gt_offsets[0][sampleIdxs[i]];
            offset_y += gt_offsets[1][sampleIdxs[i]];
        }
        float inv = 1.f / (float)Nsamples;
        LbfNode node;
        node.leaf.offset.x = inv*offset_x; // average offset of all samples in the leaves
        node.leaf.offset.y = inv*offset_y;
        node.leaf.id = (leaf_id++);
        this->nodes.push_back(node);
        return (int)(this->nodes.size() - 1);
    }

    assert(split_on_fea >= 0);
    // update sample index
    vector<int> nextlevel_idxs(Nsamples);
    int l = 0, r = Nsamples - 1;
    int i = 0;
    while (l <= r)
    {
        if (save_mFeatures[i] <= split_threshold)
            nextlevel_idxs[l++] = sampleIdxs[i];
        else
            nextlevel_idxs[r--] = sampleIdxs[i];
        ++i;
    }

    int* idxL = &nextlevel_idxs[0];
    int* idxR = &nextlevel_idxs[l];
    int left_id = this->build(leaf_id, idxL, l, gt_offsets, pixel_rel_coords, pixelValues, cur_d + 1, D);
    int right_id = this->build(leaf_id, idxR, Nsamples - l, gt_offsets, pixel_rel_coords, pixelValues, cur_d + 1, D);
    // if internal
    LbfNode node(left_id, right_id);
    node.split.pt1.x = split_pt1.x;
    node.split.pt1.y = split_pt1.y;
    node.split.pt2.x = split_pt2.x;
    node.split.pt2.y = split_pt2.y;
    node.split.threshold = split_threshold;
    this->nodes.push_back(node);
    return (int)(this->nodes.size() - 1);
}
