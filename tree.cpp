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

void LbfTree::build(
    const int* sampleIdxs, int nSample,
    const Matrix2Df & gtOffsetsRow,
    const float2 * randomPixels,
    const Matrix2Di & pixelValues
)
{
    nodes.clear();
    int leaf = 0;
    root = build(leaf, sampleIdxs, nSample, gtOffsetsRow, randomPixels, pixelValues, 0, TREE_DEPTH);
    nLeaves = leaf;
}

float2 LbfTree::run(const Image& img, const float2& landmark, int* reachedLeaf)
{
    LbfNode* node = &(this->nodes[this->root]);
    while (1)
    {
        if (node->left == -1) // leaf
        {
            *reachedLeaf = node->leaf.id;
            return float2(node->leaf.offset.x, node->leaf.offset.y);
        }
        else
        {
            float2 pt1 = landmark + float2(node->split.pt1.x, node->split.pt1.y);
            float2 pt2 = landmark + float2(node->split.pt2.x, node->split.pt2.y);
            int color1 = img[clampi(lbf_roundf(pt1.y), 0, img.height - 1)][clampi(lbf_roundf(pt1.x), 0, img.width - 1)];
            int color2 = img[clampi(lbf_roundf(pt2.y), 0, img.height - 1)][clampi(lbf_roundf(pt2.x), 0, img.width - 1)];
            int feature = color1 - color2;
            if (feature <= node->split.threshold)
                node = &(this->nodes[node->left]);
            else
                node = &(this->nodes[node->right]);
        }
    }
}

void LbfTree::load(FILE * fp)
{
    fread(&root, sizeof(int), 1, fp);
    fread(&nLeaves, sizeof(int), 1, fp);
    size_t sz;
    fread(&sz, sizeof(size_t), 1, fp); nodes.resize(sz);
    fread(&nodes[0], sizeof(LbfNode) * sz, 1, fp);
}

void LbfTree::dump(FILE * fp)
{
    fwrite(&root, sizeof(int), 1, fp);
    fwrite(&nLeaves, sizeof(int), 1, fp);
    size_t sz = nodes.size();
    fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(&nodes[0], sizeof(LbfNode) * sz, 1, fp);
}

int LbfTree::build(
    int& leafId,
    const int* sampleIdxs, int nSample,
    const Matrix2Df& gtOffsetsRow,
    const float2* randomPixels,
    const Matrix2Di& pixelValues,
    int curDepth, int maxDepth
)
{
    if (curDepth == maxDepth || nSample == 1) // leaf
    {
        float2 offset(0.0f);
        for (int i = 0; i < nSample; ++i)
        {
            offset.x += gtOffsetsRow[0][sampleIdxs[i]];
            offset.y += gtOffsetsRow[1][sampleIdxs[i]];
        }
        float inv = 1.f / (float)nSample;

        LbfNode node;
        node.leaf.offset.x = inv * offset.x; // average offset of all samples in the leaves
        node.leaf.offset.y = inv * offset.y;
        node.leaf.id = (leafId++);
        this->nodes.push_back(node);
        return (int)(this->nodes.size() - 1);
    }

    int nSamplePixelPair = lbf_C_n_2(NUM_SAMPLE_PIXELS);
    const int totalTurns = 500;
    double splitVariance = 0.f;
    int splitTurn = -1; // split on which feature
    float2 splitPt1, splitPt2; // feature by which two pixels (relative position)
    int splitThreshold = 0;
    double invN = 1.f / (double)nSample;
    vector<int> features(nSample); // features for each turn
    vector<int> splitFeatures(nSample);
    vector<int> selectedPixelPairMap(nSamplePixelPair, 0); // select m features without replacement

    const int featureRange = 255 + 256; // [-255,255]
    int featureCount[featureRange];
    double2 featureOffsetSum[featureRange];
    double2 featureOffsetSum2[featureRange];

    for (int turnId = 0; turnId < totalTurns; ++turnId)
    {
        memset(featureCount, 0, sizeof(int) * featureRange);
        memset(featureOffsetSum, 0, sizeof(double2) * featureRange);
        memset(featureOffsetSum2, 0, sizeof(double2) * featureRange);
        // !! sample without replacement
        int pixelPairIndex = randM(nSamplePixelPair); // [check] select same feature as previous levels?
        while (selectedPixelPairMap[pixelPairIndex] == -1)
            pixelPairIndex = randM(nSamplePixelPair);
        selectedPixelPairMap[pixelPairIndex] = -1;
        int2 pixelPair = getcombination(pixelPairIndex, NUM_SAMPLE_PIXELS);

        double2 offsetSum(0.), offsetSum2(0.);
        // get features
        for (int i = 0; i < nSample; ++i)
        {
            int sampleIndex = sampleIdxs[i];
            int* samplePixelValues = pixelValues[sampleIndex];
            features[i] = samplePixelValues[pixelPair.x] - samplePixelValues[pixelPair.y];
            double2 offset(gtOffsetsRow[0][sampleIndex], gtOffsetsRow[1][sampleIndex]);
            double2 offset2 = offset*offset;
            offsetSum += offset;
            offsetSum2 += offset2;
            int fv = features[i] + 255;
            ++featureCount[fv];
            featureOffsetSum[fv] += offset;
            featureOffsetSum2[fv] += offset2;
        }
        double totalVariance = _Var(offsetSum, offsetSum2, invN);

        // count
        for (int fv = 1; fv < featureRange; ++fv)
        {
            featureCount[fv] += featureCount[fv - 1];
            featureOffsetSum[fv] += featureOffsetSum[fv - 1];
            featureOffsetSum2[fv] += featureOffsetSum2[fv - 1];
        }

        double maxVarianceReduce = 0.f;
        int maxVarianceReduceFeature = -1;
        // find threshold
        for (int fv = 0; fv < featureRange; ++fv)
            if (featureCount[fv] > 0 && featureCount[fv] < nSample // not the last feature value (at least one value reserved for the right part)
                && (!fv || featureCount[fv] != featureCount[fv - 1])) // has samples for this value
            {
                double2& leftSum = featureOffsetSum[fv],
                    &leftSum2 = featureOffsetSum2[fv];
                // split include this feature value
                double invLeft = 1.f / (double)featureCount[fv];
                double invRight = 1.f / (double)(nSample - featureCount[fv]);
                double2 rightSum = offsetSum - leftSum;
                double2 rightSum2 = offsetSum2 - leftSum2;
                double leftVariance = _Var(leftSum, leftSum2, invLeft);
                double rightVariance = _Var(rightSum, rightSum2, invRight);
                double varianceReduce = totalVariance - leftVariance - rightVariance;
                if (varianceReduce > maxVarianceReduce)
                {
                    maxVarianceReduce = varianceReduce;
                    maxVarianceReduceFeature = fv;
                }
            }
        if (maxVarianceReduce > splitVariance)
        {
            splitVariance = maxVarianceReduce;
            splitTurn = turnId;
            splitThreshold = maxVarianceReduceFeature - 255;
            splitPt1 = randomPixels[pixelPair.x]; // todo: store index?
            splitPt2 = randomPixels[pixelPair.y];
            splitFeatures = features;
        }
    }

    if (splitTurn < 0) // cannot split, set as leaf
    {
        float2 offset(0.0f);
        for (int i = 0; i < nSample; ++i)
        {
            offset.x += gtOffsetsRow[0][sampleIdxs[i]];
            offset.y += gtOffsetsRow[1][sampleIdxs[i]];
        }
        float inv = 1.f / (float)nSample;
        LbfNode node;
        node.leaf.offset.x = inv * offset.x; // average offset of all samples in the leaves
        node.leaf.offset.y = inv * offset.y;
        node.leaf.id = (leafId++);
        this->nodes.push_back(node);
        return (int)(this->nodes.size() - 1);
    }

    assert(splitTurn >= 0);
    // update sample index
    vector<int> nextSampleIdxs(nSample);
    int l = 0, r = nSample - 1;
    int i = 0;
    while (l <= r)
    {
        if (splitFeatures[i] <= splitThreshold)
            nextSampleIdxs[l++] = sampleIdxs[i];
        else
            nextSampleIdxs[r--] = sampleIdxs[i];
        ++i;
    }

    int* idxL = &nextSampleIdxs[0];
    int* idxR = &nextSampleIdxs[l];
    int left = this->build(leafId, idxL, l, gtOffsetsRow, randomPixels, pixelValues, curDepth + 1, maxDepth);
    int right = this->build(leafId, idxR, nSample - l, gtOffsetsRow, randomPixels, pixelValues, curDepth + 1, maxDepth);
    // if internal
    LbfNode node(left, right);
    node.split.pt1.x = splitPt1.x;
    node.split.pt1.y = splitPt1.y;
    node.split.pt2.x = splitPt2.x;
    node.split.pt2.y = splitPt2.y;
    node.split.threshold = splitThreshold;
    this->nodes.push_back(node);
    return (int)(this->nodes.size() - 1);
}
