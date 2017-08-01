#include "lbf.h"
#include "forest.h"
#include "timer.h"
#include "3rdparty/liblinear/linear.h"
#include <omp.h>
#include <stdio.h>
#include <string>

using namespace std;

/* Assertion */
#undef assert
static void _pwassert(const wchar_t* expression, const wchar_t* file, unsigned line)
{
    printf("PW Assertion failed: %ws\n", expression);
    printf("%ws(%u)\n", file, line);
    volatile int i = 0;
    i = 1 / i;
}
#define assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_pwassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )

#if DATA_SET == SET_BIOID
const string TRAIN_DATA_PATH = ".\\TrainData\\BioID\\";
const string TRAINED_DATA_PATH = ".\\TrainedModel\\BioID\\";
const string TEST_DATA_PATH = ".\\TestData\\BioID\\";
inline string getImagePath(int i)
{
    char fname[256];
    sprintf(fname, "BioID_%04d.pgm", i);
    return TRAIN_DATA_PATH + fname;
}
inline string getShapePath(int i)
{
    char fname[256];
    sprintf(fname, "bioid_%04d.pts", i);
    return TRAIN_DATA_PATH + fname;
}
#elif DATA_SET == SET_FACE_WAREHOUSE
const string TRAIN_DATA_PATH = ".\\TrainData\\FaceWarehouse\\";
const string TRAINED_DATA_PATH = ".\\TrainedModel\\FaceWarehouse\\";
const string TEST_DATA_PATH = ".\\TestData\\FaceWarehouse\\";
inline string getImagePath(int i)
{
    char fname[256];
    string name;
    sprintf_s(fname, "pose_%d.jpg", i);
    return TRAIN_DATA_PATH + fname;
}
inline string getShapePath(int i)
{
    char fname[256];
    sprintf_s(fname, "pose_%d.land", i);
    return TRAIN_DATA_PATH + fname;
}
#else
#error DATA_SET not specified
#endif

static void mean(const Matrix2Df& data, float* res)
{
    float inv = 1.f / (float)data.cols;

    for (int i = 0; i < data.rows; ++i)
    {
        float sum = 0.f;
        float* dt = data[i];
        for (int j = 0; j < data.cols; ++j)
            sum += dt[j];
        res[i] = sum * inv;
    }
}

class LBF
{
public:
    void train();
    void test();
    void run(const string& imgPath);
private:
    void loadTrainingData(Image* trainImages, Matrix2Df& trainShapes, Matrix2Df& regShapes);
    void loadRegressor(Matrix2D<LbfRandomForest>& randomForests, Matrix2D<model*>& models);
    void loadShape(const string& fpath, int width, int height, float* shapes);
    void dumpShapeBin(string& fpath, float* shapes);
    feature_node* genGlobalFeatures(int* localFeatures, int nNonzeroLeaves);
    void ridgeRegression(string fpath, int N, int P, float* _y, const Matrix2Di& _X, float lambda, float* regResult);

    static void noprint(const char*) {}

    float meanShape[2 * NUM_LANDMARKS];
};


int main()
{
    LBF lbf;
    tic();
    //lbf.train();
    toc();
    //lbf.test();
    lbf.run("ym3.jpg");

    return 0;
}

void LBF::train()
{
    int nSample = TRAIN_DATA_SIZE * NUM_AUG;
    Image trainImages[TRAIN_DATA_SIZE];
    Matrix2Df trainShapes(2 * NUM_LANDMARKS, TRAIN_DATA_SIZE);
    Matrix2Df regShapes(2 * NUM_LANDMARKS, TRAIN_DATA_SIZE * NUM_AUG);
    Matrix2Df gtOffsets(2 * NUM_LANDMARKS, TRAIN_DATA_SIZE * NUM_AUG);
    LbfRandomForest randomForests[NUM_LANDMARKS];
    int nonzeroLeaves[NUM_LANDMARKS * NUM_TREES];

    loadTrainingData(trainImages, trainShapes, regShapes);
    for (int stageId = 1; stageId <= NUM_STAGES; ++stageId)
    {
        printf("\n\n==================== Training stage %d ====================\n", stageId);
        // Get 'gtOffset' of this stage
        tic();
        for (int coord = 0; coord < 2 * NUM_LANDMARKS; ++coord)
        {
            float* gtOffsetRow = gtOffsets[coord];
            float* trainShapeRow = trainShapes[coord];
            float* regShapeRow = regShapes[coord];
            for (int i = 0; i < nSample; ++i)
            {
                *gtOffsetRow++ = trainShapeRow[i / NUM_AUG] - (*regShapeRow++);
            }
        }
        toc("update ground-truth offset");

        // Build all forests of this stage
        tic();
        int nLeaves = 0; // for all landmarks
#pragma omp parallel for reduction(+:nLeaves)
        for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
        {
            char fileNameBuf[256];
            sprintf_s(fileNameBuf, "s%02dl%03d.rf", stageId, landmarkId);
            string fpath = TRAINED_DATA_PATH + fileNameBuf;
#ifdef NO_READ_RF
            if (0)
#else
            if (rforests[landmarkId].load(fpath))
#endif
            {
                if (0 == omp_get_thread_num())
                    printf("------ Loading random forest for landmark %d ------\n", landmarkId + 1);
            }
            else
            {
                if (0 == omp_get_thread_num())
                    printf("------ Building random forest for landmark %d ------\n", landmarkId + 1);

                Matrix2Df landRegShapes(regShapes[2 * landmarkId], 2, regShapes.cols);
                Matrix2Df landGtOffsets(gtOffsets[2 * landmarkId], 2, gtOffsets.cols);
                if (0 == omp_get_thread_num()) tic();
                randomForests[landmarkId].build(trainImages, landRegShapes, landGtOffsets);
                if (0 == omp_get_thread_num()) toc();
                randomForests[landmarkId].dump(fpath);
            }
            nLeaves += randomForests[landmarkId].nLeaves;
        }
        printf("total leaves %d\n", nLeaves);
        toc("built all forests in this stage");

        // Get global binary features
        tic();
        Matrix2Di globalFeatures(nSample, NUM_LANDMARKS * NUM_TREES);
        for (int i = 0; i < nSample; ++i)
        {
            printf("\rRun rf get local bin features %d/%d   ", i + 1, nSample);
            Image& img = trainImages[i / NUM_AUG];
            int leafIdOffset = 0;
            for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
            {
                float2 landmark(regShapes[2 * landmarkId][i], regShapes[2 * landmarkId + 1][i]);
                int* _leaves = nonzeroLeaves + landmarkId * NUM_TREES;
                randomForests[landmarkId].run(img, landmark, _leaves);
                for (int leafId = 0; leafId < NUM_TREES; ++leafId)
                    _leaves[leafId] += leafIdOffset;
                leafIdOffset += randomForests[landmarkId].nLeaves;
            }
            memcpy(globalFeatures[i], nonzeroLeaves, sizeof(int) * NUM_LANDMARKS * NUM_TREES);
        }
        printf("\n");
        toc("Run rf get local bin features");

        tic();
        // Regression
#pragma omp parallel for
        for (int coord = 0; coord < 2 * NUM_LANDMARKS; ++coord)
        {
            vector<float> regShapeDelta(nSample);
            float* gtOffsetRow = gtOffsets[coord];
            if (omp_get_thread_num() == 0)
                printf("\rRegressing W %d/%d   ", coord + 1, 2 * NUM_LANDMARKS);

            char fileNameBuf[256];
            sprintf_s(fileNameBuf, "s%02dl%03d.w", stageId, coord);
            string fpath = TRAINED_DATA_PATH + fileNameBuf;
            ridgeRegression(fpath, nSample, nLeaves, gtOffsetRow, globalFeatures, LAMBDA, &regShapeDelta[0]);
            // update shape
            float* regShapeRow = regShapes[coord];
            for (int i = 0; i < nSample; ++i)
                regShapeRow[i] += regShapeDelta[i];
        }
        printf("\n");
        toc("ridge regression");
    }
}

void LBF::test()
{
    Matrix2D<LbfRandomForest> randomForests(NUM_STAGES, NUM_LANDMARKS);
    Matrix2D<model*> models(NUM_STAGES, 2 * NUM_LANDMARKS);
    int nonzeroLeaves[NUM_LANDMARKS * NUM_TREES];
    loadRegressor(randomForests, models);

    Image img;
    float shp[2 * NUM_LANDMARKS];
    for (int i = 0; i < TRAIN_DATA_SIZE; ++i)
    {
        string imagePath = getImagePath(i);
        if (!img.load(imagePath))
        {
            printf("error loading test image %d\n", i);
            continue;
        }
        img.detectFace();
        // reset initial shape
        memcpy(shp, meanShape, sizeof(float) * 2 * NUM_LANDMARKS);
        char testName[256];
        sprintf_s(testName, "test_sample%04ds%02d.png", i, 0);
        img.plotShape(shp);
        img.saveAs(TEST_DATA_PATH + testName);
        sprintf_s(testName, "test_sample%04ds%02d.shp", i, 0);
        dumpShapeBin(TEST_DATA_PATH + testName, shp);

        tic();
        for (int stageId = 0; stageId < NUM_STAGES; ++stageId)
        {
            // regress local binary features
            int leafIdOffset = 0;
            for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
            {
                int* _leaves = nonzeroLeaves + landmarkId * NUM_TREES;
                randomForests[stageId][landmarkId].run(img, float2(shp[2 * landmarkId], shp[2 * landmarkId + 1]), _leaves);
                // id across all forests
                for (int leafId = 0; leafId < NUM_TREES; ++leafId)
                    _leaves[leafId] += leafIdOffset;
                leafIdOffset += randomForests[stageId][landmarkId].nLeaves;
            }

            feature_node* features = genGlobalFeatures(nonzeroLeaves, NUM_LANDMARKS * NUM_TREES);
            for (int coord = 0; coord < 2 * NUM_LANDMARKS; ++coord)
            {
                float delta = (float)predict(models[stageId][coord], features);
                shp[coord] += delta;
            }
            delete[] features;

            sprintf_s(testName, "test_sample%04ds%02d.png", i, stageId + 1);
            img.plotShape(shp);
            img.saveAs(TEST_DATA_PATH + testName);
            sprintf_s(testName, "test_sample%04ds%02d.shp", i, stageId + 1);
            dumpShapeBin(TEST_DATA_PATH + testName, shp);
        }
        toc();
    }
}

void LBF::run(const string & imgPath)
{
    Matrix2D<LbfRandomForest> randomForests(NUM_STAGES, NUM_LANDMARKS);
    Matrix2D<model*> models(NUM_STAGES, 2 * NUM_LANDMARKS);
    int nonzeroLeaves[NUM_LANDMARKS * NUM_TREES];
    loadRegressor(randomForests, models);

    Image img;
    float shp[2 * NUM_LANDMARKS];
    if (!img.load(imgPath))
    {
        printf("error loading image\n");
        return;
    }
    img.detectFace();
    // reset initial shape
    memcpy(shp, meanShape, sizeof(float) * 2 * NUM_LANDMARKS);
    char testName[256];
    sprintf_s(testName, "%ss%02d.png", imgPath.c_str(), 0);
    img.plotShape(shp);
    img.saveAs(TEST_DATA_PATH + testName);
    sprintf_s(testName, "%ss%02d.shp", imgPath.c_str(), 0);
    dumpShapeBin(TEST_DATA_PATH + testName, shp);

    tic();
    for (int stageId = 0; stageId < NUM_STAGES; ++stageId)
    {
        // regress local binary features
        int leafIdOffset = 0;
        Matrix2Df rotation(2, 2);
        float scale;
        {
            float s1[2 * NUM_LANDMARKS];
            float s2[2 * NUM_LANDMARKS];
            float2 center1(0.0f), center2(0.0f);
            Matrix2Df center12(2, 2), center22(2, 2);
            for (int i = 0; i < NUM_LANDMARKS; i++)
            {
                s1[2 * i] = shp[2 * i];
                s1[2 * i + 1] = shp[2 * i + 1];
                s2[2 * i] = meanShape[2 * i];
                s2[2 * i + 1] = meanShape[2 * i + 1];
                center1.x += s1[2 * i];
                center1.y += s1[2 * i + 1];
                center2.x += s2[2 * i];
                center2.y += s2[2 * i + 1];
                center12[0][0] += s1[2 * i] * s1[2 * i];
                center12[0][1] += s1[2 * i] * s1[2 * i + 1];
                center12[1][0] += s1[2 * i] * s1[2 * i + 1];
                center12[1][1] += s1[2 * i + 1] * s1[2 * i + 1];
                center22[0][0] += s2[2 * i] * s2[2 * i];
                center22[0][1] += s2[2 * i] * s2[2 * i + 1];
                center22[1][0] += s2[2 * i] * s2[2 * i + 1];
                center22[1][1] += s2[2 * i + 1] * s2[2 * i + 1];
            }
            center1 = (1.0f / NUM_LANDMARKS) * center1;
            center2 = (1.0f / NUM_LANDMARKS) * center2;
            center12[0][0] /= NUM_LANDMARKS;
            center12[0][1] /= NUM_LANDMARKS;
            center12[1][0] /= NUM_LANDMARKS;
            center12[1][1] /= NUM_LANDMARKS;
            center22[0][0] /= NUM_LANDMARKS;
            center22[0][1] /= NUM_LANDMARKS;
            center22[1][0] /= NUM_LANDMARKS;
            center22[1][1] /= NUM_LANDMARKS;
            for (int i = 0; i < NUM_LANDMARKS; i++)
            {
                s1[2 * i] -= center1.x;
                s1[2 * i + 1] -= center1.y;
                s2[2 * i] -= center2.x;
                s2[2 * i + 1] -= center2.y;
            }

            // Covariance
            Matrix2Df c1(2, 2);
            Matrix2Df c2(2, 2);
            c1[0][0] = center12[0][0] - center1.x * center1.x;
            c1[0][1] = center12[0][1] - center1.x * center1.y;
            c1[1][0] = center12[1][0] - center1.y * center1.x;
            c1[1][1] = center12[1][1] - center1.y * center1.y;
            c2[0][0] = center22[0][0] - center2.x * center2.x;
            c2[0][1] = center22[0][1] - center2.x * center2.y;
            c2[1][0] = center22[1][0] - center2.y * center2.x;
            c2[1][1] = center22[1][1] - center2.y * center2.y;
#define mypow2(n) ((n) * (n))
            float scale1 = 0.0f, scale2 = 0.0f;
            scale1 += mypow2(c1[0][0]);
            scale1 += mypow2(c1[0][1]);
            scale1 += mypow2(c1[1][0]);
            scale1 += mypow2(c1[1][1]);
            scale2 += mypow2(c2[0][0]);
            scale2 += mypow2(c2[0][1]);
            scale2 += mypow2(c2[1][0]);
            scale2 += mypow2(c2[1][1]);
            scale1 /= 4;
            scale2 /= 4;
            scale1 = sqrtf(sqrtf(scale1));
            scale2 = sqrtf(sqrtf(scale2));
#undef mypow2
            scale = scale1 / scale2;
            for (int i = 0; i < NUM_LANDMARKS; i++)
            {
                s1[2 * i] /= scale1;
                s1[2 * i + 1] /= scale1;
                s2[2 * i] /= scale2;
                s2[2 * i + 1] /= scale2;
            }

            // rotation
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < NUM_LANDMARKS; i++)
            {
                num += s1[2 * i + 1] * s2[2 * i] - s1[2 * i] * s2[2 * i + 1];
                den += s1[2 * i] * s2[2 * i] + s1[2 * i + 1] * s2[2 * i + 1];
            }
            float len = sqrtf(num * num + den * den);
            rotation[0][0] = den / len;
            rotation[0][1] = -num / len;
            rotation[1][0] = num / len;
            rotation[1][1] = den / len;
        }
        for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
        {
            float2 mean2regShape;
            mean2regShape.x = scale * (rotation[0][0] * shp[2 * landmarkId] + rotation[0][1] * shp[2 * landmarkId + 1]);
            mean2regShape.y = scale * (rotation[1][0] * shp[2 * landmarkId] + rotation[1][1] * shp[2 * landmarkId + 1]);
            int* _leaves = nonzeroLeaves + landmarkId * NUM_TREES;
            randomForests[stageId][landmarkId].run(img, mean2regShape, _leaves);
            // id across all forests
            for (int leafId = 0; leafId < NUM_TREES; ++leafId)
                _leaves[leafId] += leafIdOffset;
            leafIdOffset += randomForests[stageId][landmarkId].nLeaves;
        }

        feature_node* features = genGlobalFeatures(nonzeroLeaves, NUM_LANDMARKS * NUM_TREES);
        float ds[2 * NUM_LANDMARKS];
        for (int coord = 0; coord < 2 * NUM_LANDMARKS; ++coord)
        {
            float delta = (float)predict(models[stageId][coord], features);
            ds[coord] = delta;
        }
        for (int i = 0; i < NUM_LANDMARKS; ++i)
        {
            float2 tmp;
            tmp.x = scale * (rotation[0][0] * ds[2 * i] + rotation[0][1] * ds[2 * i + 1]);
            tmp.y = scale * (rotation[0][0] * ds[2 * i] + rotation[0][1] * ds[2 * i + 1]);
            ds[2 * i] = tmp.x;
            ds[2 * i + 1] = tmp.y;
        }
        for (int coord = 0; coord < 2 * NUM_LANDMARKS; ++coord)
        {
            shp[coord] += ds[coord];
        }
        delete[] features;

        sprintf_s(testName, "%ss%02d.png", imgPath.c_str(), stageId + 1);
        img.plotShape(shp);
        img.saveAs(TEST_DATA_PATH + testName);
        sprintf_s(testName, "%ss%02d.shp", imgPath.c_str(), stageId + 1);
        dumpShapeBin(TEST_DATA_PATH + testName, shp);
    }
    toc();
}

void LBF::loadTrainingData(Image* trainImages, Matrix2Df& trainShapes, Matrix2Df& regShapes)
{
    string imagePath, shapePath;
    float loadedShape[2 * NUM_LANDMARKS];
    for (int i = 0; i < TRAIN_DATA_SIZE; ++i)
    {
        printf("\rloading train data %04d", i);
        imagePath = getImagePath(i);
        shapePath = getShapePath(i);
        trainImages[i].load(imagePath);
        loadShape(shapePath, trainImages[i].width, trainImages[i].height, loadedShape);
        bool ret = trainImages[i].detectFaceAndNormalize(loadedShape);
        assert(ret);
        trainShapes.column(i) = loadedShape;
    }
    printf("\n");
    mean(trainShapes, meanShape);
    string meanShapeFile = TRAINED_DATA_PATH + "meanshape";
    dumpShapeBin(meanShapeFile, meanShape);
    for (int i = 0; i < TRAIN_DATA_SIZE * NUM_AUG; ++i)
    {
        int idx = randM(TRAIN_DATA_SIZE);
        regShapes.column(i) = trainShapes.column(idx);
    }
}

void LBF::loadRegressor(Matrix2D<LbfRandomForest>& randomForests, Matrix2D<model*>& models)
{
    printf("loading trained regressor...\n");
    string meanShapeFile = TRAINED_DATA_PATH + "meanshape";
    FILE* fp;
    fopen_s(&fp, meanShapeFile.c_str(), "rb");
    int L;
    fread(&L, sizeof(int), 1, fp);
    fread(meanShape, sizeof(float), 2 * NUM_LANDMARKS, fp);
    fclose(fp);

    char fileNameBuf[256];
    for (int stageId = 1; stageId <= NUM_STAGES; ++stageId)
    {
        for (int landmarkId = 0; landmarkId < NUM_LANDMARKS; ++landmarkId)
        {
            sprintf_s(fileNameBuf, "s%02dl%03d.rf", stageId, landmarkId);
            string fpath = TRAINED_DATA_PATH + fileNameBuf;
            bool ret = (randomForests[stageId - 1][landmarkId].load(fpath));
            if (!ret) { printf("error reading rf\n"); exit(-1); }

            sprintf_s(fileNameBuf, "s%02dl%03d.w", stageId, 2 * landmarkId);
            fpath = TRAINED_DATA_PATH + fileNameBuf;
            models[stageId - 1][2 * landmarkId] = load_model(fpath.c_str());
            ret = (models[stageId - 1][2 * landmarkId] == NULL) ? false : true;
            if (!ret) { printf("error reading W\n"); exit(-1); }

            sprintf_s(fileNameBuf, "s%02dl%03d.w", stageId, 2 * landmarkId + 1);
            fpath = TRAINED_DATA_PATH + fileNameBuf;
            models[stageId - 1][2 * landmarkId + 1] = load_model(fpath.c_str());
            ret = (models[stageId - 1][2 * landmarkId + 1] == NULL) ? false : true;
            if (!ret) { printf("error reading W\n"); exit(-1); }

            printf("\rstage %02d/%02d landmark %03d  ", stageId, NUM_STAGES, landmarkId);
        }
    }
    printf("\n");
}

void LBF::loadShape(const string& fpath, int width, int height, float* shapes)
{
    size_t dot = fpath.find_last_of('.');
    if (dot == fpath.npos) { printf("error loading shape\n"); exit(-1); }

    string ext = fpath.substr(dot + 1);
    if (ext == "land")
    {
        FILE* fp;
        fopen_s(&fp, fpath.c_str(), "rt");
        int nL;
        fscanf_s(fp, "%d", &nL);
        for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
        {
            float f;
            fscanf_s(fp, "%f", &f);
            //if ((i & 1) == 0) shp[i] = f * width;
            //else shp[i] = (1.f - f) * height;
            if ((i & 1) == 0) shapes[i] = f;
            else shapes[i] = (height - f);
        }
        fclose(fp);
    }
    else if (ext == "pts")
    {
        FILE* fp;
        fopen_s(&fp, fpath.c_str(), "rt");
        char buf[256];
        fgets(buf, 256, fp); fgets(buf, 256, fp); fgets(buf, 256, fp);
        for (int i = 0; i < 2 * NUM_LANDMARKS; ++i)
            fscanf_s(fp, "%f", shapes + i);
        fclose(fp);
    }
}

void LBF::dumpShapeBin(string & fpath, float * shapes)
{
    FILE* fp;
    fopen_s(&fp, fpath.c_str(), "wb");
    int nLandmark = NUM_LANDMARKS;
    fwrite(&nLandmark, sizeof(int), 1, fp);
    fwrite(shapes, sizeof(float), 2 * NUM_LANDMARKS, fp);
    fclose(fp);
}

feature_node* LBF::genGlobalFeatures(int* localFeatures, int nNonzeroLeaves)
{
    feature_node* feas = new feature_node[nNonzeroLeaves + 1];
    for (int j = 0; j < nNonzeroLeaves; ++j)
    {
        feas[j].index = localFeatures[j] + 1; // !! liblinear index starts from 1
        feas[j].value = 1.0;
    }
    feas[nNonzeroLeaves].index = -1; // mark end
    return feas;
}

void LBF::ridgeRegression(string fpath, int N, int P, float* _y, const Matrix2Di& _X, float lambda, float* regResult)
{
    set_print_string_function(&LBF::noprint); // quiet please
    problem prob;
    prob.l = N; prob.n = P;
    prob.y = new double[N];
    for (int i = 0; i < N; ++i) prob.y[i] = _y[i];
    prob.bias = -1.0;
    prob.x = new feature_node*[N];
    int nNonzeroLeaves = NUM_LANDMARKS * NUM_TREES;
    for (int i = 0; i < N; ++i)
    {
        prob.x[i] = genGlobalFeatures(_X[i], nNonzeroLeaves);
    }
    model* md = NULL;
#ifdef NO_READ_W
    if (1)
#else
    if (!(md = load_model(fpath.c_str())))
#endif
    {
        parameter params;
        params.solver_type = L2R_L2LOSS_SVR;//_DUAL;
        params.C = 0.5 / (double)lambda;
        params.nr_weight = 0;
        params.p = 0.; // => ridge regression
        params.eps = 0.001; // 0.1 for dual?

        const char* error_msg = check_parameter(&prob, &params);
        if (error_msg)
        {
            printf("%s\n", error_msg);
            exit(-1);
        }
        else
        {
            md = ::train(&prob, &params);
            save_model(fpath.c_str(), md);
        }
    }
    if (md)
    {
        for (int i = 0; i < N; ++i)
            regResult[i] = (float)predict(md, prob.x[i]);
        free_and_destroy_model(&md);
    }

    // clean up
    delete[] prob.y;
    for (int i = 0; i < N; ++i)
        delete[] prob.x[i];
    delete[] prob.x;
}
